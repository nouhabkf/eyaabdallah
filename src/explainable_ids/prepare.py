"""Nettoyage et découpages, sans entraînement.

Les CSV bruts sont lus en mode binaire. Les fichiers dérivés sont écrits
sous ``C:/data/cic-ids2017``, jamais dans ``raw`` et jamais sous OneDrive.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from explainable_ids.config import PROJECT_ROOT, resolve_paths

FILES = [
    ("Monday-WorkingHours.pcap_ISCX.csv", "Monday"),
    ("Tuesday-WorkingHours.pcap_ISCX.csv", "Tuesday"),
    ("Wednesday-workingHours.pcap_ISCX.csv", "Wednesday"),
    ("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "Thursday-Morning"),
    ("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "Thursday-Afternoon"),
    ("Friday-WorkingHours-Morning.pcap_ISCX.csv", "Friday-Morning"),
    ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "Friday-Afternoon-PortScan"),
    ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "Friday-Afternoon-DDoS"),
]
MISSING_TOKENS = {"", "NaN", "nan", "NA", "N/A", "n/a", "NULL", "null", "None"}
CHUNK_ROWS = 50_000
EXPECTED_RAW_ROWS = 2_830_743
EXPECTED_CONTRADICTORY_GROUPS = 698
EXPECTED_CONTRADICTORY_ROWS = 7_020
EXPECTED_KEPT_ROWS = 2_520_966
EXPECTED_CROSS_SET_CONTENTS_BEFORE_EXCLUSION = 28_024


def assert_raw_csv(path: Path, raw_dir: Path) -> None:
    if path.resolve().parent != raw_dir.resolve() or path.suffix.lower() != ".csv":
        raise RuntimeError(f"Lecture brute refusée : {path}")


def assert_derived_target(path: Path, raw_dir: Path) -> None:
    resolved = path.resolve()
    raw = raw_dir.resolve()
    if resolved == raw or raw in resolved.parents:
        raise RuntimeError(f"Écriture refusée dans les CSV bruts : {path}")
    if "onedrive" in str(resolved).lower():
        raise RuntimeError(f"Écriture refusée sous OneDrive : {path}")


def fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def strip_header(fields: list[str]) -> list[str]:
    return [field.strip() for field in fields]


def locate_fwd_header(names: list[str]) -> tuple[int, int]:
    indexes = [index for index, name in enumerate(names) if name == "Fwd Header Length"]
    if len(indexes) != 2:
        raise RuntimeError(
            f"La colonne Fwd Header Length apparaît {len(indexes)} fois après nettoyage des noms."
        )
    return indexes[0], indexes[1]


def feature_names(names: list[str], drop_index: int) -> list[str]:
    kept = [name for index, name in enumerate(names) if index not in {drop_index, len(names) - 1}]
    if len(kept) != len(set(kept)):
        raise RuntimeError("Des noms de caractéristiques restent en double après suppression.")
    return kept


def parse_feature(field: str) -> tuple[float | None, str]:
    """Retourne la valeur float32 et le motif : number, missing ou infinity."""
    if field in MISSING_TOKENS:
        return None, "missing"
    try:
        number = float(field)
    except ValueError as exc:
        raise RuntimeError(f"Valeur numérique illisible : {field!r}") from exc
    if math.isnan(number):
        return None, "missing"
    if math.isinf(number):
        return None, "infinity"
    return float(np.float32(number)), "number"


def value_changed_by_float32(field: str) -> bool:
    if field in MISSING_TOKENS:
        return False
    number = float(field)
    if math.isnan(number) or math.isinf(number):
        return False
    return float(np.float32(number)) != number


def binary_target(label: str) -> int:
    return 0 if label == "BENIGN" else 1


def contradictory_feature_digests(feature_labels: dict[bytes, str]) -> set[bytes]:
    return {digest for digest, label in feature_labels.items() if label == "*"}


class ChronologicalKeeper:
    """Exclut un profil contradictoire, puis ne garde que la première ligne exacte."""

    def __init__(self, contradictory: set[bytes]) -> None:
        self.contradictory = contradictory
        self.seen: dict[bytes, str] = {}
        self.cross_contents: set[bytes] = set()
        self.contradictory_exact_splits: dict[bytes, set[str]] = {}
        self.attack_cross = 0
        self.contradictory_rows = 0
        self.later_copies = 0

    def keep(self, feature_digest: bytes, exact_digest: bytes, label: str, split: str) -> bool:
        if feature_digest in self.contradictory:
            self.contradictory_rows += 1
            self.contradictory_exact_splits.setdefault(exact_digest, set()).add(split)
            return False
        previous = self.seen.get(exact_digest)
        if previous is not None:
            self.later_copies += 1
            if previous != split:
                self.cross_contents.add(exact_digest)
                if label != "BENIGN":
                    self.attack_cross += 1
            return False
        self.seen[exact_digest] = split
        return True


def grouped_allocation(count: int) -> tuple[int, int, int]:
    held_out = count * 15 // 100
    return count - 2 * held_out, held_out, held_out


def assign_grouped_indices(labels: list[str], seed: int = 42) -> dict[str, np.ndarray]:
    """Mélange reproductible à l'intérieur de chaque étiquette d'origine.

    Un seul ``random.Random(seed)`` avance dans l'ordre Unicode des étiquettes.
    Les positions d'entrée sont l'ordre chronologique des lignes conservées.
    """
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    rng = random.Random(seed)
    assigned = {"train": [], "validation": [], "test": []}
    for label in sorted(grouped):
        positions = grouped[label]
        rng.shuffle(positions)
        n_train, n_validation, _n_test = grouped_allocation(len(positions))
        assigned["train"].extend(positions[:n_train])
        assigned["validation"].extend(positions[n_train : n_train + n_validation])
        assigned["test"].extend(positions[n_train + n_validation :])
    return {name: np.sort(np.asarray(values, dtype=np.int64)) for name, values in assigned.items()}


def _assert_known_counts(stats: dict[str, Any]) -> None:
    expected = {
        "train": {
            "normal": 1_324_353,
            "attack": 202_780,
            "families": {
                "DoS GoldenEye": 10_286,
                "DoS Hulk": 172_719,
                "DoS Slowhttptest": 5_228,
                "DoS slowloris": 5_384,
                "FTP-Patator": 5_933,
                "Heartbleed": 11,
                "SSH-Patator": 3_219,
            },
        },
        "validation": {
            "normal": 395_931,
            "attack": 2_179,
            "families": {
                "Infiltration": 36,
                "Web Attack \uFFFD Brute Force": 1_470,
                "Web Attack \uFFFD Sql Injection": 21,
                "Web Attack \uFFFD XSS": 652,
            },
        },
        "test": {
            "normal": 375_502,
            "attack": 220_221,
            "families": {
                "Bot": 1_953,
                "DDoS": 128_013,
                "PortScan": 90_255,
            },
        },
    }
    for split, target in expected.items():
        counts = stats["class_counts"][split]
        if counts["normal"] != target["normal"] or counts["attack"] != target["attack"]:
            raise RuntimeError(f"Effectifs inattendus pour {split} : {counts}.")
        families = {label: count for label, count in stats["family_counts"][split].items() if label != "BENIGN"}
        if families != target["families"]:
            raise RuntimeError(f"Familles inattendues pour {split} : {families}.")


def chronological_split(day: str, config: dict[str, Any]) -> str:
    section = config["chronological_analysis"]
    if day in section["train_days"]:
        return "train"
    if day in section["validation_days"]:
        return "validation"
    if day in section["test_days"]:
        return "test"
    raise RuntimeError(f"Jour hors protocole : {day}")


def _iter_data_lines(path: Path, raw_dir: Path):
    assert_raw_csv(path, raw_dir)
    with path.open("rb") as handle:
        header_line = handle.readline().rstrip(b"\r\n").decode("utf-8")
        for line_number, line in enumerate(handle, start=1):
            raw = line.rstrip(b"\r\n")
            if not raw:
                continue
            yield line_number, raw, header_line


def run_preparation(config: dict[str, Any]) -> dict[str, Any]:
    if config["phases"]["data_preparation"] != "open":
        raise RuntimeError("Phase « data_preparation » non ouverte.")
    if config["phases"]["training"] != "blocked":
        raise RuntimeError("L'entraînement doit rester fermé pendant la préparation.")
    paths = resolve_paths(config)
    raw_dir = paths["raw"]
    interim = paths["interim"]
    interim.mkdir(parents=True, exist_ok=True)
    for directory in (paths["chronological_splits"], paths["grouped_splits"]):
        directory.mkdir(parents=True, exist_ok=True)
        assert_derived_target(directory, raw_dir)
    output = interim / "flows.parquet"
    temporary = interim / "flows.parquet.partial"
    assert_derived_target(output, raw_dir)
    assert_derived_target(temporary, raw_dir)

    before = {name: fingerprint(raw_dir / name) for name, _day in FILES}
    header, first_index, drop_index, names = _inspect_header(raw_dir)
    identities, mismatches = _scan_identities(raw_dir, first_index, drop_index)
    contradictory = contradictory_feature_digests(identities)
    del identities
    if mismatches:
        raise RuntimeError(
            f"Fwd Header Length diffère sur {mismatches} lignes. La seconde colonne n'est pas supprimée."
        )
    group_count = len(contradictory)
    if group_count != EXPECTED_CONTRADICTORY_GROUPS:
        raise RuntimeError(f"Profils contradictoires : {group_count}, attendu {EXPECTED_CONTRADICTORY_GROUPS}.")

    stats = _write_cleaned_table(
        config=config,
        raw_dir=raw_dir,
        destination=temporary,
        feature_columns=names,
        drop_index=drop_index,
        contradictory=contradictory,
    )
    if stats["contradictory_rows"] != EXPECTED_CONTRADICTORY_ROWS:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Lignes contradictoires exclues : {stats['contradictory_rows']}, attendu {EXPECTED_CONTRADICTORY_ROWS}."
        )
    if stats["kept_rows"] != EXPECTED_KEPT_ROWS:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Lignes conservées : {stats['kept_rows']}, attendu {EXPECTED_KEPT_ROWS}.")
    spanned_by_contradiction = stats["contradictory_exact_contents_spanning_sets"]
    if stats["cross_set_contents_blocked"] + spanned_by_contradiction != EXPECTED_CROSS_SET_CONTENTS_BEFORE_EXCLUSION:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Les contenus exacts qui traversent plusieurs ensembles ne se réconcilient pas avec le contrôle validé : "
            f"{stats['cross_set_contents_blocked']} + {spanned_by_contradiction}."
        )
    if stats["attack_contents_crossing_files"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Un contenu d'attaque exact traverse plusieurs fichiers.")
    temporary.replace(output)
    try:
        _assert_known_counts(stats)
        audit = _audit_parquet(output, names, contradictory)
        if audit["rows"] != stats["kept_rows"] or audit["infinite_cells"] or audit["contradictory_profiles_remaining"]:
            raise RuntimeError("L'audit du fichier dérivé a échoué.")
    except Exception:
        output.unlink(missing_ok=True)
        raise
    grouped = assign_grouped_indices(stats["label_originals"], seed=int(config["split"]["seed"]))
    _assert_grouped_contract(stats["label_originals"], grouped)
    chronological = _chronological_indices(stats["chronological_splits"])
    _write_indices(paths["chronological_splits"], chronological)
    _write_indices(paths["grouped_splits"], grouped)
    after = {name: fingerprint(raw_dir / name) for name, _day in FILES}
    if after != before:
        output.unlink(missing_ok=True)
        raise RuntimeError("L'empreinte d'un CSV brut a changé pendant la préparation.")

    manifest = _manifest(
        config=config,
        header=header,
        feature_columns=names,
        stats=stats,
        audit=audit,
        before=before,
        chronological=chronological,
        grouped=grouped,
        paths=paths,
        output=output,
    )
    _write_manifests(paths, manifest)
    _write_cleaning_log(manifest)
    print(
        f"Préparation terminée : {manifest['kept_rows']} lignes, "
        f"bruts inchangés, entraînement non lancé.",
        flush=True,
    )
    return manifest


def _inspect_header(raw_dir: Path) -> tuple[list[str], int, int, list[str]]:
    headers = []
    for name, _day in FILES:
        path = raw_dir / name
        assert_raw_csv(path, raw_dir)
        with path.open("rb") as handle:
            header = next(csv.reader([handle.readline().rstrip(b"\r\n").decode("utf-8")]))
        headers.append(strip_header(header))
    if any(header != headers[0] for header in headers[1:]):
        raise RuntimeError("Les en-têtes des huit CSV diffèrent.")
    if headers[0][-1] != "Label":
        raise RuntimeError(f"Dernière colonne inattendue : {headers[0][-1]!r}.")
    first_index, drop_index = locate_fwd_header(headers[0])
    return headers[0], first_index, drop_index, feature_names(headers[0], drop_index)


def _scan_identities(raw_dir: Path, first_index: int, drop_index: int) -> tuple[dict[bytes, str], int]:
    labels: dict[bytes, str] = {}
    mismatches = 0
    rows = 0
    for name, _day in FILES:
        path = raw_dir / name
        print(f"Identités {name}", flush=True)
        for _line_number, raw, _header in _iter_data_lines(path, raw_dir):
            if raw.count(b",") != 78:
                raise RuntimeError(f"Ligne CSV inattendue dans {name}.")
            fields = next(csv.reader([raw.decode("utf-8")]))
            if len(fields) != 79:
                raise RuntimeError(f"Parseur CSV inattendu dans {name}.")
            if fields[first_index] != fields[drop_index]:
                mismatches += 1
            comma = raw.rfind(b",")
            digest = hashlib.md5(raw[:comma]).digest()
            label = fields[-1]
            previous = labels.get(digest)
            if previous is None:
                labels[digest] = label
            elif previous != label:
                labels[digest] = "*"
            rows += 1
    if rows != EXPECTED_RAW_ROWS:
        raise RuntimeError(f"Lignes lues : {rows}, attendu {EXPECTED_RAW_ROWS}.")
    return labels, mismatches


def _write_cleaned_table(
    config: dict[str, Any],
    raw_dir: Path,
    destination: Path,
    feature_columns: list[str],
    drop_index: int,
    contradictory: set[bytes],
) -> dict[str, Any]:
    schema = pa.schema(
        [
            pa.field("row_id", pa.int64()),
            pa.field("source_file", pa.string()),
            pa.field("source_day", pa.string()),
            pa.field("source_line", pa.int64()),
            *[pa.field(name, pa.float32()) for name in feature_columns],
            pa.field("label_original", pa.string()),
            pa.field("label", pa.int8()),
            pa.field("chronological_split", pa.string()),
            pa.field("exact_md5", pa.binary(16)),
            pa.field("feature_md5", pa.binary(16)),
        ]
    )
    keeper = ChronologicalKeeper(contradictory)
    kept = 0
    infinity_cells = 0
    missing_input_cells = 0
    float32_changed_cells = 0
    label_originals: list[str] = []
    chronological_splits: list[str] = []
    family_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    class_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"normal": 0, "attack": 0})
    buffer: dict[str, list] = {field.name: [] for field in schema}
    writer = pq.ParquetWriter(destination, schema, compression="zstd")
    try:
        for name, day in FILES:
            path = raw_dir / name
            split = chronological_split(day, config)
            print(f"Nettoyage {name}", flush=True)
            for line_number, raw, _header in _iter_data_lines(path, raw_dir):
                fields = next(csv.reader([raw.decode("utf-8")]))
                comma = raw.rfind(b",")
                feature_digest = hashlib.md5(raw[:comma]).digest()
                exact_digest = hashlib.md5(raw).digest()
                if not keeper.keep(feature_digest, exact_digest, fields[-1], split):
                    continue
                values = []
                for index, field in enumerate(fields[:-1]):
                    if index == drop_index:
                        continue
                    if value_changed_by_float32(field):
                        float32_changed_cells += 1
                    parsed, kind = parse_feature(field)
                    if kind == "infinity":
                        infinity_cells += 1
                    elif kind == "missing":
                        missing_input_cells += 1
                    values.append(parsed)
                label = fields[-1]
                target = binary_target(label)
                buffer["row_id"].append(kept)
                buffer["source_file"].append(name)
                buffer["source_day"].append(day)
                buffer["source_line"].append(line_number)
                for column, value in zip(feature_columns, values, strict=True):
                    buffer[column].append(value)
                buffer["label_original"].append(label)
                buffer["label"].append(target)
                buffer["chronological_split"].append(split)
                buffer["exact_md5"].append(exact_digest)
                buffer["feature_md5"].append(feature_digest)
                label_originals.append(label)
                chronological_splits.append(split)
                family_counts[split][label] += 1
                class_counts[split]["attack" if target else "normal"] += 1
                kept += 1
                if kept % CHUNK_ROWS == 0:
                    writer.write_batch(pa.RecordBatch.from_pydict(buffer, schema=schema))
                    for column in buffer.values():
                        column.clear()
        if buffer["row_id"]:
            writer.write_batch(pa.RecordBatch.from_pydict(buffer, schema=schema))
    except Exception:
        writer.close()
        destination.unlink(missing_ok=True)
        raise
    else:
        writer.close()
    return {
        "contradictory_rows": keeper.contradictory_rows,
        "later_exact_copies": keeper.later_copies,
        "kept_rows": kept,
        "cross_set_contents_blocked": len(keeper.cross_contents),
        "contradictory_exact_contents_spanning_sets": sum(
            1 for splits in keeper.contradictory_exact_splits.values() if len(splits) > 1
        ),
        "attack_contents_crossing_files": keeper.attack_cross,
        "infinity_cells_replaced": infinity_cells,
        "missing_input_cells": missing_input_cells,
        "float32_changed_cells": float32_changed_cells,
        "label_originals": label_originals,
        "chronological_splits": chronological_splits,
        "family_counts": {split: dict(counts) for split, counts in family_counts.items()},
        "class_counts": class_counts,
        "kept_exact_contents": len(keeper.seen),
    }


def _audit_parquet(path: Path, feature_columns: list[str], contradictory: set[bytes]) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    nulls = dict.fromkeys(feature_columns, 0)
    infinite_cells = 0
    rows = 0
    contradictory_remaining = 0
    splits: dict[str, int] = defaultdict(int)
    for batch in parquet.iter_batches(batch_size=CHUNK_ROWS):
        rows += batch.num_rows
        for name in feature_columns:
            column = batch.column(name)
            nulls[name] += column.null_count
            infinite = pc.sum(pc.is_inf(column)).as_py()
            infinite_cells += 0 if infinite is None else int(infinite)
        for digest in batch.column("feature_md5").to_pylist():
            if digest in contradictory:
                contradictory_remaining += 1
        for split in batch.column("chronological_split").to_pylist():
            splits[split] += 1
    return {
        "rows": rows,
        "infinite_cells": infinite_cells,
        "missing_cells_by_column": {name: count for name, count in nulls.items() if count},
        "missing_cells": int(sum(nulls.values())),
        "contradictory_profiles_remaining": contradictory_remaining,
        "chronological_rows": dict(splits),
    }


def _chronological_indices(splits: list[str]) -> dict[str, np.ndarray]:
    grouped = {name: [] for name in ("train", "validation", "test")}
    for index, split in enumerate(splits):
        grouped[split].append(index)
    return {name: np.asarray(values, dtype=np.int64) for name, values in grouped.items()}


def _assert_grouped_contract(labels: list[str], assigned: dict[str, np.ndarray]) -> None:
    combined = np.concatenate([assigned["train"], assigned["validation"], assigned["test"]])
    if combined.size != len(labels) or np.unique(combined).size != len(labels):
        raise RuntimeError("Le découpage groupé ne forme pas une partition des lignes conservées.")
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[label].append(index)
    lookup = np.empty(len(labels), dtype=np.int8)
    for code, name in enumerate(("train", "validation", "test")):
        lookup[assigned[name]] = code
    for label, positions in by_label.items():
        expected = grouped_allocation(len(positions))
        observed = [int(np.sum(lookup[positions] == code)) for code in range(3)]
        if tuple(observed) != expected:
            raise RuntimeError(f"Allocation groupée inattendue pour {label}: {observed} != {expected}.")


def _write_indices(directory: Path, assigned: dict[str, np.ndarray]) -> None:
    for name, values in assigned.items():
        target = directory / f"{name}.npy"
        assert_derived_target(target, Path(r"C:\data\cic-ids2017\raw"))
        np.save(target, values)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _family_table(family_counts: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {
        split: {label: count for label, count in sorted(counts.items()) if label != "BENIGN"}
        for split, counts in family_counts.items()
    }


def _manifest(
    config: dict[str, Any],
    header: list[str],
    feature_columns: list[str],
    stats: dict[str, Any],
    audit: dict[str, Any],
    before: dict[str, dict[str, object]],
    chronological: dict[str, np.ndarray],
    grouped: dict[str, np.ndarray],
    paths: dict[str, Path],
    output: Path,
) -> dict[str, Any]:
    def index_record(directory: Path, assigned: dict[str, np.ndarray]) -> dict[str, Any]:
        record = {}
        for name, values in assigned.items():
            target = directory / f"{name}.npy"
            record[name] = {
                "rows": int(values.size),
                "sha256": _sha256_file(target),
                "path": str(target),
            }
        return record

    class_counts = {
        split: {key: int(value) for key, value in counts.items()}
        for split, counts in stats["class_counts"].items()
    }
    proportions = {}
    for split, counts in class_counts.items():
        total = counts["normal"] + counts["attack"]
        proportions[split] = {
            "normal": counts["normal"] / total,
            "attack": counts["attack"] / total,
        }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "shap_computed": False,
        "model_selected": False,
        "test_used_for_decisions": False,
        "raw_files_modified": False,
        "raw_fingerprints_sha256": before,
        "raw_rows": EXPECTED_RAW_ROWS,
        "header_stripped": header,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "dropped_column": "Fwd Header Length (second occurrence)",
        "fwd_header_mismatches": 0,
        "numeric_storage": config["compute"]["pandas_numeric_dtype"],
        "float32_changed_cells": stats["float32_changed_cells"],
        "infinity_cells_replaced_with_missing": stats["infinity_cells_replaced"],
        "missing_tokens_preserved_as_missing": stats["missing_input_cells"],
        "contradictory_groups_excluded": EXPECTED_CONTRADICTORY_GROUPS,
        "contradictory_rows_excluded": stats["contradictory_rows"],
        "later_exact_copies_removed": stats["later_exact_copies"],
        "cross_set_exact_contents_blocked": stats["cross_set_contents_blocked"],
        "contradictory_exact_contents_also_spanning_sets": stats[
            "contradictory_exact_contents_spanning_sets"
        ],
        "attack_exact_contents_crossing_files": stats["attack_contents_crossing_files"],
        "kept_rows": stats["kept_rows"],
        "kept_exact_contents": stats["kept_exact_contents"],
        "derived_parquet": str(output),
        "derived_parquet_sha256": _sha256_file(output),
        "audit": audit,
        "chronological": {
            "seed": None,
            "rule": "file order Monday through Friday-Afternoon-DDoS",
            "days": config["chronological_analysis"],
            "class_counts": class_counts,
            "proportions": proportions,
            "families": _family_table(stats["family_counts"]),
            "indices": index_record(paths["chronological_splits"], chronological),
            "exact_contents_shared_between_sets": 0,
        },
        "grouped": {
            "seed": int(config["split"]["seed"]),
            "generator": "random.Random",
            "python": "3.12",
            "rule": "shuffle within original label, then n*15//100 for validation and for test",
            "stratify_on": config["split"]["stratify_on"],
            "selects_official_model": False,
            "indices": index_record(paths["grouped_splits"], grouped),
            "exact_contents_shared_between_sets": 0,
            "allocation_matches_integer_rule": True,
        },
        "pipelines_fit": False,
        "models": ["logistic_regression", "random_forest", "xgboost"],
    }


def _write_manifests(paths: dict[str, Path], manifest: dict[str, Any]) -> None:
    stored = json.loads(json.dumps(manifest))
    for directory, key in (
        (paths["chronological_splits"], "chronological"),
        (paths["grouped_splits"], "grouped"),
    ):
        target = directory / "manifest.json"
        assert_derived_target(target, Path(r"C:\data\cic-ids2017\raw"))
        payload = {
            "created_utc": stored["created_utc"],
            "raw_files_modified": False,
            "training_performed": False,
            key: stored[key],
            "kept_rows": stored["kept_rows"],
            "contradictory_rows_excluded": stored["contradictory_rows_excluded"],
            "audit": stored["audit"],
        }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = paths["metrics"] / "preparation_manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(stored, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_cleaning_log(manifest: dict[str, Any]) -> None:
    chronological = manifest["chronological"]["class_counts"]
    lines = [
        "# Journal de nettoyage",
        "",
        f"Préparation du {manifest['created_utc']}. Les huit CSV bruts n'ont pas été modifiés. Leurs empreintes SHA-256 sont dans `outputs/metrics/preparation_manifest.json`. Aucun modèle n'a été ajusté.",
        "",
        "| Étape | Règle | Avant | Après | Décision |",
        "| --- | --- | ---: | ---: | --- |",
        f"| Lecture | Huit CSV bruts, ouverts en lecture seule | {manifest['raw_rows']} | {manifest['raw_rows']} | Aucune écriture dans `raw` |",
        "| Noms | Espaces de début et de fin retirés | 79 colonnes | 79 noms nets | Le nom brut n'est pas réécrit |",
        "| Fwd Header Length | Les deux colonnes sont égales sur toutes les lignes | 2 colonnes | 1 colonne | La seconde occurrence est absente du fichier dérivé |",
        f"| Infinis | Remplacés par une valeur manquante | {manifest['infinity_cells_replaced_with_missing']} cellules | 0 infini dans le parquet | Les lignes restent |",
        f"| Manquants | Jetons vides ou `NaN` conservés comme manquants | {manifest['missing_tokens_preserved_as_missing']} cellules | {manifest['audit']['missing_cells']} cellules manquantes | Pas d'imputation |",
        f"| Profils contradictoires | 698 profils, colonnes CSV identiques, étiquettes différentes | {manifest['contradictory_rows_excluded']} lignes | 0 profil restant | Groupe entier exclu |",
        f"| Doublons exacts | Première occurrence chronologique de la ligne brute | {manifest['later_exact_copies_removed']} copies | {manifest['kept_rows']} lignes | {manifest['cross_set_exact_contents_blocked']} contenus non contradictoires empêchés de traverser deux ensembles ; {manifest['contradictory_exact_contents_also_spanning_sets']} autres étaient déjà exclus avec leur profil contradictoire |",
        "",
        "## Effectifs chronologiques",
        "",
        "| Ensemble | Lignes | Normal | Attack |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split in ("train", "validation", "test"):
        counts = chronological[split]
        total = counts["normal"] + counts["attack"]
        lines.append(f"| {split} | {total} | {counts['normal']} | {counts['attack']} |")
    lines.extend(
        [
            "",
            "Le fichier dérivé est `C:/data/cic-ids2017/interim/flows.parquet`. Les indices sont sous `C:/data/cic-ids2017/splits/`.",
            "",
        ]
    )
    (PROJECT_ROOT / "docs" / "cleaning_log.md").write_text("\n".join(lines), encoding="utf-8")
