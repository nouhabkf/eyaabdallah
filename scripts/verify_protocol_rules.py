"""Vérifie trois règles, en lecture seule.

Aucun CSV n'est modifié. Aucun indice de découpage n'est enregistré.
Aucun modèle n'est ajusté.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sqlite3
import struct
from collections import defaultdict
from io import StringIO
from pathlib import Path

import pandas as pd

RAW_DIR = Path(r"C:\data\cic-ids2017\raw")
REPORT_PATH = (
    Path(__file__).resolve().parents[1] / "outputs" / "metrics" / "protocol_verification.json"
)
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
SETS = {
    "train": [0, 1, 2],
    "validation": [3, 4],
    "test": [5, 6, 7],
}
SET_OF_FILE = {file_id: name for name, ids in SETS.items() for file_id in ids}
SAMPLE_PER_FILE = 100
SAMPLE_SEED = 42


def assert_raw_csv(path: Path) -> None:
    if path.resolve().parent != RAW_DIR.resolve() or path.suffix.lower() != ".csv":
        raise SystemExit(f"Chemin refusé : {path}")


def lowest_file(mask: int) -> int:
    return (mask & -mask).bit_length() - 1


def set_count(mask: int) -> int:
    found = 0
    if mask & 0b00000111:
        found += 1
    if mask & 0b00011000:
        found += 1
    if mask & 0b11100000:
        found += 1
    return found


def numeric_digest(fields: list[str]) -> bytes | None:
    tokens = []
    for field in fields:
        if field == "":
            tokens.append(b"\x00")
            continue
        try:
            tokens.append(struct.pack("<d", float(field)))
        except ValueError:
            return None
    return hashlib.md5(b"".join(tokens)).digest()


def scan(connection: sqlite3.Connection) -> dict[str, object]:
    exact_sql = """
        INSERT INTO exact_rows(digest, label, feature_digest, first_file, file_mask, n)
        VALUES (?, ?, ?, ?, ?, 1)
        ON CONFLICT(digest) DO UPDATE SET
            n = n + 1,
            file_mask = file_mask | excluded.file_mask
    """
    feature_sql = """
        INSERT INTO feature_rows(digest, float_digest, label, file_id, n)
        VALUES (?, ?, ?, ?, 1)
        ON CONFLICT(digest, label, file_id) DO UPDATE SET
            n = n + 1
    """
    stats = {
        "rows": 0,
        "double_quote_lines": 0,
        "single_quote_lines": 0,
        "comma_anomalies": 0,
        "csv_field_anomalies": 0,
        "prefix_rejoin_mismatches": 0,
        "label_mismatches": 0,
        "numeric_failures": 0,
        "headers": [],
    }
    before: dict[int, dict[str, int]] = {file_id: defaultdict(int) for file_id in range(8)}
    pandas_checks = []
    mismatch_examples = []
    for file_id, (name, day) in enumerate(FILES):
        path = RAW_DIR / name
        assert_raw_csv(path)
        print(f"Fichier {name}", flush=True)
        exact_batch: list[tuple] = []
        feature_batch: list[tuple] = []
        reservoir: list[str] = []
        rng = random.Random(SAMPLE_SEED + file_id)
        seen = 0
        with path.open("rb") as handle:
            header = handle.readline().rstrip(b"\r\n").decode("utf-8")
            header_fields = next(csv.reader([header]))
            stats["headers"].append(
                {
                    "day": day,
                    "columns": len(header_fields),
                    "label_name": header_fields[-1],
                    "same_as_first": header_fields == stats["headers"][0]["fields"]
                    if stats["headers"]
                    else True,
                    "fields": header_fields,
                }
            )
            for line in handle:
                raw = line.rstrip(b"\r\n")
                if not raw:
                    continue
                stats["rows"] += 1
                seen += 1
                if b'"' in raw:
                    stats["double_quote_lines"] += 1
                if b"'" in raw:
                    stats["single_quote_lines"] += 1
                if raw.count(b",") != 78:
                    stats["comma_anomalies"] += 1
                    continue
                text = raw.decode("utf-8")
                fields = next(csv.reader([text]))
                if len(fields) != 79:
                    stats["csv_field_anomalies"] += 1
                    continue
                comma = raw.rfind(b",")
                prefix = raw[:comma]
                rejoined = ",".join(fields[:-1]).encode("utf-8")
                label_bytes = fields[-1].encode("utf-8")
                if rejoined != prefix:
                    stats["prefix_rejoin_mismatches"] += 1
                    if len(mismatch_examples) < 5:
                        mismatch_examples.append({"day": day, "csv_fields": len(fields)})
                if label_bytes != raw[comma + 1 :]:
                    stats["label_mismatches"] += 1
                label = fields[-1]
                feature = hashlib.md5(rejoined).digest()
                floats = numeric_digest(fields[:-1])
                if floats is None:
                    stats["numeric_failures"] += 1
                    floats = b""
                exact = hashlib.md5(raw).digest()
                exact_batch.append((exact, label, feature, file_id, 1 << file_id))
                feature_batch.append((feature, floats, label, file_id))
                before[file_id][label] += 1
                if len(reservoir) < SAMPLE_PER_FILE:
                    reservoir.append(text)
                else:
                    pick = rng.randrange(seen)
                    if pick < SAMPLE_PER_FILE:
                        reservoir[pick] = text
                if len(exact_batch) >= 20_000:
                    connection.executemany(exact_sql, exact_batch)
                    connection.executemany(feature_sql, feature_batch)
                    exact_batch.clear()
                    feature_batch.clear()
        if exact_batch:
            connection.executemany(exact_sql, exact_batch)
            connection.executemany(feature_sql, feature_batch)
        connection.commit()
        pandas_checks.append(compare_with_pandas(header, reservoir, day))
        print(f"  {seen}", flush=True)
    for item in stats["headers"]:
        item.pop("fields", None)
    stats["mismatch_examples"] = mismatch_examples
    stats["before_by_file"] = {
        FILES[file_id][1]: dict(labels) for file_id, labels in before.items()
    }
    stats["pandas_sample"] = pandas_checks
    return stats


def compare_with_pandas(header: str, raw_lines: list[str], day: str) -> dict[str, object]:
    frame = pd.read_csv(StringIO(header + "\n" + "\n".join(raw_lines) + "\n"), dtype=str, keep_default_na=False)
    numeric = pd.read_csv(
        StringIO(header + "\n" + "\n".join(raw_lines) + "\n"),
        keep_default_na=True,
    )
    string_mismatches = 0
    numeric_mismatches = 0
    by_float: dict[bytes, set[bytes]] = defaultdict(set)
    for index, raw in enumerate(raw_lines):
        fields = next(csv.reader([raw]))
        read = [frame.iat[index, column] for column in range(frame.shape[1])]
        if read != fields:
            string_mismatches += 1
        feature_hash = hashlib.md5(",".join(fields[:-1]).encode("utf-8")).digest()
        floats = numeric_digest(fields[:-1])
        if floats is not None:
            by_float[floats].add(feature_hash)
        pandas_values = numeric.iloc[index, :-1].tolist()
        if not same_numeric(fields[:-1], pandas_values):
            numeric_mismatches += 1
    collapses = sum(1 for hashes in by_float.values() if len(hashes) > 1)
    return {
        "day": day,
        "rows": len(raw_lines),
        "pandas_columns": int(frame.shape[1]),
        "string_mismatches_against_csv": string_mismatches,
        "numeric_mismatches_against_csv_floats": numeric_mismatches,
        "distinct_string_profiles_collapsed_by_float": collapses,
    }


def same_numeric(fields: list[str], pandas_values: list[object]) -> bool:
    if len(fields) != len(pandas_values):
        return False
    for field, value in zip(fields, pandas_values):
        if field == "":
            if not pd.isna(value):
                return False
            continue
        try:
            expected = float(field)
        except ValueError:
            return False
        if pd.isna(value):
            return False
        observed = float(value)
        if math.isnan(expected) and math.isnan(observed):
            continue
        if observed != expected and not math.isclose(observed, expected, rel_tol=0.0, abs_tol=0.0):
            return False
    return True


def contradictory(connection: sqlite3.Connection, digest_column: str) -> dict[str, object]:
    present = f"length({digest_column}) > 0"
    groups = connection.execute(
        f"""
        SELECT {digest_column} AS digest, COUNT(DISTINCT label), SUM(n)
        FROM feature_rows
        WHERE {present}
        GROUP BY {digest_column}
        HAVING COUNT(DISTINCT label) > 1
        """
    ).fetchall()
    by_label = connection.execute(
        f"""
        SELECT label, SUM(n)
        FROM feature_rows
        WHERE {present} AND {digest_column} IN (
            SELECT {digest_column}
            FROM feature_rows
            WHERE {present}
            GROUP BY {digest_column}
            HAVING COUNT(DISTINCT label) > 1
        )
        GROUP BY label
        ORDER BY SUM(n) DESC
        """
    ).fetchall()
    by_day = connection.execute(
        f"""
        SELECT file_id, label, SUM(n)
        FROM feature_rows
        WHERE {present} AND {digest_column} IN (
            SELECT {digest_column}
            FROM feature_rows
            WHERE {present}
            GROUP BY {digest_column}
            HAVING COUNT(DISTINCT label) > 1
        )
        GROUP BY file_id, label
        ORDER BY file_id, SUM(n) DESC
        """
    ).fetchall()
    return {
        "groups": len(groups),
        "rows": int(sum(total for _digest, _labels, total in groups)),
        "rows_by_label": {label: int(total) for label, total in by_label},
        "rows_by_day_and_label": [
            {"day": FILES[int(file_id)][1], "label": label, "rows": int(total)}
            for file_id, label, total in by_day
        ],
    }


def method_difference(connection: sqlite3.Connection) -> dict[str, int]:
    difference = connection.execute(
        """
        WITH string_bad AS (
            SELECT digest
            FROM feature_rows
            GROUP BY digest
            HAVING COUNT(DISTINCT label) > 1
        ),
        float_bad AS (
            SELECT float_digest
            FROM feature_rows
            WHERE length(float_digest) > 0
            GROUP BY float_digest
            HAVING COUNT(DISTINCT label) > 1
        )
        SELECT
            COALESCE(SUM(CASE
                WHEN string_bad.digest IS NOT NULL AND float_bad.float_digest IS NULL THEN feature_rows.n
                ELSE 0 END), 0),
            COALESCE(SUM(CASE
                WHEN string_bad.digest IS NULL AND float_bad.float_digest IS NOT NULL THEN feature_rows.n
                ELSE 0 END), 0)
        FROM feature_rows
        LEFT JOIN string_bad ON feature_rows.digest = string_bad.digest
        LEFT JOIN float_bad ON feature_rows.float_digest = float_bad.float_digest
        """
    ).fetchone()
    return {
        "rows_only_in_string_rule": int(difference[0]),
        "rows_only_in_float_rule": int(difference[1]),
    }


def verify_contradictory_strings(connection: sqlite3.Connection) -> dict[str, object]:
    bad = {
        digest
        for (digest,) in connection.execute(
            """
            SELECT digest FROM feature_rows
            GROUP BY digest
            HAVING COUNT(DISTINCT label) > 1
            """
        )
    }
    grouped: dict[bytes, dict[str, object]] = {}
    pandas_lines: list[tuple[str, str]] = []
    taken: dict[tuple[str, str], int] = defaultdict(int)
    header = ""
    for file_id, (name, day) in enumerate(FILES):
        path = RAW_DIR / name
        assert_raw_csv(path)
        with path.open("rb") as handle:
            header = handle.readline().rstrip(b"\r\n").decode("utf-8")
            for line in handle:
                raw = line.rstrip(b"\r\n")
                if not raw or raw.count(b",") != 78:
                    continue
                text = raw.decode("utf-8")
                fields = next(csv.reader([text]))
                if len(fields) != 79:
                    continue
                rejoined = ",".join(fields[:-1]).encode("utf-8")
                digest = hashlib.md5(rejoined).digest()
                if digest not in bad:
                    continue
                bucket = grouped.setdefault(
                    digest,
                    {"features": rejoined, "labels": set(), "rows": 0, "days": set()},
                )
                if bucket["features"] != rejoined:
                    bucket["feature_mismatch"] = True
                bucket["labels"].add(fields[-1])
                bucket["rows"] += 1
                bucket["days"].add(day)
                key = (day, fields[-1])
                if taken[key] < 2:
                    taken[key] += 1
                    pandas_lines.append((header, text))
    mismatched_groups = 0
    label_problem = 0
    examples = []
    for bucket in grouped.values():
        if bucket.get("feature_mismatch"):
            mismatched_groups += 1
        if len(bucket["labels"]) < 2:
            label_problem += 1
        if len(examples) < 4:
            examples.append(
                {
                    "days": sorted(bucket["days"]),
                    "labels": sorted(bucket["labels"]),
                    "rows": bucket["rows"],
                    "feature_bytes_identical_inside_group": not bucket.get("feature_mismatch", False),
                }
            )
    pandas_result = compare_saved_lines(pandas_lines)
    return {
        "groups_reloaded": len(grouped),
        "rows_reloaded": sum(int(bucket["rows"]) for bucket in grouped.values()),
        "groups_whose_feature_bytes_differ": mismatched_groups,
        "groups_with_fewer_than_two_labels": label_problem,
        "examples": examples,
        "pandas_on_contradictory_sample": pandas_result,
    }


def compare_saved_lines(pairs: list[tuple[str, str]]) -> dict[str, object]:
    if not pairs:
        return {"rows": 0}
    by_header: dict[str, list[str]] = defaultdict(list)
    for header, text in pairs:
        by_header[header].append(text)
    rows = 0
    string_mismatches = 0
    numeric_mismatches = 0
    for header, lines in by_header.items():
        result = compare_with_pandas(header, lines, "contradictory-sample")
        rows += int(result["rows"])
        string_mismatches += int(result["string_mismatches_against_csv"])
        numeric_mismatches += int(result["numeric_mismatches_against_csv_floats"])
    return {
        "rows": rows,
        "string_mismatches_against_csv": string_mismatches,
        "numeric_mismatches_against_csv_floats": numeric_mismatches,
    }


def aggregate_labels(rows: list[tuple[int, str, int]]) -> dict[str, dict[str, int]]:
    payload = {
        name: {"rows": 0, "normal": 0, "attack": 0, "labels": defaultdict(int)}
        for name in SETS
    }
    for file_id, label, total in rows:
        target = payload[SET_OF_FILE[file_id]]
        target["rows"] += total
        target["labels"][label] += total
        if label == "BENIGN":
            target["normal"] += total
        else:
            target["attack"] += total
    return {
        name: {
            "rows": values["rows"],
            "normal": values["normal"],
            "attack": values["attack"],
            "attack_rate": values["attack"] / values["rows"] if values["rows"] else None,
            "families": {
                label: count
                for label, count in sorted(values["labels"].items())
                if label != "BENIGN" and count > 0
            },
        }
        for name, values in payload.items()
    }


def deduplication(connection: sqlite3.Connection, before_by_file: dict[str, dict[str, int]]) -> dict[str, object]:
    order_errors = 0
    multi_set_contents = 0
    multi_file_contents = 0
    attack_multi_file_contents = 0
    for first_file, mask, label in connection.execute(
        "SELECT first_file, file_mask, label FROM exact_rows"
    ):
        if lowest_file(int(mask)) != int(first_file):
            order_errors += 1
        if bin(int(mask)).count("1") > 1:
            multi_file_contents += 1
            if label != "BENIGN":
                attack_multi_file_contents += 1
        if set_count(int(mask)) > 1:
            multi_set_contents += 1
    after_rows = connection.execute(
        "SELECT first_file, label, COUNT(*) FROM exact_rows GROUP BY first_file, label"
    ).fetchall()
    contradictory_digests = """
        SELECT digest FROM feature_rows
        GROUP BY digest
        HAVING COUNT(DISTINCT label) > 1
    """
    after_exclusion = connection.execute(
        f"""
        SELECT first_file, label, COUNT(*)
        FROM exact_rows
        WHERE feature_digest NOT IN ({contradictory_digests})
        GROUP BY first_file, label
        """
    ).fetchall()
    before_rows = []
    for file_id, (_name, day) in enumerate(FILES):
        for label, total in before_by_file[day].items():
            before_rows.append((file_id, label, int(total)))
    before = aggregate_labels(before_rows)
    after = aggregate_labels([(int(file_id), label, int(total)) for file_id, label, total in after_rows])
    excluded = aggregate_labels(
        [(int(file_id), label, int(total)) for file_id, label, total in after_exclusion]
    )
    families_before = sorted({family for values in before.values() for family in values["families"]})
    families_after = sorted({family for values in after.values() for family in values["families"]})
    families_joint = sorted({family for values in excluded.values() for family in values["families"]})
    return {
        "first_occurrence_is_earliest_file": order_errors == 0,
        "contents_whose_earliest_file_disagrees": order_errors,
        "exact_contents_in_several_files_before_rule": multi_file_contents,
        "attack_exact_contents_in_several_files": attack_multi_file_contents,
        "exact_contents_in_several_sets_before_rule": multi_set_contents,
        "exact_contents_in_several_sets_after_rule": 0 if order_errors == 0 else None,
        "before": before,
        "after_first_occurrence": after,
        "after_first_occurrence_and_contradictory_exclusion": excluded,
        "families_present_before": families_before,
        "families_lost_by_deduplication": sorted(set(families_before) - set(families_after)),
        "families_lost_by_deduplication_and_exclusion": sorted(set(families_before) - set(families_joint)),
        "families_by_set_lost_by_deduplication": {
            name: sorted(set(before[name]["families"]) - set(after[name]["families"]))
            for name in SETS
        },
    }


def validation_support(dedup: dict[str, object]) -> dict[str, object]:
    stages = {}
    for key in (
        "before",
        "after_first_occurrence",
        "after_first_occurrence_and_contradictory_exclusion",
    ):
        validation = dedup[key]["validation"]
        normal = validation["normal"]
        attack = validation["attack"]
        stages[key] = {
            "rows": validation["rows"],
            "normal": normal,
            "attack": attack,
            "attack_rate": validation["attack_rate"],
            "families": validation["families"],
            "attack_share_by_family": {
                label: count / attack if attack else None
                for label, count in validation["families"].items()
            },
            "bootstrap_sd_of_attack_count": math.sqrt(attack * (1 - attack / validation["rows"]))
            if validation["rows"] and attack
            else None,
            "false_positive_rate_where_fp_equals_attacks": attack / normal if normal else None,
        }
    train_families = set(dedup["before"]["train"]["families"])
    validation_families = set(dedup["before"]["validation"]["families"])
    return {
        "training_performed": False,
        "models": ["logistic_regression", "random_forest", "xgboost"],
        "same_rows_for_the_three_models": True,
        "selection_metric": "average_precision",
        "selection_input": "continuous_scores",
        "selection_split": "validation",
        "hyperparameter_split": "train_inner_cv",
        "threshold_split": "validation",
        "threshold_objective": "1 * FPR + 2 * FNR",
        "test_used_for_model_or_hyperparameters": False,
        "validation_families_absent_from_train": sorted(validation_families - train_families),
        "stages": stages,
    }


def main() -> None:
    database = Path(os.environ["TEMP"]) / "cic_ids2017_verify.sqlite"
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        """
        CREATE TABLE exact_rows (
            digest BLOB PRIMARY KEY,
            label TEXT NOT NULL,
            feature_digest BLOB NOT NULL,
            first_file INTEGER NOT NULL,
            file_mask INTEGER NOT NULL,
            n INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE feature_rows (
            digest BLOB NOT NULL,
            float_digest BLOB NOT NULL,
            label TEXT NOT NULL,
            file_id INTEGER NOT NULL,
            n INTEGER NOT NULL,
            PRIMARY KEY (digest, label, file_id)
        )
        """
    )
    try:
        stats = scan(connection)
        string_conflicts = contradictory(connection, "digest")
        float_conflicts = contradictory(connection, "float_digest")
        reloaded = verify_contradictory_strings(connection)
        dedup = deduplication(connection, stats.pop("before_by_file"))
        payload = {
            "raw_files_modified": False,
            "rows_deleted": False,
            "split_saved": False,
            "training_performed": False,
            "shap_computed": False,
            "models_retained": ["logistic_regression", "random_forest", "xgboost"],
            "csv_parser_check": {
                key: stats[key]
                for key in (
                    "rows",
                    "double_quote_lines",
                    "single_quote_lines",
                    "comma_anomalies",
                    "csv_field_anomalies",
                    "prefix_rejoin_mismatches",
                    "label_mismatches",
                    "numeric_failures",
                    "headers",
                    "mismatch_examples",
                    "pandas_sample",
                )
            },
            "contradictory_profiles_by_parsed_columns": string_conflicts,
            "contradictory_profiles_by_float_columns": float_conflicts,
            "difference_between_string_and_float_rules": method_difference(connection),
            "full_reload_of_contradictory_rows": reloaded,
            "chronological_deduplication": dedup,
            "validation_pr_auc_support": validation_support(dedup),
        }
    finally:
        connection.close()
        if database.exists():
            database.unlink()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"rows {payload['csv_parser_check']['rows']}", flush=True)
    print(
        "mismatches "
        f"{payload['csv_parser_check']['prefix_rejoin_mismatches']} "
        f"quotes {payload['csv_parser_check']['double_quote_lines']}",
        flush=True,
    )
    print(
        "conflicts "
        f"{payload['contradictory_profiles_by_parsed_columns']['groups']} "
        f"{payload['contradictory_profiles_by_parsed_columns']['rows']}",
        flush=True,
    )
    print(
        "multi-set "
        f"{payload['chronological_deduplication']['exact_contents_in_several_sets_before_rule']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
