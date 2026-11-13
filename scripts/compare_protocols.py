"""Comparaison de protocoles, en lecture seule.

Aucun CSV n'est modifié. Aucun indice de découpage n'est enregistré.
Aucun modèle n'est ajusté.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

RAW_DIR = Path(r"C:\data\cic-ids2017\raw")
REPORT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "metrics" / "protocol_comparison.json"
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


def assert_raw_csv(path: Path) -> None:
    if path.resolve().parent != RAW_DIR.resolve() or path.suffix.lower() != ".csv":
        raise SystemExit(f"Chemin refusé : {path}")


def scan(connection: sqlite3.Connection) -> None:
    exact_sql = """
        INSERT INTO exact_rows(digest, label, file_id, n)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(digest, file_id) DO UPDATE SET n = n + 1
    """
    feature_sql = """
        INSERT INTO feature_rows(digest, label, file_id, n)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(digest, label, file_id) DO UPDATE SET n = n + 1
    """
    for file_id, (name, _day) in enumerate(FILES):
        path = RAW_DIR / name
        assert_raw_csv(path)
        exact_batch: list[tuple] = []
        feature_batch: list[tuple] = []
        rows = 0
        print(f"Fichier {name}", flush=True)
        with path.open("rb") as handle:
            header = handle.readline()
            if header.count(b",") != 78:
                raise SystemExit(f"En-tête inattendu : {name}")
            for line in handle:
                payload = line.rstrip(b"\r\n")
                if not payload or payload.count(b",") != 78:
                    continue
                comma = payload.rfind(b",")
                label = payload[comma + 1 :].decode("utf-8")
                exact = hashlib.md5(payload).digest()
                feature = hashlib.md5(payload[:comma]).digest()
                exact_batch.append((exact, label, file_id))
                feature_batch.append((feature, label, file_id))
                rows += 1
                if len(exact_batch) >= 20_000:
                    connection.executemany(exact_sql, exact_batch)
                    connection.executemany(feature_sql, feature_batch)
                    exact_batch.clear()
                    feature_batch.clear()
        if exact_batch:
            connection.executemany(exact_sql, exact_batch)
            connection.executemany(feature_sql, feature_batch)
        connection.commit()
        print(f"  {rows}", flush=True)


def label_counts(connection: sqlite3.Connection) -> dict[int, dict[str, int]]:
    counts = {file_id: {} for file_id in range(len(FILES))}
    for file_id, label, total in connection.execute(
        "SELECT file_id, label, SUM(n) FROM exact_rows GROUP BY file_id, label"
    ):
        counts[int(file_id)][label] = int(total)
    return counts


def crossed_exact_copies(
    connection: sqlite3.Connection, earlier: list[int], later: list[int]
) -> dict[str, int]:
    if not earlier or not later:
        return {"contents": 0, "rows_in_later_set": 0}
    earlier_marks = ",".join("?" for _ in earlier)
    later_marks = ",".join("?" for _ in later)
    contents, later_rows = connection.execute(
        f"""
        SELECT COUNT(*), COALESCE(SUM(later_rows), 0)
        FROM (
            SELECT later.digest, SUM(later.n) AS later_rows
            FROM exact_rows AS later
            WHERE later.file_id IN ({later_marks})
              AND later.digest IN (
                    SELECT digest FROM exact_rows WHERE file_id IN ({earlier_marks})
              )
            GROUP BY later.digest
        )
        """,
        [*later, *earlier],
    ).fetchone()
    return {"contents": int(contents), "rows_in_later_set": int(later_rows)}


def contradictory(connection: sqlite3.Connection) -> dict[str, object]:
    rows = connection.execute(
        """
        SELECT digest, label, file_id, n
        FROM feature_rows
        WHERE digest IN (
            SELECT digest
            FROM feature_rows
            GROUP BY digest
            HAVING COUNT(DISTINCT label) > 1
        )
        """
    ).fetchall()
    groups: dict[bytes, dict[str, dict[int, int]]] = {}
    for digest, label, file_id, total in rows:
        groups.setdefault(digest, {}).setdefault(label, {})[int(file_id)] = int(total)
    by_label: dict[str, int] = {}
    by_file: dict[str, int] = {}
    by_file_label: list[dict[str, object]] = []
    pair_locations: dict[tuple, int] = {}
    for labels in groups.values():
        ordered = sorted(labels)
        left, right = ordered[0], ordered[1]
        left_files = tuple(sorted(labels[left]))
        right_files = tuple(sorted(labels[right]))
        key = (left, left_files, right, right_files)
        pair_locations[key] = pair_locations.get(key, 0) + 1
        for label, files in labels.items():
            for file_id, total in files.items():
                by_label[label] = by_label.get(label, 0) + total
                day = FILES[file_id][1]
                by_file[day] = by_file.get(day, 0) + total
    file_label_counts = connection.execute(
        """
        SELECT file_id, label, SUM(n)
        FROM feature_rows
        WHERE digest IN (
            SELECT digest FROM feature_rows
            GROUP BY digest HAVING COUNT(DISTINCT label) > 1
        )
        GROUP BY file_id, label
        ORDER BY file_id, SUM(n) DESC
        """
    ).fetchall()
    for file_id, label, total in file_label_counts:
        by_file_label.append(
            {"day": FILES[int(file_id)][1], "label": label, "rows": int(total)}
        )
    patterns = [
        {
            "label_a": left,
            "days_a": [FILES[file_id][1] for file_id in left_files],
            "label_b": right,
            "days_b": [FILES[file_id][1] for file_id in right_files],
            "groups": count,
        }
        for (left, left_files, right, right_files), count in sorted(
            pair_locations.items(), key=lambda item: -item[1]
        )
    ]
    return {
        "groups": len(groups),
        "rows": sum(by_label.values()),
        "rows_by_label": dict(sorted(by_label.items(), key=lambda item: -item[1])),
        "rows_by_day": by_file,
        "rows_by_day_and_label": by_file_label,
        "patterns": patterns,
    }


def scenario_rows(counts: dict[int, dict[str, int]], file_ids: list[int]) -> dict[str, object]:
    labels: dict[str, int] = {}
    for file_id in file_ids:
        for label, total in counts[file_id].items():
            labels[label] = labels.get(label, 0) + total
    normal = labels.get("BENIGN", 0)
    attack = sum(total for label, total in labels.items() if label != "BENIGN")
    families = sorted(label for label in labels if label != "BENIGN" and labels[label] > 0)
    return {
        "rows": normal + attack,
        "normal": normal,
        "attack": attack,
        "families": families,
        "labels": labels,
    }


def allocate(count: int) -> dict[str, int]:
    held_out = count * 15 // 100
    return {
        "train": count - 2 * held_out,
        "validation": held_out,
        "test": held_out,
    }


def main() -> None:
    database = Path(os.environ["TEMP"]) / "cic_ids2017_protocol.sqlite"
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        """
        CREATE TABLE exact_rows (
            digest BLOB NOT NULL,
            label TEXT NOT NULL,
            file_id INTEGER NOT NULL,
            n INTEGER NOT NULL,
            PRIMARY KEY (digest, file_id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE feature_rows (
            digest BLOB NOT NULL,
            label TEXT NOT NULL,
            file_id INTEGER NOT NULL,
            n INTEGER NOT NULL,
            PRIMARY KEY (digest, label, file_id)
        )
        """
    )
    try:
        scan(connection)
        counts = label_counts(connection)
        conflicts = contradictory(connection)
        scenarios = []
        for val_start in range(1, len(FILES) - 1):
            for test_start in range(val_start + 1, len(FILES)):
                train_ids = list(range(0, val_start))
                val_ids = list(range(val_start, test_start))
                test_ids = list(range(test_start, len(FILES)))
                train = scenario_rows(counts, train_ids)
                validation = scenario_rows(counts, val_ids)
                test = scenario_rows(counts, test_ids)
                train_families = set(train["families"])
                scenarios.append(
                    {
                        "train_days": [FILES[file_id][1] for file_id in train_ids],
                        "validation_days": [FILES[file_id][1] for file_id in val_ids],
                        "test_days": [FILES[file_id][1] for file_id in test_ids],
                        "train": train,
                        "validation": validation,
                        "test": test,
                        "families_in_train_absent_from_validation": sorted(
                            train_families - set(validation["families"])
                        ),
                        "families_in_train_absent_from_test": sorted(
                            train_families - set(test["families"])
                        ),
                        "families_in_test_absent_from_train": sorted(
                            set(test["families"]) - train_families
                        ),
                        "exact_copies_from_train_in_validation": crossed_exact_copies(
                            connection, train_ids, val_ids
                        ),
                        "exact_copies_from_train_in_test": crossed_exact_copies(
                            connection, train_ids, test_ids
                        ),
                        "exact_copies_from_validation_in_test": crossed_exact_copies(
                            connection, val_ids, test_ids
                        ),
                    }
                )
        distinct_after_exclusion = connection.execute(
            """
            SELECT label, COUNT(*)
            FROM (
                SELECT digest, label
                FROM feature_rows
                GROUP BY digest, label
            )
            WHERE digest NOT IN (
                SELECT digest FROM feature_rows
                GROUP BY digest HAVING COUNT(DISTINCT label) > 1
            )
            GROUP BY label
            """
        ).fetchall()
        payload = {
            "raw_files_modified": False,
            "rows_deleted": False,
            "split_saved": False,
            "training_performed": False,
            "models_retained": ["logistic_regression", "random_forest", "xgboost"],
            "day_order": [day for _name, day in FILES],
            "scenarios": scenarios,
            "contradictory_profiles": conflicts,
            "grouped_split_after_excluding_contradictions": [
                {"label": label, "distinct_rows": int(total), **allocate(int(total))}
                for label, total in distinct_after_exclusion
            ],
        }
    finally:
        connection.close()
        if database.exists():
            database.unlink()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"scenarios {len(scenarios)}", flush=True)
    print(json.dumps(conflicts["rows_by_label"], ensure_ascii=False), flush=True)
    print(json.dumps(conflicts["rows_by_day"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
