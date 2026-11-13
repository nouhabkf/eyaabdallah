"""Audit méthodologique en lecture seule.

Aucune ligne n'est supprimée. Les CSV bruts ne sont pas ouverts en écriture.
Aucun modèle n'est ajusté et aucun découpage n'est enregistré.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(r"C:\data\cic-ids2017\raw")
REPORT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "metrics" / "methodology_audit.json"
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


def flush(connection: sqlite3.Connection, sql: str, batch: list[tuple]) -> None:
    connection.executemany(sql, batch)
    batch.clear()


def scan(connection: sqlite3.Connection) -> dict[str, object]:
    exact_sql = """
        INSERT INTO exact_lines(digest, label, mask, n) VALUES (?, ?, ?, 1)
        ON CONFLICT(digest) DO UPDATE SET
            n = n + 1,
            mask = mask | excluded.mask
    """
    feat_sql = """
        INSERT INTO feature_lines(digest, label, mask, n) VALUES (?, ?, ?, 1)
        ON CONFLICT(digest, label) DO UPDATE SET
            n = n + 1,
            mask = mask | excluded.mask
    """
    per_file = []
    exact_label_conflicts = 0
    for index, (name, day) in enumerate(FILES):
        path = RAW_DIR / name
        assert_raw_csv(path)
        bit = 1 << index
        seen_exact: set[bytes] = set()
        seen_feat: dict[bytes, str] = {}
        within_exact: Counter[str] = Counter()
        within_feat: Counter[str] = Counter()
        feature_conflicts = 0
        rows = 0
        bad_width = 0
        exact_batch: list[tuple] = []
        feat_batch: list[tuple] = []
        print(f"Fichier {name}", flush=True)
        with path.open("rb") as handle:
            header = handle.readline().rstrip(b"\r\n")
            if header.count(b",") != 78:
                raise SystemExit(f"En-tête inattendu : {name}")
            for line in handle:
                payload = line.rstrip(b"\r\n")
                if not payload:
                    continue
                comma = payload.rfind(b",")
                if payload.count(b",") != 78 or comma < 0:
                    bad_width += 1
                    continue
                rows += 1
                label = payload[comma + 1 :].decode("utf-8")
                exact = hashlib.md5(payload).digest()
                feat = hashlib.md5(payload[:comma]).digest()
                if exact in seen_exact:
                    within_exact[label] += 1
                else:
                    seen_exact.add(exact)
                previous = seen_feat.get(feat)
                if previous is None:
                    seen_feat[feat] = label
                else:
                    within_feat[label] += 1
                    if previous != label:
                        feature_conflicts += 1
                exact_batch.append((exact, label, bit))
                feat_batch.append((feat, label, bit))
                if len(exact_batch) >= 20_000:
                    flush(connection, exact_sql, exact_batch)
                    flush(connection, feat_sql, feat_batch)
                if rows % 400_000 == 0:
                    print(f"  {rows}", flush=True)
        if exact_batch:
            flush(connection, exact_sql, exact_batch)
            flush(connection, feat_sql, feat_batch)
        connection.commit()
        per_file.append(
            {
                "file": name,
                "day": day,
                "rows": rows,
                "bad_width": bad_width,
                "distinct_exact_in_file": len(seen_exact),
                "extra_exact_in_file": int(sum(within_exact.values())),
                "extra_exact_by_label": dict(within_exact),
                "distinct_features_in_file": len(seen_feat),
                "extra_features_in_file": int(sum(within_feat.values())),
                "extra_features_by_label": dict(within_feat),
                "feature_rows_disagreeing_with_first_label_in_file": feature_conflicts,
            }
        )
        print(f"  lignes {rows} conflits internes {feature_conflicts}", flush=True)
    return {"files": per_file, "exact_hash_label_conflicts": exact_label_conflicts}


def label_summary(connection: sqlite3.Connection, table: str) -> list[dict[str, object]]:
    rows = connection.execute(
        f"""
        SELECT label, SUM(n) AS rows, COUNT(*) AS distinct_rows, SUM(n - 1) AS extra_rows
        FROM {table}
        GROUP BY label
        ORDER BY rows DESC
        """
    ).fetchall()
    return [
        {
            "label": label,
            "rows": int(total),
            "distinct_rows": int(distinct),
            "extra_rows": int(extra),
            "rows_after_keeping_one": int(distinct),
        }
        for label, total, distinct, extra in rows
    ]


def shared_pairs(connection: sqlite3.Connection, table: str) -> list[dict[str, object]]:
    pairs = []
    for left in range(len(FILES)):
        for right in range(left + 1, len(FILES)):
            contents, total_rows = connection.execute(
                f"""
                SELECT COUNT(*), COALESCE(SUM(n), 0)
                FROM {table}
                WHERE (mask & ?) != 0 AND (mask & ?) != 0
                """,
                (1 << left, 1 << right),
            ).fetchone()
            if contents:
                pairs.append(
                    {
                        "left": FILES[left][1],
                        "right": FILES[right][1],
                        "distinct_contents": int(contents),
                        "rows": int(total_rows),
                    }
                )
    return pairs


def feature_conflicts(connection: sqlite3.Connection) -> dict[str, object]:
    summary = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(group_rows), 0)
        FROM (
            SELECT digest, SUM(n) AS group_rows
            FROM feature_lines
            GROUP BY digest
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()
    pair_rows = connection.execute(
        """
        SELECT a.label, b.label, COUNT(*), SUM(a.n + b.n),
               SUM(CASE WHEN (a.mask & b.mask) = 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN (a.mask & b.mask) != 0 THEN 1 ELSE 0 END)
        FROM feature_lines AS a
        JOIN feature_lines AS b
          ON a.digest = b.digest AND a.label < b.label
        GROUP BY a.label, b.label
        ORDER BY COUNT(*) DESC
        """
    ).fetchall()
    return {
        "feature_contents_with_several_labels": int(summary[0]),
        "rows_in_those_contents": int(summary[1]),
        "label_pairs": [
            {
                "label_a": left,
                "label_b": right,
                "contents": int(contents),
                "rows": int(total),
                "contents_with_labels_in_disjoint_files": int(disjoint),
                "contents_sharing_at_least_one_file": int(shared),
            }
            for left, right, contents, total, disjoint, shared in pair_rows
        ],
    }


def cross_file_exact(connection: sqlite3.Connection) -> dict[str, object]:
    contents, total_rows = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(n), 0)
        FROM exact_lines
        WHERE mask > 0 AND (mask & (mask - 1)) != 0
        """
    ).fetchone()
    by_label = connection.execute(
        """
        SELECT label, COUNT(*), SUM(n), SUM(n - 1)
        FROM exact_lines
        WHERE mask > 0 AND (mask & (mask - 1)) != 0
        GROUP BY label
        ORDER BY SUM(n) DESC
        """
    ).fetchall()
    return {
        "distinct_contents": int(contents),
        "rows": int(total_rows),
        "by_label": [
            {
                "label": label,
                "contents": int(groups),
                "rows": int(total),
                "extra_rows": int(extra),
            }
            for label, groups, total, extra in by_label
        ],
    }


def allocate(count: int) -> dict[str, int]:
    test = count * 15 // 100
    validation = count * 15 // 100
    train = count - test - validation
    return {"train": train, "validation": validation, "test": test}


def column_anomalies() -> dict[str, object]:
    equal = 0
    unequal = 0
    left_missing = 0
    right_missing = 0
    inf_bytes = 0
    inf_packets = 0
    inf_bytes_duration_zero = 0
    inf_packets_duration_zero = 0
    missing_bytes = 0
    missing_bytes_duration_zero = 0
    missing_bytes_packets_inf = 0
    both_inf = 0
    compared = 0
    for name, _day in FILES:
        path = RAW_DIR / name
        assert_raw_csv(path)
        reader = pd.read_csv(
            path,
            usecols=[1, 14, 15, 34, 55],
            chunksize=100_000,
            low_memory=False,
            encoding="utf-8",
        )
        for chunk in reader:
            duration = chunk.iloc[:, 0]
            rates_bytes = chunk.iloc[:, 1]
            rates_packets = chunk.iloc[:, 2]
            left = chunk.iloc[:, 3]
            right = chunk.iloc[:, 4]
            compared += int(len(chunk))
            both_present = left.notna() & right.notna()
            equal += int((both_present & left.eq(right)).sum())
            unequal += int((both_present & left.ne(right)).sum())
            left_missing += int(left.isna().sum())
            right_missing += int(right.isna().sum())
            bytes_inf = np.isposinf(rates_bytes.to_numpy(dtype=float, na_value=np.nan))
            packets_inf = np.isposinf(rates_packets.to_numpy(dtype=float, na_value=np.nan))
            duration_zero = duration.eq(0).to_numpy()
            bytes_missing = rates_bytes.isna().to_numpy()
            inf_bytes += int(bytes_inf.sum())
            inf_packets += int(packets_inf.sum())
            inf_bytes_duration_zero += int((bytes_inf & duration_zero).sum())
            inf_packets_duration_zero += int((packets_inf & duration_zero).sum())
            missing_bytes += int(bytes_missing.sum())
            missing_bytes_duration_zero += int((bytes_missing & duration_zero).sum())
            missing_bytes_packets_inf += int((bytes_missing & packets_inf).sum())
            both_inf += int((bytes_inf & packets_inf).sum())
    return {
        "rows_compared": compared,
        "fwd_header_length_equal": equal,
        "fwd_header_length_different": unequal,
        "fwd_header_length_left_missing": left_missing,
        "fwd_header_length_right_missing": right_missing,
        "positive_inf_flow_bytes": inf_bytes,
        "positive_inf_flow_bytes_with_duration_zero": inf_bytes_duration_zero,
        "positive_inf_flow_packets": inf_packets,
        "positive_inf_flow_packets_with_duration_zero": inf_packets_duration_zero,
        "missing_flow_bytes": missing_bytes,
        "missing_flow_bytes_with_duration_zero": missing_bytes_duration_zero,
        "missing_flow_bytes_with_packets_inf": missing_bytes_packets_inf,
        "both_rates_positive_inf": both_inf,
    }


def main() -> None:
    database = Path(os.environ["TEMP"]) / "cic_ids2017_methodology.sqlite"
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        """
        CREATE TABLE exact_lines (
            digest BLOB PRIMARY KEY,
            label TEXT NOT NULL,
            mask INTEGER NOT NULL,
            n INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE feature_lines (
            digest BLOB NOT NULL,
            label TEXT NOT NULL,
            mask INTEGER NOT NULL,
            n INTEGER NOT NULL,
            PRIMARY KEY (digest, label)
        )
        """
    )
    try:
        scanned = scan(connection)
        exact_by_label = label_summary(connection, "exact_lines")
        totals = connection.execute(
            "SELECT COALESCE(SUM(n), 0), COUNT(*), COALESCE(SUM(n - 1), 0) FROM exact_lines"
        ).fetchone()
        feature_totals = connection.execute(
            "SELECT COALESCE(SUM(n), 0), COUNT(*), COUNT(DISTINCT digest) FROM feature_lines"
        ).fetchone()
        payload = {
            "raw_files_modified": False,
            "rows_deleted": False,
            "training_performed": False,
            "split_saved": False,
            "models_retained": ["logistic_regression", "random_forest", "xgboost"],
            "files": scanned["files"],
            "exact_duplicates": {
                "rows": int(totals[0]),
                "distinct_rows": int(totals[1]),
                "extra_rows": int(totals[2]),
                "rows_after_keeping_one_exact_copy": int(totals[1]),
                "by_label": exact_by_label,
                "across_files": cross_file_exact(connection),
                "file_pairs": shared_pairs(connection, "exact_lines"),
            },
            "feature_duplicates": {
                "rows": int(feature_totals[0]),
                "label_specific_groups": int(feature_totals[1]),
                "distinct_feature_contents": int(feature_totals[2]),
                "conflicts": feature_conflicts(connection),
                "file_pairs": shared_pairs(connection, "feature_lines"),
            },
            "column_anomalies": column_anomalies(),
        }
        by_label = {item["label"]: item for item in exact_by_label}
        payload["simulated_group_split_not_applied"] = {
            "rule": "Pour chaque contenu exact distinct, train = n - 2 * (n * 15 // 100), validation = test = n * 15 // 100. Simulation de comptage, indices non enregistrés.",
            "by_label": [
                {"label": label, **allocate(item["distinct_rows"])}
                for label, item in by_label.items()
            ],
        }
    finally:
        connection.close()
        if database.exists():
            database.unlink()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["exact_duplicates"]["across_files"], ensure_ascii=False), flush=True)
    print(json.dumps(payload["feature_duplicates"]["conflicts"], ensure_ascii=False), flush=True)
    print(json.dumps(payload["column_anomalies"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
