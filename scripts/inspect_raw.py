"""Inspection descriptive des CSV bruts CIC-IDS2017.

Lecture seule : aucun fichier de C:\\data\\cic-ids2017\\raw n'est ouvert en écriture.
Aucun modèle n'est ajusté.
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
REPORT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "metrics" / "inspection.json"
CHUNK_ROWS = 50_000
DAY_BY_FILE = {
    "Monday-WorkingHours.pcap_ISCX.csv": "Monday",
    "Tuesday-WorkingHours.pcap_ISCX.csv": "Tuesday",
    "Wednesday-workingHours.pcap_ISCX.csv": "Wednesday",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": "Thursday-Morning",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": "Thursday-Afternoon",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": "Friday-Morning",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": "Friday-Afternoon-PortScan",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": "Friday-Afternoon-DDoS",
}


def assert_raw_csv(path: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != RAW_DIR.resolve() or resolved.suffix.lower() != ".csv":
        raise SystemExit(f"Chemin refusé : {resolved}")


def header_and_newline(path: Path) -> tuple[list[str], str, int]:
    crlf = 0
    lf_only = 0
    with path.open("rb") as handle:
        sample = handle.read(1024 * 1024)
    crlf = sample.count(b"\r\n")
    lf_only = sample.count(b"\n") - crlf
    newline = "crlf" if crlf >= lf_only else "lf"
    with path.open("rb") as handle:
        header = handle.readline().rstrip(b"\r\n")
    text = header.decode("utf-8")
    columns = text.split(",")
    return columns, newline, len(header)


def scan_lines(path: Path, file_bit: int, connection: sqlite3.Connection) -> dict[str, int]:
    seen_in_file: set[bytes] = set()
    rows = 0
    extra_in_file = 0
    utf8_errors = 0
    empty_rows = 0
    batch: list[tuple[bytes, int]] = []
    with path.open("rb") as handle:
        handle.readline()
        for line in handle:
            payload = line.rstrip(b"\r\n")
            if not payload:
                empty_rows += 1
                continue
            try:
                payload.decode("utf-8")
            except UnicodeDecodeError:
                utf8_errors += 1
            rows += 1
            digest = hashlib.md5(payload).digest()
            if digest in seen_in_file:
                extra_in_file += 1
            else:
                seen_in_file.add(digest)
            batch.append((digest, file_bit))
            if len(batch) >= 20_000:
                _flush(connection, batch)
                batch.clear()
            if rows % 200_000 == 0:
                print(f"  lignes lues {rows}", flush=True)
    if batch:
        _flush(connection, batch)
    connection.commit()
    return {
        "data_rows_file": rows,
        "extra_duplicate_rows_in_file": extra_in_file,
        "unique_rows_in_file": len(seen_in_file),
        "utf8_errors": utf8_errors,
        "empty_rows": empty_rows,
    }


def _flush(connection: sqlite3.Connection, batch: list[tuple[bytes, int]]) -> None:
    connection.executemany(
        """
        INSERT INTO lines(digest, n, mask) VALUES (?, 1, ?)
        ON CONFLICT(digest) DO UPDATE SET
            n = n + 1,
            mask = mask | excluded.mask
        """,
        batch,
    )


def inspect_frame(path: Path, label_name: str) -> dict[str, object]:
    nulls: Counter[str] = Counter()
    positive_infs: Counter[str] = Counter()
    negative_infs: Counter[str] = Counter()
    dtypes: dict[str, set[str]] = {}
    labels: Counter[str] = Counter()
    parsed_rows = 0
    normal_rows = 0
    attack_rows = 0
    missing_label_rows = 0
    examples: list[dict[str, str]] = []
    datetime_columns: list[str] = []

    reader = pd.read_csv(
        path,
        chunksize=CHUNK_ROWS,
        low_memory=False,
        encoding="utf-8",
    )
    for chunk_index, chunk in enumerate(reader):
        parsed_rows += int(len(chunk))
        if chunk_index == 0:
            datetime_columns = [
                column
                for column in chunk.columns
                if any(token in column.lower() for token in ("time", "date", "stamp"))
            ]
        for column in chunk.columns:
            dtypes.setdefault(column, set()).add(str(chunk[column].dtype))
            series = chunk[column]
            nulls[column] += int(series.isna().sum())
            if pd.api.types.is_numeric_dtype(series):
                values = series.to_numpy(dtype=float, na_value=np.nan)
                positive_infs[column] += int(np.isposinf(values).sum())
                negative_infs[column] += int(np.isneginf(values).sum())
            else:
                text = series.astype("string").str.strip().str.lower()
                positive_infs[column] += int(text.isin(["inf", "+inf", "infinity", "+infinity"]).sum())
                negative_infs[column] += int(text.isin(["-inf", "-infinity"]).sum())
        label_series = chunk[label_name]
        for value in label_series:
            if pd.isna(value):
                labels["<missing>"] += 1
                missing_label_rows += 1
                continue
            text = str(value)
            labels[text] += 1
            if text.strip() == "BENIGN":
                normal_rows += 1
            else:
                attack_rows += 1
        if len(examples) < 3:
            for _, row in chunk.head(3 - len(examples)).iterrows():
                examples.append(
                    {
                        "label": "" if pd.isna(row[label_name]) else str(row[label_name]),
                        "destination_port": _cell(row, " Destination Port"),
                        "flow_duration": _cell(row, " Flow Duration"),
                    }
                )
        print(f"  morceau {chunk_index + 1} lignes {parsed_rows}", flush=True)
    return {
        "parsed_rows": parsed_rows,
        "normal_rows": normal_rows,
        "attack_rows": attack_rows,
        "missing_label_rows": missing_label_rows,
        "label_counts": dict(labels),
        "nulls": {key: value for key, value in nulls.items() if value},
        "positive_infs": {key: value for key, value in positive_infs.items() if value},
        "negative_infs": {key: value for key, value in negative_infs.items() if value},
        "dtypes": {key: sorted(value) for key, value in dtypes.items()},
        "datetime_columns": datetime_columns,
        "examples": examples,
        "null_cells": int(sum(nulls.values())),
        "positive_inf_cells": int(sum(positive_infs.values())),
        "negative_inf_cells": int(sum(negative_infs.values())),
    }


def _cell(row: pd.Series, column: str) -> str:
    if column not in row.index or pd.isna(row[column]):
        return ""
    return str(row[column])


def global_duplicates(connection: sqlite3.Connection) -> dict[str, int]:
    total_rows, distinct_rows, extra_rows = connection.execute(
        """
        SELECT COALESCE(SUM(n), 0), COUNT(*),
               COALESCE(SUM(CASE WHEN n > 1 THEN n - 1 ELSE 0 END), 0)
        FROM lines
        """
    ).fetchone()
    shared_contents, shared_rows = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(n), 0)
        FROM lines
        WHERE mask > 0 AND (mask & (mask - 1)) != 0
        """
    ).fetchone()
    return {
        "total_rows": int(total_rows),
        "distinct_rows": int(distinct_rows),
        "extra_duplicate_rows": int(extra_rows),
        "contents_found_in_several_files": int(shared_contents),
        "rows_of_contents_found_in_several_files": int(shared_rows),
    }


def main() -> None:
    files = sorted(RAW_DIR.glob("*.csv"))
    if len(files) != 8:
        raise SystemExit(f"Nombre de CSV inattendu : {len(files)}")
    for path in files:
        assert_raw_csv(path)

    database = Path(os.environ["TEMP"]) / "cic_ids2017_line_hashes.sqlite"
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE lines (digest BLOB PRIMARY KEY, n INTEGER NOT NULL, mask INTEGER NOT NULL)"
    )

    file_reports: list[dict[str, object]] = []
    try:
        reference_columns: list[str] | None = None
        for index, path in enumerate(files):
            print(f"Fichier {path.name}", flush=True)
            columns, newline, _header_bytes = header_and_newline(path)
            if reference_columns is None:
                reference_columns = columns
            line_stats = scan_lines(path, 1 << index, connection)
            label_candidates = [column for column in columns if column.strip().lower() == "label"]
            if len(label_candidates) != 1:
                raise SystemExit(f"Colonne d'étiquette introuvable dans {path.name}")
            frame_stats = inspect_frame(path, label_candidates[0])
            stripped = [column.strip() for column in columns]
            file_reports.append(
                {
                    "name": path.name,
                    "day": DAY_BY_FILE.get(path.name, path.name),
                    "bytes": path.stat().st_size,
                    "newline": newline,
                    "encoding_checked": "utf-8",
                    "columns_raw": columns,
                    "columns_stripped": stripped,
                    "n_columns": len(columns),
                    "duplicate_raw_column_names": sorted(
                        {name for name in columns if columns.count(name) > 1}
                    ),
                    "duplicate_stripped_column_names": sorted(
                        {name for name in stripped if stripped.count(name) > 1}
                    ),
                    "same_header_as_first_file": columns == reference_columns,
                    **line_stats,
                    **frame_stats,
                }
            )
        summary = global_duplicates(connection)
    finally:
        connection.close()
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(database) + suffix)
            if candidate.exists():
                candidate.unlink()

    payload = {
        "raw_dir": str(RAW_DIR),
        "csv_count": len(file_reports),
        "files": file_reports,
        "schema_identical": all(item["same_header_as_first_file"] for item in file_reports),
        "duplicates": summary,
        "models_retained": ["logistic_regression", "random_forest", "xgboost"],
        "split_chosen": False,
        "cleaning_performed": False,
        "raw_files_modified": False,
        "training_performed": False,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Rapport {REPORT_PATH}", flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
