"""Mesure l'écart float32, en lecture seule.

N'écrit ni le parquet, ni les indices, ni les CSV bruts.
N'ajuste aucun modèle.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explainable_ids.config import load_config, resolve_paths
from explainable_ids.prepare import (
    FILES,
    ChronologicalKeeper,
    _inspect_header,
    _iter_data_lines,
    _scan_identities,
    chronological_split,
    contradictory_feature_digests,
)

REPORT = Path(__file__).resolve().parents[1] / "outputs" / "metrics" / "float32_verification.json"


def is_integer_token(field: str) -> bool:
    if field.startswith("-"):
        return field[1:].isdigit()
    return field.isdigit()


def measure(config: dict) -> dict:
    raw_dir = resolve_paths(config)["raw"]
    _header, first_index, drop_index, names = _inspect_header(raw_dir)
    identities, mismatches = _scan_identities(raw_dir, first_index, drop_index)
    contradictory = contradictory_feature_digests(identities)
    del identities
    width = len(names)
    count = np.zeros(width, dtype=np.int64)
    changed = np.zeros(width, dtype=np.int64)
    integers = np.zeros(width, dtype=np.int64)
    integers_changed = np.zeros(width, dtype=np.int64)
    integers_off_by_at_least_one = np.zeros(width, dtype=np.int64)
    sum_abs = np.zeros(width, dtype=np.float64)
    max_abs = np.zeros(width, dtype=np.float64)
    sum_rel = np.zeros(width, dtype=np.float64)
    max_rel = np.zeros(width, dtype=np.float64)
    nonzero = np.zeros(width, dtype=np.int64)
    minimum = np.full(width, np.inf)
    maximum = np.full(width, -np.inf)
    mean = np.zeros(width, dtype=np.float64)
    m2 = np.zeros(width, dtype=np.float64)
    worst_original = np.zeros(width, dtype=np.float64)
    worst_stored = np.zeros(width, dtype=np.float64)
    keeper = ChronologicalKeeper(contradictory)
    kept = 0
    for name, day in FILES:
        split = chronological_split(day, config)
        print(f"Mesure {name}", flush=True)
        for _line_number, raw, _header_line in _iter_data_lines(raw_dir / name, raw_dir):
            fields = next(csv.reader([raw.decode("utf-8")]))
            comma = raw.rfind(b",")
            feature_digest = hashlib.md5(raw[:comma]).digest()
            exact_digest = hashlib.md5(raw).digest()
            if not keeper.keep(feature_digest, exact_digest, fields[-1], split):
                continue
            kept += 1
            column = 0
            for index, field in enumerate(fields[:-1]):
                if index == drop_index:
                    continue
                if field in {"", "NaN", "nan", "NA", "N/A", "n/a", "NULL", "null", "None"}:
                    column += 1
                    continue
                number = float(field)
                if not np.isfinite(number):
                    column += 1
                    continue
                stored = float(np.float32(number))
                error = abs(stored - number)
                count[column] += 1
                sum_abs[column] += error
                if error > max_abs[column]:
                    max_abs[column] = error
                    worst_original[column] = number
                    worst_stored[column] = stored
                if number != 0.0:
                    relative = error / abs(number)
                    sum_rel[column] += relative
                    nonzero[column] += 1
                    if relative > max_rel[column]:
                        max_rel[column] = relative
                if error != 0.0:
                    changed[column] += 1
                delta = number - mean[column]
                mean[column] += delta / count[column]
                m2[column] += delta * (number - mean[column])
                if number < minimum[column]:
                    minimum[column] = number
                if number > maximum[column]:
                    maximum[column] = number
                if is_integer_token(field):
                    integers[column] += 1
                    if stored != number:
                        integers_changed[column] += 1
                        if error >= 1.0:
                            integers_off_by_at_least_one[column] += 1
                column += 1
    if mismatches:
        raise RuntimeError("Les deux colonnes Fwd Header Length ne sont pas égales.")
    if kept != 2_520_966:
        raise RuntimeError(f"Lignes conservées inattendues : {kept}.")
    columns = []
    for index, name in enumerate(names):
        finite = int(count[index])
        std = float(np.sqrt(m2[index] / finite)) if finite > 1 else 0.0
        data_range = float(maximum[index] - minimum[index]) if finite else 0.0
        columns.append(
            {
                "name": name,
                "finite_values": finite,
                "changed_values": int(changed[index]),
                "max_absolute_error": float(max_abs[index]),
                "mean_absolute_error": float(sum_abs[index] / finite) if finite else 0.0,
                "mean_absolute_error_on_changed": (
                    float(sum_abs[index] / changed[index]) if changed[index] else 0.0
                ),
                "max_relative_error": float(max_rel[index]) if nonzero[index] else 0.0,
                "mean_relative_error": (
                    float(sum_rel[index] / nonzero[index]) if nonzero[index] else 0.0
                ),
                "integer_tokens": int(integers[index]),
                "integer_values_changed": int(integers_changed[index]),
                "integer_values_off_by_at_least_one": int(integers_off_by_at_least_one[index]),
                "all_tokens_are_integers": finite > 0 and int(integers[index]) == finite,
                "std": std,
                "max_error_in_standard_units": float(max_abs[index] / std) if std else 0.0,
                "range": data_range,
                "histogram_bin_width_256": data_range / 256.0 if data_range else 0.0,
        "max_error_over_bin_width": (
            float(max_abs[index] / (data_range / 256.0)) if data_range else 0.0
        ),
                "worst_original": float(worst_original[index]),
                "worst_stored": float(worst_stored[index]),
            }
        )
    return {
        "kept_rows": kept,
        "fwd_header_mismatches": mismatches,
        "changed_values": int(changed.sum()),
        "finite_values": int(count.sum()),
        "columns": columns,
    }


def benchmark() -> dict:
    rows = 400_000
    cols = 77
    generator = np.random.default_rng(42)
    timings = {}
    for dtype in (np.float32, np.float64):
        matrix = generator.standard_normal((rows, cols)).astype(dtype, copy=False)
        started = time.perf_counter()
        StandardScaler().fit(matrix)
        timings[np.dtype(dtype).name] = time.perf_counter() - started
    kept = 2_520_966
    return {
        "sample_rows": rows,
        "sample_columns": cols,
        "scaler_fit_seconds_float32": timings["float32"],
        "scaler_fit_seconds_float64": timings["float64"],
        "feature_matrix_bytes_float32": kept * cols * 4,
        "feature_matrix_bytes_float64": kept * cols * 8,
        "sklearn_tree_dtype": str(np.dtype(np.float32)),
    }


def library_probe() -> dict:
    return {
        "random_forest_fit_casts_to": "float32",
        "random_forest_source": "sklearn.tree._classes.BaseDecisionTree.fit",
        "xgboost_converts_input_to": "float32",
        "logistic_regression_accepted_dtypes": ["float64", "float32"],
        "logistic_regression_keeps_input_dtype": True,
    }


def main() -> None:
    config = load_config()
    measured = measure(config)
    measured["files_modified"] = False
    measured["training_performed"] = False
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(measured, indent=2, ensure_ascii=False), encoding="utf-8")
    measured["benchmark"] = benchmark()
    measured["libraries"] = library_probe()
    REPORT.write_text(json.dumps(measured, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"changed {measured['changed_values']}", flush=True)


if __name__ == "__main__":
    main()
