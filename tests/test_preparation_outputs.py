"""Contrats du fichier dérivé et des deux découpages enregistrés."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from explainable_ids.config import PROJECT_ROOT
from explainable_ids.prepare import grouped_allocation

MANIFEST = PROJECT_ROOT / "outputs" / "metrics" / "preparation_manifest.json"
RAW = Path(r"C:\data\cic-ids2017\raw")


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_raw_files_keep_the_recorded_sizes() -> None:
    manifest = _manifest()
    assert manifest["raw_files_modified"] is False
    assert manifest["training_performed"] is False
    assert manifest["model_selected"] is False
    assert manifest["test_used_for_decisions"] is False
    for name, fingerprint in manifest["raw_fingerprints_sha256"].items():
        assert (RAW / name).stat().st_size == fingerprint["bytes"]


def test_chronological_counts_and_partition() -> None:
    manifest = _manifest()
    assert manifest["kept_rows"] == 2_520_966
    assert manifest["contradictory_groups_excluded"] == 698
    assert manifest["contradictory_rows_excluded"] == 7_020
    assert manifest["audit"]["infinite_cells"] == 0
    assert manifest["audit"]["contradictory_profiles_remaining"] == 0
    assert manifest["attack_exact_contents_crossing_files"] == 0
    assert (
        manifest["cross_set_exact_contents_blocked"]
        + manifest["contradictory_exact_contents_also_spanning_sets"]
        == 28_024
    )
    pieces = [
        np.load(Path(manifest["chronological"]["indices"][name]["path"]))
        for name in ("train", "validation", "test")
    ]
    combined = np.concatenate(pieces)
    assert combined.size == manifest["kept_rows"]
    assert np.unique(combined).size == manifest["kept_rows"]
    assert [int(piece.size) for piece in pieces] == [1_527_133, 398_110, 595_723]


def test_grouped_indices_follow_the_integer_rule_without_overlap() -> None:
    manifest = _manifest()
    pieces = {
        name: np.load(Path(manifest["grouped"]["indices"][name]["path"]))
        for name in ("train", "validation", "test")
    }
    combined = np.concatenate(list(pieces.values()))
    assert combined.size == manifest["kept_rows"]
    assert np.unique(combined).size == manifest["kept_rows"]
    assert manifest["grouped"]["seed"] == 42
    assert manifest["grouped"]["selects_official_model"] is False
    assert [int(pieces[name].size) for name in ("train", "validation", "test")] == [
        1_764_696,
        378_135,
        378_135,
    ]
    families = manifest["chronological"]["families"]
    heartbleed = sum(families[split].get("Heartbleed", 0) for split in families)
    assert grouped_allocation(heartbleed) == (9, 1, 1)
