"""Règles de nettoyage et de découpage, sans les CSV bruts."""

from __future__ import annotations

import numpy as np

from explainable_ids.prepare import (
    ChronologicalKeeper,
    assign_grouped_indices,
    binary_target,
    contradictory_feature_digests,
    grouped_allocation,
    parse_feature,
    strip_header,
    value_changed_by_float32,
)


def test_binary_target_keeps_the_original_character_out_of_the_code() -> None:
    original = "Web Attack \uFFFD XSS"
    assert binary_target("BENIGN") == 0
    assert binary_target(original) == 1
    assert "\uFFFD" in original


def test_infinity_becomes_missing_and_missing_rows_stay_representable() -> None:
    assert parse_feature("Infinity") == (None, "infinity")
    assert parse_feature("-Infinity") == (None, "infinity")
    assert parse_feature("NaN") == (None, "missing")
    assert parse_feature("") == (None, "missing")
    value, kind = parse_feature("1.5")
    assert kind == "number"
    assert value == np.float32(1.5)


def test_float32_loss_is_detected_without_dropping_the_value() -> None:
    assert value_changed_by_float32("16777217")
    assert not value_changed_by_float32("1")
    parsed, kind = parse_feature("16777217")
    assert kind == "number"
    assert parsed == float(np.float32(16777217))


def test_header_strips_spaces() -> None:
    assert strip_header([" Label", " Flow Duration "]) == ["Label", "Flow Duration"]


def test_contradictory_profile_and_later_exact_copy_are_excluded() -> None:
    labels = {b"same": "*", b"ok": "BENIGN"}
    keeper = ChronologicalKeeper(contradictory_feature_digests(labels))
    assert keeper.keep(b"same", b"attack-line", "PortScan", "test") is False
    assert keeper.keep(b"ok", b"exact", "BENIGN", "train") is True
    assert keeper.keep(b"ok", b"exact", "BENIGN", "test") is False
    assert keeper.contradictory_rows == 1
    assert keeper.cross_contents == {b"exact"}
    assert keeper.attack_cross == 0


def test_grouped_allocation_uses_the_integer_rule_and_one_seed() -> None:
    assert grouped_allocation(11) == (9, 1, 1)
    assert grouped_allocation(21) == (15, 3, 3)
    labels = ["BENIGN", "BENIGN", "Hulk", "Hulk", "BENIGN", "Hulk"]
    first = assign_grouped_indices(labels, seed=42)
    second = assign_grouped_indices(labels, seed=42)
    for name in first:
        assert np.array_equal(first[name], second[name])
    combined = np.concatenate(list(first.values()))
    assert combined.size == len(labels)
    assert np.unique(combined).size == len(labels)
    benign = [index for index, label in enumerate(labels) if label == "BENIGN"]
    benign_train = [index for index in benign if index in set(first["train"].tolist())]
    assert len(benign_train) == grouped_allocation(len(benign))[0]
