"""Tests for src/parity/derive.py.

python -m pytest src/parity/derive_test.py -v
"""

from __future__ import annotations

import json

import pytest

from parity import derive as derive_mod


def repeats(tmp_path, tag, values, tol=None):
    """Write one file per value, all holding the same single tag."""
    paths = []
    for i, v in enumerate(values):
        row = {"tag": tag, "value": v}
        if tol is not None:
            row["error_tolerance"] = tol
        p = tmp_path / f"run{i}.json"
        p.write_text(json.dumps([row]))
        paths.append(p)
    return paths


def verdict_for(report, tag):
    return next(t for t in report.tags if t.tag == tag).verdict


def test_bit_identical_repeats_under_the_default_gate(tmp_path):
    report = derive_mod.derive(repeats(tmp_path, "loss", [2.0, 2.0, 2.0]))
    assert report.ok and verdict_for(report, "loss") == derive_mod.IDENTICAL


def test_declared_tighter_than_the_noise_is_flaky(tmp_path):
    paths = repeats(tmp_path, "loss", [2.000, 2.010, 2.005], tol={"rtol": 1e-6})
    report = derive_mod.derive(paths)
    assert not report.ok
    assert verdict_for(report, "loss") == derive_mod.FLAKY


def test_a_flaky_tag_suggests_a_tolerance_that_would_hold(tmp_path):
    paths = repeats(tmp_path, "loss", [2.000, 2.010, 2.005], tol={"rtol": 1e-6})
    noise = derive_mod.derive(paths).tags[0]
    assert noise.suggestion is not None
    assert noise.suggestion.budget(noise.magnitude) >= noise.spread


def test_declared_far_looser_than_the_noise_is_vacuous(tmp_path):
    paths = repeats(tmp_path, "loss", [2.000, 2.000001, 2.0000005], tol={"rtol": 0.10})
    report = derive_mod.derive(paths)
    assert verdict_for(report, "loss") == derive_mod.VACUOUS
    assert report.ok, "a loose gate is a warning, not a failure — slack may be intentional"


def test_a_tolerance_matched_to_the_noise_is_ok(tmp_path):
    paths = repeats(tmp_path, "loss", [2.000, 2.002, 2.001], tol={"rtol": 5e-3})
    assert verdict_for(derive_mod.derive(paths), "loss") == derive_mod.OK


def test_slack_declared_over_bit_identical_repeats_is_flagged_not_failed(tmp_path):
    paths = repeats(tmp_path, "loss", [2.0, 2.0, 2.0], tol={"rtol": 0.1})
    report = derive_mod.derive(paths)
    assert verdict_for(report, "loss") == derive_mod.SLACK and report.ok


def test_an_int_that_moves_with_the_code_unchanged_is_flaky(tmp_path):
    report = derive_mod.derive(repeats(tmp_path, "n_gaussians", [661878, 661882, 661875]))
    assert not report.ok
    assert verdict_for(report, "n_gaussians") == derive_mod.FLAKY


def test_a_string_that_moves_is_unstable(tmp_path):
    report = derive_mod.derive(repeats(tmp_path, "caption", ["a cat", "a cat", "a dog"]))
    assert not report.ok and verdict_for(report, "caption") == derive_mod.UNSTABLE


def test_a_type_change_across_repeats_is_unstable(tmp_path):
    report = derive_mod.derive(repeats(tmp_path, "n", [1, 1.0, 1]))
    assert not report.ok and verdict_for(report, "n") == derive_mod.UNSTABLE


def test_nan_in_the_baseline_stops_everything(tmp_path):
    report = derive_mod.derive(repeats(tmp_path, "loss", [2.0, float("nan"), 2.0]))
    assert not report.ok
    assert "NaN" in report.tags[0].detail


def test_a_near_zero_value_gets_an_absolute_suggestion(tmp_path):
    paths = repeats(tmp_path, "residual", [1e-9, -2e-9, 5e-10], tol={"rtol": 1e-6})
    noise = derive_mod.derive(paths).tags[0]
    assert noise.suggestion.rtol == 0.0 and noise.suggestion.atol > 0


def test_a_near_zero_value_cannot_drag_another_tag_loose(tmp_path):
    """Per-tag tolerance is the point: one ungateable value stays its own problem."""
    for i, (r, loss) in enumerate([(1e-9, 2.0), (-2e-9, 2.0), (5e-10, 2.0)]):
        (tmp_path / f"run{i}.json").write_text(json.dumps([
            {"tag": "residual", "value": r, "error_tolerance": {"rtol": 1e-6}},
            {"tag": "loss", "value": loss, "error_tolerance": {"rtol": 1e-6}},
        ]))
    report = derive_mod.derive(sorted(tmp_path.glob("run*.json")))
    assert verdict_for(report, "residual") == derive_mod.FLAKY
    assert verdict_for(report, "loss") == derive_mod.SLACK


def test_repeats_holding_different_checkpoints_are_not_comparable(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([{"tag": "x", "value": 1.0}]))
    (tmp_path / "b.json").write_text(json.dumps([{"tag": "x", "value": 1.0}]))
    (tmp_path / "c.json").write_text(json.dumps([{"tag": "y", "value": 1.0}]))
    with pytest.raises(SystemExit, match="not comparable"):
        derive_mod.derive([tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"])


def test_fewer_than_three_repeats_is_called_out(tmp_path):
    report = derive_mod.derive(repeats(tmp_path, "loss", [2.0, 2.0]))
    assert "fewer than 3 cannot measure it" in derive_mod.render(report)
