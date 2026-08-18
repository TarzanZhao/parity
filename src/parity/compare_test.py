"""Tests for src/parity/compare.py.

python -m pytest src/parity/compare_test.py -v
"""

from __future__ import annotations

import json

import pytest

from parity import compare as compare_mod


def write(path, rows):
    path.write_text(json.dumps(rows))
    return path


def rec(tag, value, tol=None):
    row = {"tag": tag, "value": value}
    if tol is not None:
        row["error_tolerance"] = tol
    return row


def test_pass_within_declared_tolerance(tmp_path):
    base = write(tmp_path / "b.json", [rec("loss_step0", 2.0, {"rtol": 0.1})])
    arm = write(tmp_path / "a.json", [rec("loss_step0", 2.1, {"rtol": 0.1})])
    report = compare_mod.compare(base, arm)
    assert report.ok and report.n_pass == 1


def test_fail_beyond_declared_tolerance(tmp_path):
    base = write(tmp_path / "b.json", [rec("loss_step0", 2.0, {"rtol": 0.01})])
    arm = write(tmp_path / "a.json", [rec("loss_step0", 2.5, {"rtol": 0.01})])
    report = compare_mod.compare(base, arm)
    assert not report.ok and report.failures[0].tag == "loss_step0"


def test_pairing_is_by_tag_not_position(tmp_path):
    base = write(tmp_path / "b.json", [rec("a", 1.0), rec("b", 2.0)])
    arm = write(tmp_path / "a.json", [rec("b", 2.0), rec("a", 1.0)])
    assert compare_mod.compare(base, arm).ok


def test_an_inserted_checkpoint_is_one_extra_tag_not_a_shift(tmp_path):
    base = write(tmp_path / "b.json", [rec("a", 1.0), rec("c", 3.0)])
    arm = write(tmp_path / "a.json", [rec("a", 1.0), rec("b", 2.0), rec("c", 3.0)])
    report = compare_mod.compare(base, arm)
    assert report.extra == ["b"] and not report.missing
    assert report.n_pass == 2 and not report.failures


def test_a_run_that_stopped_early_fails_structurally(tmp_path):
    base = write(tmp_path / "b.json",
                 [rec("loss_step0", 1.0), rec("loss_step1", 0.9), rec("loss_step2", 0.8)])
    arm = write(tmp_path / "a.json", [rec("loss_step0", 1.0), rec("loss_step1", 0.9)])
    report = compare_mod.compare(base, arm)
    assert not report.ok, "matching on the steps it managed must not pass"
    assert report.missing == ["loss_step2"]


def test_baseline_tolerance_wins_and_a_changed_one_is_reported(tmp_path):
    base = write(tmp_path / "b.json", [rec("x", 1.0, {"rtol": 1e-3})])
    arm = write(tmp_path / "a.json", [rec("x", 1.5, {"rtol": 10.0})])
    report = compare_mod.compare(base, arm)
    assert not report.ok, "the arm must not be able to loosen its own gate"
    assert [t for t, _, _ in report.retoleranced] == ["x"]
    assert "DIFFERENT tolerance" in compare_mod.render(report)


def test_rtol_override_applies_to_every_tag(tmp_path):
    base = write(tmp_path / "b.json", [rec("x", 1.0, {"rtol": 1e-9})])
    arm = write(tmp_path / "a.json", [rec("x", 1.05, {"rtol": 1e-9})])
    assert not compare_mod.compare(base, arm).ok
    assert compare_mod.compare(base, arm, rtol=0.1).ok


def test_only_and_exclude_select_tags(tmp_path):
    base = write(tmp_path / "b.json", [rec("loss_step0", 1.0), rec("wall_ms", 1000.0)])
    arm = write(tmp_path / "a.json", [rec("loss_step0", 1.0), rec("wall_ms", 250.0)])
    assert not compare_mod.compare(base, arm).ok
    assert compare_mod.compare(base, arm, exclude=["wall_ms"]).ok
    assert compare_mod.compare(base, arm, only=["loss_*"]).ok


def test_bare_float_tolerance_still_loads(tmp_path):
    base = write(tmp_path / "b.json", [rec("x", 1.0, 0.1)])
    arm = write(tmp_path / "a.json", [rec("x", 1.05, 0.1)])
    assert compare_mod.compare(base, arm).ok


def test_per_rank_directories_pair_by_filename(tmp_path):
    bdir, adir = tmp_path / "base", tmp_path / "arm"
    bdir.mkdir()
    adir.mkdir()
    for d, v in ((bdir, 1.0), (adir, 1.0)):
        write(d / "run.rank0.json", [rec("loss", v)])
        write(d / "run.rank1.json", [rec("loss", v)])
    assert compare_mod.compare(bdir, adir).ok


def test_a_missing_rank_file_is_a_structural_failure(tmp_path):
    bdir, adir = tmp_path / "base", tmp_path / "arm"
    bdir.mkdir()
    adir.mkdir()
    write(bdir / "run.rank0.json", [rec("loss", 1.0)])
    write(bdir / "run.rank1.json", [rec("loss", 1.0)])
    write(adir / "run.rank0.json", [rec("loss", 1.0)])
    report = compare_mod.compare(bdir, adir)
    assert not report.ok and report.missing == ["run.rank1.json:loss"]


def test_empty_record_is_rejected(tmp_path):
    empty = write(tmp_path / "b.json", [])
    with pytest.raises(SystemExit, match="passes trivially"):
        compare_mod.load(empty)


def test_duplicate_tag_in_a_file_is_rejected(tmp_path):
    dup = write(tmp_path / "b.json", [rec("x", 1.0), rec("x", 2.0)])
    with pytest.raises(SystemExit, match="duplicate tag"):
        compare_mod.load(dup)


def test_unknown_key_in_a_record_is_rejected(tmp_path):
    bad = tmp_path / "b.json"
    bad.write_text(json.dumps([{"tag": "x", "value": 1.0, "rtol": 1e-3}]))
    with pytest.raises(SystemExit, match="unknown key"):
        compare_mod.load(bad)


def test_render_says_fail_when_only_the_structure_is_wrong(tmp_path):
    base = write(tmp_path / "b.json", [rec("a", 1.0), rec("b", 2.0)])
    arm = write(tmp_path / "a.json", [rec("a", 1.0)])
    report = compare_mod.compare(base, arm)
    text = compare_mod.render(report)
    assert not report.ok and "FAIL" in text and "PASS" not in text


def test_render_names_the_closest_call_on_a_pass(tmp_path):
    base = write(tmp_path / "b.json", [rec("x", 1.0, {"rtol": 0.1}), rec("y", 1.0, {"rtol": 0.1})])
    arm = write(tmp_path / "a.json", [rec("x", 1.09, {"rtol": 0.1}), rec("y", 1.0, {"rtol": 0.1})])
    text = compare_mod.render(compare_mod.compare(base, arm))
    assert "PASS" in text and "closest call: x" in text
