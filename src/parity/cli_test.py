"""Tests for src/parity/cli.py.

python -m pytest src/parity/cli_test.py -v
"""

from __future__ import annotations

import json

import pytest

from parity import cli


def write(path, rows):
    path.write_text(json.dumps(rows))
    return str(path)


def test_compare_exits_zero_on_a_pass(tmp_path, capsys):
    base = write(tmp_path / "b.json", [{"tag": "x", "value": 1.0,
                                        "error_tolerance": {"rtol": 0.1}}])
    arm = write(tmp_path / "a.json", [{"tag": "x", "value": 1.05,
                                       "error_tolerance": {"rtol": 0.1}}])
    assert cli.main(["compare", base, arm]) == 0
    assert "PASS" in capsys.readouterr().out


def test_compare_exits_one_on_a_fail(tmp_path, capsys):
    base = write(tmp_path / "b.json", [{"tag": "x", "value": 1.0}])
    arm = write(tmp_path / "a.json", [{"tag": "x", "value": 2.0}])
    assert cli.main(["compare", base, arm]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_compare_accepts_overrides_and_filters(tmp_path):
    base = write(tmp_path / "b.json", [{"tag": "x", "value": 1.0},
                                       {"tag": "wall_ms", "value": 100.0}])
    arm = write(tmp_path / "a.json", [{"tag": "x", "value": 1.01},
                                      {"tag": "wall_ms", "value": 20.0}])
    assert cli.main(["compare", base, arm, "--rtol", "0.05",
                     "--exclude", "wall_ms"]) == 0


def test_derive_exits_one_when_a_gate_cannot_hold(tmp_path, capsys):
    paths = []
    for i, v in enumerate([2.0, 2.01, 2.005]):
        paths.append(write(tmp_path / f"r{i}.json",
                           [{"tag": "loss", "value": v, "error_tolerance": {"rtol": 1e-9}}]))
    assert cli.main(["derive", *paths]) == 1
    assert "FLAKY" in capsys.readouterr().out


def test_derive_exits_zero_when_every_gate_holds(tmp_path):
    paths = [write(tmp_path / f"r{i}.json", [{"tag": "loss", "value": 2.0}])
             for i in range(3)]
    assert cli.main(["derive", *paths]) == 0


def test_show_prints_every_checkpoint(tmp_path, capsys):
    rec = write(tmp_path / "r.json", [{"tag": "x", "value": 1.0},
                                      {"tag": "y", "value": 2}])
    assert cli.main(["show", rec]) == 0
    out = capsys.readouterr().out
    assert "x" in out and "y" in out and "2 checkpoint(s)" in out


def test_show_json_round_trips(tmp_path, capsys):
    rec = write(tmp_path / "r.json", [{"tag": "x", "value": 1.0,
                                       "error_tolerance": {"rtol": 0.1}}])
    cli.main(["show", rec, "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["tag"] == "x" and rows[0]["error_tolerance"]["rtol"] == 0.1


def test_no_subcommand_is_an_error(capsys):
    with pytest.raises(SystemExit):
        cli.main([])
