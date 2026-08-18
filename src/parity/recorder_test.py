"""Tests for src/parity/recorder.py.

python -m pytest src/parity/recorder_test.py -v
"""

from __future__ import annotations

import json

import pytest

import parity
from parity import recorder


class FakeTensor:
    """Stands in for a device tensor: one element, and `.item()` is the sync."""

    def __init__(self, value, numel: int = 1):
        self._value = value
        self._numel = numel
        self.item_calls = 0

    def numel(self) -> int:
        return self._numel

    def item(self):
        self.item_calls += 1
        return self._value


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.delenv("PARITY", raising=False)
    monkeypatch.delenv("PARITY_OUT", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    parity.reset()
    parity.set_enabled(None)
    parity.set_default_tolerance()
    yield
    parity.reset()
    parity.set_enabled(None)
    parity.set_default_tolerance()


def test_off_by_default_and_writes_nothing(tmp_path):
    parity.record("loss_step0", 1.0)
    assert parity.flush(tmp_path / "out.json") is None
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("value,expected", [("1", True), ("on", True), ("TRUE", True),
                                            ("0", False), ("off", False), ("", False)])
def test_parity_env_var_controls_recording(monkeypatch, value, expected):
    monkeypatch.setenv("PARITY", value)
    parity.set_enabled(None)
    assert parity.enabled() is expected


def test_record_lazy_does_not_call_the_lambda_when_off():
    calls = []
    parity.record_lazy("x", lambda: calls.append(1) or 1.0)
    assert calls == []


def test_record_lazy_calls_the_lambda_when_on(tmp_path):
    parity.set_enabled(True)
    calls = []
    parity.record_lazy("x", lambda: (calls.append(1), 2.0)[1], rtol=1e-2)
    assert calls == [1]
    rows = json.loads(parity.flush(tmp_path / "out.json").read_text())
    assert rows[0]["value"] == 2.0


def test_written_schema_is_tag_value_tolerance(tmp_path):
    parity.set_enabled(True)
    parity.record("loss_step0", 2.5, rtol=1e-3, note="5 repeats")
    parity.record("n_gaussians", 661878)
    rows = json.loads(parity.flush(tmp_path / "out.json").read_text())
    assert rows == [
        {"tag": "loss_step0", "value": 2.5,
         "error_tolerance": {"rtol": 1e-3, "atol": parity.DEFAULT_ATOL, "note": "5 repeats"}},
        {"tag": "n_gaussians", "value": 661878,
         "error_tolerance": {"rtol": 0.0, "atol": parity.DEFAULT_ATOL}},
    ]


def test_sync_is_deferred_to_flush(tmp_path):
    parity.set_enabled(True)
    t = FakeTensor(3.5)
    parity.record("x", t)
    assert t.item_calls == 0, "record must not pull the value off the device"
    parity.flush(tmp_path / "out.json")
    assert t.item_calls == 1


def test_multi_element_tensor_is_rejected_at_the_call_site():
    parity.set_enabled(True)
    with pytest.raises(TypeError, match="record_lazy"):
        parity.record("acts", FakeTensor(1.0, numel=4096))


def test_duplicate_tag_is_an_error():
    parity.set_enabled(True)
    parity.record("loss_step0", 1.0)
    with pytest.raises(ValueError, match="duplicate tag"):
        parity.record("loss_step0", 2.0)


def test_tolerance_object_and_kwargs_are_mutually_exclusive():
    parity.set_enabled(True)
    with pytest.raises(ValueError, match="not both"):
        parity.record("x", 1.0, rtol=1e-3, tolerance=parity.Tolerance(rtol=1e-2))


def test_default_tolerance_applies_to_undeclared_calls(tmp_path):
    parity.set_enabled(True)
    parity.set_default_tolerance(rtol=1e-4)
    parity.record("x", 1.0)
    parity.record("y", 2.0, rtol=0.5)
    rows = json.loads(parity.flush(tmp_path / "out.json").read_text())
    assert rows[0]["error_tolerance"]["rtol"] == 1e-4
    assert rows[1]["error_tolerance"]["rtol"] == 0.5


def test_flush_without_parity_out_says_so(monkeypatch):
    parity.set_enabled(True)
    parity.record("x", 1.0)
    with pytest.raises(RuntimeError, match="PARITY_OUT"):
        parity.flush()


def test_output_path_gets_a_rank_infix(monkeypatch, tmp_path):
    monkeypatch.setenv("PARITY_OUT", str(tmp_path / "run.json"))
    monkeypatch.setenv("RANK", "3")
    assert recorder.output_path().name == "run.rank3.json"


def test_output_path_is_bare_when_not_distributed(monkeypatch, tmp_path):
    monkeypatch.setenv("PARITY_OUT", str(tmp_path / "run.json"))
    assert recorder.output_path().name == "run.json"


def test_flush_is_a_full_snapshot_each_time(tmp_path):
    parity.set_enabled(True)
    out = tmp_path / "out.json"
    parity.record("a", 1.0)
    parity.flush(out)
    parity.record("b", 2.0)
    parity.flush(out)
    assert [r["tag"] for r in json.loads(out.read_text())] == ["a", "b"]


def test_flush_creates_the_parent_directory(tmp_path):
    parity.set_enabled(True)
    parity.record("a", 1.0)
    out = tmp_path / "deep" / "nested" / "out.json"
    parity.flush(out)
    assert out.exists()


def test_nan_is_recorded_and_warned_about(tmp_path, capsys):
    parity.set_enabled(True)
    parity.record("x", float("nan"))
    parity.flush(tmp_path / "out.json")
    assert "NaN" in capsys.readouterr().err


def test_exit_hook_is_silent_when_flush_was_given_an_explicit_path(tmp_path, capsys):
    """An explicit-path flush with no PARITY_OUT must not print a failure at exit."""
    parity.set_enabled(True)
    parity.record("x", 1.0)
    parity.flush(tmp_path / "out.json")
    capsys.readouterr()
    recorder._flush_quietly()
    assert capsys.readouterr().err == ""


def test_exit_hook_still_reports_a_real_failure(tmp_path, monkeypatch, capsys):
    parity.set_enabled(True)
    monkeypatch.setenv("PARITY_OUT", str(tmp_path / "nope" / "out.json"))
    parity.record("x", 1.0)
    monkeypatch.setattr(recorder, "_to_python", lambda tag, value: 1 / 0)
    recorder._flush_quietly()
    assert "could not flush at exit" in capsys.readouterr().err
