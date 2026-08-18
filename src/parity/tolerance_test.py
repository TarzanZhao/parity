"""Tests for src/parity/tolerance.py.

python -m pytest src/parity/tolerance_test.py -v
"""

from __future__ import annotations

import math

import pytest

from parity import tolerance as tol_mod


def test_from_json_accepts_bare_number_as_rtol():
    t = tol_mod.Tolerance.from_json(1e-3)
    assert (t.rtol, t.atol, t.note) == (1e-3, tol_mod.DEFAULT_ATOL, None)


def test_from_json_accepts_structured_form():
    t = tol_mod.Tolerance.from_json({"rtol": 0.1, "atol": 1e-6, "note": "5 repeats"})
    assert (t.rtol, t.atol, t.note) == (0.1, 1e-6, "5 repeats")


def test_from_json_none_is_the_strict_default():
    t = tol_mod.Tolerance.from_json(None)
    assert t.rtol == 0.0 and t.atol == tol_mod.DEFAULT_ATOL


@pytest.mark.parametrize("bad", [{"rtol": 1e-3, "typo": 1}, "1e-3", True, {"note": 5}])
def test_from_json_rejects_junk(bad):
    with pytest.raises(ValueError):
        tol_mod.Tolerance.from_json(bad)


def test_negative_tolerance_rejected():
    with pytest.raises(ValueError):
        tol_mod.Tolerance(rtol=-1e-3)


def test_budget_is_atol_plus_relative_part():
    t = tol_mod.Tolerance(rtol=0.1, atol=0.5)
    assert t.budget(10.0) == pytest.approx(1.5)
    assert t.budget(-10.0) == pytest.approx(1.5)


def test_default_gate_is_bit_identical():
    t = tol_mod.Tolerance()
    assert tol_mod.compare_scalar("x", 1.0, 1.0, t).ok
    assert not tol_mod.compare_scalar("x", 1.0, 1.0 + 1e-9, t).ok


def test_within_and_beyond_budget():
    t = tol_mod.Tolerance(rtol=0.1)
    assert tol_mod.compare_scalar("x", 100.0, 109.0, t).ok
    beyond = tol_mod.compare_scalar("x", 100.0, 111.0, t)
    assert not beyond.ok and beyond.ratio == pytest.approx(1.1)


def test_int_may_carry_a_relative_tolerance():
    t = tol_mod.Tolerance(rtol=0.01)
    assert tol_mod.compare_scalar("n", 1000, 1005, t).ok
    assert not tol_mod.compare_scalar("n", 1000, 1050, t).ok


def test_int_defaults_to_exact():
    assert not tol_mod.compare_scalar("n", 1000, 1001, tol_mod.Tolerance()).ok


def test_type_change_fails_however_loose_the_tolerance():
    huge = tol_mod.Tolerance(rtol=1e9, atol=1e9)
    c = tol_mod.compare_scalar("n", 661878, 661878.0, huge)
    assert not c.ok and "type changed" in c.reason


def test_bool_and_str_get_no_tolerance():
    huge = tol_mod.Tolerance(rtol=1e9, atol=1e9)
    assert tol_mod.compare_scalar("f", True, True, huge).ok
    assert not tol_mod.compare_scalar("f", True, False, huge).ok
    assert tol_mod.compare_scalar("s", "cat", "cat", huge).ok
    assert not tol_mod.compare_scalar("s", "cat", "dog", huge).ok


def test_bool_is_not_an_int():
    c = tol_mod.compare_scalar("f", 1, True, tol_mod.Tolerance())
    assert not c.ok and "type changed" in c.reason


def test_nan_never_passes():
    huge = tol_mod.Tolerance(rtol=1e9, atol=1e9)
    assert not tol_mod.compare_scalar("x", math.nan, math.nan, huge).ok
    assert not tol_mod.compare_scalar("x", 1.0, math.nan, huge).ok


def test_matching_infinities_pass_and_mismatched_do_not():
    t = tol_mod.Tolerance(rtol=0.1)
    assert tol_mod.compare_scalar("x", math.inf, math.inf, t).ok
    assert not tol_mod.compare_scalar("x", math.inf, -math.inf, t).ok
    assert not tol_mod.compare_scalar("x", math.inf, 1.0, t).ok


def test_atol_carries_a_value_that_lives_near_zero():
    relative_only = tol_mod.Tolerance(rtol=0.1)
    assert not tol_mod.compare_scalar("x", 0.0, 1e-6, relative_only).ok
    with_floor = tol_mod.Tolerance(rtol=0.1, atol=1e-5)
    assert tol_mod.compare_scalar("x", 0.0, 1e-6, with_floor).ok
