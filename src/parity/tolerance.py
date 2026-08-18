"""What `error_tolerance` means, and the one function that decides pass or fail.

A tolerance is a *declaration*: how far a number is allowed to move before the
run counts as different. It is not a measurement of how far the number actually
moves run to run — that is what `parity derive` computes, and the two disagreeing
is the single most useful thing this package reports.

## The budget

    budget = atol + rtol * |expected|
    pass   iff |actual - expected| <= budget

`atol` alone gates a value that lives near zero, where a relative test divides by
~0 and reports nonsense. `rtol` alone gates a value whose scale is not known in
advance. Most checkpoints want both, so the default carries a floor of
`DEFAULT_ATOL` and no relative slack at all: **the default gate is bit-identical
modulo denormal noise**, and every loosening is something you typed on purpose.

## Type is a declaration, and it is checked before any tolerance

An int may carry a tolerance — a count can legitimately drift under a change that
reorders work. But `661878` becoming `661878.0` still fails, because the JSON
type says what kind of quantity this is, and a quantity that changed kind is not
the same quantity. Booleans and strings are results too, and get no tolerance
ever: a flag that flipped or a caption that changed is a difference, full stop.
"""

from __future__ import annotations

import dataclasses
import math

# Absolute floor of the gate. Below this, a difference is fp noise in any dtype
# worth profiling, and a pure relative test would divide by ~0.
DEFAULT_ATOL = 1e-12

# A derived tolerance is this many times the worst spread across repeats: a
# change must move a number an order of magnitude beyond run-to-run noise before
# it counts as a change.
DERIVE_MARGIN = 10.0

_EXACT_KINDS = ("bool", "str")


@dataclasses.dataclass(frozen=True)
class Tolerance:
    """How far a recorded value may move before it counts as different.

    Attributes:
        rtol: Relative tolerance, applied to |expected|.
        atol: Absolute tolerance. Floors the test near zero.
        note: Free text — where this number came from. A tolerance nobody can
            re-derive is not a tolerance, so `parity derive` prints this back.
    """

    rtol: float = 0.0
    atol: float = DEFAULT_ATOL
    note: str | None = None

    def __post_init__(self) -> None:
        if self.rtol < 0 or self.atol < 0:
            raise ValueError(f"tolerance must be non-negative, got rtol={self.rtol} atol={self.atol}")

    def budget(self, expected: float) -> float:
        """Absolute slack allowed around `expected`."""
        return self.atol + self.rtol * abs(expected)

    def to_json(self) -> dict[str, float | str]:
        out: dict[str, float | str] = {"rtol": self.rtol, "atol": self.atol}
        if self.note is not None:
            out["note"] = self.note
        return out

    @classmethod
    def from_json(cls, obj: object) -> "Tolerance":
        """Parse a recorded `error_tolerance`.

        Accepts the structured form `{"rtol":.., "atol":.., "note":..}`, and also
        a bare number, read as `rtol` — so a hand-written file that says
        `"error_tolerance": 1e-3` still loads.
        """
        if obj is None:
            return cls()
        if isinstance(obj, bool):
            raise ValueError(f"error_tolerance must be a number or an object, got {obj!r}")
        if isinstance(obj, (int, float)):
            return cls(rtol=float(obj))
        if isinstance(obj, dict):
            unknown = set(obj) - {"rtol", "atol", "note"}
            if unknown:
                raise ValueError(f"error_tolerance has unknown key(s): {sorted(unknown)}")
            note = obj.get("note")
            if note is not None and not isinstance(note, str):
                raise ValueError(f"error_tolerance.note must be a string, got {note!r}")
            return cls(
                rtol=float(obj.get("rtol", 0.0)),
                atol=float(obj.get("atol", DEFAULT_ATOL)),
                note=note,
            )
        raise ValueError(f"error_tolerance must be a number or an object, got {type(obj).__name__}")

    def describe(self) -> str:
        body = f"rtol={self.rtol:g} atol={self.atol:g}"
        return f"{body} ({self.note})" if self.note else body


def kind(value: object) -> str:
    """JSON-level type name. `bool` is checked before `int` — it subclasses it."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return type(value).__name__


@dataclasses.dataclass(frozen=True)
class Comparison:
    """The verdict on one tag.

    Attributes:
        tag: The checkpoint's name.
        ok: Whether the arm is within budget of the baseline.
        expected: Baseline value.
        actual: Arm value.
        diff: |actual - expected|, or None for non-numeric kinds.
        budget: Slack allowed, or None for non-numeric kinds.
        ratio: diff / budget. <= 1 passes. inf when the budget is zero and the
            values differ.
        reason: Why it failed, empty when it passed.
    """

    tag: str
    ok: bool
    expected: object
    actual: object
    diff: float | None
    budget: float | None
    ratio: float
    reason: str


def compare_scalar(tag: str, expected: object, actual: object, tol: Tolerance) -> Comparison:
    """Judge one recorded value against its baseline.

    Args:
        tag: Checkpoint name, carried into the result.
        expected: Baseline value.
        actual: Value from the run under test.
        tol: The declared tolerance, normally the baseline's.

    Returns:
        A `Comparison`. Type changes, NaN, and mismatched infinities fail
        regardless of how loose the tolerance is.
    """
    ek, ak = kind(expected), kind(actual)
    if ek != ak:
        return Comparison(tag, False, expected, actual, None, None, math.inf,
                          f"type changed: {ek} -> {ak}")

    if ek in _EXACT_KINDS:
        ok = expected == actual
        return Comparison(tag, ok, expected, actual, None, None, 0.0 if ok else math.inf,
                          "" if ok else f"{ek} values differ")

    if ek not in ("int", "float"):
        return Comparison(tag, False, expected, actual, None, None, math.inf,
                          f"{ek} is not a recordable value")

    e, a = float(expected), float(actual)
    if math.isnan(e) or math.isnan(a):
        return Comparison(tag, False, expected, actual, None, None, math.inf,
                          "NaN — no tolerance covers this")
    if math.isinf(e) or math.isinf(a):
        ok = e == a
        return Comparison(tag, ok, expected, actual, None, None, 0.0 if ok else math.inf,
                          "" if ok else "infinity mismatch")

    diff = abs(a - e)
    budget = tol.budget(e)
    if diff == 0.0:
        ratio = 0.0
    elif budget == 0.0:
        ratio = math.inf
    else:
        ratio = diff / budget
    ok = diff <= budget
    return Comparison(tag, ok, expected, actual, diff, budget, ratio,
                      "" if ok else f"{ratio:.3g}x over budget")
