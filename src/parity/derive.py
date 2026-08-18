"""Measure how much each checkpoint moves on its own, and check the declared gate against it.

A tolerance you typed is a **claim**: how much drift you are willing to accept.
The spread across repeats of *unmodified* code is a **measurement**: how much the
number moves when nothing changed. They are different quantities, and the gap
between them is the only thing that says whether a gate is worth anything.

    declared << measured   the gate will flake — it fails runs that changed nothing
    declared >> measured   the gate is vacuous — it passes changes that altered numerics

Neither is visible from a single run, which is why this reads three or more.

Because every checkpoint carries its own tolerance, the classic failure of a
single global `rtol` cannot happen here: one near-zero value can no longer drag
the whole gate loose to cover itself. It can still be ungateable on its own
terms, and when a value's spread swamps its magnitude this suggests an absolute
tolerance rather than a relative one.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from pathlib import Path

from parity import compare as compare_mod
from parity import tolerance as tolerance_mod

IDENTICAL = "identical"
OK = "ok"
FLAKY = "flaky"
VACUOUS = "vacuous"
UNSTABLE = "unstable"
SLACK = "slack"

_FATAL = (FLAKY, UNSTABLE)


@dataclasses.dataclass
class TagNoise:
    """What the repeats say about one checkpoint.

    Attributes:
        tag: Checkpoint name.
        values: One value per repeat.
        spread: max - min. Zero when every repeat agreed bit for bit.
        magnitude: Median |value|, the scale a relative tolerance is measured against.
        declared: The tolerance the code declares.
        declared_budget: Slack that tolerance allows at this magnitude.
        needed_budget: Slack a gate must allow to survive the noise, with margin.
        suggestion: A tolerance that would hold, or None when the declared one is fine.
        verdict: One of the module-level constants.
        detail: Human-readable reason.
    """

    tag: str
    values: list[object]
    spread: float
    magnitude: float
    declared: tolerance_mod.Tolerance
    declared_budget: float
    needed_budget: float
    suggestion: tolerance_mod.Tolerance | None
    verdict: str
    detail: str


def _suggest(needed_budget: float, magnitude: float) -> tolerance_mod.Tolerance:
    """Turn a required absolute slack into a tolerance, relative where that makes sense."""
    if magnitude > 0 and needed_budget / magnitude <= 0.5:
        return tolerance_mod.Tolerance(rtol=needed_budget / magnitude,
                                       atol=tolerance_mod.DEFAULT_ATOL)
    return tolerance_mod.Tolerance(rtol=0.0, atol=needed_budget)


def _noise_for(tag: str, values: list[object], declared: tolerance_mod.Tolerance) -> TagNoise:
    kinds = {tolerance_mod.kind(v) for v in values}
    if len(kinds) > 1:
        return TagNoise(tag, values, math.inf, 0.0, declared, 0.0, math.inf, None, UNSTABLE,
                        f"type changes across repeats: {sorted(kinds)}")

    kind = kinds.pop()
    if kind in ("bool", "str"):
        stable = all(v == values[0] for v in values)
        return TagNoise(
            tag, values, 0.0 if stable else math.inf, 0.0, declared, 0.0,
            0.0 if stable else math.inf, None,
            IDENTICAL if stable else UNSTABLE,
            "" if stable else f"{kind} value differs across repeats and gets no tolerance",
        )
    if kind not in ("int", "float"):
        return TagNoise(tag, values, math.inf, 0.0, declared, 0.0, math.inf, None, UNSTABLE,
                        f"{kind} is not a recordable value")

    nums = [float(v) for v in values]
    if any(math.isnan(x) for x in nums):
        return TagNoise(tag, values, math.inf, 0.0, declared, 0.0, math.inf, None, UNSTABLE,
                        "NaN in the baseline — fix that before deriving anything")
    if any(math.isinf(x) for x in nums):
        stable = all(x == nums[0] for x in nums)
        return TagNoise(tag, values, 0.0 if stable else math.inf, 0.0, declared, 0.0,
                        0.0 if stable else math.inf, None,
                        IDENTICAL if stable else UNSTABLE,
                        "" if stable else "infinity differs across repeats")

    spread = max(nums) - min(nums)
    magnitude = abs(statistics.median(nums))
    declared_budget = declared.budget(magnitude)
    needed_budget = tolerance_mod.DERIVE_MARGIN * spread

    if spread == 0.0:
        if declared_budget > tolerance_mod.DEFAULT_ATOL:
            return TagNoise(tag, values, spread, magnitude, declared, declared_budget,
                            0.0, None, SLACK,
                            "repeats are bit-identical; the declared slack is for a change, "
                            "not for noise")
        return TagNoise(tag, values, spread, magnitude, declared, declared_budget,
                        0.0, None, IDENTICAL, "")

    if spread > declared_budget:
        return TagNoise(tag, values, spread, magnitude, declared, declared_budget,
                        needed_budget, _suggest(needed_budget, magnitude), FLAKY,
                        f"noise {spread:.3g} already exceeds the declared budget "
                        f"{declared_budget:.3g}")

    if declared_budget > tolerance_mod.DERIVE_MARGIN * needed_budget:
        return TagNoise(tag, values, spread, magnitude, declared, declared_budget,
                        needed_budget, _suggest(needed_budget, magnitude), VACUOUS,
                        f"budget {declared_budget:.3g} is "
                        f"{declared_budget / max(needed_budget, 1e-300):.0f}x wider than the "
                        "noise needs")

    return TagNoise(tag, values, spread, magnitude, declared, declared_budget,
                    needed_budget, None, OK, "")


@dataclasses.dataclass
class DeriveReport:
    """Noise across repeats, per checkpoint.

    Attributes:
        n_repeats: How many records were read.
        tags: Per-tag results, worst verdict first.
    """

    n_repeats: int
    tags: list[TagNoise]

    @property
    def fatal(self) -> list[TagNoise]:
        return [t for t in self.tags if t.verdict in _FATAL]

    @property
    def ok(self) -> bool:
        return not self.fatal


def derive(paths: list[str | Path]) -> DeriveReport:
    """Measure run-to-run noise from repeats of unmodified code.

    Args:
        paths: Three or more record files (or directories of per-rank files),
            each from an identical, unmodified run.

    Returns:
        A `DeriveReport`. `DeriveReport.ok` is false when a checkpoint is
        ungateable as declared.

    Raises:
        SystemExit: The repeats do not hold the same checkpoints.
    """
    runs = [compare_mod.load(p) for p in paths]
    first = runs[0]
    for i, run in enumerate(runs[1:], start=2):
        missing = sorted(set(first) - set(run))
        extra = sorted(set(run) - set(first))
        if missing or extra:
            raise SystemExit(
                f"repeat {i} does not hold the same checkpoints as repeat 1: "
                f"{len(missing)} missing, {len(extra)} extra "
                f"(e.g. {(missing or extra)[:3]}) — the repeats are not comparable"
            )

    order = {IDENTICAL: 4, OK: 3, SLACK: 2, VACUOUS: 1, FLAKY: 0, UNSTABLE: 0}
    tags = [
        _noise_for(tag, [run[tag].value for run in runs], first[tag].tolerance)
        for tag in first
    ]
    tags.sort(key=lambda t: (order[t.verdict], t.tag))
    return DeriveReport(n_repeats=len(runs), tags=tags)


def render(report: DeriveReport, limit: int = 20) -> str:
    """Format a derive report for the terminal."""
    lines: list[str] = []
    counts: dict[str, int] = {}
    for t in report.tags:
        counts[t.verdict] = counts.get(t.verdict, 0) + 1
    summary = ", ".join(f"{n} {v}" for v, n in sorted(counts.items()))
    lines.append(f"== {report.n_repeats} repeats, {len(report.tags)} checkpoints: {summary} ==")

    if report.n_repeats < 3:
        lines.append(
            f"!! {report.n_repeats} repeat(s): the spread across repeats IS the noise floor, "
            "so fewer than 3 cannot measure it"
        )

    interesting = [t for t in report.tags if t.verdict not in (IDENTICAL, OK)]
    if not interesting:
        lines.append("")
        lines.append("every checkpoint is gated at a tolerance the measured noise supports")
        return "\n".join(lines)

    for t in interesting[:limit]:
        lines.append("")
        lines.append(f"{t.verdict.upper():9s} {t.tag}")
        lines.append(f"          declared {t.declared.describe()}")
        if t.spread not in (0.0, math.inf):
            lines.append(
                f"          spread {t.spread:.3g} over {report.n_repeats} repeats "
                f"at |value| {t.magnitude:.3g}"
            )
        if t.detail:
            lines.append(f"          {t.detail}")
        if t.suggestion is not None:
            s = t.suggestion
            arg = f"rtol={s.rtol:.1e}" if s.rtol > 0 else f"atol={s.atol:.1e}"
            lines.append(f"          -> parity.record({t.tag!r}, ..., {arg})")

    if len(interesting) > limit:
        lines.append("")
        lines.append(f"... and {len(interesting) - limit} more")

    if report.fatal:
        lines.append("")
        lines.append(
            f"{len(report.fatal)} checkpoint(s) cannot be gated as declared. A tag that moves "
            "with the code unchanged is either a seeding bug or a number that does not belong "
            "in a correctness gate."
        )
    return "\n".join(lines)
