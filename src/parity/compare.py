"""Compare a recorded run against a baseline, keyed by tag.

Pairing is **by tag, never by position**. The two runs being compared differ in
code by construction, so a checkpoint added or removed in the middle would shift
every positional index and bury the one real difference under a hundred false
ones. Keyed by tag, an added checkpoint is one extra tag and nothing else moves.

What tags cannot do is cover for a run that stopped early. So the tag sets must
match **exactly**: a missing tag is a structural failure, not a skipped
comparison. An arm that died after two of three steps fails here rather than
passing because only the two steps it managed got compared — that silent pass is
how a correctness gate actually breaks in practice.

## Whose tolerance

The baseline's. It is the ground truth, and the arm is the thing on trial. When
the arm declares a *different* tolerance for a tag, that is reported: someone
edited the gate in the same change that had to pass it, and whether that was
honest is a judgement no script should make silently.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import json
from pathlib import Path

from parity import tolerance as tolerance_mod

_RECORD_KEYS = {"tag", "value", "error_tolerance"}


@dataclasses.dataclass(frozen=True)
class Record:
    """One row of a record file."""

    tag: str
    value: object
    tolerance: tolerance_mod.Tolerance


def load(path: str | Path) -> dict[str, Record]:
    """Read one record file, or a directory of per-rank files.

    A directory is read as every `*.json` inside it, with the filename prefixed
    onto each tag (`rank0.json:loss_step0`). Ranks stay separate entries rather
    than being merged, so a missing rank file is a structural mismatch like any
    other.

    Args:
        path: A `.json` file, or a directory of them.

    Returns:
        Tag -> record.

    Raises:
        SystemExit: The path is missing, empty, or holds a duplicate tag.
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.json"))
        if not files:
            raise SystemExit(f"{p}: directory holds no json record")
        out: dict[str, Record] = {}
        for f in files:
            for tag, rec in _load_file(f).items():
                out[f"{f.name}:{tag}"] = dataclasses.replace(rec, tag=f"{f.name}:{tag}")
        return out
    if not p.exists():
        raise SystemExit(f"{p}: no such record")
    return _load_file(p)


def _load_file(path: Path) -> dict[str, Record]:
    try:
        rows = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: not valid json ({exc})") from exc
    if not isinstance(rows, list):
        raise SystemExit(f"{path}: expected a list of records, got {type(rows).__name__}")
    if not rows:
        raise SystemExit(f"{path}: no records — a gate over zero checkpoints passes trivially")

    out: dict[str, Record] = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"{path}[{i}]: expected an object, got {type(row).__name__}")
        missing = {"tag", "value"} - set(row)
        if missing:
            raise SystemExit(f"{path}[{i}]: record is missing {sorted(missing)}")
        unknown = set(row) - _RECORD_KEYS
        if unknown:
            raise SystemExit(f"{path}[{i}]: unknown key(s) {sorted(unknown)}")
        tag = row["tag"]
        if not isinstance(tag, str):
            raise SystemExit(f"{path}[{i}]: tag must be a string, got {tag!r}")
        if tag in out:
            raise SystemExit(
                f"{path}: duplicate tag {tag!r} — a tag is the identity of a checkpoint"
            )
        try:
            tol = tolerance_mod.Tolerance.from_json(row.get("error_tolerance"))
        except ValueError as exc:
            raise SystemExit(f"{path}[{i}] ({tag}): {exc}") from exc
        out[tag] = Record(tag=tag, value=row["value"], tolerance=tol)
    return out


def _select(tags: set[str], only: list[str], exclude: list[str]) -> set[str]:
    kept = tags
    if only:
        kept = {t for t in kept if any(fnmatch.fnmatch(t, pat) for pat in only)}
    if exclude:
        kept = {t for t in kept if not any(fnmatch.fnmatch(t, pat) for pat in exclude)}
    return kept


@dataclasses.dataclass
class Report:
    """The outcome of one comparison.

    Attributes:
        missing: Tags in the baseline and not in the arm.
        extra: Tags in the arm and not in the baseline.
        retoleranced: Tags whose declared tolerance differs between the files.
        comparisons: Per-tag verdicts, failures first.
        n_pass: How many tags passed.
    """

    missing: list[str]
    extra: list[str]
    retoleranced: list[tuple[str, tolerance_mod.Tolerance, tolerance_mod.Tolerance]]
    comparisons: list[tolerance_mod.Comparison]
    n_pass: int

    @property
    def failures(self) -> list[tolerance_mod.Comparison]:
        return [c for c in self.comparisons if not c.ok]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.extra and not self.failures


def compare(
    baseline: str | Path,
    arm: str | Path,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
) -> Report:
    """Judge `arm` against `baseline`.

    Args:
        baseline: Ground-truth record, from unmodified code.
        arm: Record from the run under test.
        rtol: Override every declared relative tolerance with this one.
        atol: Override every declared absolute tolerance with this one.
        only: Glob(s); when given, only matching tags are compared.
        exclude: Glob(s) of tags to drop.

    Returns:
        A `Report`. `Report.ok` is the gate.
    """
    base = load(baseline)
    test = load(arm)

    keep = _select(set(base) | set(test), only or [], exclude or [])
    base = {t: r for t, r in base.items() if t in keep}
    test = {t: r for t, r in test.items() if t in keep}

    missing = sorted(set(base) - set(test))
    extra = sorted(set(test) - set(base))

    retoleranced: list[tuple[str, tolerance_mod.Tolerance, tolerance_mod.Tolerance]] = []
    comparisons: list[tolerance_mod.Comparison] = []
    for tag in sorted(set(base) & set(test)):
        b, t = base[tag], test[tag]
        if (b.tolerance.rtol, b.tolerance.atol) != (t.tolerance.rtol, t.tolerance.atol):
            retoleranced.append((tag, b.tolerance, t.tolerance))
        effective = tolerance_mod.Tolerance(
            rtol=b.tolerance.rtol if rtol is None else rtol,
            atol=b.tolerance.atol if atol is None else atol,
            note=b.tolerance.note,
        )
        comparisons.append(tolerance_mod.compare_scalar(tag, b.value, t.value, effective))

    comparisons.sort(key=lambda c: (c.ok, -c.ratio if c.ratio != float("inf") else -1e308))
    return Report(
        missing=missing,
        extra=extra,
        retoleranced=retoleranced,
        comparisons=comparisons,
        n_pass=sum(1 for c in comparisons if c.ok),
    )


def render(report: Report, limit: int = 20) -> str:
    """Format a report for the terminal."""
    lines: list[str] = []
    n = len(report.comparisons)
    lines.append(f"== {n} tag(s) compared, {report.n_pass} pass, {len(report.failures)} fail ==")

    if report.missing or report.extra:
        lines.append("")
        lines.append(
            f"!! the two runs do not hold the same checkpoints: "
            f"{len(report.missing)} missing, {len(report.extra)} extra"
        )
        for tag in report.missing[:limit]:
            lines.append(f"   - missing from arm:      {tag}")
        for tag in report.extra[:limit]:
            lines.append(f"   - not in the baseline:   {tag}")
        lines.append("   a run that stopped early looks exactly like this — check that first")

    if report.retoleranced:
        lines.append("")
        lines.append(
            f"!! {len(report.retoleranced)} tag(s) declare a DIFFERENT tolerance in the arm. "
            "The baseline's is used; a loosened gate in the same change that had to pass it "
            "is worth a second look:"
        )
        for tag, b, t in report.retoleranced[:limit]:
            lines.append(f"   {tag}: baseline {b.describe()}  |  arm {t.describe()}")

    if not report.ok:
        lines.append("")
        lines.append("FAIL")
    if report.failures:
        for c in report.failures[:limit]:
            lines.append(f"   {c.tag}")
            lines.append(f"      expected {c.expected!r}   actual {c.actual!r}")
            if c.diff is not None:
                lines.append(f"      |diff| {c.diff:.6g}   budget {c.budget:.6g}   {c.reason}")
            else:
                lines.append(f"      {c.reason}")
        if len(report.failures) > limit:
            lines.append(f"   ... and {len(report.failures) - limit} more")

    if report.ok:
        worst = max((c for c in report.comparisons), key=lambda c: c.ratio, default=None)
        lines.append("")
        lines.append("PASS")
        if worst is not None and worst.ratio > 0:
            lines.append(f"   closest call: {worst.tag} at {worst.ratio:.3g}x of its budget")
        else:
            lines.append("   every value is identical")
    return "\n".join(lines)
