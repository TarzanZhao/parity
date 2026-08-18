"""Record scalar checkpoints from a running job, gated by one environment variable.

    PARITY=1 PARITY_OUT=run.json python train.py

With `PARITY` unset every entry point returns on its first line and the job pays
nothing — not even the reduction, if it was passed as a lambda to `record_lazy`.
That matters more than it sounds: this package exists to serve performance work,
and an instrument that perturbs the thing it measures is worthless. The intended
shape is **two runs of the same binary differing only by an env var** — one timed
with recording off, one with it on.

## The sync is deferred

`.item()` on a CUDA tensor is a device synchronisation. `record` therefore keeps
whatever it was handed and converts at `flush()`, so a step loop that records ten
checkpoints pays one sync at the end rather than ten inside the loop. The shape
check (is this really a scalar?) still happens at call time, where the traceback
points at the offending line.

## The tag is the identity

Nothing is keyed by call site, call order, or line number, because the two runs
being compared are *by construction* running different code. A tag is a string
you wrote, and recording the same one twice is an error — a duplicate means you
believed two checkpoints were distinct and they were not.
"""

from __future__ import annotations

import atexit
import dataclasses
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

from parity import tolerance as tolerance_mod

_TRUTHY = {"1", "on", "true", "yes", "y"}


@dataclasses.dataclass
class _Pending:
    """One checkpoint, value not yet pulled off the device."""

    tag: str
    value: Any
    tolerance: tolerance_mod.Tolerance


_records: list[_Pending] = []
_seen: set[str] = set()
_default_tolerance = tolerance_mod.Tolerance()
_enabled: bool | None = None
_atexit_registered = False


def enabled() -> bool:
    """Whether recording is on. Resolved from `PARITY` once, then cached."""
    global _enabled
    if _enabled is None:
        _enabled = os.environ.get("PARITY", "").strip().lower() in _TRUTHY
    return _enabled


def set_enabled(value: bool | None) -> None:
    """Force recording on or off, ignoring `PARITY`. `None` re-reads the env var."""
    global _enabled
    _enabled = value


def set_default_tolerance(
    rtol: float = 0.0,
    atol: float = tolerance_mod.DEFAULT_ATOL,
    note: str | None = None,
) -> None:
    """Set the tolerance used by calls that declare none of their own."""
    global _default_tolerance
    _default_tolerance = tolerance_mod.Tolerance(rtol=rtol, atol=atol, note=note)


def reset() -> None:
    """Drop every pending record. For tests, and for a harness that runs twice."""
    _records.clear()
    _seen.clear()


def _resolve_tolerance(
    rtol: float | None,
    atol: float | None,
    note: str | None,
    tolerance: tolerance_mod.Tolerance | None,
) -> tolerance_mod.Tolerance:
    if tolerance is not None:
        if rtol is not None or atol is not None or note is not None:
            raise ValueError("pass either `tolerance=` or rtol/atol/note, not both")
        return tolerance
    if rtol is None and atol is None and note is None:
        return _default_tolerance
    return tolerance_mod.Tolerance(
        rtol=_default_tolerance.rtol if rtol is None else rtol,
        atol=_default_tolerance.atol if atol is None else atol,
        note=_default_tolerance.note if note is None else note,
    )


def _numel(value: Any) -> int | None:
    """Element count for a tensor-like, or None if the object does not say."""
    numel = getattr(value, "numel", None)
    if callable(numel):  # torch
        try:
            return int(numel())
        except Exception:
            return None
    size = getattr(value, "size", None)
    if isinstance(size, int):  # numpy
        return size
    return None


def _check_scalar(tag: str, value: Any) -> None:
    """Reject a non-scalar at call time, where the traceback is useful."""
    if isinstance(value, (bool, int, float, str)):
        return
    n = _numel(value)
    if n is not None and n != 1:
        raise TypeError(
            f"parity: {tag!r} got a {n}-element tensor, but a checkpoint is a SCALAR. "
            "Reduce it at the call site so the reduction is skipped when PARITY is off:\n"
            f"    parity.record_lazy({tag!r}, lambda: x.abs().max().item(), rtol=...)"
        )
    if not hasattr(value, "item"):
        raise TypeError(
            f"parity: {tag!r} got {type(value).__name__}; expected int/float/bool/str "
            "or a 1-element tensor"
        )


def _to_python(tag: str, value: Any) -> int | float | bool | str:
    """Pull the value off the device. This is where the sync happens."""
    if isinstance(value, (bool, int, float, str)):
        return value
    out = value.item()
    if isinstance(out, (bool, int, float, str)):
        return out
    if isinstance(out, complex):
        raise TypeError(f"parity: {tag!r} is complex; record magnitude and phase separately")
    raise TypeError(f"parity: {tag!r} resolved to {type(out).__name__}, which is not a scalar")


def record(
    tag: str,
    value: Any,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    note: str | None = None,
    tolerance: tolerance_mod.Tolerance | None = None,
) -> None:
    """Record one scalar checkpoint under `tag`.

    Args:
        tag: Unique name. Fold the step or layer index into the string —
            `f"loss_step{step}"` — since nothing else distinguishes two calls.
        value: A Python scalar, or a 1-element tensor. Multi-element tensors are
            rejected; reduce them yourself via `record_lazy`.
        rtol: Relative tolerance for this checkpoint.
        atol: Absolute tolerance. Use this, not `rtol`, for values near zero.
        note: Where the tolerance came from. Printed back by `parity derive`.
        tolerance: A prebuilt `Tolerance`, mutually exclusive with rtol/atol/note.

    Raises:
        ValueError: `tag` was already recorded.
        TypeError: `value` is not a scalar.
    """
    if not enabled():
        return
    if tag in _seen:
        raise ValueError(
            f"parity: duplicate tag {tag!r}. A tag is the identity of a checkpoint — "
            "fold the step or index into it, e.g. f'{tag}_step{step}'."
        )
    _check_scalar(tag, value)
    _seen.add(tag)
    _records.append(_Pending(tag, value, _resolve_tolerance(rtol, atol, note, tolerance)))
    _register_atexit()


def record_lazy(
    tag: str,
    fn: Callable[[], Any],
    *,
    rtol: float | None = None,
    atol: float | None = None,
    note: str | None = None,
    tolerance: tolerance_mod.Tolerance | None = None,
) -> None:
    """Record `fn()` under `tag`, calling `fn` only when recording is on.

    The point is the *reduction*: `lambda: x.abs().max()` runs a kernel, and with
    `PARITY` unset it must not run at all. `fn` is called immediately when
    recording is on — the result stays on its device until `flush()`.

    Args:
        tag: Unique name, as in `record`.
        fn: Zero-argument callable returning a scalar or 1-element tensor.
        rtol: Relative tolerance for this checkpoint.
        atol: Absolute tolerance.
        note: Where the tolerance came from.
        tolerance: A prebuilt `Tolerance`, mutually exclusive with rtol/atol/note.
    """
    if not enabled():
        return
    record(tag, fn(), rtol=rtol, atol=atol, note=note, tolerance=tolerance)


def _rank() -> int | None:
    """This process's rank, or None if the job is not distributed."""
    torch_dist = sys.modules.get("torch.distributed")
    if torch_dist is not None:
        try:
            if torch_dist.is_available() and torch_dist.is_initialized():
                return int(torch_dist.get_rank())
        except Exception:
            pass
    env = os.environ.get("RANK", "")
    return int(env) if env.isdigit() else None


def output_path() -> Path:
    """Where `flush()` will write, `PARITY_OUT` with a `.rank<N>` infix if distributed."""
    raw = os.environ.get("PARITY_OUT", "").strip()
    if not raw:
        raise RuntimeError(
            "parity: PARITY is on but PARITY_OUT is unset — there is nowhere to write. "
            "Set PARITY_OUT=<path>.json, or call parity.flush(path)."
        )
    path = Path(raw)
    rank = _rank()
    if rank is not None:
        path = path.with_suffix(f".rank{rank}{path.suffix}")
    return path


def flush(path: str | Path | None = None) -> Path | None:
    """Resolve every pending value and write the record file.

    Writes the full accumulated list each time, so calling it once per step is a
    cheap way to survive a crash. Returns None when recording is off.

    Args:
        path: Override for `PARITY_OUT`. The rank infix is not applied to it.

    Returns:
        The path written, or None.
    """
    if not enabled():
        return None
    out = Path(path) if path is not None else output_path()
    payload = [
        {
            "tag": r.tag,
            "value": _to_python(r.tag, r.value),
            "error_tolerance": r.tolerance.to_json(),
        }
        for r in _records
    ]
    n_nan = sum(
        1 for row in payload if isinstance(row["value"], float) and math.isnan(row["value"])
    )
    if n_nan:
        print(f"parity: {n_nan} recorded value(s) are NaN — no tolerance covers those",
              file=sys.stderr)
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def _register_atexit() -> None:
    """Flush on exit, so a forgotten `flush()` is not a lost run."""
    global _atexit_registered
    if _atexit_registered:
        return
    _atexit_registered = True
    atexit.register(_flush_quietly)


def _flush_quietly() -> None:
    # No PARITY_OUT means the exit hook has nowhere to write, and the only way
    # records exist at all is that the caller passed an explicit path to flush().
    # Complaining here would print a failure at the end of every such run.
    if not enabled() or not os.environ.get("PARITY_OUT", "").strip():
        return
    try:
        flush()
    except Exception as exc:  # a failed atexit must not mask the job's own error
        print(f"parity: could not flush at exit: {exc}", file=sys.stderr)
