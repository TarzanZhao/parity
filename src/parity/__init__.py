"""Scalar checkpoints, gated by an env var, compared against a baseline.

    import parity
    parity.record(f"loss_step{step}", loss, rtol=1e-3)
    parity.record_lazy(f"actmax_step{step}", lambda: h.abs().max(), rtol=1e-2)
    parity.flush()

    PARITY=1 PARITY_OUT=base.json python train.py
    parity compare base.json arm.json
"""

from parity.recorder import (
    enabled,
    flush,
    output_path,
    record,
    record_lazy,
    reset,
    set_default_tolerance,
    set_enabled,
)
from parity.tolerance import DEFAULT_ATOL, DERIVE_MARGIN, Comparison, Tolerance

__all__ = [
    "Comparison",
    "DEFAULT_ATOL",
    "DERIVE_MARGIN",
    "Tolerance",
    "enabled",
    "flush",
    "output_path",
    "record",
    "record_lazy",
    "reset",
    "set_default_tolerance",
    "set_enabled",
]
