# parity

Prove a change did not alter what a job computes. You mark the checkpoints; the
package records them when an env var says so, and compares a later run against
the baseline within a tolerance you declared at the call site.

Built for performance work, where the question after every change is *is this
faster, or is it just wrong* — but nothing here is specific to that. It has no
dependencies and does not import torch.

## Quick start

Mark the checkpoints. Fold the step into the tag; nothing else distinguishes two
calls.

```python
import parity

for step, batch in enumerate(loader):
    loss = model(batch)
    parity.record(f"loss_step{step}", loss, rtol=1e-3)
    parity.record_lazy(f"actmax_step{step}", lambda: h.abs().max(), rtol=1e-2)

parity.record("n_gaussians", len(gaussians))     # int, exact by default
parity.flush()
```

Record the baseline from unmodified code, three times, and find out how much the
numbers move on their own:

```bash
for i in 1 2 3; do PARITY=1 PARITY_OUT=base$i.json python train.py; done
parity derive base1.json base2.json base3.json
```

```
== 3 repeats, 3 checkpoints: 1 identical, 1 ok, 1 vacuous ==

VACUOUS   actmax_step0
          declared rtol=0.01 atol=1e-12
          spread 3e-07 over 3 repeats at |value| 4.51
          budget 0.0451 is 15033x wider than the noise needs
          -> parity.record('actmax_step0', ..., rtol=6.7e-07)
```

Then gate the change:

```bash
PARITY=1 PARITY_OUT=arm.json python train.py     # with your fix applied
parity compare base1.json arm.json              # exit 0 passes, 1 fails
```

## Two rules worth knowing before you use it

**A tolerance you typed is a claim, not a measurement.** `rtol=0.1` says *I will
accept 10% drift*. It says nothing about whether the number actually moves 10% or
1e-9 between two identical runs. When the gap is large in either direction the
gate is broken — too tight and it fails runs that changed nothing, too loose and
it passes changes that altered numerics. `parity derive` is the only thing that
tells you which, and it needs three runs of **unmodified** code to do it.

**Turn it off for the timed run.** `.item()` on a CUDA tensor is a device
synchronisation. The intended shape is two runs of the same binary differing only
by an env var: one timed with `PARITY` unset, one recorded with it set. With
`PARITY` unset every entry point returns on its first line, and a `record_lazy`
lambda is never called at all — so the reduction inside it costs nothing.

## What gets recorded

A flat list. Nothing is keyed by call site, call order, or line number, because
the two runs being compared are by construction running different code.

```json
[
  {"tag": "loss_step0", "value": 2.431,
   "error_tolerance": {"rtol": 0.001, "atol": 1e-12, "note": "5 repeats"}},
  {"tag": "n_gaussians", "value": 661878,
   "error_tolerance": {"rtol": 0.0, "atol": 1e-12}}
]
```

Comparison pairs **by tag**, so inserting a checkpoint in the middle shows up as
one extra tag rather than shifting everything after it. But the tag sets must
match exactly: a run that stopped after two of three steps fails structurally
instead of passing on the two steps it managed.

## Rules the comparison will not bend

| | |
|---|---|
| `budget = atol + rtol * \|expected\|`; pass iff `\|diff\| <= budget` | `atol` alone gates a value living near zero, where a relative test divides by ~0 |
| The default is `rtol=0, atol=1e-12` | Bit-identical modulo denormals. Every loosening is something you typed on purpose |
| An int may carry `rtol` | A count can legitimately drift under a change that reorders work |
| `661878` → `661878.0` fails anyway | The JSON type declares what kind of quantity this is |
| `bool` and `str` get no tolerance, ever | A flipped flag or a changed caption is a difference, full stop |
| NaN never passes, however loose the gate | |
| The **baseline's** tolerance is used | The arm cannot loosen the gate it has to pass. A tolerance that differs between the two files is reported |
| A duplicate tag is an error | A tag is the identity of a checkpoint |
| An empty record file is an error | A gate over zero checkpoints passes trivially |

## Distributed

`PARITY_OUT=run.json` writes `run.rank3.json` when `torch.distributed` is
initialised (or `RANK` is set). Point `compare` at the two *directories*: files
pair by name, ranks stay separate, and a missing rank file is a structural
failure like any other.

Nothing is all-reduced inside parity. A collective would change the timing and
can deadlock outright when ranks reach a different number of `record` calls.

## API

| | |
|---|---|
| `parity.record(tag, value, *, rtol=, atol=, note=, tolerance=)` | Record a scalar or 1-element tensor. Multi-element tensors are rejected — reduce them yourself |
| `parity.record_lazy(tag, fn, ...)` | Same, but `fn` is called only when recording is on |
| `parity.flush(path=None)` | Resolve every value and write. A full snapshot each time, so calling it per step survives a crash. Also runs at exit |
| `parity.enabled()` | Whether `PARITY` is on. Guard an expensive block with it |
| `parity.set_enabled(bool \| None)` | Force on/off, ignoring the env var. `None` re-reads it |
| `parity.set_default_tolerance(rtol=, atol=, note=)` | For calls that declare none of their own |
| `parity.reset()` | Drop pending records |

| | |
|---|---|
| `parity compare BASE ARM [--rtol X] [--atol Y] [--only GLOB] [--exclude GLOB]` | Exit 1 on any difference beyond tolerance |
| `parity derive REC REC REC...` | Exit 1 when a checkpoint cannot be gated as declared |
| `parity show REC [--json]` | Print one record |

`compare` and `derive` are importable as `parity.compare.compare()` and
`parity.derive.derive()`, both returning a report object with `.ok`.

### Verdicts from `derive`

| | |
|---|---|
| `identical` | Bit-identical across repeats, gated exactly |
| `ok` | Declared tolerance matches the measured noise |
| `slack` | Repeats are bit-identical but slack was declared — fine if it is there for an expected change |
| `vacuous` | Budget is >10x wider than the noise needs. Warning, not a failure |
| `flaky` | Noise already exceeds the declared budget. **Failure** |
| `unstable` | Type, string or bool changed across repeats. A seeding bug, or not a result. **Failure** |

## Notes

Reduce in fp32. A bf16 scalar carries about three decimal digits, so a value
recorded from bf16 has a noise floor near `1e-2` on its own and any `rtol`
tighter than that will flake — `x.float().mean()`, not `x.mean()`.

Prefer `absmean` over `mean` for an activation. A tensor centred near zero has a
mean that is nearly zero by construction, and its relative error explodes for
reasons that have nothing to do with your change.

Put checkpoints on **boundaries**, not wherever looks suspicious: batch out of
the loader, input to the model, each block's output, loss, grad norm, parameter
norm after the optimizer step. Boundaries let you bisect a difference to between
two of them. A checkpoint in the middle of a block only says "different here".

Recording only the loss will miss a broken gradient all-reduce — rank 0's loss
looks fine for a surprisingly long time.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest src -q
```

## Who uses it

`parity` is standalone and has no opinion about who calls it. It was written for
end-to-end performance work, where the question after every change is *is this
faster, or is it just wrong*, and it is the correctness gate of
[torch-performance-agent](https://github.com/TarzanZhao/torch-performance-agent),
which pins this repository as a submodule. Nothing here depends on that.
