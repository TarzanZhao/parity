"""Command line: `parity compare`, `parity derive`, `parity show`.

Exit status is the gate — 0 passes, 1 fails — so these drop straight into a
shell script without parsing any output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from parity import compare as compare_mod
from parity import derive as derive_mod


def _cmd_compare(args: argparse.Namespace) -> int:
    report = compare_mod.compare(
        args.baseline,
        args.arm,
        rtol=args.rtol,
        atol=args.atol,
        only=args.only,
        exclude=args.exclude,
    )
    print(compare_mod.render(report, limit=args.limit))
    return 0 if report.ok else 1


def _cmd_derive(args: argparse.Namespace) -> int:
    report = derive_mod.derive(args.records)
    print(derive_mod.render(report, limit=args.limit))
    return 0 if report.ok else 1


def _cmd_show(args: argparse.Namespace) -> int:
    records = compare_mod.load(args.record)
    if args.json:
        print(json.dumps(
            [{"tag": r.tag, "value": r.value, "error_tolerance": r.tolerance.to_json()}
             for r in records.values()],
            indent=2,
        ))
        return 0
    width = max((len(t) for t in records), default=0)
    for tag, rec in records.items():
        print(f"{tag:<{width}}  {rec.value!r:>16}   {rec.tolerance.describe()}")
    print(f"-- {len(records)} checkpoint(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the `parity` command."""
    parser = argparse.ArgumentParser(
        prog="parity",
        description="Compare recorded scalar checkpoints against a baseline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cmp = sub.add_parser(
        "compare",
        help="judge a run against a baseline",
        description="Judge a run against a baseline. Exit 1 when it differs beyond tolerance.",
    )
    p_cmp.add_argument("baseline", type=Path, help="record from unmodified code")
    p_cmp.add_argument("arm", type=Path, help="record from the run under test")
    p_cmp.add_argument("--rtol", type=float, default=None,
                       help="override every declared relative tolerance")
    p_cmp.add_argument("--atol", type=float, default=None,
                       help="override every declared absolute tolerance")
    p_cmp.add_argument("--only", action="append", default=[], metavar="GLOB",
                       help="compare only tags matching this (repeatable)")
    p_cmp.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                       help="drop tags matching this (repeatable)")
    p_cmp.add_argument("--limit", type=int, default=20, help="rows printed per section")
    p_cmp.set_defaults(func=_cmd_compare)

    p_der = sub.add_parser(
        "derive",
        help="measure run-to-run noise and check the declared tolerances against it",
        description="Read >= 3 repeats of UNMODIFIED code and report, per checkpoint, "
                    "whether its declared tolerance matches the noise actually measured.",
    )
    p_der.add_argument("records", type=Path, nargs="+", help="three or more repeats")
    p_der.add_argument("--limit", type=int, default=20, help="rows printed")
    p_der.set_defaults(func=_cmd_derive)

    p_show = sub.add_parser("show", help="print one record")
    p_show.add_argument("record", type=Path)
    p_show.add_argument("--json", action="store_true", help="print as json")
    p_show.set_defaults(func=_cmd_show)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `parity` console script."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
