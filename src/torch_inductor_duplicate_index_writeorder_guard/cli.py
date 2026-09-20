"""Command-line interface: run the from-scratch diagnosis of the
Inductor computed-duplicate-index write-order miscompilation against
the currently installed torch build, using the shared semantic-color
design system.
"""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-inductor-duplicate-index-writeorder-guard",
        description=(
            "Diagnose whether the currently installed torch build's "
            "Inductor backend miscompiles an advanced-indexing "
            "read-modify-write assignment (v[idx] = v[idx] + delta) "
            "when idx contains a COMPUTED duplicate index "
            "(pytorch/pytorch#197582), confirm a literal-index control "
            "case is unaffected (isolating the defect), and verify the "
            "safe_dup_index_assign() guard (torch._dynamo.disable "
            "graph-break wrapper) restores eager semantics. Never "
            "trusts a cached or previously-reported result, always "
            "re-runs the repro on THIS host's actual installed torch "
            "version."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-inductor-duplicate-index-writeorder-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_correct"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["bug_reproduced"]:
        print(status_headline(style, "fail", "computed-duplicate-index write-order miscompilation reproduced on this host"))
    else:
        print(status_headline(style, "info", "no miscompilation reproduced on this host's installed torch build"))

    if report["isolation_confirmed"]:
        print(status_headline(style, "ok", "literal-duplicate-index control case correctly unaffected (defect isolated to computed indices)"))
    else:
        print(status_headline(style, "warn", "literal-duplicate-index control case did NOT behave as the upstream issue describes"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "guard restores eager write-order semantics"))
    else:
        print(status_headline(style, "fail", "guard did NOT restore eager semantics"))

    section("cases (kind -> eager vs native vs guard)")
    for c in report["cases"]:
        native_flag = "DIVERGED" if not c["native_matches_eager"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    c["kind"],
                    f"native={native_flag:9s}  guard_matches_eager={str(c['guard_matches_eager']):5s}  {guard_flag}",
                )
            ]
        )

    return 0 if report["guard_fully_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
