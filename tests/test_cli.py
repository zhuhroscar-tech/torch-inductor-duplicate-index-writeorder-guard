"""Tests for the CLI entry point: argument parsing, --version, --json,
--no-color, and exit codes -- independent of whether torch is
installed. Mocks core.diagnose so every CLI message path (torch-
unavailable, no-bug "info" line, isolation-warn line, guard-mismatch
"fail" line) is exercised deterministically, regardless of whether
this host's installed torch build happens to reproduce the underlying
bug."""
from __future__ import annotations

import json
import runpy
import sys

import pytest

from torch_inductor_duplicate_index_writeorder_guard import core
from torch_inductor_duplicate_index_writeorder_guard.cli import main


def _fake_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_url": "https://github.com/pytorch/pytorch/issues/197582",
        "cases": [
            {
                "kind": "computed_duplicate_index",
                "description": "fake case",
                "eager_result": [2.0, 3.0, 3.0, 4.0],
                "native_compiled_result": [3.0, 4.0, 3.0, 4.0],
                "guarded_compiled_result": [2.0, 3.0, 3.0, 4.0],
                "native_matches_eager": False,
                "guard_matches_eager": True,
                "guard_correct": True,
            }
        ],
        "bug_reproduced": True,
        "isolation_confirmed": True,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_version_flag(capsys):
    code = main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert "torch-inductor-duplicate-index-writeorder-guard" in out


def test_json_output_is_valid_json_and_reports_guard_status(capsys):
    torch = pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert "torch_version" in report
    assert report["torch_version"] == torch.__version__
    assert "guard_fully_correct" in report
    assert code in (0, 1)


def test_json_exit_code_matches_guard_fully_correct(capsys):
    pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert code == (0 if report["guard_fully_correct"] else 1)


def test_text_output_no_color_has_no_ansi_escapes(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "\x1b[" not in out


def test_text_output_reports_case_section(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "cases" in out


def test_torch_unavailable_json_mode_reports_error_and_exit_2(monkeypatch, capsys):
    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload == {"error": "torch is required for diagnosis"}
    assert code == 2


def test_torch_unavailable_text_mode_reports_fail_headline_and_exit_2(monkeypatch, capsys):
    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "torch unavailable: torch is required for diagnosis" in out
    assert "[X]" in out
    assert code == 2


def test_no_bug_prints_info_line(monkeypatch, capsys):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(bug_reproduced=False))
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "no miscompilation reproduced on this host's installed torch build" in out


def test_bug_reproduced_prints_fail_line(monkeypatch, capsys):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(bug_reproduced=True))
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "computed-duplicate-index write-order miscompilation reproduced on this host" in out


def test_isolation_not_confirmed_prints_warn_line(monkeypatch, capsys):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(isolation_confirmed=False))
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "did NOT behave as the upstream issue describes" in out


def test_guard_mismatch_prints_fail_line_and_exit_1(monkeypatch, capsys):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guard_fully_correct=False))
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "guard did NOT restore eager semantics" in out
    assert "guard restores eager write-order semantics" not in out
    assert code == 1

    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guard_fully_correct=False))
    code_json = main(["--json"])
    capsys.readouterr()
    assert code_json == 1


def test_module_entry_point_runs_main_and_exits_with_its_code(monkeypatch):
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guard_fully_correct=False))
    monkeypatch.setattr(sys, "argv", ["torch-inductor-duplicate-index-writeorder-guard", "--no-color"])
    monkeypatch.delitem(sys.modules, "torch_inductor_duplicate_index_writeorder_guard.cli", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("torch_inductor_duplicate_index_writeorder_guard.cli", run_name="__main__")
    assert exc_info.value.code == 1
