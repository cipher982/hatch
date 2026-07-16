"""Tests for live provider-contract checks."""

from __future__ import annotations

import subprocess
from unittest import mock

from hatch.cli import EXIT_CONFIG_ERROR, EXIT_SUCCESS, main
from hatch.doctor import DoctorCheck, check_cursor_model, parse_cursor_model_ids


def test_parse_cursor_model_ids() -> None:
    output = """Available models

auto - Auto (default)
cursor-grok-4.5-high - Cursor Grok 4.5
"""
    assert parse_cursor_model_ids(output) == {"auto", "cursor-grok-4.5-high"}


def test_cursor_model_check_passes_for_configured_model() -> None:
    result = subprocess.CompletedProcess(
        ["cursor-agent", "models"],
        0,
        stdout="cursor-grok-4.5-high - Cursor Grok 4.5\n",
        stderr="",
    )
    with mock.patch("hatch.doctor.subprocess.run", return_value=result):
        check = check_cursor_model()

    assert check.ok is True
    assert "cursor-grok-4.5-high is available" in check.detail


def test_cursor_model_check_explains_stale_mapping() -> None:
    result = subprocess.CompletedProcess(
        ["cursor-agent", "models"],
        0,
        stdout="auto - Auto (default)\n",
        stderr="",
    )
    with mock.patch("hatch.doctor.subprocess.run", return_value=result):
        check = check_cursor_model()

    assert check.ok is False
    assert "configured model 'cursor-grok-4.5-high' is unavailable" in check.detail


def test_cursor_model_check_handles_missing_cli() -> None:
    with mock.patch("hatch.doctor.subprocess.run", side_effect=FileNotFoundError):
        check = check_cursor_model()

    assert check.ok is False
    assert check.detail == "cursor-agent is not installed"


def test_doctor_command_reports_success(capsys) -> None:
    with mock.patch(
        "hatch.cli.run_doctor",
        return_value=[DoctorCheck("cursor.grok", True, "current model is available")],
    ):
        exit_code = main(["doctor"])

    assert exit_code == EXIT_SUCCESS
    assert "PASS cursor.grok: current model is available" in capsys.readouterr().out


def test_doctor_command_reports_json_failure(capsys) -> None:
    with mock.patch(
        "hatch.cli.run_doctor",
        return_value=[DoctorCheck("cursor.grok", False, "configured model is unavailable")],
    ):
        exit_code = main(["doctor", "--json"])

    assert exit_code == EXIT_CONFIG_ERROR
    assert capsys.readouterr().out == (
        '{"ok": false, "checks": [{"name": "cursor.grok", "ok": false, '
        '"detail": "configured model is unavailable"}]}\n'
    )
