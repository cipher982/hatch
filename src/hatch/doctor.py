"""Live checks for Hatch's local provider integrations."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass

from hatch.models import CURSOR_GROK


@dataclass(frozen=True)
class DoctorCheck:
    """One provider-contract check."""

    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def parse_cursor_model_ids(output: str) -> set[str]:
    """Extract model IDs from ``cursor-agent models`` output."""
    return {
        line.split(" - ", 1)[0].strip()
        for line in output.splitlines()
        if " - " in line and line.split(" - ", 1)[0].strip()
    }


def check_cursor_model() -> DoctorCheck:
    """Verify that Hatch's stable Grok alias still resolves on this account."""
    try:
        result = subprocess.run(
            ["cursor-agent", "models"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        return DoctorCheck("cursor.grok", False, "cursor-agent is not installed")
    except subprocess.TimeoutExpired:
        return DoctorCheck("cursor.grok", False, "cursor-agent models timed out after 30s")

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
        return DoctorCheck("cursor.grok", False, f"could not list Cursor models: {detail}")

    available = parse_cursor_model_ids(result.stdout)
    if CURSOR_GROK not in available:
        return DoctorCheck(
            "cursor.grok",
            False,
            f"configured model {CURSOR_GROK!r} is unavailable; run `cursor-agent models` and update CURSOR_GROK",
        )
    return DoctorCheck("cursor.grok", True, f"{CURSOR_GROK} is available")


def run_doctor() -> list[DoctorCheck]:
    """Run live integration checks."""
    return [check_cursor_model()]
