"""Language-neutral black-box contract harness for the Go migration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
CASE_ROOT = ROOT / "testdata" / "contracts" / "cases"


def test_python_hatch_executes_shared_fake_provider(tmp_path: Path) -> None:
    case = json.loads((CASE_ROOT / "raw_gemini_success.json").read_text())
    fake = tmp_path / "testprovider"
    subprocess.run(
        ["go", "build", "-o", str(fake), "./internal/testprovider"],
        cwd=ROOT,
        check=True,
    )
    (tmp_path / "gemini").symlink_to(fake)

    record = tmp_path / "invocation.json"
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{tmp_path}{os.pathsep}{env['PATH']}",
            "HATCH_DISABLE_SECRET_HELPER": "1",
            "HATCH_TEST_RECORD": str(record),
            "HATCH_TEST_SCENARIO": case["scenario"],
        }
    )
    completed = subprocess.run(
        [sys.executable, "-m", "hatch", *case["arguments"]],
        cwd=ROOT,
        env=env,
        input=case["stdin"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == case["expected"]["exit_code"], completed.stderr
    result = json.loads(completed.stdout)
    assert result["ok"] is case["expected"]["ok"]
    assert result["status"] == case["expected"]["status"]
    assert result["output"] == case["expected"]["output"]

    invocation = json.loads(record.read_text())
    assert invocation["argv"] == case["expected"]["provider_argv"]
    assert invocation["stdin_sha256"] == case["expected"]["provider_stdin_sha256"]
    assert invocation["stdin_bytes"] == case["expected"]["provider_stdin_bytes"]
    assert invocation["environment"]["DCG_NO_SELF_HEAL"] == "1"
    assert "DCG_BYPASS" not in invocation["environment"]
