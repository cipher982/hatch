"""Language-neutral black-box contract harness for the Go migration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent
CASE_ROOT = ROOT / "testdata" / "contracts" / "cases"
PREPARED_PROMPT = (
    ROOT / "testdata" / "contracts" / "fixtures" / "oracle_prepared_prompt.txt"
).read_text().rstrip("\n")


def contract_cases() -> list[Path]:
    return sorted(CASE_ROOT.glob("*.json"))


@pytest.fixture(scope="session")
def fake_provider(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("fake-provider")
    fake = root / "testprovider"
    subprocess.run(
        ["go", "build", "-o", str(fake), "./internal/testprovider"],
        cwd=ROOT,
        check=True,
    )
    return fake


@pytest.mark.parametrize("case_path", contract_cases(), ids=lambda path: path.stem)
def test_python_hatch_executes_shared_fake_provider(
    tmp_path: Path, fake_provider: Path, case_path: Path
) -> None:
    case = json.loads(case_path.read_text())
    for binary in case["provider_binaries"]:
        (tmp_path / binary).symlink_to(fake_provider)

    record = tmp_path / "invocation.json"
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ)
    env.update(
        {
            "HOME": str(home),
            "PATH": f"{tmp_path}{os.pathsep}{env['PATH']}",
            "HATCH_DISABLE_SECRET_HELPER": "1",
            "HATCH_RUN_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
            "HATCH_TEST_RECORD": str(record),
            "HATCH_TEST_SCENARIO": case["scenario"],
        }
    )
    env.update(case.get("environment", {}))
    completed = subprocess.run(
        [sys.executable, "-m", "hatch", *case["arguments"]],
        cwd=ROOT,
        env=env,
        input=case["stdin"],
        text=True,
        capture_output=True,
        check=False,
    )

    expected = case["expected"]
    assert completed.returncode == expected["exit_code"], completed.stderr
    for fragment in expected.get("stderr_contains", []):
        assert fragment in completed.stderr
    result = json.loads(completed.stdout)
    result.pop("duration_ms")
    if expected["result"]["artifact_path"] == "$ARTIFACT":
        artifact = Path(result["artifact_path"])
        assert artifact.is_dir()
        assert (artifact / "metadata.json").is_file()
        result["artifact_path"] = "$ARTIFACT"
    assert result == expected["result"]

    invocation = json.loads(record.read_text())
    normalized_argv = [
        "$PREPARED_PROMPT" if arg == PREPARED_PROMPT else arg
        for arg in invocation["argv"]
    ]
    assert normalized_argv == expected["provider_argv"]
    assert invocation["stdin_sha256"] == expected["provider_stdin_sha256"]
    assert invocation["stdin_bytes"] == expected["provider_stdin_bytes"]
    assert invocation["environment"]["DCG_NO_SELF_HEAL"] == "1"
    assert "DCG_BYPASS" not in invocation["environment"]
