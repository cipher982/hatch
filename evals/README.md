# Harness evaluation

This directory holds reproducible comparisons of the coding harnesses behind
Hatch. The public switch is:

```bash
hatch codex sol --harness opencode --json "task"
hatch codex sol --harness pi --json "task"
hatch codex sol --harness omp --json "task"
```

OpenCode remains the default for the `codex` and `openrouter` surfaces. The
evaluation harness must always make the selected implementation explicit in
its output metadata.

## Evaluation rules

Each run should record:

- task ID, harness, model, provider, and installed tool versions;
- fresh checkout or worktree path;
- prompt text and validation command;
- Hatch JSON result and artifact path;
- wall-clock duration, exit status, token usage when the provider exposes it;
- validation result and a human correctness judgment.

Use two lanes:

1. Product lane: each harness runs with its intended defaults.
2. Normalized lane: the same model, task prompt, isolated home/configuration,
   no optional extensions or fallback models, and the same network policy.

Do not reuse a modified checkout between harnesses. Do not hide retries or
provider fallback. A failed launch is a recorded result, not a reason to rerun
the task silently.

## Starter task shape

Task records belong in `evals/tasks/`. Keep prompts and validation commands
small enough that a task can be repeated. A task should have one primary
success condition and may include secondary observations.

The first useful comparison should use 8 to 12 tasks, repeated three times per
harness and model. Expand only after the task set produces distinguishable
results.

Recommended categories:

- repository navigation and diagnosis;
- single-file bug fix with an existing regression test;
- multi-file feature with a narrow validation command;
- refactor with behavior-preserving tests;
- test failure investigation;
- task requiring careful editing of unfamiliar code.

The first benchmark should compare Pi, OpenCode, and OMP on the same OpenRouter
model. A second pass can compare the same three harnesses on an OpenAI API
model. Claude subscription runs stay on the Claude Code surface.
