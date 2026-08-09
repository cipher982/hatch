# Evaluation task records

Keep one task per file or one JSON object per line. The task must identify the
workspace independently of the harness. Do not encode a harness-specific
solution in the prompt.

Suggested record:

```json
{
  "id": "repo-bug-001",
  "category": "bug-fix",
  "prompt": "Find and fix the regression described in ...",
  "validation": "go test ./path/to/package -run TestName -count=1"
}
```

Before running a task, create a clean worktree from the same base revision for
each harness and repetition. Store the worktree path in the run metadata rather
than changing the task prompt.
