# Hatch Credential Helper Protocol

Status: Go contract

Hatch never discovers a secret manager. An operator opts in by either setting
`HATCH_CREDENTIAL_HELPER` to an absolute executable path or installing that
path in the private plain-text pointer
`${XDG_CONFIG_HOME:-$HOME/.config}/hatch/credential-helper`. The environment setting
takes precedence over the file. Hatch consults the helper only after an explicit
CLI credential and a non-empty provider environment variable are both absent.
The pointer must be an owner-only regular file (normally mode `0600`), may not be
a symlink, and contains only the absolute helper path plus an optional newline.

Hatch starts the executable with no arguments and writes one JSON object plus a
newline to stdin:

```json
{"environment":"OPENAI_API_KEY","project":"personal-shell"}
```

The helper returns the secret bytes on stdout and no diagnostic output on
success. Exit statuses are:

- `0`: found; trimmed stdout must be non-empty;
- `3`: absent; stdout is ignored;
- any other status: authority error; Hatch fails closed and may report bounded
  stderr as a diagnostic.

The call has a ten-second deadline. Secret bytes are placed only in the child
provider environment. They must never enter argv, run manifests, artifacts,
logs, or test goldens.

Installing or selecting a helper changes credential authority and is therefore
an explicit Agent Home operation. Merely shipping this protocol does not select
one.
