# Hatch Go Rewrite Epic

Status: Go 0.2.0 cut over; Phase 7 field soak in progress (1/50 contract-complete; 8 observed)

Decision owner: David Rose

Scope: replace Hatch's Python production implementation with one Go CLI while
preserving its supported behavior and implementing the unified durable-run
contract

Companion specification: `docs/durable-run-contract.md`

## Decision

Rewrite Hatch in Go through a contract-first, differential migration. Do not
translate Python files line by line, do not run paid provider calls twice by
default, and do not cut over because the Go code compiles or its unit tests pass.

The released product remains the `hatch` CLI. The Go implementation becomes one
standalone binary with a small internal package graph, no embedded model
judgment, and no required language runtime. Provider executables and explicitly
configured external integrations remain separate dependencies.

The migration has two independent promises:

1. **Compatibility:** every supported observable behavior in the Python CLI is
   classified and either preserved or deliberately changed.
2. **Completion:** every semantic requirement in the unified durable-run
   contract passes against the Go implementation, including requirements that
   the current Python implementation does not satisfy.

The Python implementation is the legacy behavior oracle, not the target
architecture. We will add only the characterization seams needed to observe it.
We will not first implement the whole new architecture in Python and then pay to
implement it again in Go.

## Sequencing decision: change the system in Go

The durable-run design was approved immediately before the language decision.
That creates three plausible sequences:

| Sequence | Benefit | Cost/risk | Decision |
|---|---|---|---|
| Implement the unified system fully in Python, then port it | Production exercises the new semantics before the rewrite; Python becomes a complete differential oracle | Builds the coordinator, store, adapters, schema, and tests twice; creates two opportunities for semantic drift; delays the distribution benefit | Reject |
| Port today's Python architecture to Go, then refactor it | The initial translation has the smallest conceptual delta | Intentionally reproduces separate stream paths and the async bypass; then rewrites the Go code again; old defects can become accidental compatibility | Reject |
| Freeze current Python behavior and implement the approved target directly in Go | Pays for the new architecture once while retaining a real executable oracle for existing behavior | Requires two explicit test authorities and careful classification where new semantics differ | Accept |

This is not a big-bang trust model. It is a single target implementation behind
two independent verification suites:

```text
current Python executable -> legacy parity oracle
approved durable-run spec  -> target contract oracle
                              ↓
                         Go implementation
```

Allowed Python changes after this decision are narrow:

- security or data-loss fixes that cannot wait for cutover;
- characterization hooks needed by the fake-provider/differential harness;
- fixture capture and stable normalization of inherently variable fields;
- an explicit implementation selector used during validation;
- documentation that deprecates the Python import API.

Do not add Python `RunStore`, `RunCoordinator`, universal adapters, new run
commands, archive receipts, or retention machinery merely to make them an
intermediate porting source. Those belong in Go. If urgent production need makes
one of them necessary before Go is ready, implement the smallest Python fix,
add it to both oracles, and record the exception in the migration ledger.

## Why Go

Hatch is deterministic infrastructure around nondeterministic providers. Its
work is process launch, environment construction, concurrent stream capture,
HTTP polling, cancellation, exact event recognition, private filesystem state,
and stable JSON. It is not a numerical workload, web application, scheduler, or
place for model judgment.

Go fits that shape:

- one fast-starting binary for macOS and Linux;
- a sufficient standard library for CLI parsing, subprocesses, HTTP, JSON,
  hashing, files, synchronization, and tests;
- goroutines make simultaneous stdout, stderr, progress, and artifact capture
  direct without splitting the product across async and threaded implementations;
- static types make lifecycle axes and provider claims harder to conflate;
- cross-compilation and installation are simpler than shipping a Python tool
  environment;
- the language remains small enough for agents and humans to review the entire
  execution path.

Go does not make the provider faster and does not automatically make Hatch
correct. Provider calls dominate latency. Atomic persistence, redaction, process
groups, and adapter semantics still require deliberate design and adversarial
tests. The rewrite is justified by a simpler executable and stronger mechanical
core, not a promised orchestration speedup.

The radically simpler alternative remains implementing the approved design in
Python. Phase 1 is therefore a real continuation gate, not ceremony: the first
Go vertical slice must demonstrate a materially simpler single coordinator,
standalone installation, and equal or better failure semantics. If it instead
needs a framework, sprawling platform layer, or compatibility shim, stop the Go
rewrite and apply the durable-run design in Python.

## Research spike

One Exa-backed Hatch research worker reviewed current documented AI-assisted
rewrites. The durable local research artifact remains operational evidence; the
decision-relevant public sources and conclusions are preserved here.

The strongest cases were:

| Case | Evidence | Useful lesson | Limitation |
|---|---|---|---|
| [Wiz Python-to-Go PDF library](https://claude.com/customers/wiz) | Vendor-published production case: roughly 50,000 Python lines became 18,413 Go lines; hundreds of PDFs and 500 pathological cases were tested; a feature-flag rollout found a remaining 1% mismatch and 20–30 issues; production processing was reported at least 2x faster | Real-data parallel comparison found defects the preproduction corpus missed | Quantitative productivity claims are vendor/customer reports, not an independent audit |
| [Bun Zig-to-Rust rewrite](https://bun.sh/blog/bun-in-rust), [merged PR](https://github.com/oven-sh/bun/pull/30412) | Production rewrite across 1,448 files with a pre-existing language-independent suite, adversarial agent review, isolated worktrees, multi-platform CI, benchmarks, and later fuzzing | Massive agent parallelism only worked after work was sharded and Git state isolated; test execution itself was manually verified | Extraordinary scale and stated API cost make it a feasibility proof, not Hatch's operating template; 19 known regressions still shipped and were fixed |
| [Kodit Python-to-Go field report](https://winder.ai/python-to-go-migration-with-claude-code/) | Independent self-report using a dependency-aware migration ledger and test-first agent tasks | Compilation and unit tests passed while schema, indexing, embedding, and end-to-end behavior were wrong; side-by-side real-data tests exposed the failures | One team's report, not a controlled comparison |
| [OpenAI modernization guide](https://developers.openai.com/cookbook/examples/codex/code_modernization) | Vendor method: language-neutral contract, legacy/modern parallel runs, checkpointed plan, separate validation | Supports a parity-first execution model and independent validator | Guidance and examples, not proof of a completed rewrite |

Research conclusions we accept:

- Behavioral oracles beat translation confidence.
- A green unit suite is necessary but insufficient.
- Agent work must be sliced by independently verifiable ownership boundaries.
- The worker that writes an implementation must not silently bless its new
  golden output.
- Real execution and malformed/error paths reveal integration mistakes that
  type checking cannot.
- Performance is a separate measured claim, never inferred from language choice.
- The old implementation must remain an explicit rollback binary until the new
  one has field evidence.

Research advice we reject:

- We will not fully refactor Python onto the new durable-run architecture before
  starting Go. Hatch is small enough that this would duplicate the dominant
  implementation work. A language-neutral oracle plus the current Python
  executable provides the needed reference.
- We will not imitate Bun's 64-agent fleet. Hatch's shared coordinator and
  storage semantics are too coupled for that parallelism to be economical.
- We will not treat vendor-reported duration, cost, or output multipliers as a
  plan or acceptance criterion.

## Baseline and compatibility boundary

At the start of this epic:

- the production code is 3,941 lines of Python;
- the suite collects 304 tests;
- the hermetic baseline is `295 passed, 9 skipped`;
- Python has no runtime package dependencies;
- the public command surfaces are `claude`, `codex`, `cursor`, `openrouter`,
  `expert`, `doctor`, and advanced raw backend flags;
- surfaced Claude, Cursor, and OpenCode paths have separate streaming code;
- Expert is an HTTP execution path;
- only OpenCode currently has the first durable artifact implementation;
- the async Python `run()` function bypasses the streamed CLI path;
- no other checked-out repository under `~/git` imports Hatch's Python library.

### Supported compatibility

The stable public product contract is the CLI:

- command names, aliases, arguments, defaults, stdin behavior, and exit codes;
- machine-mode stdout containing exactly one JSON document;
- human output and stderr progress/diagnostics;
- provider/model resolution;
- credential precedence and environment sanitization;
- cwd, timeout, cancellation, and subprocess behavior;
- exact command/env/stdin construction for provider executables;
- Expert request/poll behavior;
- Longhouse origin metadata and DCG/Observatory integration;
- existing artifact readability and the new durable-run schema.

Help whitespace and incidental Python exception wording are not automatically
stable. The compatibility ledger must classify them rather than accidentally
copying them.

### Python library decision

`from hatch import run, Backend, AgentResult` is documented but has no known
consumer in the checked-out project inventory. It is not possible to replace a
Python import with a pure Go binary while claiming literal compatibility.

Therefore:

- announce the Python library as deprecated when the Go preview lands;
- keep the final Python release and source tag available;
- do not ship a permanent Python shim around the Go binary;
- do not create a public Go library merely to mirror an unused Python API;
- require an explicit consumer report before adding a compatibility bridge.

The migration ledger must still translate every Python library test into either
a CLI-observable invariant, an internal Go coordinator test, or an explicitly
retired Python-only behavior. Zero tests may disappear unclassified.

## First-principles invariants

1. **Tests are claims, not line counts.** “304 tests ported” is insufficient;
   every test needs a named behavioral claim and a Go proof.
2. **The old implementation cannot define new correctness.** Existing behavior
   is preserved unless it conflicts with the approved durable-run contract,
   security, or an explicitly recorded correction.
3. **One provider call has one owner.** Differential tests use fake providers.
   Live parity calls are explicit and sequential, never hidden double-spending.
4. **Raw evidence precedes interpretation.** Tests and production artifacts keep
   provider streams intact; adapters derive facts without deleting evidence.
5. **Mechanics stay deterministic.** Go handles storage, schemas, auth plumbing,
   timeouts, and exact events. The invoked agent decides whether an answer is
   useful.
6. **No skipped-success theater.** CI must prove required tests executed. A
   zero-test package or silently skipped provider corpus is a failure.
7. **Goldens have independent authority.** Implementation code cannot regenerate
   expected output during the same test run.
8. **Every phase is releasable or disposable.** A failed phase can be reverted
   without mutating Python artifacts or provider-native history.
9. **No hybrid production runtime.** During development two binaries coexist,
   but one invocation executes wholly in Python or wholly in Go.
10. **Deletion is a gate, not cleanup.** Python is removed only after Go cutover
    evidence, rollback proof, and artifact compatibility are complete.

## Target Go system

### Repository shape

Keep the package graph deliberately small:

```text
cmd/hatch/                 process entrypoint and build metadata
internal/cli/              argument normalization, CLI rendering, dispatch,
                           and small host-integration preflights
internal/run/              coordinator, store, schema, execution mechanics
internal/provider/         catalog, command builders, stream adapters
internal/expert/           Responses HTTP request and polling adapter
internal/doctor/           live provider contract diagnostics
internal/testprovider/     hermetic fake executable used by contract tests
testdata/contracts/        language-neutral cases, fixtures, and goldens
testdata/legacy/           sanitized Python event/artifact fixtures
```

Do not add a framework, dependency-injection container, event bus, daemon,
plugin loader, or generic session manager. Start with the Go standard library.
A dependency needs a written reason and must remove more code/risk than it adds.

### Executable boundary

`main` does only four things:

1. normalize arguments and machine defaults;
2. construct an application with explicit OS/clock/HTTP dependencies;
3. execute one command;
4. render one exit code.

All surfaced and raw provider commands pass through one `run.Coordinator`.
Expert uses the same run identity/store but an HTTP execution record rather than
a fake process record. No CLI branch may launch a provider below the coordinator.

### Core types

Use concrete structs and validated named string enums for persisted state.
Reject unknown schema major versions, preserve unknown additive JSON fields when
reading legacy records, and never derive one lifecycle axis from another.

The core concepts remain those in the durable-run contract:

```text
RunID
Lifecycle + Outcome
ResultState
CaptureState
ProviderState
ArchiveState
Invocation
ExecutionOutcome (subprocess or HTTP)
Observation
Warning
RunManifest
```

Do not model sufficiency, answer quality, or “partial usefulness.” The mechanical
facts are output presence, terminal-marker observation, process/HTTP completion,
and retained evidence.

### Interfaces only at real seams

Prefer concrete types. Introduce small interfaces only where tests or distinct
mechanisms require substitution:

```go
type Adapter interface {
    ObserveStdout([]byte) []Observation
    ObserveStderr([]byte) []Observation
    Finalize(ExecutionOutcome) ProviderOutcome
    StateClaim(Artifact) ProviderStateClaim
}

type Store interface {
    Prepare(context.Context, PreparedRun) (*Artifact, error)
    OpenStreams(*Artifact) (StreamSinks, error)
    CommitResult(*Artifact, Result) error
    CommitTerminal(*Artifact, TerminalRun) error
}
```

The subprocess executor is concrete and is tested through the fake-provider
process boundary. A fake store injects exact persistence failures. Do not add an
executor interface merely to avoid launching hermetic subprocesses, and do not
make every function mockable.

### Concurrency model

One invocation owns one coordinator goroutine. Subprocess execution may use
bounded goroutines for stdout, stderr, process wait, and caller progress. Every
goroutine has a clear owner, cancellation path, and join before terminal commit.

Rules:

- use `context.Context` for cancellation and deadlines;
- never let a failed artifact sink kill the provider as an accidental side
  effect;
- never use `bufio.Scanner` with its incidental default token limit for provider
  streams; preserve long logical lines without an undocumented truncation
  boundary (any intentional evidence budget is a separate public guardrail);
- send immutable observations or own mutable adapter state in one goroutine;
- close channels from the producer side only;
- run the complete suite with the race detector;
- use OS-specific files for process-group creation/signaling on supported
  platforms;
- record PID plus process start evidence when the OS exposes it;
- do not claim unknown descendants were killed.

### Durable storage

Implement the companion specification directly in Go:

- mint one run ID before provider launch;
- create mode-`0700` run directories and mode-`0600` evidence files;
- tee streams to disk before adapter interpretation;
- atomically replace manifests using a same-directory temporary file, file
  sync, rename, and directory sync at lifecycle boundaries;
- close and hash the evidence set before the terminal transition;
- preserve incomplete artifacts after crashes;
- use allowlisted provider snapshots;
- never serialize credential values or prompt-bearing argv;
- keep current Python/OpenCode and Expert artifacts read-only.

The run store is a filesystem protocol, not a database. Do not introduce SQLite
for Hatch metadata merely because an OpenCode snapshot contains SQLite files.

### Provider adapters

Each adapter owns only exact provider facts:

| Adapter | Execution | Required proof |
|---|---|---|
| Raw text | subprocess | stdout/stderr, exit, timeout, no invented native ID |
| Claude | subprocess JSONL | init/session identity, final result, structured error, long lines, unknown event retention |
| Cursor | subprocess JSONL | init/session identity, tool-event deduplication, result/error, no false recovery claim |
| OpenCode | subprocess JSONL plus isolated XDG state | session identity, structured/transient errors, allowlisted snapshot, same-version recovery hint, concurrency isolation |
| Expert | HTTP | request shape, response ID, polling, timeout without cancellation, stored snapshots, HTTP failures before/after identity |

Provider command builders remain pure: input configuration produces argv,
environment additions/removals, stdin bytes, and risk metadata. Credential
hydration happens before builders. Adapters cannot mutate storage or archive
policy.

### Credential authority boundary

The current Python implementation imports David's shared
`~/git/me/scripts/infisical_cache.py` module in-process. Go cannot preserve that
mechanism, and the public Hatch repository must not absorb a personal secret
manager or silently choose a new credential authority.

The target order remains:

1. explicit CLI credential override where already supported;
2. existing non-empty environment value;
3. explicitly configured external credential helper;
4. provider-native login for surfaces that already use it;
5. fail closed with the current actionable error.

Inapplicable steps are skipped per surface. Claude, Cursor, and Gemini native
login paths do not begin consulting the external helper merely because it
exists.

Define a narrow helper protocol outside Hatch:

```text
input:  requested environment name + logical project/profile
output: secret bytes on stdout, no logging
status: found | absent | authority_error
```

The executable wire contract is specified in
[`credential-helper-protocol.md`](credential-helper-protocol.md).

The command path/configuration is explicit; Hatch never searches for arbitrary
secret tools. The helper inherits the current machine-token/cache authority and
must preserve its error distinctions. Secret bytes never enter argv, manifests,
logs, or test goldens.

Creating or retargeting that helper is a separately authorized Agent Home task.
Until it exists, the Go preview may use explicit environment credentials for
tests and live proofs, but cutover on David's machine is blocked. Do not invoke a
Python module through an implicit `uv` fallback and then claim the installed Go
system has no Python runtime dependency.

### Distribution

Produce reproducible release binaries for:

- `darwin/arm64` as the primary local target;
- `darwin/amd64` while Intel macOS remains supported;
- `linux/amd64` and `linux/arm64` for containers/hosts.

Build metadata includes version, commit, dirty state, Go version, and target.
Pin the initially verified `go1.26.3` toolchain in the module/build workflow;
toolchain upgrades are explicit changes.
Release checks verify `hatch --version`, help, doctor startup, checksum files,
and execution on each available target. Signing/notarization and a Homebrew
formula are separate delivery decisions; they must not block behavioral parity.

## Executable verification system

### 1. Compatibility ledger

Create `testdata/contracts/python-test-ledger.json`. It contains every collected
pytest node ID and exactly one disposition:

```text
preserved_by_contract
preserved_by_go_unit
intentional_change
retired_python_library_only
obsolete_implementation_detail
```

Each row names the Go test or contract case that proves the claim. Intentional
changes require a reason and specification link. CI compares the ledger with
fresh `pytest --collect-only` output until Python deletion; missing and duplicate
rows fail.

### 2. Hermetic fake-provider executable

Build one Go test helper that can impersonate `claude`, `cursor-agent`,
`opencode`, `codex`, and `gemini` based on argv zero plus a scenario variable.
It records:

- argv as an array;
- selected non-secret environment names/values;
- removed-variable assertions;
- cwd;
- stdin bytes/digest;
- signals received and child survival;

It can emit fixture stdout/stderr with controlled timing, malformed JSON, long
lines, structured errors, late success after transient failure, partial output,
nonzero exits, hanging descendants, and concurrent run IDs.

Both Python Hatch and Go Hatch execute this same binary. This tests real process
boundaries without paid providers or language-specific mocks.

### 3. Language-neutral contract cases

Each JSON case specifies inputs, fake-provider scenario, and observable outputs:

```text
CLI arguments and stdin
initial environment and filesystem
expected provider argv/env/stdin/cwd
expected stdout/stderr/exit semantics
expected normalized public JSON
expected artifact tree, modes, manifest transitions, and evidence digests
```

Nondeterminism is normalized only through a reviewed field list: run ID,
timestamps, PID/start identity, duration, and temporary root. Paths become
artifact-relative. Error text, null/unknown distinctions, warnings, raw stream
bytes, ordering where meaningful, and security decisions are not normalized.

Goldens are reviewed inputs. Tests never update them automatically. An agent
changing implementation code may propose a golden diff but a separate reviewer
must validate it against Python evidence or the approved new specification.

### 4. Two suites, not one confused oracle

The migration runs two explicit suites:

- **Legacy parity suite:** Python and Go must match every preserved current
  behavior.
- **Target contract suite:** Go must satisfy the durable-run specification,
  including new behavior absent from Python.

When the target contract intentionally corrects Python, the case belongs to the
target suite and its ledger row is `intentional_change`; differential mismatch
is expected and explained. This prevents both blindly canonizing Python bugs and
silently changing existing behavior.

### 5. Go verification ladder

Required presubmit checks:

```text
go test ./...
go test -race ./...
go vet ./...
go test ./... -run Contract
go test ./... -run LegacyParity
```

`TestContractCorpus` and `TestLegacyParity` load explicit fixture indexes, fail
when the index is empty or a referenced case did not execute, and report one
named subtest per case. A passing `-run` command with no matching test is not
accepted as evidence. A ledger-validation test likewise proves that all 304
baseline node IDs have exactly one live disposition.

Add fuzz targets for provider event parsers, manifest readers, path validation,
and redaction. Seed them with the sanitized real event corpus. Persist every
discovered regression as a normal fixture.

Security/adversarial cases include:

- symlink substitution and traversal;
- hostile artifact roots;
- umask independence;
- permission, write, sync, rename, and close failures;
- credential-bearing env and argv;
- prompts containing shell metacharacters and invalid UTF-8 stream bytes;
- stdout/stderr larger than memory comfort and single lines larger than scanner
  defaults;
- timeout races, cancellation, caller disconnect, process-group survivors, and
  PID reuse evidence;
- concurrent runs with no shared writable provider state;
- malformed, duplicated, reordered, unknown, and version-drifted events;
- legacy artifact reads with unknown additive fields and unsupported majors.

### 6. Live proofs

Credentialed provider checks are explicit, sequential, and harmless. They prove
current provider seams, not core correctness. Each surfaced provider needs:

- one successful bounded prompt;
- native identity observation where exposed;
- terminal output and artifact inspection;
- a deliberate non-billing failure where feasible (missing binary, invalid
  fixture/config, or locally induced timeout rather than wasting tokens);
- recorded provider CLI/API version and proof date.

A provider outage produces an unavailable live proof, not a false failure of the
hermetic core. Cutover still requires a current successful proof for every
enabled surfaced provider.

### 7. Performance and operational proof

Measure rather than assume:

- cold `hatch --help` and `--version` startup;
- idle binary size and peak resident memory;
- sustained stdout/stderr tee throughput;
- 32 concurrent fake-provider runs for race/isolation evidence;
- timeout-to-process-group-cleanup latency;
- artifact create/terminal commit latency;
- Expert polling cancellation and connection reuse.

Go must not regress provider-visible behavior or artifact safety. Provider call
latency is reported separately because it overwhelms launcher overhead.

## Migration execution

### Phase 0 — Freeze and classify

Status: complete.

- Tag the Python baseline and record its build/test commands.
- Save `pytest --collect-only` and the green baseline result.
- Build the compatibility ledger with all 304 tests classified.
- Capture sanitized real provider fixtures and existing artifact examples.
- Define the exact intentional changes introduced by the durable-run contract.
- Add the fake-provider executable and first black-box cases against Python.

Gate: the Python binary is reproducibly green, every test is classified, and the
fake provider can exercise every execution family without credentials.

### Phase 1 — Go skeleton and public CLI

Status: complete.

- Add `go.mod`, `cmd/hatch`, build metadata, and the minimal package graph.
- Implement argument normalization, aliases, help, stdin selection, machine
  defaults, JSON/human rendering, and exit-code mapping.
- Port pure model catalog and backend command construction.
- Port context, credential precedence, environment stripping, DCG/Observatory,
  AWS preflight, Longhouse origin, and doctor mechanics.
- Specify and separately authorize the external credential-helper protocol;
  preserve existing authority while using explicit environment credentials in
  hermetic tests.
- Run command/env cases against Python and Go.

Gate: all applicable legacy CLI/backend/context/credential/doctor claims match;
no provider call can bypass the future coordinator entrypoint. Go cutover
remains blocked until the configured credential helper is proven or the user
explicitly selects a different credential authority.

Continuation gate: build/install the vertical slice and compare its code path,
failure behavior, startup, and operational dependencies with Python. Continue
only if Go is delivering the promised simpler executable and coordinator. A
failure here sends the durable-run implementation back to Python rather than
expanding the Go architecture to justify sunk cost.

### Phase 2 — Execution kernel

Status: complete.

- Implement the prelaunch `RunStore` and one run identity.
- Implement subprocess execution, process groups, concurrent stream sinks,
  cancellation, timeout, and terminal commit.
- Implement the raw-text adapter first.
- Add fault injection, crash subprocess tests, race tests, and fuzz seeds.

Gate: the raw Codex/Gemini fixture matrix and target durable-storage suite pass,
including large streams, concurrency, crash evidence, and every persistence
failure boundary.

### Phase 3 — Structured provider adapters

Status: complete.

- Port Claude, Cursor, and OpenCode using sanitized fixtures.
- Keep adapters thin; share only mechanical parsing helpers.
- Preserve all unknown events as raw evidence.
- Implement OpenCode's allowlisted isolated state snapshot and version-bound
  recovery hint.
- Add an explicit adapter drift tripwire for recognized structured output that
  produces no terminal interpretation.

Gate: each adapter passes success, failure, transient recovery, missing terminal,
unknown event, long-line, timeout, and identity/capability cases. Differential
legacy cases have zero unexplained divergence.

### Phase 4 — Expert and complete target contract

Status: complete.

- Port Expert using `net/http` with explicit polling and timeout semantics.
- Store response snapshots in the common run artifact.
- Finish public nested run JSON and compatibility aliases.
- Add local `hatch runs list|inspect`.
- Add read-only compatibility readers for Python/OpenCode and Expert artifacts.

Gate: every V1 requirement in `docs/durable-run-contract.md` maps to a passing Go
test; existing artifacts are inspectable without mutation.

### Phase 5 — Independent validation

Status: complete. See `docs/validation-report.md`.

- Run the complete legacy parity and target contract suites.
- Assign validation to an agent/context that did not write the relevant package.
- Perform adversarial review of process cleanup, persistence ordering,
  redaction, provider capability claims, and test skip behavior.
- Run benchmarks and compare declared objectives.
- Run current live proofs for every enabled surfaced provider.
- Build all release targets and smoke them where runners exist.

Gate: zero unexplained differential results; all required tests demonstrably
execute; race/vet/fuzz seed suites pass; live proof and release matrix are
complete.

### Phase 6 — Cutover with explicit rollback

Status: complete. Go is selected locally; the real Go → Python → Go rollback
was rehearsed without artifact or credential migration.

- Install the Go binary as `hatch` while retaining the last Python release as
  `hatch-python` or an exact recoverable package version.
- Provide an explicit operator-only implementation selector during validation;
  never silently fall back per invocation.
- Verify `which hatch`, version/commit, help, doctor, fake-provider smoke, and one
  real surfaced run from the installed artifact.
- Keep all artifact readers backward-compatible.

Gate: rollback to the Python binary is rehearsed and requires no data mutation,
credential change, or provider-state conversion.

### Phase 7 — Retire Python

Status: in progress. Eight genuine durable Go runs have accumulated, but the
seven runs before the post-cutover contract audit are provider evidence rather
than Python-retirement credit. One run currently has the final persisted hash
manifest and canonical axes. Python is retained only as the frozen compatibility
oracle and rollback release until the cryptographic
`scripts/check-field-evidence.sh` gate passes.

- Observe at least 50 real Go Hatch invocations spanning every enabled surfaced
  provider, with at least five per surface.
- Require zero unexplained result/capture loss, credential exposure, duplicate
  provider execution, or process cleanup incident.
- Close every migration ledger row and remove obsolete Python-only CI.
- Preserve the Python source tag, release instructions, fixtures, and artifact
  reader tests.
- Remove Python production code, package metadata, and compatibility selector in
  one reviewable commit.

Gate: Go is the only production implementation, rollback remains available from
the preserved release, and a clean checkout can build/test/install Hatch without
Python.

## Agent execution protocol

AI makes this rewrite cheaper only if ownership and validation remain explicit.

- One lead owns the epic, contract ledger, core types, and final integration.
- Before Phase 2 stabilizes, parallel work is limited to the oracle/fixtures and
  non-overlapping pure command builders.
- After the kernel contract freezes, adapters may be ported in separate
  worktrees with one adapter per task.
- Every task names source files, ledger rows, target package, fixtures, commands,
  and acceptance tests.
- Agents stage only explicit paths and commit one coherent slice.
- The implementation agent cannot approve its own golden changes.
- A separate adversarial reviewer examines semantic parity, test omissions, dead
  code, and unjustified abstraction; findings are verified against current code.
- Shared-branch merge retries and conflict strategies that discard sibling
  changes are forbidden.
- Failed work is fixed at the task/spec/harness level when the failure pattern
  repeats; do not accumulate hand patches around a broken generation process.

Useful parallel lanes:

```text
contract corpus / ledger
pure provider command builders
one provider adapter per isolated worktree after kernel freeze
independent adversarial verification
```

Keep the coordinator, store, schema, and process-group code single-owner until
their contracts are stable.

## Stop, rollback, and correction rules

Stop the rewrite before cutover if:

- the current Python behavior cannot be classified into a finite oracle;
- the Go design requires a daemon, plugin framework, or hybrid runtime to match
  this small CLI;
- a provider can only be supported by weakening raw evidence, credential safety,
  or lifecycle truthfulness;
- differential normalization grows to hide substantive output differences;
- agents repeatedly change tests to match their implementation rather than the
  approved contract;
- the release binary cannot reproduce artifact/process semantics on supported
  targets.

Roll back cutover immediately for:

- lost final output or artifacts;
- leaked credentials or prompts through argv/metadata;
- duplicate paid provider execution;
- an unbounded child process after timeout/cancellation;
- corrupt or destructively migrated legacy artifacts;
- an unexplained public JSON or exit-code incompatibility affecting callers;
- provider fallback or authority changes not selected by the user.

When Python and the approved target disagree, classify the discrepancy:

```text
Go bug -> fix Go and add regression fixture
Python bug corrected by target spec -> record intentional change
underspecified behavior -> stop, decide, then update spec and both test authorities
provider drift -> update fixture from raw evidence and adapter claim, never guess
```

## Definition of done

- One Go binary implements every supported Hatch CLI surface.
- Hatch itself requires no Python runtime or package; the installed credential
  path also has no hidden Python fallback at final cutover.
- Every original pytest node has a reviewed disposition and Go/contract proof.
- The legacy parity suite has zero unexplained differences.
- Every unified durable-run V1 requirement has a named passing Go test.
- `go test`, race, vet, contract, parity, fuzz seeds, adversarial storage, and
  release-target checks pass without silently skipped required cases.
- Current live proofs pass for every enabled surfaced provider.
- Existing Python/OpenCode/Expert artifacts remain readable and unmodified.
- The installed Go binary identifies its source commit and passes an installed
  end-to-end smoke.
- Rollback to the preserved Python release has been rehearsed.
- Field evidence satisfies the Python deletion gate.
- Documentation, AGENTS guidance, install commands, and CI describe only the
  final Go system, while the migration ledger and source tag preserve history.
