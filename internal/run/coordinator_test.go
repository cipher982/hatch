package run

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cipher982/hatch/internal/provider"
)

func TestCoordinatorRawSuccess(t *testing.T) {
	t.Setenv("LONGHOUSE_MANAGED_SESSION_ID", "parent-session")
	t.Setenv("LONGHOUSE_THREAD_ID", "parent-thread")
	t.Setenv("LONGHOUSE_PROVIDER_SESSION_ID", "parent-provider")
	fake := buildTestProvider(t)
	record := filepath.Join(t.TempDir(), "record.json")
	invocation := provider.Invocation{
		Argv: []string{fake}, Stdin: []byte(provider.PreparePrompt("oracle prompt")),
		SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "success_text", "HATCH_TEST_RECORD": record},
	}
	root := filepath.Join(t.TempDir(), "runs")
	coordinator := NewCoordinator(NewStore(root))
	result := coordinator.Execute(Request{
		Surface: "codex.raw", Provider: "openai", Model: "gpt-test",
		Prompt: "oracle prompt", Timeout: 5 * time.Second, Invocation: invocation, Automation: true,
	})
	if !result.OK || result.Output != "fake provider output\n" || result.Run == nil {
		t.Fatalf("unexpected result: %#v", result)
	}
	if result.ArtifactPath == nil || result.Run.Lifecycle != LifecycleTerminal || result.Run.Outcome == nil || *result.Run.Outcome != OutcomeSucceeded {
		t.Fatalf("incomplete durable result: %#v", result)
	}
	if _, err := os.Stat(filepath.Join(*result.ArtifactPath, "result.json")); err != nil {
		t.Fatalf("public projection missing: %v", err)
	}
	var observed struct {
		Environment map[string]string `json:"environment"`
	}
	data, err := os.ReadFile(record)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &observed); err != nil {
		t.Fatal(err)
	}
	if observed.Environment["LONGHOUSE_HATCH_RUN_ID"] != result.Run.RunID {
		t.Fatalf("Longhouse run id = %q, manifest = %q", observed.Environment["LONGHOUSE_HATCH_RUN_ID"], result.Run.RunID)
	}
	if observed.Environment["LONGHOUSE_IS_SIDECHAIN"] != "1" || observed.Environment["LONGHOUSE_ORIGIN_KIND"] != "hatch_automation" ||
		observed.Environment["LONGHOUSE_PARENT_SESSION_ID"] != "parent-session" ||
		observed.Environment["LONGHOUSE_PARENT_THREAD_ID"] != "parent-thread" ||
		observed.Environment["LONGHOUSE_PARENT_PROVIDER_SESSION_ID"] != "parent-provider" {
		t.Fatalf("Longhouse automation environment missing: %#v", observed.Environment)
	}
	if observed.Environment["DCG_NO_SELF_HEAL"] != "1" {
		t.Fatalf("guard environment missing: %#v", observed.Environment)
	}
}

type alwaysFailWriter struct{}

func (alwaysFailWriter) Write([]byte) (int, error) { return 0, errors.New("sink failed") }

type failingStreamSink struct{}

func (failingStreamSink) Write([]byte) (int, error) { return 0, errors.New("stream sink failed") }
func (failingStreamSink) Sync() error               { return nil }
func (failingStreamSink) Close() error              { return nil }

type failingStreamStore struct{ Store }

func (f failingStreamStore) OpenStreams(*Artifact) (StreamSink, StreamSink, error) {
	return failingStreamSink{}, failingStreamSink{}, nil
}

type unavailableStreamStore struct{ Store }

func (u unavailableStreamStore) OpenStreams(*Artifact) (StreamSink, StreamSink, error) {
	return nil, nil, errors.New("streams unavailable")
}

type orderingStore struct {
	Store
	events *[]string
}

func (s orderingStore) WritePublicProjection(artifact *Artifact, result PublicResult) error {
	*s.events = append(*s.events, "result.json")
	return s.Store.WritePublicProjection(artifact, result)
}

func (s orderingStore) CommitTerminal(artifact *Artifact, outcome Outcome, exitCode int, result Result, state State, warnings []Warning) error {
	*s.events = append(*s.events, "terminal manifest")
	return s.Store.CommitTerminal(artifact, outcome, exitCode, result, state, warnings)
}

type transientCommitStore struct {
	Store
	attempts *int
}

type markRunningFailureStore struct{ Store }

func (s markRunningFailureStore) MarkRunning(*Artifact, int, time.Time, string) error {
	return errors.New("injected mark-running failure")
}

type writeResultFailureStore struct{ Store }

func (s writeResultFailureStore) WriteResult(*Artifact, []byte) (string, error) {
	return "", errors.New("injected result-write failure")
}

func (s transientCommitStore) CommitTerminal(artifact *Artifact, outcome Outcome, exitCode int, result Result, state State, warnings []Warning) error {
	(*s.attempts)++
	if *s.attempts == 1 {
		return errors.New("injected terminal commit failure")
	}
	return s.Store.CommitTerminal(artifact, outcome, exitCode, result, state, warnings)
}

func TestCoordinatorPersistsLaunchFailure(t *testing.T) {
	root := filepath.Join(t.TempDir(), "runs")
	result := NewCoordinator(NewStore(root)).Execute(Request{
		Surface: "codex.raw", Provider: "openai", Prompt: "prompt", Timeout: time.Second,
		Invocation: provider.Invocation{Argv: []string{filepath.Join(t.TempDir(), "missing-provider")}},
	})
	if result.OK || result.Status != "not_found" || result.ExitCode != -2 || result.Run == nil || result.ArtifactPath == nil ||
		result.Run.Lifecycle != LifecycleTerminal || result.Run.Outcome == nil || *result.Run.Outcome != OutcomeLaunch {
		t.Fatalf("launch result = %#v", result)
	}
	for _, name := range []string{"manifest.json", "result.json", "result.txt"} {
		if _, err := os.Stat(filepath.Join(*result.ArtifactPath, name)); err != nil {
			t.Fatalf("%s missing after launch failure: %v", name, err)
		}
	}
}

func TestCoordinatorReturnsCanonicalRunWhenStreamsUnavailable(t *testing.T) {
	store := unavailableStreamStore{Store: NewStore(filepath.Join(t.TempDir(), "runs"))}
	result := NewCoordinator(store).Execute(Request{
		Surface: "codex.raw", Provider: "openai", Prompt: "prompt", Timeout: time.Second,
		Invocation: provider.Invocation{Argv: []string{"unused-provider"}},
	})
	if result.OK || result.ExitCode != -3 || result.Run == nil || result.ArtifactPath != nil ||
		result.Run.Lifecycle != LifecycleTerminal || result.Run.Outcome == nil || *result.Run.Outcome != OutcomeFailed ||
		result.Run.Capture.State != "degraded" {
		t.Fatalf("stream-open result = %#v", result)
	}
}

func TestCoordinatorWritesProjectionBeforeTerminalManifest(t *testing.T) {
	fake := buildTestProvider(t)
	events := []string{}
	store := orderingStore{Store: NewStore(filepath.Join(t.TempDir(), "runs")), events: &events}
	result := NewCoordinator(store).Execute(Request{
		Surface: "codex.raw", Provider: "openai", Prompt: "prompt", Timeout: 5 * time.Second,
		Invocation: provider.Invocation{Argv: []string{fake}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "success_text"}},
	})
	if !result.OK || len(events) < 2 || events[0] != "result.json" || events[1] != "terminal manifest" {
		t.Fatalf("result=%#v events=%#v", result, events)
	}
}

func TestCoordinatorRetriesTerminalCommitAsDegraded(t *testing.T) {
	fake := buildTestProvider(t)
	attempts := 0
	store := transientCommitStore{Store: NewStore(filepath.Join(t.TempDir(), "runs")), attempts: &attempts}
	result := NewCoordinator(store).Execute(Request{
		Surface: "codex.raw", Provider: "openai", Prompt: "prompt", Timeout: 5 * time.Second,
		Invocation: provider.Invocation{Argv: []string{fake}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "success_text"}},
	})
	if !result.OK || attempts != 2 || result.Run == nil || result.Run.Lifecycle != LifecycleTerminal ||
		result.Run.Capture.State != "degraded" || len(result.Run.Warnings) == 0 || result.ArtifactPath != nil {
		t.Fatalf("result=%#v attempts=%d", result, attempts)
	}
	data, err := os.ReadFile(filepath.Join(result.Run.Capture.ArtifactPath, "manifest.json"))
	if err != nil || !bytes.Contains(data, []byte(`"lifecycle": "terminal"`)) || !bytes.Contains(data, []byte("injected terminal commit failure")) {
		t.Fatalf("disk manifest=%s err=%v", data, err)
	}
}

func TestCoordinatorReturnsAnswerWhenMarkRunningPersistenceFails(t *testing.T) {
	fake := buildTestProvider(t)
	store := markRunningFailureStore{Store: NewStore(filepath.Join(t.TempDir(), "runs"))}
	result := NewCoordinator(store).Execute(Request{
		Surface: "codex.raw", Provider: "openai", Prompt: "prompt", Timeout: 5 * time.Second,
		Invocation: provider.Invocation{Argv: []string{fake}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "success_text"}},
	})
	if !result.OK || result.Output != "fake provider output\n" || result.Run == nil || result.Run.Capture.State != "degraded" || result.ArtifactPath != nil {
		t.Fatalf("result=%#v", result)
	}
}

func TestCoordinatorCleansUpTimeoutWhenMarkRunningPersistenceFails(t *testing.T) {
	fake := buildTestProvider(t)
	store := markRunningFailureStore{Store: NewStore(filepath.Join(t.TempDir(), "runs"))}
	result := NewCoordinator(store).Execute(Request{
		Surface: "codex.raw", Provider: "openai", Prompt: "prompt", Timeout: 50 * time.Millisecond,
		Invocation: provider.Invocation{Argv: []string{fake}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "hang"}},
	})
	if result.Run == nil || result.Run.Outcome == nil || *result.Run.Outcome != OutcomeTimedOut || result.Run.Process == nil ||
		result.Run.Process.TimeoutCleanup == nil || result.Run.Capture.State != "degraded" {
		t.Fatalf("result=%#v", result)
	}
}

func TestCoordinatorReturnsAnswerWhenResultPersistenceFails(t *testing.T) {
	fake := buildTestProvider(t)
	store := writeResultFailureStore{Store: NewStore(filepath.Join(t.TempDir(), "runs"))}
	result := NewCoordinator(store).Execute(Request{
		Surface: "codex.raw", Provider: "openai", Prompt: "prompt", Timeout: 5 * time.Second,
		Invocation: provider.Invocation{Argv: []string{fake}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "success_text"}},
	})
	if !result.OK || result.Output != "fake provider output\n" || result.Run == nil || result.Run.Result.OutputFile != nil || result.Run.Capture.State != "degraded" || result.ArtifactPath != nil {
		t.Fatalf("result=%#v", result)
	}
}

func TestCaptureWriterPreservesAnswerAfterSinkFailure(t *testing.T) {
	var memory bytes.Buffer
	w := &captureWriter{memory: &memory, sink: alwaysFailWriter{}}
	data := []byte("complete provider answer")
	written, err := w.Write(data)
	if err != nil || written != len(data) || memory.String() != string(data) || w.sinkErr == nil {
		t.Fatalf("write=%d err=%v memory=%q sinkErr=%v", written, err, memory.String(), w.sinkErr)
	}
	more := []byte(" and more")
	_, _ = w.Write(more)
	if memory.String() != string(append(data, more...)) {
		t.Fatalf("continued capture = %q", memory.String())
	}
}

func TestCoordinatorReturnsAnswerWhenArtifactStreamFails(t *testing.T) {
	fake := buildTestProvider(t)
	store := failingStreamStore{Store: NewStore(filepath.Join(t.TempDir(), "runs"))}
	result := NewCoordinator(store).Execute(Request{
		Surface: "codex.raw", Provider: "openai", Prompt: "prompt", Timeout: 5 * time.Second,
		Invocation: provider.Invocation{Argv: []string{fake}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "success_text"}},
	})
	if !result.OK || result.Output != "fake provider output\n" || result.ArtifactPath != nil || result.Run == nil ||
		result.Run.Capture.State != "degraded" || result.Run.Outcome == nil || *result.Run.Outcome != OutcomeSucceededWarnings {
		t.Fatalf("result = %#v", result)
	}
}

func TestCoordinatorRawFailure(t *testing.T) {
	fake := buildTestProvider(t)
	coordinator := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs")))
	result := coordinator.Execute(Request{
		Surface: "codex.raw", Provider: "openai", Prompt: "prompt", Timeout: 5 * time.Second,
		Invocation: provider.Invocation{Argv: []string{fake}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "stderr_nonzero"}},
	})
	if result.OK || result.ExitCode != 23 || result.Error == nil || *result.Error != "fake provider failure\n" {
		t.Fatalf("unexpected failure: %#v", result)
	}
}

func TestCoordinatorStructuredProviders(t *testing.T) {
	fake := buildTestProvider(t)
	tests := []struct {
		name, backend, model, scenario, output, session string
	}{
		{"claude", "claude", "haiku", "success_claude", "fake claude output", "claude-session-oracle"},
		{"cursor", "cursor", "grok-4.7-high", "success_cursor", "fake cursor output", "cursor-session-oracle"},
		{"pi", "pi", "openai/gpt-6-astra", "success_pi", "fake success_pi output", ""},
		{"omp", "omp", "openai/gpt-6-astra", "success_omp", "fake success_omp output", ""},
		{"cursor-kimi", "cursor", "kimi-k3", "success_cursor_kimi", "fake cursor kimi output", "cursor-kimi-session-oracle"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var progress []string
			invocation, err := provider.Build(provider.Request{Backend: test.backend, Model: test.model, Prompt: "oracle prompt", APIKey: "fake"})
			if err != nil {
				t.Fatal(err)
			}
			invocation.Argv[0] = fake
			if invocation.SetEnv == nil {
				invocation.SetEnv = map[string]string{}
			}
			invocation.SetEnv["HATCH_TEST_SCENARIO"] = test.scenario
			if test.backend == "pi" {
				invocation.SetEnv["HATCH_TEST_SESSION"] = "pi-session-oracle"
			}
			if test.backend == "omp" {
				invocation.SetEnv["HATCH_TEST_SESSION"] = "omp-session-oracle"
			}
			coordinator := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs")))
			result := coordinator.Execute(Request{
				Surface: test.name, Provider: test.name, Model: test.model, Prompt: "oracle prompt",
				Timeout: 5 * time.Second, Invocation: invocation,
				ProgressLabel: test.name, Progress: func(message string) { progress = append(progress, message) },
			})
			if !result.OK || result.Output != test.output || (test.session != "" && (result.SessionID == nil || *result.SessionID != test.session)) {
				t.Fatalf("unexpected result: ok=%v output=%q session=%v error=%s stderr=%s", result.OK, result.Output, result.SessionID, stringValue(result.Error), stringValue(result.Stderr))
			}
			if result.Run.Result.TerminalMarker != "observed" || result.Run.Capture.StdoutFile != "stdout.jsonl" {
				t.Fatalf("structured evidence mismatch: %#v", result.Run)
			}
			if test.backend == "pi" || test.backend == "omp" {
				if result.Run.ProviderState.Retention != "unavailable" || result.Run.ProviderState.Capabilities["state_isolation"] != "hatch_per_run" || result.Run.ProviderState.Capabilities["session_persistence"] != "disabled_ephemeral" {
					t.Fatalf("%s state policy = %#v", test.backend, result.Run.ProviderState)
				}
				if _, err := os.Stat(filepath.Join(*result.ArtifactPath, "provider")); !os.IsNotExist(err) {
					t.Fatalf("%s provider runtime retained: %v", test.backend, err)
				}
			}
		})
	}
}

func TestCoordinatorStructuredFailureAndRecovery(t *testing.T) {
	fake := buildTestProvider(t)
	tests := []struct {
		name, backend, model, scenario, output, errorText string
		ok                                                bool
		outcome                                           Outcome
		warnings                                          int
	}{
		{"cursor error", "cursor", "grok-4.7-high", "cursor_error", "", "request rejected", false, OutcomeFailed, 0},
		{"pi error", "pi", "openai/gpt-6-astra", "pi_error", "", "provider unavailable", false, OutcomeFailed, 0},
		{"pi recovered", "pi", "openai/gpt-6-astra", "pi_transient_then_success", "recovered answer", "", true, OutcomeSucceededWarnings, 1},
		{"pi missing terminal", "pi", "openai/gpt-6-astra", "pi_missing_terminal", "useful evidence", "structured provider output did not contain a terminal marker", false, OutcomeFailed, 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			invocation, err := provider.Build(provider.Request{Backend: test.backend, Model: test.model, Prompt: "oracle", APIKey: "fake"})
			if err != nil {
				t.Fatal(err)
			}
			invocation.Argv[0] = fake
			invocation.SetEnv["HATCH_TEST_SCENARIO"] = test.scenario
			result := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs"))).Execute(Request{
				Surface: test.backend, Provider: test.backend, Model: test.model, Prompt: "oracle",
				Timeout: 5 * time.Second, Invocation: invocation,
			})
			if result.OK != test.ok || result.Output != test.output || stringValue(result.Error) != chooseError(test.errorText) ||
				result.Run == nil || result.Run.Outcome == nil || *result.Run.Outcome != test.outcome || len(result.Run.Warnings) != test.warnings {
				t.Fatalf("result = %#v", result)
			}
			for _, warning := range result.Run.Warnings {
				if warning.Code != "capture_persistence_failed" && warning.EvidenceFile == nil {
					t.Fatalf("warning lacks evidence provenance: %#v", warning)
				}
			}
		})
	}
}

func chooseError(value string) string {
	if value == "" {
		return "<nil>"
	}
	return value
}

func stringValue(value *string) string {
	if value == nil {
		return "<nil>"
	}
	return *value
}

func TestCoordinatorTimeoutKillsProcessGroup(t *testing.T) {
	fake := buildTestProvider(t)
	coordinator := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs")))
	started := time.Now()
	result := coordinator.Execute(Request{
		Surface: "codex.raw", Provider: "openai", Prompt: "prompt", Timeout: time.Second,
		Invocation: provider.Invocation{Argv: []string{fake}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "hang"}},
	})
	if result.OK || result.ExitCode != -1 || result.Status != "timeout" {
		t.Fatalf("unexpected timeout: %#v", result)
	}
	if time.Since(started) > 5*time.Second {
		t.Fatal("timeout did not terminate promptly")
	}
	if result.Output != "partial output\n" {
		t.Fatalf("partial output lost: %q", result.Output)
	}
	if result.Run == nil || result.Run.Process == nil || result.Run.Process.ProcessGroup == nil || *result.Run.Process.ProcessGroup != result.Run.Process.PID || result.Run.Process.StartIdentity == nil || *result.Run.Process.StartIdentity == "" {
		t.Fatalf("process identity missing: %#v", result.Run)
	}
}

func TestCoordinatorCancellationKillsProcessGroup(t *testing.T) {
	fake := buildTestProvider(t)
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(100*time.Millisecond, cancel)
	result := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs"))).Execute(Request{
		Context: ctx, Surface: "codex.raw", Backend: "codex", Provider: "openai", Prompt: "prompt", Timeout: 5 * time.Second,
		Invocation: provider.Invocation{Argv: []string{fake}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "hang"}},
	})
	if result.OK || result.Status != "cancelled" || result.ExitCode != -4 || result.CLIExitCode() != 130 || result.Run == nil || result.Run.Outcome == nil ||
		*result.Run.Outcome != OutcomeCancelled || result.Run.Process == nil || result.Run.Process.CancelCleanup == nil ||
		!result.Run.Process.CancelCleanup.WaitBounded || result.Run.Process.CancelCleanup.Signal != "SIGKILL" {
		t.Fatalf("cancelled result = %#v", result)
	}
}

func TestCompletedProviderResultWinsCancellationRace(t *testing.T) {
	complete := provider.Interpretation{Output: []byte("answer"), TerminalMarker: "observed"}
	if cancellationWins(true, 0, complete) {
		t.Fatal("completed provider result was relabelled as cancelled")
	}
	if !cancellationWins(true, 0, provider.Interpretation{Output: []byte("partial"), TerminalMarker: "not_observed"}) {
		t.Fatal("incomplete provider result incorrectly won cancellation race")
	}
}

func TestCoordinatorCancelledBeforeProviderLaunch(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs"))).Execute(Request{
		Context: ctx, Surface: "codex.raw", Backend: "codex", Provider: "openai", Prompt: "prompt",
		Invocation: provider.Invocation{Argv: []string{filepath.Join(t.TempDir(), "provider-must-not-start")}},
	})
	if result.OK || result.Status != "cancelled" || result.CLIExitCode() != 130 || result.Run == nil || result.Run.Outcome == nil ||
		*result.Run.Outcome != OutcomeCancelled || result.Run.Process != nil {
		t.Fatalf("prelaunch cancellation = %#v", result)
	}
}

func TestProviderStateIsolationUsesPerRunNamespaces(t *testing.T) {
	store := NewStore(filepath.Join(t.TempDir(), "runs"))
	pi, err := provider.Build(provider.Request{Backend: "pi", Model: "openrouter/deepseek/deepseek-v4.1-flash-1017", Prompt: "prompt", APIKey: "fake"})
	if err != nil {
		t.Fatal(err)
	}
	pi.PiDCGRunner = "/home/u/.config/dcg/hook-runner"
	artifact, err := store.Prepare(PreparedRun{Surface: "openrouter.deepseek", Backend: "pi", Provider: "openrouter", Model: "openrouter/deepseek/deepseek-v4.1-flash-1017", Request: "prompt", ReasoningPolicy: pi.ReasoningPolicy})
	if err != nil {
		t.Fatal(err)
	}
	cleanup, err := prepareProviderState(artifact, &pi)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	configDir := pi.SetEnv["PI_CODING_AGENT_DIR"]
	if !strings.HasPrefix(configDir, artifact.Path) || !strings.HasPrefix(pi.SetEnv["PI_CODING_AGENT_SESSION_DIR"], artifact.Path) {
		t.Fatalf("Pi isolation = %#v", pi.SetEnv)
	}
	models, err := os.ReadFile(filepath.Join(configDir, "models.json"))
	if err != nil || !bytes.Contains(models, []byte(`"order":["DeepSeek"]`)) || !bytes.Contains(models, []byte(`"allow_fallbacks":false`)) {
		t.Fatalf("per-run models.json = %s (%v)", models, err)
	}
	if info, err := os.Lstat(filepath.Join(configDir, "models.json")); err != nil || info.Mode()&os.ModeSymlink != 0 {
		t.Fatalf("models.json must be a real per-run file, not a link to the user's: %v", err)
	}
	guard := filepath.Join(configDir, "dcg-guard.mjs")
	last := len(pi.Argv) - 1
	if pi.SetEnv["HATCH_DCG_RUNNER"] != "/home/u/.config/dcg/hook-runner" || pi.Argv[last-2] != "-e" || pi.Argv[last-1] != guard || pi.Argv[last] != provider.PreparePrompt("prompt") {
		t.Fatalf("DCG guard not loaded before the prompt: env=%#v argv=%#v", pi.SetEnv, pi.Argv)
	}
	if pi.SetEnv["DCG_HARNESS"] != "hatch-pi" {
		t.Fatalf("DCG harness label = %q", pi.SetEnv["DCG_HARNESS"])
	}
	if len(pi.RedactedArgv) != len(pi.Argv) || pi.RedactedArgv[len(pi.RedactedArgv)-1] != "<prompt>" {
		t.Fatalf("redaction drift: %#v", pi.RedactedArgv)
	}

	codex, err := provider.Build(provider.Request{Backend: "codex", Model: "gpt-6.1-sol", Prompt: "prompt", APIKey: "fake"})
	if err != nil {
		t.Fatal(err)
	}
	codexArtifact, err := store.Prepare(PreparedRun{Surface: "codex.raw", Backend: "codex", Provider: "openai", Model: "gpt-6.1-sol", Request: "prompt", ReasoningPolicy: codex.ReasoningPolicy})
	if err != nil {
		t.Fatal(err)
	}
	codexCleanup, err := prepareProviderState(codexArtifact, &codex)
	if err != nil {
		t.Fatal(err)
	}
	defer codexCleanup()
	if !strings.HasPrefix(codex.SetEnv["CODEX_HOME"], codexArtifact.Path) || codex.SetEnv["CODEX_HOME"] == configDir {
		t.Fatalf("Codex isolation = %#v", codex.SetEnv)
	}

	cursor, err := provider.Build(provider.Request{Backend: "cursor", Model: "grok-4.7-high", Prompt: "prompt"})
	if err != nil {
		t.Fatal(err)
	}
	cursorArtifact, err := store.Prepare(PreparedRun{Surface: "cursor.grok", Backend: "cursor", Provider: "cursor", Model: "grok-4.7-high", Request: "prompt", ReasoningPolicy: cursor.ReasoningPolicy})
	if err != nil {
		t.Fatal(err)
	}
	cursorCleanup, err := prepareProviderState(cursorArtifact, &cursor)
	if err != nil {
		t.Fatal(err)
	}
	defer cursorCleanup()
	if !strings.HasPrefix(cursor.SetEnv["HOME"], cursorArtifact.Path) || cursor.SetEnv["HOME"] == codex.SetEnv["CODEX_HOME"] {
		t.Fatalf("Cursor isolation = %#v", cursor.SetEnv)
	}
	if _, err := os.Stat(filepath.Join(cursor.SetEnv["HOME"], ".cursor")); err != nil {
		t.Fatalf("Cursor home missing .cursor: %v", err)
	}
}

func TestRawCodexRunRecordsEphemeralIsolation(t *testing.T) {
	fake := buildTestProvider(t)
	invocation, err := provider.Build(provider.Request{Backend: "codex", Model: "gpt-6.1-sol", Prompt: "prompt", APIKey: "fake"})
	if err != nil {
		t.Fatal(err)
	}
	invocation.Argv[0] = fake
	invocation.SetEnv["HATCH_TEST_SCENARIO"] = "success_text"
	result := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs"))).Execute(Request{
		Surface: "codex.raw", Backend: "codex", Provider: "openai", Model: "gpt-6.1-sol", Prompt: "prompt", Invocation: invocation,
	})
	if !result.OK || result.Run == nil || result.ArtifactPath == nil || result.Run.ProviderState.Capabilities["state_isolation"] != "hatch_per_run" ||
		result.Run.ProviderState.Capabilities["session_persistence"] != "disabled_ephemeral" || result.Run.ProviderState.Capabilities["recovery"] != "unsupported_ephemeral" {
		t.Fatalf("raw Codex result = %#v", result)
	}
	if _, err := os.Stat(filepath.Join(*result.ArtifactPath, "provider")); !os.IsNotExist(err) {
		t.Fatalf("Codex provider runtime retained: %v", err)
	}
}

func TestCoordinatorTimeoutKillsDescendants(t *testing.T) {
	fake := buildTestProvider(t)
	sentinel := filepath.Join(t.TempDir(), "child-survived")
	result := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs"))).Execute(Request{
		Surface: "codex.raw", Provider: "openai", Prompt: "prompt", Timeout: 300 * time.Millisecond,
		Invocation: provider.Invocation{Argv: []string{fake}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "hang_with_child", "HATCH_CHILD_SENTINEL": sentinel}},
	})
	if result.Status != "timeout" {
		t.Fatalf("result = %#v", result)
	}
	time.Sleep(2200 * time.Millisecond)
	if _, err := os.Stat(sentinel); !os.IsNotExist(err) {
		t.Fatalf("descendant survived process-group timeout: %v", err)
	}
}

func TestCoordinatorTimeoutKillsDetachedDescendants(t *testing.T) {
	fake := buildTestProvider(t)
	sentinel := filepath.Join(t.TempDir(), "child-survived")
	var progress []string
	result := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs"))).Execute(Request{
		Surface: "codex.raw", Provider: "openai", Prompt: "prompt", Timeout: 300 * time.Millisecond,
		Progress:   func(line string) { progress = append(progress, line) },
		Invocation: provider.Invocation{Argv: []string{fake}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "hang_with_detached_sentinel_child", "HATCH_CHILD_SENTINEL": sentinel}},
	})
	if result.Status != "timeout" || result.Output != "partial output\n" {
		t.Fatalf("result = %#v", result)
	}
	time.Sleep(2500 * time.Millisecond)
	if _, err := os.Stat(sentinel); !os.IsNotExist(err) {
		t.Fatalf("detached descendant (own session) survived the timeout: %v", err)
	}
	if !slices.ContainsFunc(progress, func(line string) bool { return strings.Contains(line, "[hatch] killed ") }) {
		t.Fatalf("killed survivors not reported: %q", progress)
	}
}

func TestCoordinatorKillsDescendantsLeftAfterProviderExits(t *testing.T) {
	fake := buildTestProvider(t)
	sentinel := filepath.Join(t.TempDir(), "child-survived")
	var progress []string
	result := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs"))).Execute(Request{
		Surface: "codex.raw", Provider: "openai", Prompt: "prompt", Timeout: 10 * time.Second,
		Progress:   func(line string) { progress = append(progress, line) },
		Invocation: provider.Invocation{Argv: []string{fake}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "exit_with_detached_sentinel_child", "HATCH_CHILD_SENTINEL": sentinel}},
	})
	if !result.OK || result.Output != "answer\n" {
		t.Fatalf("result = %#v", result)
	}
	time.Sleep(2500 * time.Millisecond)
	if _, err := os.Stat(sentinel); !os.IsNotExist(err) {
		t.Fatalf("child left running after the provider exited survived: %v", err)
	}
	if !slices.ContainsFunc(progress, func(line string) bool { return strings.Contains(line, "[hatch] killed 1 process") }) {
		t.Fatalf("killed survivor not reported: %q", progress)
	}
}

func TestCoordinatorTimeoutBoundsDetachedDescendantPipes(t *testing.T) {
	fake := buildTestProvider(t)
	started := time.Now()
	result := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs"))).Execute(Request{
		Surface: "codex.raw", Provider: "openai", Prompt: "prompt", Timeout: 250 * time.Millisecond,
		Invocation: provider.Invocation{Argv: []string{fake}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "hang_with_detached_child"}},
	})
	if result.Status != "timeout" || time.Since(started) > 3*time.Second {
		t.Fatalf("detached timeout result=%#v elapsed=%s", result, time.Since(started))
	}
	if result.Run == nil || result.Run.Process == nil || result.Run.Process.TimeoutCleanup == nil ||
		!result.Run.Process.TimeoutCleanup.WaitBounded ||
		result.Run.Process.TimeoutCleanup.SurvivorState != "unknown" {
		t.Fatalf("timeout cleanup evidence = %#v", result.Run)
	}
}

func TestCoordinatorPreservesInvalidUTF8Evidence(t *testing.T) {
	fake := buildTestProvider(t)
	result := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs"))).Execute(Request{
		Surface: "codex.raw", Provider: "openai", Prompt: "prompt", Timeout: 5 * time.Second,
		Invocation: provider.Invocation{Argv: []string{fake}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "invalid_utf8"}},
	})
	if !result.OK || result.ArtifactPath == nil {
		t.Fatalf("result = %#v", result)
	}
	raw, err := os.ReadFile(filepath.Join(*result.ArtifactPath, "stdout.log"))
	want := []byte{'o', 'k', ':', 0xff, 0xfe, '\n'}
	if err != nil || !bytes.Equal(raw, want) {
		t.Fatalf("raw = %v, %v", raw, err)
	}
}

func TestCoordinatorBoundsPublicMemoryWhilePreservingLargeRawEvidence(t *testing.T) {
	fake := buildTestProvider(t)
	result := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs"))).Execute(Request{
		Surface: "codex.raw", Provider: "openai", Prompt: "prompt", Timeout: 10 * time.Second,
		Invocation: provider.Invocation{Argv: []string{fake}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "large_output"}},
	})
	if result.OK || result.Error == nil || !strings.Contains(*result.Error, "public interpretation limit") || result.ArtifactPath == nil {
		t.Fatalf("result=%#v", result)
	}
	info, err := os.Stat(filepath.Join(*result.ArtifactPath, "stdout.log"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() != 34*1024*1024 {
		t.Fatalf("raw evidence size=%d", info.Size())
	}
}

func TestCoordinatorRedactsPromptAndCredentialValues(t *testing.T) {
	fake := buildTestProvider(t)
	secret := "sk-secret-never-persist"
	prompt := "prompt with $(shell) and 'quotes'"
	invocation, err := provider.Build(provider.Request{Backend: "pi", Model: "openrouter/deepseek/deepseek-v4.1-flash-1017", Prompt: prompt, APIKey: secret})
	if err != nil {
		t.Fatal(err)
	}
	invocation.Argv[0] = fake
	invocation.SetEnv["HATCH_TEST_SCENARIO"] = "success_pi"
	result := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs"))).Execute(Request{
		Surface: "openrouter.deepseek", Provider: "openrouter", Model: "openrouter/deepseek/deepseek-v4.1-flash-1017", Prompt: prompt,
		Timeout: 5 * time.Second, Invocation: invocation, CredentialNames: []string{"OPENROUTER_API_KEY"},
	})
	if !result.OK || result.ArtifactPath == nil {
		t.Fatalf("result = %#v", result)
	}
	manifest, err := os.ReadFile(filepath.Join(*result.ArtifactPath, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(manifest, []byte(secret)) || bytes.Contains(manifest, []byte(prompt)) || !bytes.Contains(manifest, []byte("OPENROUTER_API_KEY")) {
		t.Fatalf("unsafe manifest: %s", manifest)
	}
}

func TestCoordinatorFailsClosedOnRedactionMetadataDrift(t *testing.T) {
	fake := buildTestProvider(t)
	record := filepath.Join(t.TempDir(), "provider-ran")
	result := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs"))).Execute(Request{
		Surface: "openrouter.deepseek", Provider: "openrouter", Prompt: "sensitive prompt", Timeout: time.Second,
		Invocation: provider.Invocation{
			Argv: []string{fake, "sensitive prompt"}, RedactedArgv: []string{fake},
			SetEnv: map[string]string{"HATCH_TEST_RECORD": record},
		},
	})
	if result.OK || result.ExitCode != -3 || result.ArtifactPath != nil || result.Error == nil || !strings.Contains(*result.Error, "redaction metadata") {
		t.Fatalf("result=%#v", result)
	}
	if _, err := os.Stat(record); !os.IsNotExist(err) {
		t.Fatalf("provider ran despite unsafe redaction metadata: %v", err)
	}
}

func TestCoordinatorThirtyTwoConcurrentRunsAreIsolated(t *testing.T) {
	fake := buildTestProvider(t)
	store := NewStore(filepath.Join(t.TempDir(), "runs"))
	const count = 32
	results := make(chan PublicResult, count)
	var group sync.WaitGroup
	for index := 0; index < count; index++ {
		group.Add(1)
		go func() {
			defer group.Done()
			results <- NewCoordinator(store).Execute(Request{
				Surface: "codex.raw", Provider: "openai", Prompt: "concurrent", Timeout: 5 * time.Second,
				Invocation: provider.Invocation{Argv: []string{fake}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "success_text"}},
			})
		}()
	}
	group.Wait()
	close(results)
	ids, paths := map[string]bool{}, map[string]bool{}
	for result := range results {
		if !result.OK || result.Run == nil || result.ArtifactPath == nil {
			t.Fatalf("concurrent result = %#v", result)
		}
		if ids[result.Run.RunID] || paths[*result.ArtifactPath] {
			t.Fatalf("duplicate identity: %s %s", result.Run.RunID, *result.ArtifactPath)
		}
		ids[result.Run.RunID], paths[*result.ArtifactPath] = true, true
	}
	if len(ids) != count || len(paths) != count {
		t.Fatalf("ids=%d paths=%d", len(ids), len(paths))
	}
}

func buildTestProvider(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "testprovider")
	root := filepath.Join("..", "..")
	command := exec.Command("go", "build", "-o", path, "./internal/testprovider")
	command.Dir = root
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("build test provider: %v\n%s", err, output)
	}
	return path
}

func TestCursorRunHomeLinksDCGHooks(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := os.MkdirAll(filepath.Join(home, ".cursor"), 0o700); err != nil {
		t.Fatal(err)
	}
	hooks := filepath.Join(home, ".cursor", "hooks.json")
	if err := os.WriteFile(hooks, []byte(`{"version":1}`), 0o600); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	linkUserCursorConfig(root)
	target, err := os.Readlink(filepath.Join(root, ".cursor", "hooks.json"))
	if err != nil || target != hooks {
		t.Fatalf("Cursor per-run home lacks the user's hooks.json (guard would not run): %q %v", target, err)
	}
}
