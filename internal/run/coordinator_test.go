package run

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
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
		Surface: "gemini.raw", Provider: "google", Model: "gemini-3-pro-preview",
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
		Surface: "gemini.raw", Provider: "google", Prompt: "prompt", Timeout: time.Second,
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
		Surface: "gemini.raw", Provider: "google", Prompt: "prompt", Timeout: time.Second,
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
		Surface: "gemini.raw", Provider: "google", Prompt: "prompt", Timeout: time.Second,
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
		Surface: "gemini.raw", Provider: "google", Prompt: "prompt", Timeout: time.Second,
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
		Surface: "gemini.raw", Provider: "google", Prompt: "prompt", Timeout: time.Second,
		Invocation: provider.Invocation{Argv: []string{fake}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "success_text"}},
	})
	if !result.OK || result.Output != "fake provider output\n" || result.Run == nil || result.Run.Capture.State != "degraded" || result.ArtifactPath != nil {
		t.Fatalf("result=%#v", result)
	}
}

func TestCoordinatorReturnsAnswerWhenResultPersistenceFails(t *testing.T) {
	fake := buildTestProvider(t)
	store := writeResultFailureStore{Store: NewStore(filepath.Join(t.TempDir(), "runs"))}
	result := NewCoordinator(store).Execute(Request{
		Surface: "gemini.raw", Provider: "google", Prompt: "prompt", Timeout: time.Second,
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
		Surface: "gemini.raw", Provider: "google", Prompt: "prompt", Timeout: 5 * time.Second,
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
		Surface: "gemini.raw", Provider: "google", Prompt: "prompt", Timeout: 5 * time.Second,
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
		{"cursor", "cursor", "cursor-grok-4.5-high", "success_cursor", "fake cursor output", "cursor-session-oracle"},
		{"opencode", "opencode", "openrouter/moonshotai/kimi-k3", "success_opencode", "fake opencode output", "ses_oracle1234"},
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
			coordinator := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs")))
			result := coordinator.Execute(Request{
				Surface: test.name, Provider: test.name, Model: test.model, Prompt: "oracle prompt",
				Timeout: 5 * time.Second, Invocation: invocation,
				ProgressLabel: test.name, Progress: func(message string) { progress = append(progress, message) },
			})
			if !result.OK || result.Output != test.output || result.SessionID == nil || *result.SessionID != test.session {
				t.Fatalf("unexpected result: ok=%v output=%q session=%v error=%s stderr=%s", result.OK, result.Output, result.SessionID, stringValue(result.Error), stringValue(result.Stderr))
			}
			if result.Run.Result.TerminalMarker != "observed" || result.Run.Capture.StdoutFile != "stdout.jsonl" {
				t.Fatalf("structured evidence mismatch: %#v", result.Run)
			}
			if test.backend == "opencode" {
				stateFile := filepath.Join(*result.ArtifactPath, "provider", "opencode", "data", "opencode", "session.db")
				if data, err := os.ReadFile(stateFile); err != nil || string(data) != "fake opencode state" {
					t.Fatalf("provider state not preserved: %q, %v", data, err)
				}
				if len(progress) < 3 || progress[0][:12] != "[hatch] run " {
					t.Fatalf("progress = %#v", progress)
				}
				if _, err := os.Stat(filepath.Join(*result.ArtifactPath, "provider", "opencode", "data", "opencode", "auth.json")); !os.IsNotExist(err) {
					t.Fatalf("untrusted provider auth state retained: %v", err)
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
		{"cursor error", "cursor", "cursor-grok-4.5-high", "cursor_error", "", "request rejected", false, OutcomeFailed, 0},
		{"opencode error", "opencode", "openrouter/moonshotai/kimi-k3", "opencode_error", "", "provider unavailable", false, OutcomeFailed, 0},
		{"opencode recovered", "opencode", "openrouter/moonshotai/kimi-k3", "opencode_transient_then_success", "recovered answer", "", true, OutcomeSucceededWarnings, 1},
		{"opencode missing terminal", "opencode", "openrouter/moonshotai/kimi-k3", "opencode_missing_terminal", "useful evidence", "structured provider output did not contain a terminal marker", false, OutcomeFailed, 1},
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
		Surface: "gemini.raw", Provider: "google", Prompt: "prompt", Timeout: 500 * time.Millisecond,
		Invocation: provider.Invocation{Argv: []string{fake}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "hang"}},
	})
	if result.OK || result.ExitCode != -1 || result.Status != "timeout" {
		t.Fatalf("unexpected timeout: %#v", result)
	}
	if time.Since(started) > 3*time.Second {
		t.Fatal("timeout did not terminate promptly")
	}
	if result.Output != "partial output\n" {
		t.Fatalf("partial output lost: %q", result.Output)
	}
	if result.Run == nil || result.Run.Process == nil || result.Run.Process.ProcessGroup == nil || *result.Run.Process.ProcessGroup != result.Run.Process.PID || result.Run.Process.StartIdentity == nil || *result.Run.Process.StartIdentity == "" {
		t.Fatalf("process identity missing: %#v", result.Run)
	}
}

func TestOpenCodeTimeoutProducesVersionBoundRecoveryHint(t *testing.T) {
	fake := buildTestProvider(t)
	invocation, err := provider.Build(provider.Request{Backend: "opencode", Model: "openrouter/moonshotai/kimi-k3", Prompt: "prompt", APIKey: "fake"})
	if err != nil {
		t.Fatal(err)
	}
	invocation.Argv[0] = fake
	invocation.ProviderVersion = "opencode 1.2.3"
	invocation.SetEnv["HATCH_TEST_SCENARIO"] = "hang_opencode"
	result := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs"))).Execute(Request{
		Surface: "openrouter.kimi-k3", Provider: "openrouter", Model: "openrouter/moonshotai/kimi-k3", CWD: t.TempDir(), Prompt: "prompt",
		Timeout: 300 * time.Millisecond, Invocation: invocation,
	})
	if result.Status != "timeout" || result.Run == nil || result.Run.ProviderState.RecoveryHint == nil ||
		result.Run.ProviderState.InspectHint == nil || result.Run.ProviderState.ProviderVersion == nil ||
		*result.Run.ProviderState.ProviderVersion != "opencode 1.2.3" || result.ResumeCommand == nil ||
		!result.Run.ProviderState.RecoveryHint.VersionBound || !result.Run.ProviderState.RecoveryHint.RequiresApprovalBypass {
		t.Fatalf("result = %#v", result)
	}
	for _, arg := range result.Run.ProviderState.RecoveryHint.Argv {
		if strings.Contains(arg, "fake") || strings.Contains(arg, "KEY") {
			t.Fatalf("credential-like recovery arg: %q", arg)
		}
	}
}

func TestCoordinatorTimeoutKillsDescendants(t *testing.T) {
	fake := buildTestProvider(t)
	sentinel := filepath.Join(t.TempDir(), "child-survived")
	result := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs"))).Execute(Request{
		Surface: "gemini.raw", Provider: "google", Prompt: "prompt", Timeout: 300 * time.Millisecond,
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

func TestCoordinatorTimeoutBoundsDetachedDescendantPipes(t *testing.T) {
	fake := buildTestProvider(t)
	started := time.Now()
	result := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs"))).Execute(Request{
		Surface: "gemini.raw", Provider: "google", Prompt: "prompt", Timeout: 250 * time.Millisecond,
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
		Surface: "gemini.raw", Provider: "google", Prompt: "prompt", Timeout: 5 * time.Second,
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
		Surface: "gemini.raw", Provider: "google", Prompt: "prompt", Timeout: 10 * time.Second,
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
	invocation, err := provider.Build(provider.Request{Backend: "opencode", Model: "openrouter/moonshotai/kimi-k3", Prompt: prompt, APIKey: secret})
	if err != nil {
		t.Fatal(err)
	}
	invocation.Argv[0] = fake
	invocation.SetEnv["HATCH_TEST_SCENARIO"] = "success_opencode"
	result := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs"))).Execute(Request{
		Surface: "openrouter.kimi-k3", Provider: "openrouter", Model: "openrouter/moonshotai/kimi-k3", Prompt: prompt,
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
		Surface: "openrouter.kimi-k3", Provider: "openrouter", Prompt: "sensitive prompt", Timeout: time.Second,
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
				Surface: "gemini.raw", Provider: "google", Prompt: "concurrent", Timeout: 5 * time.Second,
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

func TestPruneOpenCodeStateUsesExplicitAllowlist(t *testing.T) {
	artifact := &Artifact{Path: t.TempDir()}
	root := filepath.Join(artifact.Path, "provider", "opencode", "data", "opencode")
	if err := os.MkdirAll(root, 0o700); err != nil {
		t.Fatal(err)
	}
	for name, value := range map[string]string{"opencode.db": "db", "opencode.db-wal": "wal", "auth.json": "secret", "log.txt": "prompt"} {
		if err := os.WriteFile(filepath.Join(root, name), []byte(value), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Symlink(filepath.Join(root, "opencode.db"), filepath.Join(root, "opencode.db-shm")); err != nil {
		t.Fatal(err)
	}
	approved, err := pruneOpenCodeState(artifact)
	if err != nil || approved != 2 {
		t.Fatalf("approved=%d err=%v", approved, err)
	}
	for _, name := range []string{"auth.json", "log.txt", "opencode.db-shm"} {
		if _, err := os.Lstat(filepath.Join(root, name)); !os.IsNotExist(err) {
			t.Fatalf("%s retained: %v", name, err)
		}
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
