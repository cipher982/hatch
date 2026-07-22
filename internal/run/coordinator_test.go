package run

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
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
			}
		})
	}
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
