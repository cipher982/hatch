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
		Prompt: "oracle prompt", Timeout: 5 * time.Second, Invocation: invocation,
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
