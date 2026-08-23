package run

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCollectGarbageIsDryRunByDefaultAndPreservesEvidence(t *testing.T) {
	root := filepath.Join(t.TempDir(), "runs")
	artifact := createGarbageRun(t, root, true)
	config := filepath.Join(artifact.Path, "provider", "opencode-config")
	report, err := CollectGarbage(root, false)
	if err != nil {
		t.Fatal(err)
	}
	if report.TotalPaths != 3 || report.TotalFiles != 3 || report.TotalLogicalBytes != 13 || report.RemovedLogicalBytes != 0 {
		t.Fatalf("dry-run report = %#v", report)
	}
	if _, err := os.Stat(config); err != nil {
		t.Fatalf("dry-run removed config: %v", err)
	}

	report, err = CollectGarbage(root, true)
	if err != nil || report.RemovedLogicalBytes != 13 || len(report.Errors) != 0 {
		t.Fatalf("apply report = %#v err=%v", report, err)
	}
	for _, path := range []string{config, filepath.Join(artifact.Path, "provider", "opencode-cache"), filepath.Join(artifact.Path, "provider", "omp")} {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("garbage remains at %s: %v", path, err)
		}
	}
	for _, name := range []string{"manifest.json", "request.txt", "result.txt", "stdout.log", "stderr.log", "evidence.sha256"} {
		if _, err := os.Stat(filepath.Join(artifact.Path, name)); err != nil {
			t.Fatalf("evidence %s was removed: %v", name, err)
		}
	}
}

func TestCollectGarbageSkipsNonterminalAndPinnedRuns(t *testing.T) {
	root := filepath.Join(t.TempDir(), "runs")
	nonterminal := createGarbageRun(t, root, false)
	pinned := createGarbageRun(t, root, true)
	if err := os.WriteFile(filepath.Join(pinned.Path, ".hatch-pin"), []byte("keep\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	report, err := CollectGarbage(root, true)
	if err != nil {
		t.Fatal(err)
	}
	if report.RunsSkippedNonterminal != 1 || report.RunsSkippedPinned != 1 || report.TotalPaths != 0 {
		t.Fatalf("report = %#v", report)
	}
	for _, artifact := range []*Artifact{nonterminal, pinned} {
		if _, err := os.Stat(filepath.Join(artifact.Path, "provider", "opencode-config")); err != nil {
			t.Fatalf("skipped runtime missing: %v", err)
		}
	}
}

func createGarbageRun(t *testing.T, root string, terminal bool) *Artifact {
	t.Helper()
	store := NewStore(root)
	artifact, err := store.Prepare(PreparedRun{Surface: "codex.sol", Backend: "opencode", Provider: "openai", Model: "openai/gpt-5.6-sol", Request: "prompt"})
	if err != nil {
		t.Fatal(err)
	}
	stdout, stderr, err := store.OpenStreams(artifact)
	if err != nil {
		t.Fatal(err)
	}
	_ = stdout.Close()
	_ = stderr.Close()
	if terminal {
		result, err := store.WriteResult(artifact, []byte("answer"))
		if err != nil {
			t.Fatal(err)
		}
		if err := store.CommitTerminal(artifact, OutcomeSucceeded, 0, Result{Output: "present", TerminalMarker: "observed", OutputBytes: 6, OutputFile: &result}, State{Retention: "unavailable", NativeIDState: "observed", NativeID: stringPointer("session"), Capabilities: map[string]string{}}, nil); err != nil {
			t.Fatal(err)
		}
	}
	for path, contents := range map[string]string{
		filepath.Join(artifact.Path, "provider", "opencode-config", "opencode", "node_modules", "dependency.js"): "1234567",
		filepath.Join(artifact.Path, "provider", "opencode-cache", "opencode", "models.json"):                    "12345",
		filepath.Join(artifact.Path, "provider", "omp", "state.json"):                                            "1",
	} {
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return artifact
}

func stringPointer(value string) *string { return &value }
