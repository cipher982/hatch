package run

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestStorePreparesPrivateDurableRun(t *testing.T) {
	root := filepath.Join(t.TempDir(), "runs")
	fixed := time.Date(2026, 7, 22, 15, 0, 0, 0, time.UTC)
	store := NewStore(root)
	store.Now = func() time.Time { return fixed }
	store.IDGen = func(time.Time) (string, error) { return "hatch_test_run", nil }

	artifact, err := store.Prepare(PreparedRun{
		Surface: "gemini.raw", Provider: "google", Model: "gemini-3-pro-preview", CWD: "/repo",
		Request: "secret prompt", RedactedArgv: []string{"gemini", "<prompt>"}, CredentialNames: []string{"GEMINI_API_KEY"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if artifact.Manifest.Lifecycle != LifecyclePrepared || artifact.Manifest.RunID != "hatch_test_run" {
		t.Fatalf("unexpected manifest: %#v", artifact.Manifest)
	}
	assertMode(t, root, 0o700)
	assertMode(t, artifact.Path, 0o700)
	assertMode(t, filepath.Join(artifact.Path, "request.txt"), 0o600)
	assertMode(t, filepath.Join(artifact.Path, "manifest.json"), 0o600)
	if got, err := os.ReadFile(filepath.Join(artifact.Path, "request.txt")); err != nil || string(got) != "secret prompt" {
		t.Fatalf("request evidence = %q, %v", got, err)
	}

	var disk Manifest
	data, err := os.ReadFile(filepath.Join(artifact.Path, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &disk); err != nil {
		t.Fatal(err)
	}
	if disk.RunID != artifact.Manifest.RunID || disk.Capture.ArtifactPath != artifact.Path {
		t.Fatalf("disk manifest mismatch: %#v", disk)
	}
}

func TestStoreLifecycleAndEvidenceDigest(t *testing.T) {
	store := NewStore(filepath.Join(t.TempDir(), "runs"))
	store.IDGen = func(time.Time) (string, error) { return "hatch_lifecycle", nil }
	artifact, err := store.Prepare(PreparedRun{Surface: "gemini.raw", Provider: "google", Model: "model", CWD: "/repo", Request: "prompt", RedactedArgv: []string{"gemini"}})
	if err != nil {
		t.Fatal(err)
	}
	stdout, stderr, err := store.OpenStreams(artifact)
	if err != nil {
		t.Fatal(err)
	}
	stdout.WriteString("answer\n")
	stderr.WriteString("warning\n")
	stdout.Close()
	stderr.Close()
	started := time.Now()
	if err := store.MarkRunning(artifact, 123, started); err != nil {
		t.Fatal(err)
	}
	resultFile, err := store.WriteResult(artifact, []byte("answer\n"))
	if err != nil {
		t.Fatal(err)
	}
	state := State{Retention: "provider_owned", NativeIDState: "not_exposed", Capabilities: map[string]string{}}
	if err := store.CommitTerminal(artifact, OutcomeSucceeded, 0, Result{
		Output: "present", TerminalMarker: "not_applicable", OutputBytes: 7, OutputFile: &resultFile,
	}, state, nil); err != nil {
		t.Fatal(err)
	}
	if artifact.Manifest.Lifecycle != LifecycleTerminal || artifact.Manifest.Capture.EvidenceSHA256 == nil {
		t.Fatalf("terminal manifest incomplete: %#v", artifact.Manifest)
	}
}

func TestStoreFailsBeforeLaunchWhenRootIsAFile(t *testing.T) {
	root := filepath.Join(t.TempDir(), "not-a-directory")
	if err := os.WriteFile(root, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	store := NewStore(root)
	if _, err := store.Prepare(PreparedRun{Surface: "raw", Provider: "unknown", Request: "prompt"}); err == nil {
		t.Fatal("Prepare succeeded with an unusable root")
	}
}

func assertMode(t *testing.T, path string, want os.FileMode) {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != want {
		t.Fatalf("%s mode = %#o, want %#o", path, got, want)
	}
}
