package run

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestCollectGarbageAgedSnapshotRetainsAuditableEvidence(t *testing.T) {
	root := filepath.Join(t.TempDir(), "runs")
	store := NewStore(root)
	store.Now = func() time.Time { return time.Now().UTC().Add(-20 * 24 * time.Hour) }
	a, err := store.Prepare(PreparedRun{Surface: "codex.astra", Backend: "opencode", Provider: "openai", Model: "model", Request: "prompt"})
	if err != nil {
		t.Fatal(err)
	}
	stdout, stderr, err := store.OpenStreams(a)
	if err != nil {
		t.Fatal(err)
	}
	_ = stdout.Close()
	_ = stderr.Close()
	if err := store.MarkRunning(a, 123, store.Now(), "identity"); err != nil {
		t.Fatal(err)
	}
	snapshot := filepath.Join(a.Path, "provider", "opencode-snapshot", "data")
	if err := os.MkdirAll(snapshot, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(snapshot, "session.db"), []byte("native state"), 0o600); err != nil {
		t.Fatal(err)
	}
	result, err := store.WriteResult(a, []byte("answer"))
	if err != nil {
		t.Fatal(err)
	}
	state := State{Retention: "hatch_preserved", NativeIDState: "observed", NativeID: stringPointer("session"), Capabilities: map[string]string{}, SnapshotPath: stringPointer("provider/opencode-snapshot")}
	if err := store.CommitTerminal(a, OutcomeSucceeded, 0, Result{Output: "present", TerminalMarker: "observed", OutputBytes: 6, OutputFile: &result}, state, nil); err != nil {
		t.Fatal(err)
	}
	if err := store.WritePublicProjection(a, PublicResult{OK: true, Run: &a.Manifest}); err != nil {
		t.Fatal(err)
	}
	before, err := AuditFieldEvidence(root, 0, 0)
	if err != nil || !before.Passed() {
		t.Fatalf("before audit: %+v %v", before, err)
	}
	dry, err := CollectGarbage(root, false)
	if err != nil || dry.Classes[GarbageProviderPayload].LogicalBytes != int64(len("native state")) || len(dry.Candidates) != 1 {
		t.Fatalf("dry: %+v %v", dry, err)
	}
	if _, err := os.Stat(filepath.Join(snapshot, "session.db")); err != nil {
		t.Fatal(err)
	}
	applied, err := CollectGarbage(root, true)
	if err != nil || len(applied.Errors) != 0 || applied.RemovedLogicalBytes != int64(len("native state")) {
		t.Fatalf("apply: %+v %v", applied, err)
	}
	if _, err := os.Stat(snapshot); !os.IsNotExist(err) {
		t.Fatalf("snapshot still exists: %v", err)
	}
	after, err := AuditFieldEvidence(root, 0, 0)
	if err != nil || !after.Passed() || after.Eligible != before.Eligible {
		t.Fatalf("after audit: %+v %v", after, err)
	}
	if _, err := InspectRecord(root, "", a.Manifest.RunID); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadContent(root, "", a.Manifest.RunID, ContentOptions{Part: "result", Limit: 8192}); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadContent(root, "", a.Manifest.RunID, ContentOptions{Part: "evidence", Limit: 8192}); err != nil {
		t.Fatal(err)
	}
	again, err := CollectGarbage(root, true)
	if err != nil || again.TotalPaths != 0 {
		t.Fatalf("idempotence: %+v %v", again, err)
	}
}

func TestCollectGarbageDoesNotRemoveRecentOrCorruptSnapshots(t *testing.T) {
	root := filepath.Join(t.TempDir(), "runs")
	store := NewStore(root)
	a, err := store.Prepare(PreparedRun{Surface: "codex.astra", Backend: "opencode", Provider: "openai", Model: "model", Request: "prompt"})
	if err != nil {
		t.Fatal(err)
	}
	stdout, stderr, err := store.OpenStreams(a)
	if err != nil {
		t.Fatal(err)
	}
	_ = stdout.Close()
	_ = stderr.Close()
	result, err := store.WriteResult(a, []byte("answer"))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(a.Path, "provider", "opencode-snapshot", "data")
	if err := os.MkdirAll(path, 0o700); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(path, "session.db")
	if err := os.WriteFile(file, []byte("original"), 0o600); err != nil {
		t.Fatal(err)
	}
	state := State{Retention: "hatch_preserved", NativeIDState: "unavailable", Capabilities: map[string]string{}, SnapshotPath: stringPointer("provider/opencode-snapshot")}
	if err := store.CommitTerminal(a, OutcomeFailed, 1, Result{Output: "present", TerminalMarker: "not_observed", OutputBytes: 6, OutputFile: &result}, state, nil); err != nil {
		t.Fatal(err)
	}
	if err := store.WritePublicProjection(a, PublicResult{Run: &a.Manifest}); err != nil {
		t.Fatal(err)
	}
	recent, err := CollectGarbage(root, true)
	if err != nil || recent.TotalPaths != 0 {
		t.Fatalf("recent: %+v %v", recent, err)
	}
	a.Manifest.CreatedAt = time.Now().UTC().Add(-20 * 24 * time.Hour)
	if err := store.writeManifest(a); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file, []byte("tampered"), 0o600); err != nil {
		t.Fatal(err)
	}
	corrupt, err := CollectGarbage(root, true)
	if err != nil || len(corrupt.Errors) != 1 {
		t.Fatalf("corrupt: %+v %v", corrupt, err)
	}
	if _, err := os.Stat(file); err != nil {
		t.Fatalf("corrupt payload removed: %v", err)
	}
}

func TestCollectGarbageAgedCursorRuntime(t *testing.T) {
	root := filepath.Join(t.TempDir(), "runs")
	store := NewStore(root)
	store.Now = func() time.Time { return time.Now().UTC().Add(-20 * 24 * time.Hour) }
	a, err := store.Prepare(PreparedRun{Surface: "cursor.grok", Backend: "cursor", Provider: "cursor", Model: "grok", Request: "prompt"})
	if err != nil {
		t.Fatal(err)
	}
	stdout, stderr, err := store.OpenStreams(a)
	if err != nil {
		t.Fatal(err)
	}
	_ = stdout.Close()
	_ = stderr.Close()
	if err := store.MarkRunning(a, 123, store.Now(), "identity"); err != nil {
		t.Fatal(err)
	}
	result, err := store.WriteResult(a, []byte("answer"))
	if err != nil {
		t.Fatal(err)
	}
	if err := store.CommitTerminal(a, OutcomeSucceeded, 0, Result{Output: "present", TerminalMarker: "observed", OutputBytes: 6, OutputFile: &result}, State{Retention: "unknown", NativeIDState: "unavailable", Capabilities: map[string]string{}}, nil); err != nil {
		t.Fatal(err)
	}
	if err := store.WritePublicProjection(a, PublicResult{Run: &a.Manifest}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(a.Path, "provider", "cursor", ".local", "share")
	if err := os.MkdirAll(path, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(path, "cache"), []byte("runtime"), 0o600); err != nil {
		t.Fatal(err)
	}
	report, err := CollectGarbage(root, true)
	if err != nil || len(report.Errors) != 0 || report.Classes[GarbageProviderPayload].LogicalBytes != 7 {
		t.Fatalf("cursor report: %+v %v", report, err)
	}
	audit, err := AuditFieldEvidence(root, 0, 0)
	if err != nil || !audit.Passed() {
		t.Fatalf("cursor audit: %+v %v", audit, err)
	}
}

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
	artifact, err := store.Prepare(PreparedRun{Surface: "codex.astra", Backend: "opencode", Provider: "openai", Model: "openai/gpt-6-astra", Request: "prompt"})
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
