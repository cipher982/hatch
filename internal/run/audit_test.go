package run

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestAuditFieldEvidenceClassifiesAndPassesCompleteFixture(t *testing.T) {
	root := filepath.Join(t.TempDir(), "runs")
	store := NewStore(root)
	for _, surface := range []string{"claude.haiku", "codex.terra", "cursor.grok", "openrouter.deepseek-v4-pro", "expert"} {
		for range 5 {
			createAuditRun(t, store, surface, OutcomeSucceeded)
		}
	}
	createAuditRun(t, store, "claude.haiku", OutcomeFailed)
	createAuditRun(t, store, "openrouter.raw", OutcomeSucceeded)
	if _, err := store.Prepare(PreparedRun{Surface: "expert", Backend: "http", Provider: "openai", Model: "expert", Request: "prompt"}); err != nil {
		t.Fatal(err)
	}
	writePrecontractAuditManifest(t, root, "old-terminal", "terminal")
	writePrecontractAuditManifest(t, root, "old-crash", "running")

	audit, err := AuditFieldEvidence(root, 25, 5)
	if err != nil {
		t.Fatal(err)
	}
	if !audit.Passed() || audit.Eligible != 25 || audit.Observed != 30 || audit.ExcludedPreContract != 2 || audit.Incomplete != 1 || audit.NonSuccess != 1 || audit.NonSurfaced != 1 || audit.Unsafe != 0 || audit.ExplainedUnsafe != 0 || audit.UnexplainedUnsafe != 0 {
		t.Fatalf("audit = %#v", audit)
	}
}

func TestAuditFieldEvidenceRejectsCorruption(t *testing.T) {
	tests := map[string]func(*testing.T, *Artifact){
		"evidence bytes": func(t *testing.T, artifact *Artifact) {
			appendFile(t, filepath.Join(artifact.Path, artifact.Manifest.Capture.StdoutFile), []byte("tampered\n"))
		},
		"recorded digest": func(t *testing.T, artifact *Artifact) {
			digest := "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
			artifact.Manifest.Capture.EvidenceSHA256 = &digest
			writeAuditManifest(t, artifact)
		},
		"request digest": func(t *testing.T, artifact *Artifact) {
			artifact.Manifest.Invocation.RequestSHA256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
			writeAuditManifest(t, artifact)
		},
		"result size": func(t *testing.T, artifact *Artifact) {
			artifact.Manifest.Result.OutputBytes++
			writeAuditManifest(t, artifact)
		},
		"missing process evidence": func(t *testing.T, artifact *Artifact) {
			artifact.Manifest.Process = nil
			writeAuditManifest(t, artifact)
		},
		"undeclared file": func(t *testing.T, artifact *Artifact) {
			if err := os.WriteFile(filepath.Join(artifact.Path, "extra.log"), []byte("extra"), 0o600); err != nil {
				t.Fatal(err)
			}
		},
		"missing projection": func(t *testing.T, artifact *Artifact) {
			if err := os.Remove(filepath.Join(artifact.Path, "result.json")); err != nil {
				t.Fatal(err)
			}
		},
		"public evidence permissions": func(t *testing.T, artifact *Artifact) {
			if err := os.Chmod(filepath.Join(artifact.Path, artifact.Manifest.Capture.StdoutFile), 0o644); err != nil {
				t.Fatal(err)
			}
		},
		"traversal entry": func(t *testing.T, artifact *Artifact) {
			path := filepath.Join(artifact.Path, artifact.Manifest.Capture.EvidenceManifestFile)
			appendFile(t, path, []byte("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  ../../outside\n"))
			updateEvidenceManifestDigest(t, artifact, path)
		},
		"duplicate entry": func(t *testing.T, artifact *Artifact) {
			path := filepath.Join(artifact.Path, artifact.Manifest.Capture.EvidenceManifestFile)
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			end := bytes.IndexByte(data, '\n') + 1
			if end <= 0 {
				t.Fatal("evidence manifest has no complete entry")
			}
			appendFile(t, path, data[:end])
			updateEvidenceManifestDigest(t, artifact, path)
		},
		"detached identity": func(t *testing.T, artifact *Artifact) {
			artifact.Manifest.RunID = "copied-run"
			writeAuditManifest(t, artifact)
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			root := filepath.Join(t.TempDir(), "runs")
			artifact := createAuditRun(t, NewStore(root), "claude.haiku", OutcomeSucceeded)
			mutate(t, artifact)
			audit, err := AuditFieldEvidence(root, 0, 0)
			if err != nil {
				t.Fatal(err)
			}
			if audit.Unsafe != 1 || audit.UnexplainedUnsafe != 1 || audit.Passed() {
				t.Fatalf("audit accepted %s: %#v", name, audit)
			}
		})
	}
}

func TestAuditFieldEvidenceAcceptsOnlyExactReviewedIncident(t *testing.T) {
	root := filepath.Join(t.TempDir(), "runs")
	artifact := createAuditRun(t, NewStore(root), "claude.haiku", OutcomeSucceeded)
	appendFile(t, filepath.Join(artifact.Path, artifact.Manifest.Capture.StdoutFile), []byte("late write\n"))

	initial, err := auditFieldEvidence(root, 0, 0, nil)
	if err != nil || len(initial.UnsafeRuns) != 1 || initial.Passed() {
		t.Fatalf("initial audit=%#v err=%v", initial, err)
	}
	disposition := FieldIncidentDisposition{Kind: "test", FixedByCommit: "deadbeef", Explanation: "reviewed fixture"}
	key := fieldDispositionKey(artifact.Manifest.RunID, *artifact.Manifest.Capture.EvidenceSHA256, initial.UnsafeRuns[0].Reason)
	reviewed, err := auditFieldEvidence(root, 0, 0, map[string]FieldIncidentDisposition{key: disposition})
	if err != nil || !reviewed.Passed() || reviewed.Unsafe != 1 || reviewed.ExplainedUnsafe != 1 || reviewed.UnexplainedUnsafe != 0 || reviewed.UnsafeRuns[0].Disposition == nil {
		t.Fatalf("reviewed audit=%#v err=%v", reviewed, err)
	}

	appendFile(t, filepath.Join(artifact.Path, artifact.Manifest.Capture.StdoutFile), []byte("another write\n"))
	changed, err := auditFieldEvidence(root, 0, 0, map[string]FieldIncidentDisposition{key: disposition})
	if err != nil || changed.Passed() || changed.ExplainedUnsafe != 0 || changed.UnexplainedUnsafe != 1 {
		t.Fatalf("changed audit=%#v err=%v", changed, err)
	}
}

func createAuditRun(t *testing.T, store Store, surface string, outcome Outcome) *Artifact {
	t.Helper()
	execution := "subprocess"
	if surface == "expert" {
		execution = "http"
	}
	artifact, err := store.Prepare(PreparedRun{Surface: surface, Backend: "fake", Provider: "fake", Model: "fake", Request: "prompt", Execution: execution})
	if err != nil {
		t.Fatal(err)
	}
	if execution == "http" {
		if err := store.MarkHTTPRunning(artifact, store.Now()); err != nil {
			t.Fatal(err)
		}
	} else if err := store.MarkRunning(artifact, 123, store.Now(), "test-process"); err != nil {
		t.Fatal(err)
	}
	stdout, stderr, err := store.OpenStreams(artifact)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stdout.Write([]byte("output\n")); err != nil {
		t.Fatal(err)
	}
	if err := stdout.Close(); err != nil {
		t.Fatal(err)
	}
	if err := stderr.Close(); err != nil {
		t.Fatal(err)
	}
	resultFile, err := store.WriteResult(artifact, []byte("output\n"))
	if err != nil {
		t.Fatal(err)
	}
	result := Result{Output: "present", TerminalMarker: "not_applicable", OutputBytes: 7, OutputFile: &resultFile}
	state := State{Retention: "unknown", NativeIDState: "not_exposed", Capabilities: map[string]string{}}
	if execution == "http" {
		completed := store.Now().UTC()
		artifact.Manifest.HTTP.CompletedAt = &completed
	}
	if err := store.CommitTerminal(artifact, outcome, 0, result, state, nil); err != nil {
		t.Fatal(err)
	}
	public := PublicResult{OK: outcome == OutcomeSucceeded, Status: "ok", Output: "output\n", ExitCode: 0, Run: &artifact.Manifest}
	if err := store.WritePublicProjection(artifact, public); err != nil {
		t.Fatal(err)
	}
	return artifact
}

func writePrecontractAuditManifest(t *testing.T, root, id, lifecycle string) {
	t.Helper()
	directory := filepath.Join(root, id)
	if err := os.MkdirAll(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	data := []byte(`{"schema_version":1,"run_id":"` + id + `","lifecycle":"` + lifecycle + `"}`)
	if err := os.WriteFile(filepath.Join(directory, "manifest.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func appendFile(t *testing.T, path string, data []byte) {
	t.Helper()
	file, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.Write(data); err != nil {
		file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
}

func updateEvidenceManifestDigest(t *testing.T, artifact *Artifact, path string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(data)
	encoded := hex.EncodeToString(digest[:])
	artifact.Manifest.Capture.EvidenceSHA256 = &encoded
	writeAuditManifest(t, artifact)
}

func writeAuditManifest(t *testing.T, artifact *Artifact) {
	t.Helper()
	data, err := json.Marshal(artifact.Manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(artifact.Path, "manifest.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
}
