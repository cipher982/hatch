package run

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestListAndInspectCurrentAndLegacyRecords(t *testing.T) {
	root := filepath.Join(t.TempDir(), "runs")
	store := NewStore(root)
	store.IDGen = func(time.Time) (string, error) { return "hatch_current", nil }
	artifact, err := store.Prepare(PreparedRun{Surface: "gemini.raw", Provider: "google", Request: "prompt"})
	if err != nil {
		t.Fatal(err)
	}
	stdout, stderr, err := store.OpenStreams(artifact)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = stdout.Write([]byte("answer"))
	_ = stdout.Close()
	_ = stderr.Close()
	resultFile, _ := store.WriteResult(artifact, []byte("answer"))
	if err := store.CommitTerminal(artifact, OutcomeSucceeded, 0, Result{Output: "present", TerminalMarker: "not_applicable", OutputFile: &resultFile}, State{Retention: "unknown", NativeIDState: "not_exposed", Capabilities: map[string]string{}}, nil); err != nil {
		t.Fatal(err)
	}
	legacyDirectory := filepath.Join(root, "old-directory")
	if err := os.MkdirAll(legacyDirectory, 0o700); err != nil {
		t.Fatal(err)
	}
	writeJSONFile(t, filepath.Join(legacyDirectory, "metadata.json"), map[string]any{"artifact_kind": "hatch_opencode_run", "run_id": "legacy_run", "outcome": "succeeded", "session_id": "ses_old"})
	expertCache := filepath.Join(t.TempDir(), "expert")
	if err := os.MkdirAll(expertCache, 0o700); err != nil {
		t.Fatal(err)
	}
	writeJSONFile(t, filepath.Join(expertCache, "resp_old.json"), map[string]any{"response_id": "resp_old", "status": "queued"})

	summaries, err := ListRecords(root, expertCache)
	if err != nil || len(summaries) != 3 {
		t.Fatalf("summaries = %#v, %v", summaries, err)
	}
	current, err := InspectRecord(root, expertCache, "hatch_current")
	if err != nil || current.Manifest == nil || current.Manifest.RunID != "hatch_current" ||
		!contains(current.Files, "evidence.sha256") || !contains(current.Files, "manifest.json") {
		t.Fatalf("current = %#v, %v", current, err)
	}
	legacy, err := InspectRecord(root, expertCache, "legacy_run")
	if err != nil || legacy.Kind != "legacy_opencode" {
		t.Fatalf("legacy = %#v, %v", legacy, err)
	}
	expert, err := InspectRecord(root, expertCache, "resp_old")
	if err != nil || expert.Kind != "legacy_expert" {
		t.Fatalf("expert = %#v, %v", expert, err)
	}
}

func contains(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}

func TestReadRecordRetainsRawAndNormalizesUnknownEnums(t *testing.T) {
	directory := t.TempDir()
	writeJSONFile(t, filepath.Join(directory, "manifest.json"), map[string]any{
		"schema_version": 1, "run_id": "hatch_unknown", "lifecycle": "future", "outcome": "future",
		"capture": map[string]any{"state": "future"}, "provider_state": map[string]any{"retention": "future"},
		"future_field": map[string]any{"kept": true},
	})
	record, err := ReadRecord(directory)
	if err != nil {
		t.Fatal(err)
	}
	if record.Manifest.Lifecycle != "unknown" || record.Manifest.Outcome == nil || *record.Manifest.Outcome != "unknown" || record.Manifest.Capture.State != "unknown" || record.Raw["future_field"] == nil {
		t.Fatalf("record = %#v", record)
	}
}

func TestReadRecordAcceptsPreviewProviderVersionField(t *testing.T) {
	directory := t.TempDir()
	writeJSONFile(t, filepath.Join(directory, "manifest.json"), map[string]any{
		"schema_version": 1, "run_id": "hatch_preview", "lifecycle": "terminal",
		"capture": map[string]any{"state": "durable"},
		"provider_state": map[string]any{
			"retention": "hatch_preserved", "provider_version": "opencode preview",
		},
	})
	record, err := ReadRecord(directory)
	if err != nil || record.Manifest.ProviderState.ProviderVersion == nil ||
		*record.Manifest.ProviderState.ProviderVersion != "opencode preview" {
		t.Fatalf("record = %#v, %v", record, err)
	}
}

func TestReadRecordRejectsUnsupportedSchemaAndTraversal(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "run")
	if err := os.Mkdir(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	writeJSONFile(t, filepath.Join(directory, "manifest.json"), map[string]any{"schema_version": 2})
	if _, err := ReadRecord(directory); err == nil {
		t.Fatal("unsupported schema accepted")
	}
	if _, err := InspectRecord(root, t.TempDir(), "../run"); err == nil {
		t.Fatal("traversal run id accepted")
	}
}

func writeJSONFile(t *testing.T, path string, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}
