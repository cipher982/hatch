package run

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestManifestWriterRejectsInvalidAxesAndPaths(t *testing.T) {
	store := NewStore(filepath.Join(t.TempDir(), "runs"))
	artifact, err := store.Prepare(PreparedRun{Surface: "test", Backend: "raw", Provider: "test", Model: "model", Request: "prompt"})
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]func(*Manifest){
		"unsafe path": func(m *Manifest) { m.Capture.StdoutFile = "../secret" },
		"unsafe result path": func(m *Manifest) {
			value := "../result"
			m.Result.OutputFile = &value
		},
		"unsafe snapshot path": func(m *Manifest) {
			value := "../snapshot"
			m.ProviderState.SnapshotPath = &value
		},
		"unsafe archive path": func(m *Manifest) {
			value := "../receipt"
			m.Archive.ReceiptFile = &value
		},
		"non-hex request digest":  func(m *Manifest) { m.Invocation.RequestSHA256 = strings.Repeat("z", 64) },
		"collapsed axes":          func(m *Manifest) { m.Capture.State = "succeeded" },
		"unknown warning":         func(m *Manifest) { m.Warnings = []Warning{{Code: "free_form"}} },
		"early outcome":           func(m *Manifest) { value := OutcomeFailed; m.Outcome = &value },
		"missing backend":         func(m *Manifest) { m.Backend = "" },
		"missing writer contract": func(m *Manifest) { m.Writer = Writer{} },
		"identity without id": func(m *Manifest) {
			m.ProviderState.NativeIDState = "observed"
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			candidate := artifact.Manifest
			mutate(&candidate)
			if err := ValidateManifest(candidate); err == nil {
				t.Fatalf("invalid manifest accepted: %#v", candidate)
			}
		})
	}
}

func TestManifestWriterRejectsMismatchedArtifactPath(t *testing.T) {
	store := NewStore(filepath.Join(t.TempDir(), "runs"))
	artifact, err := store.Prepare(PreparedRun{Surface: "test", Backend: "raw", Provider: "test", Model: "model", Request: "prompt"})
	if err != nil {
		t.Fatal(err)
	}
	artifact.Manifest.Capture.ArtifactPath = filepath.Join(t.TempDir(), "copied-run")
	if err := store.writeManifest(artifact); err == nil {
		t.Fatal("writer accepted a manifest detached from its run directory")
	}
}
