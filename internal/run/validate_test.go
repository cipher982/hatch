package run

import (
	"path/filepath"
	"testing"
)

func TestManifestWriterRejectsInvalidAxesAndPaths(t *testing.T) {
	store := NewStore(filepath.Join(t.TempDir(), "runs"))
	artifact, err := store.Prepare(PreparedRun{Surface: "test", Backend: "raw", Provider: "test", Model: "model", Request: "prompt"})
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]func(*Manifest){
		"unsafe path":     func(m *Manifest) { m.Capture.StdoutFile = "../secret" },
		"collapsed axes":  func(m *Manifest) { m.Capture.State = "succeeded" },
		"unknown warning": func(m *Manifest) { m.Warnings = []Warning{{Code: "free_form"}} },
		"early outcome":   func(m *Manifest) { value := OutcomeFailed; m.Outcome = &value },
		"missing backend": func(m *Manifest) { m.Backend = "" },
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
