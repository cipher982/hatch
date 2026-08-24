package cli

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func installAdmissionFixture(t *testing.T, body string, required bool) ExecutionContext {
	t.Helper()
	home := t.TempDir()
	hook := filepath.Join(home, ".local", "bin", "agent-storage-admit")
	if err := os.MkdirAll(filepath.Dir(hook), 0o700); err != nil {
		t.Fatal(err)
	}
	if body != "" {
		if err := os.WriteFile(hook, []byte("#!/bin/sh\n"+body+"\n"), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	if required {
		declaration := filepath.Join(home, "git", "me", "config", "storage-admission", "release.json")
		if err := os.MkdirAll(filepath.Dir(declaration), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(declaration, []byte("{\"required\":true}\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return ExecutionContext{Home: home, HomeWritable: true}
}

func TestStorageAdmissionAllows(t *testing.T) {
	sentinel := filepath.Join(t.TempDir(), "parent")
	t.Setenv("STORAGE_ADMISSION_SENTINEL", sentinel)
	ctx := installAdmissionFixture(t, "printf '%s' \"$AGENT_STORAGE_PARENT_PID\" > \"$STORAGE_ADMISSION_SENTINEL\"\nexit 0", true)
	active, err := applyStorageAdmission(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !active {
		t.Fatal("configured hook should report active admission")
	}
	parent, err := os.ReadFile(sentinel)
	if err != nil || string(parent) != fmt.Sprint(os.Getpid()) {
		t.Fatalf("admission parent = %q, err=%v", parent, err)
	}
}

func TestStorageAdmissionDenialIsReturned(t *testing.T) {
	ctx := installAdmissionFixture(t, "echo below-floor >&2; exit 75", true)
	_, err := applyStorageAdmission(ctx)
	if err == nil || !strings.Contains(err.Error(), "below-floor") {
		t.Fatalf("error = %v", err)
	}
}

func TestStorageAdmissionRequiredFailsClosed(t *testing.T) {
	ctx := installAdmissionFixture(t, "", true)
	_, err := applyStorageAdmission(ctx)
	if err == nil || !strings.Contains(err.Error(), "requires storage admission") {
		t.Fatalf("error = %v", err)
	}
}

func TestStorageAdmissionOptionalWhenUndeclared(t *testing.T) {
	ctx := ExecutionContext{Home: t.TempDir(), HomeWritable: true}
	active, err := applyStorageAdmission(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if active {
		t.Fatal("undeclared hook should report inactive admission")
	}
}
