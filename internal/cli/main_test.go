package cli

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestMainRawGeminiVerticalSlice(t *testing.T) {
	root := t.TempDir()
	fake := filepath.Join(root, "testprovider")
	command := exec.Command("go", "build", "-o", fake, "./internal/testprovider")
	command.Dir = filepath.Join("..", "..")
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("build test provider: %v\n%s", err, output)
	}
	if err := os.Symlink(fake, filepath.Join(root, "gemini")); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", root+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("HATCH_RUN_ARTIFACT_ROOT", filepath.Join(root, "runs"))
	t.Setenv("HATCH_TEST_SCENARIO", "success_text")
	t.Setenv("HATCH_TEST_RECORD", filepath.Join(root, "invocation.json"))

	var stdout, stderr bytes.Buffer
	exitCode := Main([]string{"-b", "gemini", "--json", "-"}, bytes.NewBufferString("oracle prompt"), &stdout, &stderr, true)
	if exitCode != 0 {
		t.Fatalf("exit=%d stderr=%s stdout=%s", exitCode, stderr.String(), stdout.String())
	}
	var result struct {
		OK           bool   `json:"ok"`
		Output       string `json:"output"`
		ArtifactPath string `json:"artifact_path"`
		Run          struct {
			RunID     string `json:"run_id"`
			Lifecycle string `json:"lifecycle"`
			Outcome   string `json:"outcome"`
		} `json:"run"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if !result.OK || result.Output != "fake provider output\n" || result.Run.Lifecycle != "terminal" || result.Run.Outcome != "succeeded" {
		t.Fatalf("unexpected result: %#v", result)
	}
	if result.ArtifactPath == "" || result.Run.RunID == "" {
		t.Fatalf("durable identity missing: %#v", result)
	}
}

func TestMainFailsClosedBeforeProviderWhenArtifactRootUnavailable(t *testing.T) {
	root := filepath.Join(t.TempDir(), "root-file")
	if err := os.WriteFile(root, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HATCH_RUN_ARTIFACT_ROOT", root)
	var stdout, stderr bytes.Buffer
	exitCode := Main([]string{"-b", "gemini", "--json", "prompt"}, bytes.NewReader(nil), &stdout, &stderr, true)
	if exitCode == 0 {
		t.Fatalf("unexpected success: %s", stdout.String())
	}
	var result map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result["ok"] != false {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func TestMainDoctorJSON(t *testing.T) {
	directory := t.TempDir()
	binary := filepath.Join(directory, "cursor-agent")
	if err := os.WriteFile(binary, []byte("#!/bin/sh\nprintf '%s\\n' 'cursor-grok-4.5-high - Grok'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	var stdout, stderr bytes.Buffer
	if exit := Main([]string{"doctor", "--json"}, bytes.NewReader(nil), &stdout, &stderr, true); exit != 0 {
		t.Fatalf("exit=%d stdout=%s stderr=%s", exit, stdout.String(), stderr.String())
	}
	var result struct {
		OK     bool `json:"ok"`
		Checks []struct {
			Name string `json:"name"`
			OK   bool   `json:"ok"`
		} `json:"checks"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if !result.OK || len(result.Checks) != 1 || result.Checks[0].Name != "cursor.grok" || !result.Checks[0].OK {
		t.Fatalf("doctor = %#v", result)
	}
}
