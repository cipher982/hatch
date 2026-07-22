package contracts

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestReleaseInstall(t *testing.T) {
	root := repoRoot(t)
	build := func(directory string) {
		command := exec.Command("sh", filepath.Join(root, "scripts", "build-release.sh"))
		command.Dir = root
		command.Env = append(os.Environ(), "DIST_DIR="+directory, "VERSION=test-release", "COMMIT=test-commit")
		if data, err := command.CombinedOutput(); err != nil {
			t.Fatalf("release build: %v\n%s", err, data)
		}
	}
	firstDir, secondDir := t.TempDir(), t.TempDir()
	build(firstDir)
	build(secondDir)
	for _, target := range []string{"darwin_arm64", "darwin_amd64", "linux_amd64", "linux_arm64"} {
		first := mustRead(t, filepath.Join(firstDir, "hatch_test-release_"+target, "hatch"))
		second := mustRead(t, filepath.Join(secondDir, "hatch_test-release_"+target, "hatch"))
		if len(first) == 0 || sha256.Sum256(first) != sha256.Sum256(second) {
			t.Fatalf("release target %s was empty or non-reproducible", target)
		}
	}
	secondPath := filepath.Join(secondDir, "hatch_test-release_"+runtime.GOOS+"_"+runtime.GOARCH, "hatch")

	version := exec.Command(secondPath, "--version")
	versionOutput, err := version.Output()
	if err != nil || !bytes.Contains(versionOutput, []byte("test-release")) || !bytes.Contains(versionOutput, []byte("test-commit")) || !bytes.Contains(versionOutput, []byte(runtime.GOOS+"/"+runtime.GOARCH)) {
		t.Fatalf("version=%q err=%v", versionOutput, err)
	}
	help := exec.Command(secondPath, "--help")
	helpOutput, err := help.Output()
	if err != nil || !bytes.Contains(helpOutput, []byte("hatch codex")) {
		t.Fatalf("help=%q err=%v", helpOutput, err)
	}
	doctor := exec.Command(secondPath, "doctor", "--json")
	doctor.Env = append(os.Environ(), "PATH="+t.TempDir())
	doctorOutput, doctorErr := doctor.Output()
	if doctorErr == nil || !strings.Contains(doctorErr.Error(), "exit status 4") {
		t.Fatalf("doctor startup should report unavailable provider: %v", doctorErr)
	}
	var doctorResult map[string]any
	if err := json.Unmarshal(doctorOutput, &doctorResult); err != nil || doctorResult["ok"] != false {
		t.Fatalf("doctor=%s err=%v", doctorOutput, err)
	}
	rehearsal := exec.Command("sh", filepath.Join(root, "scripts", "test-install-local.sh"))
	rehearsal.Dir = root
	if output, err := rehearsal.CombinedOutput(); err != nil || !bytes.Contains(output, []byte("local Go install smoke passed")) {
		t.Fatalf("local install smoke: %v\n%s", err, output)
	}
}
