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
	targets := [][2]string{{"darwin", "arm64"}, {"darwin", "amd64"}, {"linux", "amd64"}, {"linux", "arm64"}}
	for _, target := range targets {
		name := target[0] + "_" + target[1]
		t.Run(name, func(t *testing.T) {
			output := filepath.Join(t.TempDir(), "hatch")
			build := exec.Command("go", "build", "-buildvcs=false", "-trimpath", "-o", output, "./cmd/hatch")
			build.Dir = root
			build.Env = append(os.Environ(), "CGO_ENABLED=0", "GOOS="+target[0], "GOARCH="+target[1])
			if data, err := build.CombinedOutput(); err != nil {
				t.Fatalf("cross-build: %v\n%s", err, data)
			}
			if info, err := os.Stat(output); err != nil || info.Size() == 0 {
				t.Fatalf("binary: %v %#v", err, info)
			}
		})
	}

	build := func(path string) []byte {
		command := exec.Command("go", "build", "-buildvcs=false", "-trimpath", "-ldflags", "-X github.com/cipher982/hatch/internal/cli.Version=test-release -X github.com/cipher982/hatch/internal/cli.Commit=test-commit -X github.com/cipher982/hatch/internal/cli.Dirty=false", "-o", path, "./cmd/hatch")
		command.Dir = root
		command.Env = append(os.Environ(), "CGO_ENABLED=0")
		if data, err := command.CombinedOutput(); err != nil {
			t.Fatalf("native build: %v\n%s", err, data)
		}
		return mustRead(t, path)
	}
	first := build(filepath.Join(t.TempDir(), "hatch-one"))
	secondPath := filepath.Join(t.TempDir(), "hatch-two")
	second := build(secondPath)
	if sha256.Sum256(first) != sha256.Sum256(second) {
		t.Fatal("identical release inputs produced different binaries")
	}

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
}
