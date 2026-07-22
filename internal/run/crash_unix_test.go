//go:build darwin || linux

package run

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

func TestCrashLeavesInspectableRunningArtifact(t *testing.T) {
	repo := filepath.Join("..", "..")
	directory := t.TempDir()
	hatchBinary := filepath.Join(directory, "hatch")
	fakeBinary := filepath.Join(directory, "testprovider")
	for output, command := range map[string]*exec.Cmd{
		hatchBinary: exec.Command("go", "build", "-o", hatchBinary, "./cmd/hatch"),
		fakeBinary:  exec.Command("go", "build", "-o", fakeBinary, "./internal/testprovider"),
	} {
		command.Dir = repo
		if data, err := command.CombinedOutput(); err != nil {
			t.Fatalf("build %s: %v\n%s", output, err, data)
		}
	}
	if err := os.Symlink(fakeBinary, filepath.Join(directory, "gemini")); err != nil {
		t.Fatal(err)
	}
	root := filepath.Join(directory, "runs")
	home := filepath.Join(directory, "home")
	if err := os.Mkdir(home, 0o700); err != nil {
		t.Fatal(err)
	}
	command := exec.Command(hatchBinary, "-b", "gemini", "--json", "--timeout", "30", "crash proof")
	command.Env = append(os.Environ(),
		"HOME="+home,
		"PATH="+directory+string(os.PathListSeparator)+os.Getenv("PATH"),
		"HATCH_RUN_ARTIFACT_ROOT="+root,
		"HATCH_TEST_SCENARIO=hang",
	)
	var stdout, stderr bytes.Buffer
	command.Stdout, command.Stderr = &stdout, &stderr
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	var record Record
	deadline := time.Now().Add(8 * time.Second)
	for time.Now().Before(deadline) {
		entries, _ := os.ReadDir(root)
		if len(entries) == 1 {
			candidate, err := ReadRecord(filepath.Join(root, entries[0].Name()))
			if err == nil && candidate.Manifest != nil && candidate.Manifest.Lifecycle == LifecycleRunning {
				raw, _ := os.ReadFile(filepath.Join(candidate.Path, candidate.Manifest.Capture.StdoutFile))
				if bytes.Contains(raw, []byte("partial output")) {
					record = candidate
					break
				}
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	if record.Manifest == nil {
		_ = command.Process.Kill()
		_, _ = command.Process.Wait()
		t.Fatalf("running artifact not observed; stdout=%s stderr=%s", stdout.String(), stderr.String())
	}
	if err := command.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	_, _ = command.Process.Wait()
	if record.Manifest.Process != nil && record.Manifest.Process.ProcessGroup != nil {
		_ = syscall.Kill(-*record.Manifest.Process.ProcessGroup, syscall.SIGKILL)
	}
	after, err := ReadRecord(record.Path)
	if err != nil {
		t.Fatal(err)
	}
	if after.Manifest.Lifecycle != LifecycleRunning || after.Manifest.Outcome != nil || after.Manifest.Process == nil || after.Manifest.Process.StartIdentity == nil {
		t.Fatalf("crash artifact = %#v", after.Manifest)
	}
	raw, err := os.ReadFile(filepath.Join(record.Path, after.Manifest.Capture.StdoutFile))
	if err != nil || !bytes.Contains(raw, []byte("partial output")) {
		t.Fatalf("crash evidence = %q, %v", raw, err)
	}
}
