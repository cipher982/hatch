//go:build darwin || linux

package run

import (
	"bytes"
	"encoding/json"
	"errors"
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
	observationDeadline := time.Now().Add(2 * time.Second)
	for (after.Observation == nil || !after.Observation.SuspectedOrphan) && time.Now().Before(observationDeadline) {
		time.Sleep(20 * time.Millisecond)
		after, err = ReadRecord(record.Path)
		if err != nil {
			t.Fatal(err)
		}
	}
	if after.Observation == nil || after.Observation.ProcessAlive == nil || *after.Observation.ProcessAlive ||
		!after.Observation.SuspectedOrphan || after.Manifest.Lifecycle != LifecycleRunning {
		t.Fatalf("nonterminal observation = %#v manifest=%#v", after.Observation, after.Manifest)
	}
	raw, err := os.ReadFile(filepath.Join(record.Path, after.Manifest.Capture.StdoutFile))
	if err != nil || !bytes.Contains(raw, []byte("partial output")) {
		t.Fatalf("crash evidence = %q, %v", raw, err)
	}
}

func TestCLIInterruptCommitsCancelledArtifact(t *testing.T) {
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
	command := exec.Command(hatchBinary, "-b", "gemini", "--json", "--timeout", "30", "interrupt proof")
	command.Env = append(os.Environ(),
		"PATH="+directory+string(os.PathListSeparator)+os.Getenv("PATH"),
		"HATCH_RUN_ARTIFACT_ROOT="+root,
		"HATCH_TEST_SCENARIO=hang",
	)
	var stdout, stderr bytes.Buffer
	command.Stdout, command.Stderr = &stdout, &stderr
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(8 * time.Second)
	for time.Now().Before(deadline) {
		entries, _ := os.ReadDir(root)
		if len(entries) == 1 {
			record, err := ReadRecord(filepath.Join(root, entries[0].Name()))
			if err == nil && record.Manifest != nil && record.Manifest.Lifecycle == LifecycleRunning {
				break
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err := command.Process.Signal(os.Interrupt); err != nil {
		t.Fatal(err)
	}
	waitErr := command.Wait()
	var exit *exec.ExitError
	if !errors.As(waitErr, &exit) || exit.ExitCode() != 130 {
		t.Fatalf("interrupt exit=%v stdout=%s stderr=%s", waitErr, stdout.String(), stderr.String())
	}
	var result PublicResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("decode result: %v\nstdout=%s\nstderr=%s", err, stdout.String(), stderr.String())
	}
	if result.Status != "cancelled" || result.Run == nil || result.Run.Outcome == nil ||
		*result.Run.Outcome != OutcomeCancelled || result.Run.Process == nil || result.Run.Process.CancelCleanup == nil {
		t.Fatalf("interrupt result = %#v", result)
	}
}
