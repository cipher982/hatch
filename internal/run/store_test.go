package run

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestStorePreparesPrivateDurableRun(t *testing.T) {
	root := filepath.Join(t.TempDir(), "runs")
	fixed := time.Date(2026, 7, 22, 15, 0, 0, 0, time.UTC)
	store := NewStore(root)
	store.Now = func() time.Time { return fixed }
	store.IDGen = func(time.Time) (string, error) { return "hatch_test_run", nil }

	artifact, err := store.Prepare(PreparedRun{
		Surface: "gemini.raw", Provider: "google", Model: "gemini-3-pro-preview", CWD: "/repo",
		Request: "secret prompt", RedactedArgv: []string{"gemini", "<prompt>"}, CredentialNames: []string{"GEMINI_API_KEY"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if artifact.Manifest.Lifecycle != LifecyclePrepared || artifact.Manifest.RunID != "hatch_test_run" {
		t.Fatalf("unexpected manifest: %#v", artifact.Manifest)
	}
	assertMode(t, root, 0o700)
	assertMode(t, artifact.Path, 0o700)
	assertMode(t, filepath.Join(artifact.Path, "request.txt"), 0o600)
	assertMode(t, filepath.Join(artifact.Path, "manifest.json"), 0o600)
	if got, err := os.ReadFile(filepath.Join(artifact.Path, "request.txt")); err != nil || string(got) != "secret prompt" {
		t.Fatalf("request evidence = %q, %v", got, err)
	}

	var disk Manifest
	data, err := os.ReadFile(filepath.Join(artifact.Path, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &disk); err != nil {
		t.Fatal(err)
	}
	if disk.RunID != artifact.Manifest.RunID || disk.Capture.ArtifactPath != artifact.Path {
		t.Fatalf("disk manifest mismatch: %#v", disk)
	}
}

func TestStorePermissionsIgnoreCallerUmask(t *testing.T) {
	for _, mask := range []int{0o000, 0o077} {
		old := syscall.Umask(mask)
		store := NewStore(filepath.Join(t.TempDir(), "runs"))
		artifact, err := store.Prepare(PreparedRun{Request: "prompt"})
		syscall.Umask(old)
		if err != nil {
			t.Fatalf("umask %#o: %v", mask, err)
		}
		assertMode(t, store.Root, 0o700)
		assertMode(t, artifact.Path, 0o700)
		assertMode(t, filepath.Join(artifact.Path, "request.txt"), 0o600)
	}
}

type faultFile struct {
	bytes.Buffer
	fail string
}

func (f *faultFile) Name() string { return "/tmp/fault-temp" }
func (f *faultFile) Chmod(os.FileMode) error {
	if f.fail == "chmod" {
		return errors.New("chmod")
	}
	return nil
}
func (f *faultFile) Sync() error {
	if f.fail == "sync" {
		return errors.New("sync")
	}
	return nil
}
func (f *faultFile) Close() error {
	if f.fail == "close" {
		return errors.New("close")
	}
	return nil
}
func (f *faultFile) Write(data []byte) (int, error) {
	if f.fail == "write" {
		return 0, errors.New("write")
	}
	return f.Buffer.Write(data)
}

type faultDir struct{ fail bool }

func (d faultDir) Sync() error {
	if d.fail {
		return errors.New("dir sync")
	}
	return nil
}
func (faultDir) Close() error { return nil }

func TestAtomicPrivatePropagatesEveryPersistenceBoundary(t *testing.T) {
	tests := []string{"create", "chmod", "write", "sync", "close", "rename", "open_dir", "dir_sync"}
	for _, fail := range tests {
		t.Run(fail, func(t *testing.T) {
			ops := atomicFileOps{
				createTemp: func(string, string) (atomicTempFile, error) {
					if fail == "create" {
						return nil, errors.New("create")
					}
					return &faultFile{fail: fail}, nil
				},
				rename: func(string, string) error {
					if fail == "rename" {
						return errors.New("rename")
					}
					return nil
				},
				openDir: func(string) (atomicSyncDir, error) {
					if fail == "open_dir" {
						return nil, errors.New("open dir")
					}
					return faultDir{fail: fail == "dir_sync"}, nil
				},
				remove: func(string) error { return nil },
			}
			if err := atomicPrivateWithOps(filepath.Join(t.TempDir(), "manifest.json"), []byte("data"), ops); err == nil {
				t.Fatalf("%s failure was ignored", fail)
			}
		})
	}
}

func TestStoreLifecycleAndEvidenceDigest(t *testing.T) {
	store := NewStore(filepath.Join(t.TempDir(), "runs"))
	store.IDGen = func(time.Time) (string, error) { return "hatch_lifecycle", nil }
	artifact, err := store.Prepare(PreparedRun{Surface: "gemini.raw", Provider: "google", Model: "model", CWD: "/repo", Request: "prompt", RedactedArgv: []string{"gemini"}})
	if err != nil {
		t.Fatal(err)
	}
	stdout, stderr, err := store.OpenStreams(artifact)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.WriteString(stdout, "answer\n")
	_, _ = io.WriteString(stderr, "warning\n")
	stdout.Close()
	stderr.Close()
	started := time.Now()
	if err := store.MarkRunning(artifact, 123, started, "test-start"); err != nil {
		t.Fatal(err)
	}
	resultFile, err := store.WriteResult(artifact, []byte("answer\n"))
	if err != nil {
		t.Fatal(err)
	}
	state := State{Retention: "provider_owned", NativeIDState: "not_exposed", Capabilities: map[string]string{}}
	if err := store.CommitTerminal(artifact, OutcomeSucceeded, 0, Result{
		Output: "present", TerminalMarker: "not_applicable", OutputBytes: 7, OutputFile: &resultFile,
	}, state, nil); err != nil {
		t.Fatal(err)
	}
	if artifact.Manifest.Lifecycle != LifecycleTerminal || artifact.Manifest.Capture.EvidenceSHA256 == nil {
		t.Fatalf("terminal manifest incomplete: %#v", artifact.Manifest)
	}
	if artifact.Manifest.Warnings == nil {
		t.Fatal("terminal warnings must be an explicit empty array")
	}
	manifestPath := filepath.Join(artifact.Path, artifact.Manifest.Capture.EvidenceManifestFile)
	manifest, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	assertMode(t, manifestPath, 0o600)
	lines := strings.Split(strings.TrimSuffix(string(manifest), "\n"), "\n")
	if len(lines) != 4 || !strings.HasSuffix(lines[0], "  request.txt") ||
		!strings.HasSuffix(lines[1], "  result.txt") ||
		!strings.HasSuffix(lines[2], "  stderr.log") ||
		!strings.HasSuffix(lines[3], "  stdout.log") {
		t.Fatalf("evidence manifest is not the sorted closed set: %q", manifest)
	}
	digest := sha256.Sum256(manifest)
	if got, want := *artifact.Manifest.Capture.EvidenceSHA256, fmt.Sprintf("%x", digest); got != want {
		t.Fatalf("evidence digest = %s, want %s", got, want)
	}
}

func TestStoreFailsBeforeLaunchWhenRootIsAFile(t *testing.T) {
	root := filepath.Join(t.TempDir(), "not-a-directory")
	if err := os.WriteFile(root, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	store := NewStore(root)
	if _, err := store.Prepare(PreparedRun{Surface: "raw", Provider: "unknown", Request: "prompt"}); err == nil {
		t.Fatal("Prepare succeeded with an unusable root")
	}
}

func TestStoreRejectsSymlinkArtifactRoot(t *testing.T) {
	parent := t.TempDir()
	target := filepath.Join(parent, "target")
	if err := os.Mkdir(target, 0o700); err != nil {
		t.Fatal(err)
	}
	root := filepath.Join(parent, "runs")
	if err := os.Symlink(target, root); err != nil {
		t.Fatal(err)
	}
	if _, err := NewStore(root).Prepare(PreparedRun{Request: "prompt"}); err == nil {
		t.Fatal("Prepare followed a symlink artifact root")
	}
}

func TestTerminalHashFailureDegradesCapture(t *testing.T) {
	store := NewStore(filepath.Join(t.TempDir(), "runs"))
	store.IDGen = func(time.Time) (string, error) { return "hatch_missing_streams", nil }
	artifact, err := store.Prepare(PreparedRun{Request: "prompt"})
	if err != nil {
		t.Fatal(err)
	}
	resultFile, err := store.WriteResult(artifact, []byte("answer"))
	if err != nil {
		t.Fatal(err)
	}
	state := State{Retention: "unknown", NativeIDState: "unknown", Capabilities: map[string]string{}}
	if err := store.CommitTerminal(artifact, OutcomeSucceeded, 0, Result{Output: "present", TerminalMarker: "not_applicable", OutputFile: &resultFile}, state, nil); err != nil {
		t.Fatal(err)
	}
	if artifact.Manifest.Capture.State != "degraded" || artifact.Manifest.Outcome == nil || *artifact.Manifest.Outcome != OutcomeSucceededWarnings || len(artifact.Manifest.Warnings) == 0 {
		t.Fatalf("manifest = %#v", artifact.Manifest)
	}
}

func TestEvidenceDigestRejectsTraversal(t *testing.T) {
	if _, err := evidenceDigest(t.TempDir(), []string{"../secret"}); err == nil {
		t.Fatal("traversal evidence path accepted")
	}
}

func assertMode(t *testing.T, path string, want os.FileMode) {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != want {
		t.Fatalf("%s mode = %#o, want %#o", path, got, want)
	}
}
