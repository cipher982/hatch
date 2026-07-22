package run

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type Store struct {
	Root  string
	Now   func() time.Time
	IDGen func(time.Time) (string, error)
}

type StreamSink interface {
	io.Writer
	Sync() error
	Close() error
}

type RunStore interface {
	Prepare(PreparedRun) (*Artifact, error)
	OpenStreams(*Artifact) (StreamSink, StreamSink, error)
	MarkRunning(*Artifact, int, time.Time, string) error
	MarkHTTPRunning(*Artifact, time.Time) error
	WriteResult(*Artifact, []byte) (string, error)
	CommitTerminal(*Artifact, Outcome, int, Result, State, []Warning) error
	WritePublicProjection(*Artifact, PublicResult) error
	MarkCaptureDegraded(*Artifact, Warning) error
}

type Artifact struct {
	Path     string
	Manifest Manifest
}

type PreparedRun struct {
	Surface, Provider, Model, CWD, Request string
	Execution                              string
	RedactedArgv                           []string
	CredentialNames                        []string
	StructuredStdout                       bool
}

func DefaultRoot() (string, error) {
	if configured := strings.TrimSpace(os.Getenv("HATCH_RUN_ARTIFACT_ROOT")); configured != "" {
		return filepath.Abs(configured)
	}
	if state := strings.TrimSpace(os.Getenv("XDG_STATE_HOME")); state != "" {
		return filepath.Abs(filepath.Join(state, "hatch", "runs"))
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".local", "state", "hatch", "runs"), nil
}

func NewStore(root string) Store {
	return Store{Root: root, Now: time.Now, IDGen: newRunID}
}

func (s Store) Prepare(spec PreparedRun) (*Artifact, error) {
	now := s.Now().UTC()
	runID, err := s.IDGen(now)
	if err != nil {
		return nil, err
	}
	if err := secureMkdirAll(s.Root); err != nil {
		return nil, fmt.Errorf("prepare artifact root: %w", err)
	}
	path := filepath.Join(s.Root, runID)
	if err := os.Mkdir(path, 0o700); err != nil {
		return nil, fmt.Errorf("create run artifact: %w", err)
	}
	if err := os.Chmod(path, 0o700); err != nil {
		return nil, err
	}
	requestPath := filepath.Join(path, "request.txt")
	if err := writePrivate(requestPath, []byte(spec.Request)); err != nil {
		return nil, err
	}
	digest := sha256.Sum256([]byte(spec.Request))
	stdoutFile := "stdout.log"
	if spec.StructuredStdout {
		stdoutFile = "stdout.jsonl"
	}
	manifest := Manifest{
		SchemaVersion: 1, RunID: runID, CreatedAt: now, UpdatedAt: now,
		Lifecycle: LifecyclePrepared, Surface: spec.Surface, Provider: spec.Provider,
		Model: spec.Model, CWD: spec.CWD, Execution: executionOrDefault(spec.Execution),
		Invocation: Invocation{
			RequestFile: "request.txt", RequestSHA256: hex.EncodeToString(digest[:]),
			RedactedArgv: append([]string(nil), spec.RedactedArgv...), CredentialEnvNames: append([]string(nil), spec.CredentialNames...),
		},
		Result:        Result{Output: "absent", TerminalMarker: "not_observed"},
		Capture:       Capture{State: "durable", ArtifactPath: path, StdoutFile: stdoutFile, StderrFile: "stderr.log"},
		ProviderState: State{Retention: "unknown", NativeIDState: "unknown", Capabilities: map[string]string{}},
		Archive:       Archive{State: "not_requested"}, Warnings: []Warning{},
	}
	artifact := &Artifact{Path: path, Manifest: manifest}
	if err := s.writeManifest(artifact); err != nil {
		return nil, err
	}
	return artifact, nil
}

func executionOrDefault(value string) string {
	if value == "" {
		return "subprocess"
	}
	return value
}

func (s Store) MarkRunning(artifact *Artifact, pid int, started time.Time, identity string) error {
	artifact.Manifest.Lifecycle = LifecycleRunning
	processGroup := pid
	artifact.Manifest.Process = &Process{PID: pid, ProcessGroup: &processGroup, StartedAt: started.UTC()}
	if identity != "" {
		artifact.Manifest.Process.StartIdentity = &identity
	}
	artifact.Manifest.UpdatedAt = s.Now().UTC()
	return s.writeManifest(artifact)
}

func (s Store) MarkHTTPRunning(artifact *Artifact, started time.Time) error {
	artifact.Manifest.Lifecycle = LifecycleRunning
	artifact.Manifest.HTTP = &HTTP{StartedAt: started.UTC()}
	artifact.Manifest.UpdatedAt = s.Now().UTC()
	return s.writeManifest(artifact)
}

func (s Store) WriteResult(artifact *Artifact, output []byte) (string, error) {
	name := "result.txt"
	if err := writePrivate(filepath.Join(artifact.Path, name), output); err != nil {
		return "", err
	}
	return name, nil
}

func (s Store) WritePublicProjection(artifact *Artifact, result PublicResult) error {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return atomicPrivate(filepath.Join(artifact.Path, "result.json"), data)
}

func (s Store) CommitTerminal(artifact *Artifact, outcome Outcome, exitCode int, result Result, state State, warnings []Warning) error {
	now := s.Now().UTC()
	artifact.Manifest.Lifecycle = LifecycleTerminal
	artifact.Manifest.Outcome = &outcome
	artifact.Manifest.UpdatedAt = now
	artifact.Manifest.Result = result
	artifact.Manifest.ProviderState = state
	artifact.Manifest.Warnings = append([]Warning(nil), warnings...)
	if artifact.Manifest.Capture.State == "degraded" && outcome == OutcomeSucceeded {
		updated := OutcomeSucceededWarnings
		artifact.Manifest.Outcome = &updated
	}
	if artifact.Manifest.Process != nil {
		artifact.Manifest.Process.ExitedAt = &now
		artifact.Manifest.Process.ExitCode = &exitCode
	}
	evidence := []string{"request.txt", artifact.Manifest.Capture.StdoutFile, artifact.Manifest.Capture.StderrFile}
	if result.OutputFile != nil {
		evidence = append(evidence, *result.OutputFile)
	}
	if state.SnapshotPath != nil {
		for _, subtree := range []string{"data", "state"} {
			base := filepath.Join(artifact.Path, *state.SnapshotPath, subtree)
			err := filepath.WalkDir(base, func(path string, entry os.DirEntry, walkErr error) error {
				if walkErr != nil || entry.IsDir() {
					return walkErr
				}
				relative, err := filepath.Rel(artifact.Path, path)
				if err == nil {
					evidence = append(evidence, relative)
				}
				return err
			})
			if err != nil && !os.IsNotExist(err) {
				s.markCaptureDegraded(artifact, outcome, fmt.Errorf("walk provider evidence: %w", err))
			}
		}
	}
	digest, err := evidenceDigest(artifact.Path, evidence)
	if err == nil {
		artifact.Manifest.Capture.EvidenceSHA256 = &digest
	} else {
		s.markCaptureDegraded(artifact, outcome, fmt.Errorf("hash evidence: %w", err))
	}
	return s.writeManifest(artifact)
}

func (s Store) markCaptureDegraded(artifact *Artifact, outcome Outcome, err error) {
	artifact.Manifest.Capture.State = "degraded"
	artifact.Manifest.Warnings = append(artifact.Manifest.Warnings, Warning{Code: "capture_persistence_failed", Message: err.Error()})
	if outcome == OutcomeSucceeded {
		updated := OutcomeSucceededWarnings
		artifact.Manifest.Outcome = &updated
	}
}

func (s Store) OpenStreams(artifact *Artifact) (StreamSink, StreamSink, error) {
	stdout, err := openPrivate(filepath.Join(artifact.Path, artifact.Manifest.Capture.StdoutFile))
	if err != nil {
		return nil, nil, err
	}
	stderr, err := openPrivate(filepath.Join(artifact.Path, artifact.Manifest.Capture.StderrFile))
	if err != nil {
		stdout.Close()
		return nil, nil, err
	}
	return stdout, stderr, nil
}

func (s Store) MarkCaptureDegraded(artifact *Artifact, warning Warning) error {
	artifact.Manifest.Capture.State = "degraded"
	artifact.Manifest.Warnings = append(artifact.Manifest.Warnings, warning)
	if artifact.Manifest.Outcome != nil && *artifact.Manifest.Outcome == OutcomeSucceeded {
		outcome := OutcomeSucceededWarnings
		artifact.Manifest.Outcome = &outcome
	}
	artifact.Manifest.UpdatedAt = s.Now().UTC()
	return s.writeManifest(artifact)
}

func (s Store) writeManifest(artifact *Artifact) error {
	data, err := json.MarshalIndent(artifact.Manifest, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return atomicPrivate(filepath.Join(artifact.Path, "manifest.json"), data)
}

func newRunID(now time.Time) (string, error) {
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		return "", err
	}
	stamp := now.UTC().Format("20060102T150405.000000000Z")
	return "hatch_" + stamp + "_" + hex.EncodeToString(random), nil
}

func secureMkdirAll(path string) error {
	if info, err := os.Lstat(path); err == nil {
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("artifact root must not be a symlink: %s", path)
		}
		if !info.IsDir() {
			return fmt.Errorf("artifact root is not a directory: %s", path)
		}
	} else if !os.IsNotExist(err) {
		return err
	}
	if err := os.MkdirAll(path, 0o700); err != nil {
		return err
	}
	if info, err := os.Lstat(path); err != nil || info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return fmt.Errorf("artifact root is not a private directory: %s", path)
	}
	return os.Chmod(path, 0o700)
}

func openPrivate(path string) (*os.File, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return nil, err
	}
	if err := file.Chmod(0o600); err != nil {
		file.Close()
		return nil, err
	}
	return file, nil
}

func writePrivate(path string, data []byte) error {
	file, err := openPrivate(path)
	if err != nil {
		return err
	}
	if _, err := file.Write(data); err != nil {
		file.Close()
		return err
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}

func atomicPrivate(path string, data []byte) error {
	dir := filepath.Dir(path)
	temp, err := os.CreateTemp(dir, ".manifest-*")
	if err != nil {
		return err
	}
	tempPath := temp.Name()
	defer os.Remove(tempPath)
	if err := temp.Chmod(0o600); err != nil {
		temp.Close()
		return err
	}
	if _, err := temp.Write(data); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Sync(); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tempPath, path); err != nil {
		return err
	}
	directory, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer directory.Close()
	return directory.Sync()
}

func evidenceDigest(root string, names []string) (string, error) {
	sort.Strings(names)
	hash := sha256.New()
	for _, name := range names {
		path, err := evidencePath(root, name)
		if err != nil {
			return "", err
		}
		file, err := os.Open(path)
		if err != nil {
			return "", err
		}
		fileHash := sha256.New()
		_, copyErr := io.Copy(fileHash, file)
		closeErr := file.Close()
		if copyErr != nil {
			return "", copyErr
		}
		if closeErr != nil {
			return "", closeErr
		}
		fmt.Fprintf(hash, "%s  %s\n", hex.EncodeToString(fileHash.Sum(nil)), filepath.ToSlash(name))
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func evidencePath(root, name string) (string, error) {
	if name == "" || filepath.IsAbs(name) || filepath.Clean(name) != name || name == ".." || strings.HasPrefix(name, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("unsafe evidence path %q", name)
	}
	return filepath.Join(root, name), nil
}
