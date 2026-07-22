package run

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
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

type Artifact struct {
	Path     string
	Manifest Manifest
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

func (s Store) Prepare(surface, provider, model, cwd, request string, redactedArgv []string, credentialNames []string) (*Artifact, error) {
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
	if err := writePrivate(requestPath, []byte(request)); err != nil {
		return nil, err
	}
	digest := sha256.Sum256([]byte(request))
	manifest := Manifest{
		SchemaVersion: 1, RunID: runID, CreatedAt: now, UpdatedAt: now,
		Lifecycle: LifecyclePrepared, Surface: surface, Provider: provider,
		Model: model, CWD: cwd, Execution: "subprocess",
		Invocation: Invocation{
			RequestFile: "request.txt", RequestSHA256: hex.EncodeToString(digest[:]),
			RedactedArgv: append([]string(nil), redactedArgv...), CredentialEnvNames: append([]string(nil), credentialNames...),
		},
		Result:        Result{Output: "absent", TerminalMarker: "not_observed"},
		Capture:       Capture{State: "durable", ArtifactPath: path, StdoutFile: "stdout.log", StderrFile: "stderr.log"},
		ProviderState: State{Retention: "unknown", NativeIDState: "unknown", Capabilities: map[string]string{}},
		Archive:       Archive{State: "not_requested"}, Warnings: []Warning{},
	}
	artifact := &Artifact{Path: path, Manifest: manifest}
	if err := s.writeManifest(artifact); err != nil {
		return nil, err
	}
	return artifact, nil
}

func (s Store) MarkRunning(artifact *Artifact, pid int, started time.Time) error {
	artifact.Manifest.Lifecycle = LifecycleRunning
	artifact.Manifest.Process = &Process{PID: pid, StartedAt: started.UTC()}
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
	if artifact.Manifest.Process != nil {
		artifact.Manifest.Process.ExitedAt = &now
		artifact.Manifest.Process.ExitCode = &exitCode
	}
	evidence := []string{"request.txt", artifact.Manifest.Capture.StdoutFile, artifact.Manifest.Capture.StderrFile}
	if result.OutputFile != nil {
		evidence = append(evidence, *result.OutputFile)
	}
	digest, err := evidenceDigest(artifact.Path, evidence)
	if err == nil {
		artifact.Manifest.Capture.EvidenceSHA256 = &digest
	}
	return s.writeManifest(artifact)
}

func (s Store) OpenStreams(artifact *Artifact) (*os.File, *os.File, error) {
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
	if err := os.MkdirAll(path, 0o700); err != nil {
		return err
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
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			return "", err
		}
		fileDigest := sha256.Sum256(data)
		fmt.Fprintf(hash, "%s  %s\n", hex.EncodeToString(fileDigest[:]), name)
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}
