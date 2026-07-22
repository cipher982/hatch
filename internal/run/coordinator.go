package run

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/cipher982/hatch/internal/provider"
)

type Coordinator struct {
	Store RunStore
	Now   func() time.Time
}

type Request struct {
	Surface         string
	Provider        string
	Model           string
	CWD             string
	Prompt          string
	Timeout         time.Duration
	Invocation      provider.Invocation
	CredentialNames []string
	Automation      bool
	Progress        func(string)
	ProgressLabel   string
}

func NewCoordinator(store RunStore) Coordinator {
	return Coordinator{Store: store, Now: time.Now}
}

func (c Coordinator) Execute(req Request) PublicResult {
	started := c.Now()
	redacted := redactArgv(req.Invocation.Argv)
	artifact, err := c.Store.Prepare(PreparedRun{
		Surface: req.Surface, Provider: req.Provider, Model: req.Model, CWD: effectiveCWD(req.CWD),
		Request: req.Prompt, RedactedArgv: redacted, CredentialNames: req.CredentialNames,
		StructuredStdout: req.Invocation.StreamFormat == "jsonl",
	})
	if err != nil {
		return failedResult(-3, started, c.Now(), fmt.Sprintf("prepare durable run: %v", err), nil)
	}
	artifactPath := artifact.Path
	if req.Progress != nil {
		req.Progress(fmt.Sprintf("[hatch] run %s artifact %s", artifact.Manifest.RunID, artifact.Path))
		if req.ProgressLabel != "" && (req.Invocation.Adapter == "" || req.Invocation.Adapter == "raw") {
			req.Progress(fmt.Sprintf("[hatch] %s started", req.ProgressLabel))
		}
	}
	stderrText := ""
	warnings := []Warning{}

	stdoutFile, stderrFile, err := c.Store.OpenStreams(artifact)
	if err != nil {
		message := fmt.Sprintf("open durable streams: %v", err)
		return failedResult(-3, started, c.Now(), message, &artifactPath)
	}
	cleanupProviderState, err := prepareProviderState(artifact, &req.Invocation)
	if err != nil {
		_ = stdoutFile.Close()
		_ = stderrFile.Close()
		message := fmt.Sprintf("prepare provider state: %v", err)
		return failedResult(-3, started, c.Now(), message, &artifactPath)
	}
	defer cleanupProviderState()

	if len(req.Invocation.Argv) == 0 {
		message := "provider invocation is empty"
		return failedResult(-2, started, c.Now(), message, &artifactPath)
	}
	cmd := exec.Command(req.Invocation.Argv[0], req.Invocation.Argv[1:]...)
	configureProcess(cmd)
	cmd.Dir = req.CWD
	cmd.Env = buildEnvironment(req.Invocation, artifact.Manifest.RunID, req.Automation)
	if req.Invocation.Stdin != nil {
		cmd.Stdin = bytes.NewReader(req.Invocation.Stdin)
	}
	var stdout, stderr bytes.Buffer
	stdoutCapture := &captureWriter{memory: &stdout, sink: stdoutFile}
	if req.Progress != nil && req.Invocation.Adapter != "" && req.Invocation.Adapter != "raw" {
		progress := provider.NewProgressParser(req.Invocation.Adapter, req.ProgressLabel)
		stdoutCapture.observeLine = func(line []byte) {
			for _, message := range progress.Observe(line) {
				req.Progress(message)
			}
		}
	}
	stderrCapture := &captureWriter{memory: &stderr, sink: stderrFile}
	cmd.Stdout = stdoutCapture
	cmd.Stderr = stderrCapture

	if err := cmd.Start(); err != nil {
		_ = stdoutFile.Close()
		_ = stderrFile.Close()
		message := err.Error()
		outcome := OutcomeLaunch
		resultState := Result{Output: "absent", TerminalMarker: "not_applicable", Error: &message}
		_ = c.Store.CommitTerminal(artifact, outcome, -2, resultState, unknownState(), nil)
		result := failedResult(-2, started, c.Now(), message, &artifactPath)
		result.Run = &artifact.Manifest
		return result
	}
	if err := c.Store.MarkRunning(artifact, cmd.Process.Pid, c.Now(), processStartIdentity(cmd.Process.Pid)); err != nil {
		warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: err.Error()})
		artifact.Manifest.Capture.State = "degraded"
	}

	waited := make(chan error, 1)
	go func() { waited <- cmd.Wait() }()
	var waitErr error
	timedOut := false
	if req.Timeout > 0 {
		timer := time.NewTimer(req.Timeout)
		select {
		case waitErr = <-waited:
			timer.Stop()
		case <-timer.C:
			timedOut = true
			_ = killProcessGroup(cmd)
			waitErr = <-waited
		}
	} else {
		waitErr = <-waited
	}
	for _, capture := range []*captureWriter{stdoutCapture, stderrCapture} {
		capture.Flush()
		if capture.sinkErr != nil {
			warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: capture.sinkErr.Error()})
			artifact.Manifest.Capture.State = "degraded"
		}
	}
	for _, file := range []StreamSink{stdoutFile, stderrFile} {
		if err := file.Sync(); err != nil {
			warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: err.Error()})
			artifact.Manifest.Capture.State = "degraded"
		}
		if err := file.Close(); err != nil {
			warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: err.Error()})
			artifact.Manifest.Capture.State = "degraded"
		}
	}
	stderrText = stderr.String()
	exitCode := processExitCode(waitErr)
	if timedOut {
		exitCode = -1
	}

	interpretation := provider.Interpret(req.Invocation.Adapter, stdout.Bytes(), stderr.Bytes())
	for _, message := range interpretation.Warnings {
		warnings = append(warnings, Warning{Code: "provider_warning", Message: message})
	}
	output := interpretation.Output
	resultFile, resultWriteErr := c.Store.WriteResult(artifact, output)
	if resultWriteErr != nil {
		warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: resultWriteErr.Error()})
		artifact.Manifest.Capture.State = "degraded"
	}
	resultState := Result{Output: "absent", TerminalMarker: interpretation.TerminalMarker, OutputBytes: int64(len(output))}
	if len(output) > 0 {
		resultState.Output = "present"
	}
	if resultWriteErr == nil {
		resultState.OutputFile = &resultFile
	}

	outcome := OutcomeSucceeded
	ok := exitCode == 0 && !timedOut && interpretation.Error == "" && interpretation.TerminalMarker != "not_observed" && len(output) > 0
	status := "ok"
	var resultErr *string
	if timedOut {
		outcome = OutcomeTimedOut
		status = "timeout"
		message := fmt.Sprintf("Process timed out after %ds", int(req.Timeout.Seconds()))
		resultErr = &message
	} else if exitCode != 0 {
		outcome = OutcomeFailed
		status = "error"
		message := strings.TrimSpace(stderrText)
		if message == "" {
			message = fmt.Sprintf("Process exited with code %d", exitCode)
		} else {
			message = stderrText
		}
		resultErr = &message
	} else if interpretation.Error != "" {
		outcome = OutcomeFailed
		status = "error"
		resultErr = &interpretation.Error
	} else if interpretation.TerminalMarker == "not_observed" {
		outcome = OutcomeFailed
		status = "error"
		message := "structured provider output did not contain a terminal marker"
		resultErr = &message
	} else if len(output) == 0 {
		outcome = OutcomeFailed
		status = "error"
		message := "Empty output from agent"
		resultErr = &message
	} else if len(interpretation.Warnings) > 0 {
		outcome = OutcomeSucceededWarnings
	}
	resultState.Error = resultErr
	state := State{
		Retention: interpretation.Retention, NativeIDState: interpretation.NativeIDState,
		Capabilities: interpretation.Capabilities,
	}
	if interpretation.NativeID != "" {
		state.NativeID = &interpretation.NativeID
	}
	if req.Invocation.Adapter == "opencode" {
		approved, pruneErr := pruneOpenCodeState(artifact)
		if pruneErr != nil {
			warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: pruneErr.Error()})
			artifact.Manifest.Capture.State = "degraded"
		}
		if approved > 0 {
			snapshot := "provider/opencode"
			state.SnapshotPath = &snapshot
			state.Retention = "hatch_preserved"
			state.Capabilities["snapshot"] = "supported"
		} else {
			state.Retention = "unavailable"
			state.Capabilities["snapshot"] = "unsupported"
		}
	}
	if err := c.Store.CommitTerminal(artifact, outcome, exitCode, resultState, state, warnings); err != nil {
		warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: err.Error()})
		artifact.Manifest.Capture.State = "degraded"
	}

	stderrCopy := stderrText
	result := PublicResult{
		OK: ok, Status: status, Output: string(output), ExitCode: exitCode,
		DurationMS: c.Now().Sub(started).Milliseconds(), Error: resultErr,
		Stderr: &stderrCopy, ArtifactPath: &artifactPath, Run: &artifact.Manifest,
	}
	if artifact.Manifest.Capture.State != "durable" {
		result.ArtifactPath = nil
	}
	result.SessionID = state.NativeID
	if err := c.Store.WritePublicProjection(artifact, result); err != nil {
		_ = c.Store.MarkCaptureDegraded(artifact, Warning{Code: "capture_persistence_failed", Message: err.Error()})
		result.ArtifactPath = nil
	}
	if req.Progress != nil && req.ProgressLabel != "" && (req.Invocation.Adapter == "" || req.Invocation.Adapter == "raw") {
		req.Progress(fmt.Sprintf("[hatch] %s completed", req.ProgressLabel))
	}
	return result
}

type captureWriter struct {
	memory      *bytes.Buffer
	sink        io.Writer
	sinkErr     error
	observeLine func([]byte)
	pending     []byte
}

func (w *captureWriter) Write(data []byte) (int, error) {
	_, _ = w.memory.Write(data)
	if w.sinkErr == nil {
		if written, err := w.sink.Write(data); err != nil {
			w.sinkErr = err
		} else if written != len(data) {
			w.sinkErr = io.ErrShortWrite
		}
	}
	if w.observeLine != nil {
		w.pending = append(w.pending, data...)
		for {
			index := bytes.IndexByte(w.pending, '\n')
			if index < 0 {
				break
			}
			line := append([]byte(nil), w.pending[:index]...)
			w.pending = w.pending[index+1:]
			w.observeLine(line)
		}
	}
	// Storage degradation must not make os/exec stop draining provider output.
	return len(data), nil
}

func (w *captureWriter) Flush() {
	if w.observeLine != nil && len(w.pending) > 0 {
		w.observeLine(append([]byte(nil), w.pending...))
		w.pending = nil
	}
}

func buildEnvironment(invocation provider.Invocation, runID string, automation bool) []string {
	values := make(map[string]string)
	for _, entry := range os.Environ() {
		if index := strings.IndexByte(entry, '='); index >= 0 {
			values[entry[:index]] = entry[index+1:]
		}
	}
	delete(values, "DCG_BYPASS")
	values["DCG_NO_SELF_HEAL"] = "1"
	values["LONGHOUSE_HATCH_RUN_ID"] = runID
	if automation {
		values["LONGHOUSE_IS_SIDECHAIN"] = "1"
		values["LONGHOUSE_ORIGIN_KIND"] = "hatch_automation"
		copyFirstEnvironment(values, "LONGHOUSE_PARENT_SESSION_ID", "LONGHOUSE_MANAGED_SESSION_ID", "LONGHOUSE_SESSION_ID", "LONGHOUSE_CHANNEL_SESSION_ID")
		copyFirstEnvironment(values, "LONGHOUSE_PARENT_THREAD_ID", "LONGHOUSE_THREAD_ID")
		copyFirstEnvironment(values, "LONGHOUSE_PARENT_PROVIDER_SESSION_ID", "LONGHOUSE_PROVIDER_SESSION_ID")
		if strings.TrimSpace(values["LONGHOUSE_OPENCODE_SESSION_METADATA_ROOT"]) == "" {
			home := strings.TrimSpace(values["LONGHOUSE_HOME"])
			if home == "" {
				home = filepath.Join(strings.TrimSpace(values["HOME"]), ".longhouse")
			}
			values["LONGHOUSE_OPENCODE_SESSION_METADATA_ROOT"] = filepath.Join(home, "provider-session-metadata", "opencode")
		}
	}
	for _, name := range invocation.UnsetEnv {
		delete(values, name)
	}
	for name, value := range invocation.SetEnv {
		values[name] = value
	}
	result := make([]string, 0, len(values))
	for name, value := range values {
		result = append(result, name+"="+value)
	}
	return result
}

func copyFirstEnvironment(values map[string]string, target string, sources ...string) {
	for _, source := range sources {
		if value := strings.TrimSpace(values[source]); value != "" {
			values[target] = value
			return
		}
	}
}

func redactArgv(argv []string) []string {
	result := append([]string(nil), argv...)
	for index, value := range result {
		if strings.HasPrefix(value, "Hatch execution contract:\n") {
			result[index] = "<prompt>"
		}
	}
	return result
}

func unknownState() State {
	return State{Retention: "unknown", NativeIDState: "not_exposed", Capabilities: map[string]string{}}
}

func failedResult(exitCode int, started, finished time.Time, message string, artifactPath *string) PublicResult {
	return PublicResult{OK: false, Status: "error", Output: "", ExitCode: exitCode,
		DurationMS: finished.Sub(started).Milliseconds(), Error: &message, ArtifactPath: artifactPath}
}

func processExitCode(err error) int {
	if err == nil {
		return 0
	}
	var exit *exec.ExitError
	if errors.As(err, &exit) {
		return exit.ExitCode()
	}
	return 1
}

func effectiveCWD(cwd string) string {
	if cwd != "" {
		if absolute, err := filepath.Abs(cwd); err == nil {
			return absolute
		}
		return cwd
	}
	current, err := os.Getwd()
	if err != nil {
		return ""
	}
	return current
}

func prepareProviderState(artifact *Artifact, invocation *provider.Invocation) (func(), error) {
	if invocation.Adapter != "opencode" {
		return func() {}, nil
	}
	if invocation.SetEnv == nil {
		invocation.SetEnv = map[string]string{}
	}
	root := filepath.Join(artifact.Path, "provider", "opencode")
	for name, envName := range map[string]string{
		"data": "XDG_DATA_HOME", "state": "XDG_STATE_HOME",
	} {
		path := filepath.Join(root, name)
		if err := os.MkdirAll(path, 0o700); err != nil {
			return nil, err
		}
		if err := os.Chmod(path, 0o700); err != nil {
			return nil, err
		}
		invocation.SetEnv[envName] = path
	}
	cache, err := os.MkdirTemp("", "hatch-opencode-cache-*")
	if err != nil {
		return nil, err
	}
	if err := os.Chmod(cache, 0o700); err != nil {
		os.RemoveAll(cache)
		return nil, err
	}
	invocation.SetEnv["XDG_CACHE_HOME"] = cache
	return func() { _ = os.RemoveAll(cache) }, nil
}

var openCodeStateAllowlist = map[string]bool{
	"data/opencode/opencode.db":     true,
	"data/opencode/opencode.db-shm": true,
	"data/opencode/opencode.db-wal": true,
	"data/opencode/session.db":      true, // Hermetic compatibility oracle.
}

func pruneOpenCodeState(artifact *Artifact) (int, error) {
	root := filepath.Join(artifact.Path, "provider", "opencode")
	approved := 0
	err := filepath.WalkDir(root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			return nil
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		relative = filepath.ToSlash(relative)
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if openCodeStateAllowlist[relative] && info.Mode().IsRegular() {
			approved++
			return os.Chmod(path, 0o600)
		}
		return os.Remove(path)
	})
	return approved, err
}
