package run

import (
	"bytes"
	"context"
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

const publicCaptureLimit = 32 << 20

type Request struct {
	Context         context.Context
	Surface         string
	Backend         string
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
	ctx := req.Context
	if ctx == nil {
		ctx = context.Background()
	}
	redacted, err := validatedRedactedArgv(req.Invocation)
	if err != nil {
		return failedResult(-3, started, c.Now(), err.Error(), nil)
	}
	artifact, err := c.Store.Prepare(PreparedRun{
		Surface: req.Surface, Backend: req.Backend, Provider: req.Provider, Model: req.Model, CWD: effectiveCWD(req.CWD),
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
		return c.finalizePrelaunchFailure(artifact, started, -3, OutcomeFailed, message, []Warning{{Code: "capture_persistence_failed", Message: err.Error()}})
	}
	cleanupProviderState, err := prepareProviderState(artifact, &req.Invocation)
	if err != nil {
		_ = stdoutFile.Close()
		_ = stderrFile.Close()
		message := fmt.Sprintf("prepare provider state: %v", err)
		return c.finalizePrelaunchFailure(artifact, started, -3, OutcomeFailed, message, nil)
	}
	defer cleanupProviderState()

	if len(req.Invocation.Argv) == 0 {
		message := "provider invocation is empty"
		_ = stdoutFile.Close()
		_ = stderrFile.Close()
		return c.finalizePrelaunchFailure(artifact, started, -2, OutcomeLaunch, message, nil)
	}
	if err := ctx.Err(); err != nil {
		_ = stdoutFile.Close()
		_ = stderrFile.Close()
		return c.finalizePrelaunchFailure(artifact, started, -4, OutcomeCancelled, "Agent cancelled before provider launch", nil)
	}
	cmd := exec.Command(req.Invocation.Argv[0], req.Invocation.Argv[1:]...)
	configureProcess(cmd)
	cmd.WaitDelay = time.Second
	cmd.Dir = req.CWD
	cmd.Env = buildEnvironment(req.Invocation, artifact.Manifest.RunID, req.Automation)
	if req.Invocation.Stdin != nil {
		cmd.Stdin = bytes.NewReader(req.Invocation.Stdin)
	}
	stdout := newBoundedCapture(publicCaptureLimit)
	stderr := newBoundedCapture(publicCaptureLimit)
	stdoutCapture := &captureWriter{memory: stdout, sink: stdoutFile}
	if req.Progress != nil && req.Invocation.Adapter != "" && req.Invocation.Adapter != "raw" {
		progress := provider.NewProgressParser(req.Invocation.Adapter, req.ProgressLabel)
		stdoutCapture.observeLine = func(line []byte) {
			for _, message := range progress.Observe(line) {
				req.Progress(message)
			}
		}
	}
	stderrCapture := &captureWriter{memory: stderr, sink: stderrFile}
	cmd.Stdout = stdoutCapture
	cmd.Stderr = stderrCapture

	if err := cmd.Start(); err != nil {
		_ = stdoutFile.Close()
		_ = stderrFile.Close()
		return c.finalizePrelaunchFailure(artifact, started, -2, OutcomeLaunch, err.Error(), nil)
	}
	providerStarted := c.Now()
	processIdentity := processStartIdentity(cmd.Process.Pid)
	if err := c.Store.MarkRunning(artifact, cmd.Process.Pid, providerStarted, processIdentity); err != nil {
		warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: err.Error()})
		artifact.Manifest.Capture.State = "degraded"
	}
	// A persistence failure must not erase the in-memory ownership observation
	// needed to clean up a later timeout or cancellation. Store implementations
	// normally populate this while writing the running manifest, but the
	// coordinator retains the process truth independently of that write.
	if artifact.Manifest.Process == nil {
		processGroup := cmd.Process.Pid
		artifact.Manifest.Process = &Process{PID: cmd.Process.Pid, ProcessGroup: &processGroup, StartedAt: providerStarted.UTC()}
		if processIdentity != "" {
			artifact.Manifest.Process.StartIdentity = &processIdentity
		}
	}

	waited := make(chan error, 1)
	go func() { waited <- cmd.Wait() }()
	var waitErr error
	timedOut := false
	cancelled := false
	cleanupSignal := ""
	if req.Timeout > 0 {
		timer := time.NewTimer(req.Timeout)
		select {
		case waitErr = <-waited:
			timer.Stop()
		case <-timer.C:
			timedOut = true
			cleanupSignal, _ = killProcessGroup(cmd)
			waitErr = <-waited
		case <-ctx.Done():
			timer.Stop()
			cancelled = true
			cleanupSignal, _ = killProcessGroup(cmd)
			waitErr = <-waited
		}
	} else {
		select {
		case waitErr = <-waited:
		case <-ctx.Done():
			cancelled = true
			cleanupSignal, _ = killProcessGroup(cmd)
			waitErr = <-waited
		}
	}
	if timedOut || cancelled {
		cleanup := &TimeoutCleanup{
			Signal: cleanupSignal, WaitBounded: true, SurvivorState: "unknown",
			PipeClosureForced: errors.Is(waitErr, exec.ErrWaitDelay),
		}
		if timedOut {
			artifact.Manifest.Process.TimeoutCleanup = cleanup
		} else {
			artifact.Manifest.Process.CancelCleanup = cleanup
		}
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
	if stdout.Overflowed() {
		interpretation.Output = nil
		interpretation.Error = fmt.Sprintf("provider stdout exceeded the %d MiB public interpretation limit; full raw evidence is retained in the run artifact", publicCaptureLimit>>20)
		interpretation.TerminalMarker = "not_applicable"
	}
	for _, warning := range interpretation.Warnings {
		evidenceFile := artifact.Manifest.Capture.StdoutFile
		if warning.Code == "stderr_error_recovered" {
			evidenceFile = artifact.Manifest.Capture.StderrFile
		}
		warnings = append(warnings, Warning{Code: warning.Code, Message: warning.Message, EvidenceFile: &evidenceFile})
	}
	output := interpretation.Output
	// A provider completion already available when cancellation races with the
	// wait wins. Cancel cleanup remains in process evidence, but a complete
	// terminal answer is not relabelled or discarded.
	cancelledOutcome := cancellationWins(cancelled, exitCode, interpretation)
	if cancelledOutcome {
		exitCode = -4
	}
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
	ok := exitCode == 0 && !timedOut && !cancelledOutcome && interpretation.Error == "" && interpretation.TerminalMarker != "not_observed" && len(output) > 0
	status := "ok"
	var resultErr *string
	if timedOut {
		outcome = OutcomeTimedOut
		status = "timeout"
		message := fmt.Sprintf("Agent timed out after %ds", int(req.Timeout.Seconds()))
		resultErr = &message
	} else if cancelledOutcome {
		outcome = OutcomeCancelled
		status = "cancelled"
		message := "Agent cancelled"
		resultErr = &message
	} else if exitCode != 0 {
		outcome = OutcomeFailed
		status = "error"
		message := strings.TrimSpace(stderrText)
		if message == "" {
			message = fmt.Sprintf("Exit code %d", exitCode)
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
			if req.Invocation.ProviderVersion != "" {
				state.ProviderVersion = &req.Invocation.ProviderVersion
			}
			if state.NativeID != nil && req.Invocation.ProviderVersion != "" {
				dataEnv := "XDG_DATA_HOME=" + filepath.Join(artifact.Path, "provider", "opencode", "data")
				stateEnv := "XDG_STATE_HOME=" + filepath.Join(artifact.Path, "provider", "opencode", "state")
				state.InspectHint = &OperatorHint{
					Argv:         []string{"env", dataEnv, stateEnv, "opencode", "export", *state.NativeID},
					VersionBound: true, ProviderVersion: req.Invocation.ProviderVersion,
				}
				state.Capabilities["inspect"] = "supported_same_version"
				if timedOut {
					argv := []string{"env", dataEnv, stateEnv, "opencode", "run", "--dangerously-skip-permissions"}
					if req.CWD != "" {
						argv = append(argv, "--dir", req.CWD)
					}
					argv = append(argv, "--print-logs", "--log-level", "ERROR", "--format", "json", "-m", req.Model, "--session", *state.NativeID, "Return only the concise final answer from the evidence already gathered. Do not use tools or expand the investigation.")
					state.RecoveryHint = &OperatorHint{Argv: argv, VersionBound: true, ProviderVersion: req.Invocation.ProviderVersion, RequiresApprovalBypass: true}
					state.Capabilities["recovery_hint"] = "best_effort_same_version"
				}
			} else if state.NativeID != nil {
				state.Capabilities["inspect"] = "unsupported"
			}
		} else {
			state.Retention = "unavailable"
			state.Capabilities["snapshot"] = "unsupported"
		}
	}
	c.Store.StageTerminal(artifact, outcome, exitCode, resultState, state, warnings)
	stderrCopy := stderrText
	var publicStderr *string = &stderrCopy
	if timedOut {
		publicStderr = nil
	}
	result := PublicResult{
		OK: ok, Status: status, Output: string(output), ExitCode: exitCode,
		DurationMS: c.Now().Sub(started).Milliseconds(), Error: resultErr,
		Stderr: publicStderr, ArtifactPath: &artifactPath, Run: &artifact.Manifest,
	}
	if artifact.Manifest.Capture.State != "durable" {
		result.ArtifactPath = nil
	}
	result.SessionID = state.NativeID
	if state.RecoveryHint != nil {
		rendered := shellJoin(state.RecoveryHint.Argv)
		result.ResumeCommand = &rendered
	}
	if err := c.Store.WritePublicProjection(artifact, result); err != nil {
		warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: err.Error()})
		artifact.Manifest.Capture.State = "degraded"
		c.Store.StageTerminal(artifact, outcome, exitCode, resultState, state, warnings)
		result.ArtifactPath = nil
	}
	if err := c.Store.CommitTerminal(artifact, outcome, exitCode, resultState, state, warnings); err != nil {
		warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: err.Error()})
		artifact.Manifest.Capture.State = "degraded"
		c.Store.StageTerminal(artifact, outcome, exitCode, resultState, state, warnings)
		result.ArtifactPath = nil
		_ = c.Store.WritePublicProjection(artifact, result)
		_ = c.Store.CommitTerminal(artifact, outcome, exitCode, resultState, state, warnings)
	}
	if req.Progress != nil && req.ProgressLabel != "" && (req.Invocation.Adapter == "" || req.Invocation.Adapter == "raw") {
		req.Progress(fmt.Sprintf("[hatch] %s completed", req.ProgressLabel))
	}
	return result
}

func cancellationWins(requested bool, exitCode int, interpretation provider.Interpretation) bool {
	return requested && !(exitCode == 0 && interpretation.Error == "" && interpretation.TerminalMarker != "not_observed" && len(interpretation.Output) > 0)
}

func (c Coordinator) finalizePrelaunchFailure(artifact *Artifact, started time.Time, exitCode int, outcome Outcome, message string, warnings []Warning) PublicResult {
	resultFile, writeErr := c.Store.WriteResult(artifact, nil)
	resultState := Result{Output: "absent", TerminalMarker: "not_applicable", Error: &message}
	if writeErr == nil {
		resultState.OutputFile = &resultFile
	} else {
		artifact.Manifest.Capture.State = "degraded"
		warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: writeErr.Error()})
	}
	state := unknownState()
	c.Store.StageTerminal(artifact, outcome, exitCode, resultState, state, warnings)
	artifactPath := artifact.Path
	result := failedResult(exitCode, started, c.Now(), message, &artifactPath)
	if outcome == OutcomeCancelled {
		result.Status = "cancelled"
	} else if outcome == OutcomeTimedOut {
		result.Status = "timeout"
	}
	result.Run = &artifact.Manifest
	if artifact.Manifest.Capture.State != "durable" {
		result.ArtifactPath = nil
	}
	if err := c.Store.WritePublicProjection(artifact, result); err != nil {
		warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: err.Error()})
		artifact.Manifest.Capture.State = "degraded"
		c.Store.StageTerminal(artifact, outcome, exitCode, resultState, state, warnings)
		result.ArtifactPath = nil
	}
	if err := c.Store.CommitTerminal(artifact, outcome, exitCode, resultState, state, warnings); err != nil {
		warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: err.Error()})
		artifact.Manifest.Capture.State = "degraded"
		c.Store.StageTerminal(artifact, outcome, exitCode, resultState, state, warnings)
		result.ArtifactPath = nil
		_ = c.Store.WritePublicProjection(artifact, result)
		_ = c.Store.CommitTerminal(artifact, outcome, exitCode, resultState, state, warnings)
	}
	return result
}

func shellJoin(argv []string) string {
	quoted := make([]string, len(argv))
	for index, arg := range argv {
		if arg != "" && strings.IndexFunc(arg, func(r rune) bool {
			return !(r >= 'a' && r <= 'z') && !(r >= 'A' && r <= 'Z') && !(r >= '0' && r <= '9') && !strings.ContainsRune("_@%+=:,./-", r)
		}) == -1 {
			quoted[index] = arg
		} else {
			quoted[index] = "'" + strings.ReplaceAll(arg, "'", "'\"'\"'") + "'"
		}
	}
	return strings.Join(quoted, " ")
}

type captureWriter struct {
	memory      io.Writer
	sink        io.Writer
	sinkErr     error
	observeLine func([]byte)
	pending     []byte
}

type boundedCapture struct {
	buffer   bytes.Buffer
	limit    int
	overflow bool
}

func newBoundedCapture(limit int) *boundedCapture { return &boundedCapture{limit: limit} }

func (b *boundedCapture) Write(data []byte) (int, error) {
	remaining := b.limit - b.buffer.Len()
	if remaining > 0 {
		keep := len(data)
		if keep > remaining {
			keep = remaining
		}
		_, _ = b.buffer.Write(data[:keep])
	}
	if len(data) > remaining {
		b.overflow = true
	}
	return len(data), nil
}

func (b *boundedCapture) Bytes() []byte    { return b.buffer.Bytes() }
func (b *boundedCapture) String() string   { return b.buffer.String() }
func (b *boundedCapture) Overflowed() bool { return b.overflow }

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
		if len(w.pending)+len(data) > publicCaptureLimit {
			w.pending = nil
			w.observeLine = nil
			return len(data), nil
		}
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

func validatedRedactedArgv(invocation provider.Invocation) ([]string, error) {
	if invocation.RedactedArgv == nil {
		return append([]string(nil), invocation.Argv...), nil
	}
	if len(invocation.RedactedArgv) != len(invocation.Argv) {
		return nil, fmt.Errorf("provider invocation redaction metadata does not match argv")
	}
	return append([]string(nil), invocation.RedactedArgv...), nil
}

func unknownState() State {
	return State{Retention: "unknown", NativeIDState: "not_exposed", Capabilities: map[string]string{}}
}

func failedResult(exitCode int, started, finished time.Time, message string, artifactPath *string) PublicResult {
	status := "error"
	if exitCode == -2 {
		status = "not_found"
	}
	return PublicResult{OK: false, Status: status, Output: "", ExitCode: exitCode,
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
