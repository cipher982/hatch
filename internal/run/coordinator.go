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
	"sync"
	"time"

	"github.com/cipher982/hatch/internal/provider"
)

type Coordinator struct {
	Store Store
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
}

func NewCoordinator(store Store) Coordinator {
	return Coordinator{Store: store, Now: time.Now}
}

func (c Coordinator) Execute(req Request) PublicResult {
	started := c.Now()
	redacted := redactArgv(req.Invocation.Argv)
	artifact, err := c.Store.Prepare(req.Surface, req.Provider, req.Model, effectiveCWD(req.CWD), req.Prompt, redacted, req.CredentialNames)
	if err != nil {
		return failedResult(-3, started, c.Now(), fmt.Sprintf("prepare durable run: %v", err), nil)
	}
	artifactPath := artifact.Path
	stderrText := ""
	warnings := []Warning{}

	stdoutFile, stderrFile, err := c.Store.OpenStreams(artifact)
	if err != nil {
		message := fmt.Sprintf("open durable streams: %v", err)
		return failedResult(-3, started, c.Now(), message, &artifactPath)
	}
	defer stdoutFile.Close()
	defer stderrFile.Close()

	if len(req.Invocation.Argv) == 0 {
		message := "provider invocation is empty"
		return failedResult(-2, started, c.Now(), message, &artifactPath)
	}
	cmd := exec.Command(req.Invocation.Argv[0], req.Invocation.Argv[1:]...)
	configureProcess(cmd)
	cmd.Dir = req.CWD
	cmd.Env = buildEnvironment(req.Invocation, artifact.Manifest.RunID)
	if req.Invocation.Stdin != nil {
		cmd.Stdin = bytes.NewReader(req.Invocation.Stdin)
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return failedResult(-2, started, c.Now(), err.Error(), &artifactPath)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return failedResult(-2, started, c.Now(), err.Error(), &artifactPath)
	}

	if err := cmd.Start(); err != nil {
		message := err.Error()
		outcome := OutcomeLaunch
		resultState := Result{Output: "absent", TerminalMarker: "not_applicable", Error: &message}
		_ = c.Store.CommitTerminal(artifact, outcome, -2, resultState, unknownState(), nil)
		result := failedResult(-2, started, c.Now(), message, &artifactPath)
		result.Run = &artifact.Manifest
		return result
	}
	if err := c.Store.MarkRunning(artifact, cmd.Process.Pid, c.Now()); err != nil {
		warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: err.Error()})
		artifact.Manifest.Capture.State = "degraded"
	}

	var stdout, stderr bytes.Buffer
	var copies sync.WaitGroup
	copies.Add(2)
	go copyStream(&copies, io.MultiWriter(stdoutFile, &stdout), stdoutPipe)
	go copyStream(&copies, io.MultiWriter(stderrFile, &stderr), stderrPipe)

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
	copies.Wait()
	_ = stdoutFile.Sync()
	_ = stderrFile.Sync()
	stderrText = stderr.String()
	exitCode := processExitCode(waitErr)
	if timedOut {
		exitCode = -1
	}

	output := stdout.Bytes()
	resultFile, resultWriteErr := c.Store.WriteResult(artifact, output)
	if resultWriteErr != nil {
		warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: resultWriteErr.Error()})
		artifact.Manifest.Capture.State = "degraded"
	}
	resultState := Result{Output: "absent", TerminalMarker: "not_applicable", OutputBytes: int64(len(output))}
	if len(output) > 0 {
		resultState.Output = "present"
	}
	if resultWriteErr == nil {
		resultState.OutputFile = &resultFile
	}

	outcome := OutcomeSucceeded
	ok := exitCode == 0 && !timedOut
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
	}
	resultState.Error = resultErr
	if err := c.Store.CommitTerminal(artifact, outcome, exitCode, resultState, unknownState(), warnings); err != nil {
		warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: err.Error()})
		artifact.Manifest.Capture.State = "degraded"
	}

	stderrCopy := stderrText
	result := PublicResult{
		OK: ok, Status: status, Output: string(output), ExitCode: exitCode,
		DurationMS: c.Now().Sub(started).Milliseconds(), Error: resultErr,
		Stderr: &stderrCopy, ArtifactPath: &artifactPath, Run: &artifact.Manifest,
	}
	if err := c.Store.WritePublicProjection(artifact, result); err != nil {
		artifact.Manifest.Capture.State = "degraded"
		artifact.Manifest.Warnings = append(artifact.Manifest.Warnings, Warning{Code: "capture_persistence_failed", Message: err.Error()})
		_ = c.Store.writeManifest(artifact)
	}
	return result
}

func copyStream(group *sync.WaitGroup, destination io.Writer, source io.Reader) {
	defer group.Done()
	_, _ = io.Copy(destination, source)
}

func buildEnvironment(invocation provider.Invocation, runID string) []string {
	values := make(map[string]string)
	for _, entry := range os.Environ() {
		if index := strings.IndexByte(entry, '='); index >= 0 {
			values[entry[:index]] = entry[index+1:]
		}
	}
	delete(values, "DCG_BYPASS")
	values["DCG_NO_SELF_HEAL"] = "1"
	values["LONGHOUSE_HATCH_RUN_ID"] = runID
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
