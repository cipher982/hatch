package run

import (
	"bytes"
	"context"
	"fmt"
	"time"
)

type HTTPRequest struct {
	Surface, Provider, Model, CWD, Prompt string
	Timeout                               time.Duration
	CredentialNames                       []string
	Progress                              func(string)
	Execute                               func(context.Context, func([]byte) error) HTTPOutcome
}

type HTTPOutcome struct {
	Output        string
	Error         string
	NativeID      string
	NativeIDState string
	Retention     string
	Capabilities  map[string]string
	Warnings      []Warning
	Attempts      int
	LastStatus    int
	TimedOut      bool
}

func (c Coordinator) ExecuteHTTP(req HTTPRequest) PublicResult {
	started := c.Now()
	artifact, err := c.Store.Prepare(PreparedRun{
		Surface: req.Surface, Provider: req.Provider, Model: req.Model, CWD: effectiveCWD(req.CWD),
		Request: req.Prompt, RedactedArgv: []string{"POST", "<responses-url>"}, CredentialNames: req.CredentialNames,
		StructuredStdout: true, Execution: "http",
	})
	if err != nil {
		return failedResult(-3, started, c.Now(), fmt.Sprintf("prepare durable run: %v", err), nil)
	}
	artifactPath := artifact.Path
	if req.Progress != nil {
		req.Progress(fmt.Sprintf("[hatch] run %s artifact %s", artifact.Manifest.RunID, artifact.Path))
	}
	stdoutSink, stderrSink, err := c.Store.OpenStreams(artifact)
	if err != nil {
		return failedResult(-3, started, c.Now(), fmt.Sprintf("open durable streams: %v", err), &artifactPath)
	}
	warnings := []Warning{}
	if err := c.Store.MarkHTTPRunning(artifact, started); err != nil {
		artifact.Manifest.Capture.State = "degraded"
		warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: err.Error()})
	}
	var stdout bytes.Buffer
	capture := &captureWriter{memory: &stdout, sink: stdoutSink}
	record := func(snapshot []byte) error {
		if len(snapshot) == 0 {
			return nil
		}
		_, _ = capture.Write(snapshot)
		if snapshot[len(snapshot)-1] != '\n' {
			_, _ = capture.Write([]byte{'\n'})
		}
		return nil
	}
	ctx := context.Background()
	cancel := func() {}
	if req.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, req.Timeout)
	}
	defer cancel()
	outcome := req.Execute(ctx, record)
	if ctx.Err() == context.DeadlineExceeded {
		outcome.TimedOut = true
	}
	warnings = append(warnings, outcome.Warnings...)
	if capture.sinkErr != nil {
		artifact.Manifest.Capture.State = "degraded"
		warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: capture.sinkErr.Error()})
	}
	for _, sink := range []StreamSink{stdoutSink, stderrSink} {
		if err := sink.Sync(); err != nil {
			artifact.Manifest.Capture.State = "degraded"
			warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: err.Error()})
		}
		if err := sink.Close(); err != nil {
			artifact.Manifest.Capture.State = "degraded"
			warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: err.Error()})
		}
	}
	resultFile, writeErr := c.Store.WriteResult(artifact, []byte(outcome.Output))
	resultState := Result{Output: "absent", TerminalMarker: "not_applicable", OutputBytes: int64(len(outcome.Output))}
	if outcome.Output != "" {
		resultState.Output = "present"
	}
	if writeErr == nil {
		resultState.OutputFile = &resultFile
	} else {
		artifact.Manifest.Capture.State = "degraded"
		warnings = append(warnings, Warning{Code: "capture_persistence_failed", Message: writeErr.Error()})
	}
	terminalOutcome, exitCode, status := OutcomeSucceeded, 0, "ok"
	ok := outcome.Error == "" && !outcome.TimedOut && outcome.Output != ""
	var resultErr *string
	if outcome.TimedOut {
		terminalOutcome, exitCode, status = OutcomeTimedOut, -1, "timeout"
		message := outcome.Error
		if message == "" {
			message = fmt.Sprintf("HTTP execution timed out after %ds", int(req.Timeout.Seconds()))
		}
		resultErr = &message
	} else if outcome.Error != "" || outcome.Output == "" {
		terminalOutcome, exitCode, status = OutcomeFailed, 1, "error"
		message := outcome.Error
		if message == "" {
			message = "HTTP provider returned no final output"
		}
		resultErr = &message
	} else if len(warnings) > 0 {
		terminalOutcome = OutcomeSucceededWarnings
	}
	resultState.Error = resultErr
	state := State{Retention: outcome.Retention, NativeIDState: outcome.NativeIDState, Capabilities: outcome.Capabilities}
	if state.Retention == "" {
		state.Retention = "remote_provider"
	}
	if state.NativeIDState == "" {
		state.NativeIDState = "unavailable"
	}
	if state.Capabilities == nil {
		state.Capabilities = map[string]string{}
	}
	if outcome.NativeID != "" {
		state.NativeID = &outcome.NativeID
	}
	completed := c.Now().UTC()
	artifact.Manifest.HTTP.Attempts = outcome.Attempts
	artifact.Manifest.HTTP.CompletedAt = &completed
	if outcome.LastStatus != 0 {
		artifact.Manifest.HTTP.LastStatus = &outcome.LastStatus
	}
	if err := c.Store.CommitTerminal(artifact, terminalOutcome, exitCode, resultState, state, warnings); err != nil {
		_ = c.Store.MarkCaptureDegraded(artifact, Warning{Code: "capture_persistence_failed", Message: err.Error()})
	}
	result := PublicResult{OK: ok, Status: status, Output: outcome.Output, ExitCode: exitCode, DurationMS: c.Now().Sub(started).Milliseconds(), Error: resultErr, ArtifactPath: &artifactPath, Run: &artifact.Manifest, SessionID: state.NativeID}
	if artifact.Manifest.Capture.State != "durable" {
		result.ArtifactPath = nil
	}
	if err := c.Store.WritePublicProjection(artifact, result); err != nil {
		_ = c.Store.MarkCaptureDegraded(artifact, Warning{Code: "capture_persistence_failed", Message: err.Error()})
		result.ArtifactPath = nil
	}
	return result
}
