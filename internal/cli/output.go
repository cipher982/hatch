package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/cipher982/hatch/internal/expert"
	runner "github.com/cipher982/hatch/internal/run"
)

const (
	defaultOutputBytes = 8192
	maxOutputBytes     = 32768
	maxResponseBytes   = 64 << 10
)

func parseOutputLimit(value string) (int, error) {
	n, err := strconv.Atoi(value)
	if err != nil || n < 256 || n > maxOutputBytes {
		return 0, fmt.Errorf("--max-output-bytes must be between 256 and %d; full output remains in the run artifact", maxOutputBytes)
	}
	return n, nil
}

// Display limits never change the coordinator result or its durable projection.
type resultDisplay struct {
	runner.PublicResult
	OutputTruncated   bool     `json:"output_truncated"`
	StderrTruncated   bool     `json:"stderr_truncated"`
	ErrorTruncated    bool     `json:"error_truncated"`
	MetadataTruncated bool     `json:"metadata_truncated"`
	ReadCommand       []string `json:"read_command,omitempty"`
	ManifestCommand   []string `json:"manifest_command,omitempty"`
	StderrCommand     []string `json:"stderr_command,omitempty"`
}

func displayResult(result runner.PublicResult, limit int) resultDisplay {
	d := resultDisplay{PublicResult: result}
	d.Output, d.OutputTruncated = runner.ClipText(result.Output, limit, false)
	if result.Stderr != nil {
		text, clipped := runner.ClipText(*result.Stderr, 2048, true)
		d.Stderr, d.StderrTruncated = &text, clipped
	}
	if result.Error != nil {
		text, clipped := runner.ClipText(*result.Error, 2048, true)
		d.Error, d.ErrorTruncated = &text, clipped
	}
	if result.Run != nil {
		d.Run, d.MetadataTruncated = displayManifest(result.Run)
		d.ReadCommand = []string{"hatch", "runs", "read", result.Run.RunID, "--part", "result", "--json"}
		d.ManifestCommand = []string{"hatch", "runs", "read", result.Run.RunID, "--part", "manifest", "--json"}
		if d.StderrTruncated || d.ErrorTruncated {
			d.StderrCommand = []string{"hatch", "runs", "read", result.Run.RunID, "--part", "stderr", "--json"}
		}
	}
	// Compatibility fields can contain arbitrarily large provider recovery hints.
	if d.ResumeCommand != nil {
		value, clipped := runner.ClipText(*d.ResumeCommand, 1024, false)
		d.ResumeCommand = &value
		d.MetadataTruncated = d.MetadataTruncated || clipped
	}
	return d
}

func renderResult(result runner.PublicResult, limit int, jsonOutput bool, stdout, stderr io.Writer) int {
	d := displayResult(result, limit)
	if jsonOutput {
		for {
			encoded, err := json.Marshal(d)
			if err != nil {
				return renderConfigError(true, stdout, stderr, err)
			}
			if len(encoded)+1 <= maxResponseBytes {
				if _, err := stdout.Write(append(encoded, '\n')); err != nil {
					return 1
				}
				break
			}
			if len(d.Output) > 256 {
				d.Output, _ = runner.ClipText(d.Output, len(d.Output)/2, false)
				d.OutputTruncated = true
				continue
			}
			// Pathological metadata is available through the separate manifest reader.
			d.Run = minimalManifest(result.Run)
			d.ArtifactPath, d.SessionID, d.ResumeCommand = nil, nil, nil
			d.MetadataTruncated = true
		}
	} else {
		if result.OK {
			fmt.Fprintln(stdout, strings.TrimRight(d.Output, "\n"))
		}
		if d.Error != nil {
			fmt.Fprintf(stderr, "Error: %s\n", *d.Error)
		}
		if d.OutputTruncated {
			fmt.Fprintf(stderr, "Output preview truncated; read the full result with: %s\n", commandText(d.ReadCommand))
		}
		if d.ErrorTruncated || d.StderrTruncated {
			fmt.Fprintf(stderr, "Diagnostics preview truncated; read: %s\n", commandText(d.StderrCommand))
		}
	}
	return result.CLIExitCode()
}

func minimalManifest(m *runner.Manifest) *runner.Manifest {
	if m == nil {
		return nil
	}
	title, _ := runner.ClipText(m.Title, 512, false)
	surface, _ := runner.ClipText(m.Surface, 128, false)
	backend, _ := runner.ClipText(m.Backend, 128, false)
	provider, _ := runner.ClipText(m.Provider, 128, false)
	model, _ := runner.ClipText(m.Model, 256, false)
	cwd, _ := runner.ClipText(m.CWD, 1024, false)
	execution, _ := runner.ClipText(m.Execution, 128, false)
	output, _ := runner.ClipText(m.Result.Output, 1024, false)
	marker, _ := runner.ClipText(m.Result.TerminalMarker, 128, false)
	var provenance *runner.Provenance
	if m.Provenance != nil {
		copy := *m.Provenance
		copy.CallerCWD, _ = runner.ClipText(copy.CallerCWD, 1024, false)
		copy.CallerKind, _ = runner.ClipText(copy.CallerKind, 128, false)
		copy.CallerSessionID, _ = runner.ClipText(copy.CallerSessionID, 512, false)
		copy.CallerRequestID, _ = runner.ClipText(copy.CallerRequestID, 512, false)
		copy.ParentRunID, _ = runner.ClipText(copy.ParentRunID, 512, false)
		provenance = &copy
	}
	return &runner.Manifest{
		SchemaVersion: m.SchemaVersion, Writer: m.Writer, RunID: m.RunID,
		CreatedAt: m.CreatedAt, UpdatedAt: m.UpdatedAt, Lifecycle: m.Lifecycle, Outcome: m.Outcome,
		Surface: surface, Backend: backend, Provider: provider, Model: model, Title: title,
		ReasoningPolicy: m.ReasoningPolicy, CWD: cwd, Provenance: provenance, Execution: execution,
		Result: runner.Result{Output: output, TerminalMarker: marker, OutputBytes: m.Result.OutputBytes},
	}
}

func displayManifest(m *runner.Manifest) (*runner.Manifest, bool) {
	// Ordinary manifests remain exact; large ones use an explicitly partial projection.
	// Error text can be the full provider stderr. Do not serialize it just to measure it.
	if m.Result.Error != nil && len(*m.Result.Error) > 2048 {
		return minimalManifest(m), true
	}
	if len(m.Warnings) > 16 || len(m.Invocation.RedactedArgv) > 64 {
		return minimalManifest(m), true
	}
	for _, warning := range m.Warnings {
		if len(warning.Message) > 2048 {
			return minimalManifest(m), true
		}
	}
	for _, arg := range m.Invocation.RedactedArgv {
		if len(arg) > 4096 {
			return minimalManifest(m), true
		}
	}
	data, err := json.Marshal(m)
	if err == nil && len(data) <= 12<<10 {
		return m, false
	}
	return minimalManifest(m), true
}

func renderExpertResult(result expert.Result, limit int, jsonOutput bool, stdout, stderr io.Writer) int {
	if !jsonOutput {
		public := runner.PublicResult{OK: result.OK, Status: result.Status, Output: result.Output, Error: result.Error, Run: result.Run}
		renderResult(public, limit, false, stdout, stderr)
		return result.ExitCode
	}
	copy := result
	var clipped bool
	copy.Output, clipped = runner.ClipText(result.Output, limit, false)
	metadataClipped := false
	if copy.Error != nil {
		text, c := runner.ClipText(*copy.Error, 2048, true)
		copy.Error = &text
		metadataClipped = c
	}
	if copy.Run != nil {
		var manifestClipped bool
		copy.Run, manifestClipped = displayManifest(copy.Run)
		metadataClipped = metadataClipped || manifestClipped
	}
	// Each evidence field gets its own allowance. A large citation payload must
	// not evict otherwise useful sources or usage.
	for _, field := range []struct {
		value any
		clear func()
	}{
		{copy.Citations, func() { copy.Citations = nil }},
		{copy.Sources, func() { copy.Sources = nil }},
		{copy.Usage, func() { copy.Usage = nil }},
	} {
		metadataBudget := 16 << 10
		if !metadataFits(field.value, &metadataBudget, 0) {
			field.clear()
			metadataClipped = true
		}
	}
	payload := struct {
		expert.Result
		OutputTruncated   bool     `json:"output_truncated"`
		MetadataTruncated bool     `json:"metadata_truncated"`
		ReadCommand       []string `json:"read_command,omitempty"`
		EvidenceCommand   []string `json:"evidence_command,omitempty"`
	}{Result: copy, OutputTruncated: clipped, MetadataTruncated: metadataClipped}
	if result.Run != nil {
		payload.ReadCommand = []string{"hatch", "runs", "read", result.Run.RunID, "--part", "result", "--json"}
	}
	if result.Run != nil {
		payload.EvidenceCommand = []string{"hatch", "runs", "read", result.Run.RunID, "--part", "stdout", "--json"}
	}
	minimalized := false
	for {
		data, err := json.Marshal(payload)
		if err != nil {
			return renderConfigError(true, stdout, stderr, err)
		}
		if len(data)+1 <= maxResponseBytes {
			if _, err := stdout.Write(append(data, '\n')); err != nil {
				return 1
			}
			return result.ExitCode
		}
		if len(payload.Output) > 256 {
			payload.Output, _ = runner.ClipText(payload.Output, len(payload.Output)/2, false)
			payload.OutputTruncated = true
			continue
		}
		// Remove only the metadata field that still prevents a bounded response.
		// The other fields remain useful and are independently recoverable from
		// the stored response.
		switch {
		case payload.Citations != nil:
			payload.Citations = nil
		case payload.Sources != nil:
			payload.Sources = nil
		case payload.Usage != nil:
			payload.Usage = nil
		default:
			if !minimalized {
				payload.Run = minimalManifest(result.Run)
				payload.Model, _ = runner.ClipText(payload.Model, 512, false)
				payload.ResolvedModel = nil
				payload.ResponseID = nil
				payload.ArtifactPath = nil
				minimalized = true
			} else {
				payload.Run = nil
				payload.ReadCommand = nil
				payload.EvidenceCommand = nil
			}
		}
		payload.MetadataTruncated = true
	}
}

const (
	maxMetadataDepth = 16
	maxMetadataNodes = 8192
)

// metadataFits applies cheap shape limits before measuring the actual JSON
// representation. Responses are already decoded, so charging six bytes per
// source byte needlessly discarded ordinary citations.
func metadataFits(value any, budget *int, depth int) bool {
	if budget == nil || *budget < 0 || depth > maxMetadataDepth {
		return false
	}
	nodes, stringBytes := 0, 0
	var guard func(any, int) bool
	guard = func(value any, depth int) bool {
		if depth > maxMetadataDepth || nodes >= maxMetadataNodes {
			return false
		}
		nodes++
		switch v := value.(type) {
		case nil, bool, float64, float32, int, int64, uint64, json.Number:
			return true
		case string:
			stringBytes += len(v)
			return stringBytes <= *budget
		case []map[string]any:
			for _, item := range v {
				if !guard(item, depth+1) {
					return false
				}
			}
		case []any:
			for _, item := range v {
				if !guard(item, depth+1) {
					return false
				}
			}
		case map[string]any:
			for key, item := range v {
				if !guard(key, depth+1) || !guard(item, depth+1) {
					return false
				}
			}
		default:
			return false
		}
		return true
	}
	if !guard(value, depth) {
		return false
	}
	encoded, err := json.Marshal(value)
	if err != nil || len(encoded) > *budget {
		return false
	}
	*budget -= len(encoded)
	return true
}

func commandText(args []string) string {
	parts := make([]string, len(args))
	for i, arg := range args {
		if arg != "" && !strings.ContainsAny(arg, " \t\n'\"$`\\;&|()<>*?[]{}!") {
			parts[i] = arg
		} else {
			parts[i] = "'" + strings.ReplaceAll(arg, "'", "'\\''") + "'"
		}
	}
	return strings.Join(parts, " ")
}

func newProgressSink(w io.Writer) func(string) {
	remaining := 16 << 10
	notified := false
	return func(message string) {
		if remaining <= 0 {
			if !notified {
				fmt.Fprintln(w, "[hatch] progress preview limit reached; inspect the announced run ID for durable results")
				notified = true
			}
			return
		}
		message, _ = runner.ClipText(message, min(remaining, 1024), false)
		fmt.Fprintln(w, message)
		remaining -= len(message) + 1
	}
}
