package run

import (
	"encoding/hex"
	"fmt"
	"path/filepath"
	"strings"
)

var warningCodesV1 = map[string]bool{
	"capture_persistence_failed": true,
	"transient_provider_error":   true,
	"stderr_error_recovered":     true,
	"adapter_recognition_empty":  true,
}

// ValidateManifest enforces records emitted by the V1 writer. Compatibility
// readers intentionally normalize older/additive records separately.
func ValidateManifest(manifest Manifest) error {
	if manifest.SchemaVersion != 1 {
		return fmt.Errorf("unsupported schema version %d", manifest.SchemaVersion)
	}
	if manifest.Writer.Implementation != "go" || manifest.Writer.ContractRevision != 1 {
		return fmt.Errorf("unsupported writer contract %q revision %d", manifest.Writer.Implementation, manifest.Writer.ContractRevision)
	}
	for name, value := range map[string]string{
		"run_id": manifest.RunID, "surface": manifest.Surface, "backend": manifest.Backend,
		"provider": manifest.Provider, "model": manifest.Model,
	} {
		if strings.TrimSpace(value) == "" {
			return fmt.Errorf("%s is required", name)
		}
	}
	_, createdOffset := manifest.CreatedAt.Zone()
	_, updatedOffset := manifest.UpdatedAt.Zone()
	if manifest.CreatedAt.IsZero() || manifest.UpdatedAt.IsZero() || createdOffset != 0 || updatedOffset != 0 {
		return fmt.Errorf("created_at and updated_at must be nonzero UTC timestamps")
	}
	if !oneOfString(string(manifest.Lifecycle), "prepared", "running", "terminal") {
		return fmt.Errorf("invalid lifecycle %q", manifest.Lifecycle)
	}
	if manifest.Lifecycle == LifecycleTerminal {
		if manifest.Outcome == nil || !oneOfString(string(*manifest.Outcome), "succeeded", "succeeded_with_warnings", "failed", "timed_out", "cancelled", "launch_failed", "abandoned") {
			return fmt.Errorf("terminal manifest requires a valid outcome")
		}
	} else if manifest.Outcome != nil {
		return fmt.Errorf("nonterminal manifest cannot have an outcome")
	}
	if !oneOfString(manifest.Execution, "subprocess", "http") {
		return fmt.Errorf("invalid execution %q", manifest.Execution)
	}
	if manifest.Execution == "http" && manifest.Process != nil {
		return fmt.Errorf("HTTP manifest cannot contain process evidence")
	}
	if manifest.Execution == "subprocess" && manifest.HTTP != nil {
		return fmt.Errorf("subprocess manifest cannot contain HTTP evidence")
	}
	if manifest.Lifecycle == LifecycleRunning && manifest.Execution == "subprocess" && manifest.Process == nil {
		return fmt.Errorf("running subprocess requires process evidence")
	}
	if manifest.Lifecycle == LifecycleRunning && manifest.Execution == "http" && manifest.HTTP == nil {
		return fmt.Errorf("running HTTP execution requires HTTP evidence")
	}
	for name, value := range map[string]string{
		"request_file":           manifest.Invocation.RequestFile,
		"stdout_file":            manifest.Capture.StdoutFile,
		"stderr_file":            manifest.Capture.StderrFile,
		"evidence_manifest_file": manifest.Capture.EvidenceManifestFile,
	} {
		if !safeRelativeFile(value) {
			return fmt.Errorf("%s is unsafe: %q", name, value)
		}
	}
	if !validSHA256(manifest.Invocation.RequestSHA256) {
		return fmt.Errorf("request_sha256 must be a SHA-256 digest")
	}
	if manifest.Capture.ArtifactPath == "" || !filepath.IsAbs(manifest.Capture.ArtifactPath) {
		return fmt.Errorf("artifact_path must be absolute")
	}
	if !oneOfString(manifest.Capture.State, "durable", "degraded", "disabled") {
		return fmt.Errorf("invalid capture state %q", manifest.Capture.State)
	}
	if !oneOfString(manifest.Result.Output, "present", "absent") || !oneOfString(manifest.Result.TerminalMarker, "observed", "not_observed", "not_applicable") {
		return fmt.Errorf("invalid result axes")
	}
	if manifest.Lifecycle == LifecycleTerminal && manifest.Capture.State == "durable" {
		if manifest.Capture.EvidenceSHA256 == nil || !validSHA256(*manifest.Capture.EvidenceSHA256) {
			return fmt.Errorf("durable terminal manifest requires an evidence digest")
		}
		if manifest.Result.Output == "present" && manifest.Result.OutputFile == nil {
			return fmt.Errorf("durably captured output requires an output file")
		}
	}
	for name, value := range map[string]*string{
		"result.output_file":           manifest.Result.OutputFile,
		"provider_state.snapshot_path": manifest.ProviderState.SnapshotPath,
		"archive.receipt_file":         manifest.Archive.ReceiptFile,
	} {
		if value != nil && !safeRelativeFile(*value) {
			return fmt.Errorf("%s is unsafe: %q", name, *value)
		}
	}
	if !oneOfString(manifest.ProviderState.Retention, "hatch_preserved", "provider_owned", "remote_provider", "unavailable", "unknown") {
		return fmt.Errorf("invalid provider retention %q", manifest.ProviderState.Retention)
	}
	if !oneOfString(manifest.ProviderState.NativeIDState, "observed", "not_exposed", "unavailable", "unknown") {
		return fmt.Errorf("invalid native identity state %q", manifest.ProviderState.NativeIDState)
	}
	if manifest.ProviderState.Capabilities == nil {
		return fmt.Errorf("provider capabilities must be explicit")
	}
	if manifest.ProviderState.NativeIDState == "observed" && manifest.ProviderState.NativeID == nil {
		return fmt.Errorf("observed native identity requires native_id")
	}
	if !oneOfString(manifest.Archive.State, "not_requested", "pending", "acknowledged", "failed") {
		return fmt.Errorf("invalid archive state %q", manifest.Archive.State)
	}
	if manifest.Archive.State == "acknowledged" && manifest.Archive.ReceiptFile == nil {
		return fmt.Errorf("acknowledged archive requires a receipt")
	}
	for _, warning := range manifest.Warnings {
		if !warningCodesV1[warning.Code] {
			return fmt.Errorf("invalid warning code %q", warning.Code)
		}
		if warning.EvidenceFile != nil && !safeRelativeFile(*warning.EvidenceFile) {
			return fmt.Errorf("warning evidence path is unsafe: %q", *warning.EvidenceFile)
		}
	}
	return nil
}

func safeRelativeFile(name string) bool {
	return name != "" && !filepath.IsAbs(name) && filepath.Clean(name) == name && name != "." && name != ".." && !strings.HasPrefix(name, ".."+string(filepath.Separator))
}

func validSHA256(value string) bool {
	if len(value) != 64 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}
