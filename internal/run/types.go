package run

import (
	"encoding/json"
	"time"
)

type Lifecycle string
type Outcome string

const (
	LifecyclePrepared Lifecycle = "prepared"
	LifecycleRunning  Lifecycle = "running"
	LifecycleTerminal Lifecycle = "terminal"

	OutcomeSucceeded         Outcome = "succeeded"
	OutcomeSucceededWarnings Outcome = "succeeded_with_warnings"
	OutcomeFailed            Outcome = "failed"
	OutcomeTimedOut          Outcome = "timed_out"
	OutcomeCancelled         Outcome = "cancelled"
	OutcomeLaunch            Outcome = "launch_failed"
)

type Manifest struct {
	SchemaVersion int        `json:"schema_version"`
	Writer        Writer     `json:"writer"`
	RunID         string     `json:"run_id"`
	CreatedAt     time.Time  `json:"created_at"`
	UpdatedAt     time.Time  `json:"updated_at"`
	Lifecycle     Lifecycle  `json:"lifecycle"`
	Outcome       *Outcome   `json:"outcome"`
	Surface       string     `json:"surface"`
	Backend       string     `json:"backend"`
	Provider      string     `json:"provider"`
	Model         string     `json:"model"`
	CWD           string     `json:"cwd"`
	Execution     string     `json:"execution"`
	Invocation    Invocation `json:"invocation"`
	Process       *Process   `json:"process"`
	HTTP          *HTTP      `json:"http,omitempty"`
	Result        Result     `json:"result"`
	Capture       Capture    `json:"capture"`
	ProviderState State      `json:"provider_state"`
	Archive       Archive    `json:"archive"`
	Warnings      []Warning  `json:"warnings"`
}

type Writer struct {
	Implementation   string `json:"implementation"`
	ContractRevision int    `json:"contract_revision"`
}

type HTTP struct {
	StartedAt   time.Time  `json:"started_at"`
	CompletedAt *time.Time `json:"completed_at"`
	Attempts    int        `json:"attempts"`
	LastStatus  *int       `json:"last_status"`
}

type Invocation struct {
	RequestFile        string   `json:"request_file"`
	RequestSHA256      string   `json:"request_sha256"`
	RedactedArgv       []string `json:"redacted_argv"`
	CredentialEnvNames []string `json:"credential_env_names"`
}

type Process struct {
	PID            int             `json:"pid"`
	ProcessGroup   *int            `json:"process_group_id,omitempty"`
	StartIdentity  *string         `json:"start_identity,omitempty"`
	StartedAt      time.Time       `json:"started_at"`
	ExitedAt       *time.Time      `json:"exited_at"`
	ExitCode       *int            `json:"exit_code"`
	TimeoutCleanup *TimeoutCleanup `json:"timeout_cleanup,omitempty"`
	CancelCleanup  *TimeoutCleanup `json:"cancel_cleanup,omitempty"`
}

type TimeoutCleanup struct {
	Signal            string `json:"signal"`
	WaitBounded       bool   `json:"wait_bounded"`
	PipeClosureForced bool   `json:"pipe_closure_forced"`
	SurvivorState     string `json:"survivor_state"`
}

type Result struct {
	Output         string  `json:"output"`
	TerminalMarker string  `json:"terminal_marker"`
	OutputBytes    int64   `json:"output_bytes"`
	OutputFile     *string `json:"output_file"`
	Error          *string `json:"error"`
}

type Capture struct {
	State                string  `json:"state"`
	ArtifactPath         string  `json:"artifact_path"`
	EvidenceManifestFile string  `json:"evidence_manifest_file"`
	EvidenceSHA256       *string `json:"evidence_sha256"`
	StdoutFile           string  `json:"stdout_file"`
	StderrFile           string  `json:"stderr_file"`
}

type State struct {
	Retention       string            `json:"retention"`
	NativeID        *string           `json:"native_id"`
	NativeIDState   string            `json:"native_id_state"`
	Capabilities    map[string]string `json:"capabilities"`
	SnapshotPath    *string           `json:"snapshot_path"`
	ProviderVersion *string           `json:"provider_tool_version,omitempty"`
	InspectHint     *OperatorHint     `json:"inspect_hint,omitempty"`
	RecoveryHint    *OperatorHint     `json:"recovery_hint,omitempty"`
}

// UnmarshalJSON accepts the brief Go-preview spelling so artifacts written
// before the V1 field-name audit remain readable without rewriting them.
func (s *State) UnmarshalJSON(data []byte) error {
	type stateWire State
	var current stateWire
	if err := json.Unmarshal(data, &current); err != nil {
		return err
	}
	if current.ProviderVersion == nil {
		var legacy struct {
			ProviderVersion *string `json:"provider_version"`
		}
		if err := json.Unmarshal(data, &legacy); err != nil {
			return err
		}
		current.ProviderVersion = legacy.ProviderVersion
	}
	*s = State(current)
	return nil
}

// MarshalJSON keeps the short-lived Go-preview spelling additive while V1
// consumers move to provider_tool_version.
func (s State) MarshalJSON() ([]byte, error) {
	type stateWire State
	return json.Marshal(struct {
		stateWire
		PreviewProviderVersion *string `json:"provider_version,omitempty"`
	}{stateWire: stateWire(s), PreviewProviderVersion: s.ProviderVersion})
}

type OperatorHint struct {
	Argv                   []string `json:"argv"`
	VersionBound           bool     `json:"version_bound"`
	ProviderVersion        string   `json:"provider_version,omitempty"`
	RequiresApprovalBypass bool     `json:"requires_approval_bypass"`
}

type Archive struct {
	State       string  `json:"state"`
	ReceiptFile *string `json:"receipt_file"`
}

type Warning struct {
	Code         string  `json:"code"`
	Message      string  `json:"message"`
	EvidenceFile *string `json:"evidence_file"`
}

type PublicResult struct {
	OK            bool      `json:"ok"`
	Status        string    `json:"status"`
	Output        string    `json:"output"`
	ExitCode      int       `json:"exit_code"`
	DurationMS    int64     `json:"duration_ms"`
	Error         *string   `json:"error"`
	Stderr        *string   `json:"stderr"`
	ArtifactPath  *string   `json:"artifact_path"`
	SessionID     *string   `json:"session_id"`
	ResumeCommand *string   `json:"resume_command"`
	Run           *Manifest `json:"run,omitempty"`
}

func (r PublicResult) CLIExitCode() int {
	if r.OK {
		return 0
	}
	switch r.ExitCode {
	case -1:
		return 2
	case -2:
		return 3
	case -4:
		return 130
	default:
		return 1
	}
}
