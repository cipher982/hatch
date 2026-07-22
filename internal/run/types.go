package run

import "time"

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
	OutcomeLaunch            Outcome = "launch_failed"
)

type Manifest struct {
	SchemaVersion int        `json:"schema_version"`
	RunID         string     `json:"run_id"`
	CreatedAt     time.Time  `json:"created_at"`
	UpdatedAt     time.Time  `json:"updated_at"`
	Lifecycle     Lifecycle  `json:"lifecycle"`
	Outcome       *Outcome   `json:"outcome"`
	Surface       string     `json:"surface"`
	Provider      string     `json:"provider"`
	Model         string     `json:"model"`
	CWD           string     `json:"cwd"`
	Execution     string     `json:"execution"`
	Invocation    Invocation `json:"invocation"`
	Process       *Process   `json:"process"`
	Result        Result     `json:"result"`
	Capture       Capture    `json:"capture"`
	ProviderState State      `json:"provider_state"`
	Archive       Archive    `json:"archive"`
	Warnings      []Warning  `json:"warnings"`
}

type Invocation struct {
	RequestFile        string   `json:"request_file"`
	RequestSHA256      string   `json:"request_sha256"`
	RedactedArgv       []string `json:"redacted_argv"`
	CredentialEnvNames []string `json:"credential_env_names"`
}

type Process struct {
	PID       int        `json:"pid"`
	StartedAt time.Time  `json:"started_at"`
	ExitedAt  *time.Time `json:"exited_at"`
	ExitCode  *int       `json:"exit_code"`
}

type Result struct {
	Output         string  `json:"output"`
	TerminalMarker string  `json:"terminal_marker"`
	OutputBytes    int64   `json:"output_bytes"`
	OutputFile     *string `json:"output_file"`
	Error          *string `json:"error"`
}

type Capture struct {
	State          string  `json:"state"`
	ArtifactPath   string  `json:"artifact_path"`
	EvidenceSHA256 *string `json:"evidence_sha256"`
	StdoutFile     string  `json:"stdout_file"`
	StderrFile     string  `json:"stderr_file"`
}

type State struct {
	Retention     string            `json:"retention"`
	NativeID      *string           `json:"native_id"`
	NativeIDState string            `json:"native_id_state"`
	Capabilities  map[string]string `json:"capabilities"`
	SnapshotPath  *string           `json:"snapshot_path"`
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
	default:
		return 1
	}
}
