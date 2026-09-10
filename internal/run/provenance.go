package run

import (
	"fmt"
	"os"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Provenance identifies the caller that initiated a run and, when applicable,
// the immediately enclosing Hatch run. It is deliberately separate from the
// target CWD recorded on a run manifest.
type Provenance struct {
	CallerCWD       string `json:"caller_cwd"`
	CallerKind      string `json:"caller_kind"`
	CallerSessionID string `json:"caller_session_id,omitempty"`
	CallerRequestID string `json:"caller_request_id,omitempty"`
	ParentRunID     string `json:"parent_run_id,omitempty"`
}

const (
	maxProvenanceIDBytes = 512
	maxTitleBytes        = 160
	maxCallerKindBytes   = 64
)

// ValidateTitle validates the optional user-facing run title without changing
// it. Empty titles are valid; callers that accept a title must reject rather
// than truncate malformed or oversized values.
func ValidateTitle(title string) error {
	if title == "" {
		return nil
	}
	if !utf8.ValidString(title) {
		return fmt.Errorf("title must be valid UTF-8")
	}
	if len([]byte(title)) > maxTitleBytes {
		return fmt.Errorf("title must be at most %d UTF-8 bytes", maxTitleBytes)
	}
	for _, r := range title {
		if unicode.IsControl(r) || r == '\u2028' || r == '\u2029' {
			return fmt.Errorf("title must be a single line without control characters")
		}
	}
	return nil
}

func validateProvenanceID(name, value string) error {
	if value == "" {
		return nil
	}
	if !utf8.ValidString(value) {
		return fmt.Errorf("%s must be valid UTF-8", name)
	}
	if len([]byte(value)) > maxProvenanceIDBytes {
		return fmt.Errorf("%s must be at most %d UTF-8 bytes", name, maxProvenanceIDBytes)
	}
	for _, r := range value {
		if unicode.IsControl(r) || r == '\u2028' || r == '\u2029' {
			return fmt.Errorf("%s must be a single line without control characters", name)
		}
	}
	return nil
}

func validateCallerKind(value string) error {
	if len(value) > maxCallerKindBytes {
		return fmt.Errorf("caller kind must be at most %d UTF-8 bytes", maxCallerKindBytes)
	}
	return validateProvenanceID("caller kind", value)
}

func normalizedCallerCWD() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	return normalizePath(cwd), nil
}

func normalizedIdentity(value string) string {
	return strings.TrimSpace(value)
}

// ResolveProvenance resolves caller identity from explicit values first,
// Hatch propagation values second, and Longhouse's existing session chain last.
// The parent run is always the direct incoming Hatch run, not an inherited
// ancestor or provider-native identity.
func ResolveProvenance(session, request string) (*Provenance, error) {
	callerCWD, err := normalizedCallerCWD()
	if err != nil {
		return nil, fmt.Errorf("resolve caller cwd: %w", err)
	}

	p := &Provenance{CallerCWD: callerCWD, CallerKind: "unknown"}
	sessionKind := ""
	requestKind := ""
	inheritedKind := normalizedIdentity(os.Getenv("HATCH_CALLER_KIND"))
	if inheritedKind == "" {
		inheritedKind = "cli"
	}
	session = normalizedIdentity(session)
	request = normalizedIdentity(request)
	if session != "" {
		p.CallerSessionID = session
		sessionKind = "cli"
	} else if inherited := normalizedIdentity(os.Getenv("HATCH_CALLER_SESSION_ID")); inherited != "" {
		p.CallerSessionID = inherited
		sessionKind = inheritedKind
	} else {
		for _, name := range []string{"LONGHOUSE_MANAGED_SESSION_ID", "LONGHOUSE_SESSION_ID", "LONGHOUSE_CHANNEL_SESSION_ID"} {
			if inherited := normalizedIdentity(os.Getenv(name)); inherited != "" {
				p.CallerSessionID = inherited
				sessionKind = "longhouse"
				break
			}
		}
	}

	if request != "" {
		p.CallerRequestID = request
		requestKind = "cli"
	} else if inherited := normalizedIdentity(os.Getenv("HATCH_CALLER_REQUEST_ID")); inherited != "" {
		p.CallerRequestID = inherited
		requestKind = inheritedKind
	} else if inherited := normalizedIdentity(os.Getenv("LONGHOUSE_THREAD_ID")); inherited != "" {
		p.CallerRequestID = inherited
		requestKind = "longhouse"
	}

	parent := normalizedIdentity(os.Getenv("HATCH_RUN_ID"))
	if parent == "" {
		parent = normalizedIdentity(os.Getenv("LONGHOUSE_HATCH_RUN_ID"))
	}
	p.ParentRunID = parent

	if err := validateProvenanceID("caller session id", p.CallerSessionID); err != nil {
		return nil, err
	}
	if err := validateProvenanceID("caller request id", p.CallerRequestID); err != nil {
		return nil, err
	}
	if err := validateProvenanceID("parent run id", p.ParentRunID); err != nil {
		return nil, err
	}
	kind := sessionKind
	if kind == "" {
		kind = requestKind
	}
	if kind != "" {
		p.CallerKind = kind
	}
	if err := validateCallerKind(p.CallerKind); err != nil {
		return nil, err
	}
	return p, nil
}

func cloneProvenance(value *Provenance) *Provenance {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}
