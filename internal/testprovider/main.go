// Command testprovider is a hermetic stand-in for Hatch provider CLIs.
// It records the process boundary and emits deterministic fixture output.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

type invocation struct {
	Argv        []string          `json:"argv"`
	CWD         string            `json:"cwd"`
	StdinSHA256 string            `json:"stdin_sha256"`
	StdinBytes  int               `json:"stdin_bytes"`
	Environment map[string]string `json:"environment"`
}

func main() {
	stdin, err := io.ReadAll(os.Stdin)
	if err != nil {
		fmt.Fprintf(os.Stderr, "testprovider: read stdin: %v\n", err)
		os.Exit(98)
	}

	if recordPath := os.Getenv("HATCH_TEST_RECORD"); recordPath != "" {
		if err := writeRecord(recordPath, stdin); err != nil {
			fmt.Fprintf(os.Stderr, "testprovider: write record: %v\n", err)
			os.Exit(97)
		}
	}

	switch os.Getenv("HATCH_TEST_SCENARIO") {
	case "", "success_text":
		fmt.Fprintln(os.Stdout, "fake provider output")
	case "success_claude":
		emitJSON(map[string]any{
			"type": "system", "subtype": "init", "model": "haiku",
			"session_id": "claude-session-oracle",
		})
		emitJSON(map[string]any{
			"type": "result", "result": "fake claude output", "duration_ms": 42,
		})
	case "success_cursor":
		emitJSON(map[string]any{
			"type": "system", "subtype": "init", "model": "cursor-grok-4.5-high",
			"session_id": "cursor-session-oracle",
		})
		emitJSON(map[string]any{
			"type": "result", "subtype": "success", "is_error": false,
			"duration_ms": 1250, "result": "fake cursor output",
		})
	case "success_opencode":
		if dataHome := os.Getenv("XDG_DATA_HOME"); dataHome != "" {
			stateDir := filepath.Join(dataHome, "opencode")
			if err := os.MkdirAll(stateDir, 0o700); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(94)
			}
			if err := os.WriteFile(filepath.Join(stateDir, "session.db"), []byte("fake opencode state"), 0o600); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(94)
			}
		}
		emitJSON(map[string]any{
			"type": "step_start", "sessionID": "ses_oracle1234",
		})
		emitJSON(map[string]any{
			"type": "text",
			"part": map[string]any{
				"text":     "fake opencode output",
				"metadata": map[string]any{"openai": map[string]any{"phase": "final_answer"}},
			},
		})
		emitJSON(map[string]any{
			"type": "step_finish", "part": map[string]any{"reason": "stop"},
		})
	case "cursor_error":
		emitJSON(map[string]any{"type": "system", "subtype": "init", "session_id": "cursor-error-session"})
		emitJSON(map[string]any{"type": "result", "subtype": "error", "is_error": true, "result": "request rejected"})
	case "opencode_error":
		emitJSON(map[string]any{"type": "step_start", "sessionID": "ses_error"})
		emitJSON(map[string]any{"type": "error", "error": map[string]any{"data": map[string]any{"message": "provider unavailable"}}})
	case "opencode_transient_then_success":
		emitJSON(map[string]any{"type": "step_start", "sessionID": "ses_recovered"})
		emitJSON(map[string]any{"type": "error", "error": map[string]any{"data": map[string]any{"message": "transient transport error"}}})
		emitJSON(map[string]any{"type": "text", "part": map[string]any{"text": "recovered answer", "metadata": map[string]any{"openai": map[string]any{"phase": "final_answer"}}}})
		emitJSON(map[string]any{"type": "step_finish", "part": map[string]any{"reason": "stop"}})
	case "opencode_missing_terminal":
		emitJSON(map[string]any{"type": "step_start", "sessionID": "ses_incomplete"})
		emitJSON(map[string]any{"type": "text", "part": map[string]any{"text": "useful evidence"}})
	case "stderr_nonzero":
		fmt.Fprintln(os.Stderr, "fake provider failure")
		os.Exit(23)
	case "malformed_then_text":
		fmt.Fprintln(os.Stdout, "{not-json")
		fmt.Fprintln(os.Stdout, "fake provider output")
	case "hang":
		fmt.Fprintln(os.Stdout, "partial output")
		time.Sleep(10 * time.Second)
	default:
		fmt.Fprintf(os.Stderr, "testprovider: unknown scenario %q\n", os.Getenv("HATCH_TEST_SCENARIO"))
		os.Exit(96)
	}
}

func emitJSON(value any) {
	if err := json.NewEncoder(os.Stdout).Encode(value); err != nil {
		fmt.Fprintf(os.Stderr, "testprovider: encode output: %v\n", err)
		os.Exit(95)
	}
}

func writeRecord(path string, stdin []byte) error {
	cwd, err := os.Getwd()
	if err != nil {
		return err
	}
	digest := sha256.Sum256(stdin)
	record := invocation{
		Argv:        append([]string(nil), os.Args...),
		CWD:         cwd,
		StdinSHA256: hex.EncodeToString(digest[:]),
		StdinBytes:  len(stdin),
		Environment: selectedEnvironment(),
	}
	encoded, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		return err
	}
	encoded = append(encoded, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	return os.WriteFile(path, encoded, 0o600)
}

func selectedEnvironment() map[string]string {
	result := make(map[string]string)
	for _, name := range []string{
		"DCG_BYPASS",
		"DCG_NO_SELF_HEAL",
		"GEMINI_API_KEY",
		"HATCH_AUTOMATION",
		"LONGHOUSE_HATCH_RUN_ID",
		"LONGHOUSE_IS_SIDECHAIN",
		"LONGHOUSE_ORIGIN_KIND",
		"LONGHOUSE_PARENT_SESSION_ID",
		"LONGHOUSE_PARENT_THREAD_ID",
		"LONGHOUSE_PARENT_PROVIDER_SESSION_ID",
		"LONGHOUSE_OPENCODE_SESSION_METADATA_ROOT",
	} {
		if value, ok := os.LookupEnv(name); ok {
			result[name] = value
		}
	}
	return result
}
