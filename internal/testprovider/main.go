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
	case "stderr_nonzero":
		fmt.Fprintln(os.Stderr, "fake provider failure")
		os.Exit(23)
	case "malformed_then_text":
		fmt.Fprintln(os.Stdout, "{not-json")
		fmt.Fprintln(os.Stdout, "fake provider output")
	default:
		fmt.Fprintf(os.Stderr, "testprovider: unknown scenario %q\n", os.Getenv("HATCH_TEST_SCENARIO"))
		os.Exit(96)
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
	} {
		if value, ok := os.LookupEnv(name); ok {
			result[name] = value
		}
	}
	return result
}
