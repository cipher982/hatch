package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	runner "github.com/cipher982/hatch/internal/run"
)

func runRuns(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" {
		fmt.Fprint(stdout, RunsHelp)
		return 0
	}
	root, err := runner.DefaultRoot()
	if err != nil {
		return renderConfigError(false, stdout, stderr, err)
	}
	expertCache := legacyExpertCache()
	switch args[0] {
	case "list":
		jsonOutput, status, err := parseRunsList(args[1:])
		if err != nil {
			return renderConfigError(jsonOutput, stdout, stderr, err)
		}
		records, err := runner.ListRecords(root, expertCache)
		if err != nil {
			return renderConfigError(jsonOutput, stdout, stderr, err)
		}
		if status != "" {
			filtered := records[:0]
			for _, record := range records {
				if record.Outcome == status || record.Lifecycle == status || record.Capture == status {
					filtered = append(filtered, record)
				}
			}
			records = filtered
		}
		if jsonOutput {
			_ = json.NewEncoder(stdout).Encode(map[string]any{"runs": records})
		} else {
			for _, record := range records {
				fmt.Fprintf(stdout, "%s\t%s\t%s\t%s\n", record.RunID, record.Outcome, record.Surface, record.Path)
			}
		}
		return 0
	case "inspect":
		id, jsonOutput, err := parseRunsInspect(args[1:])
		if err != nil {
			return renderConfigError(jsonOutput, stdout, stderr, err)
		}
		record, err := runner.InspectRecord(root, expertCache, id)
		if err != nil {
			return renderConfigError(jsonOutput, stdout, stderr, err)
		}
		if jsonOutput {
			_ = json.NewEncoder(stdout).Encode(record)
		} else if record.Manifest != nil {
			manifest := record.Manifest
			fmt.Fprintf(stdout, "Run: %s\nPath: %s\nLifecycle: %s\nOutcome: %s\nCapture: %s\n", manifest.RunID, record.Path, manifest.Lifecycle, outcomeString(manifest.Outcome), manifest.Capture.State)
			if manifest.ProviderState.NativeID != nil {
				fmt.Fprintf(stdout, "Native ID: %s\n", *manifest.ProviderState.NativeID)
			}
			if record.Observation != nil {
				fmt.Fprintf(stdout, "Process observed: pid=%d suspected_orphan=%t\n", record.Observation.PID, record.Observation.SuspectedOrphan)
			}
			fmt.Fprintln(stdout, "Files:")
			for _, name := range record.Files {
				fmt.Fprintf(stdout, "  %s\n", name)
			}
		} else {
			fmt.Fprintf(stdout, "Legacy record: %s\nPath: %s\n", record.Kind, record.Path)
			encoded, _ := json.MarshalIndent(record.Legacy, "", "  ")
			fmt.Fprintln(stdout, string(encoded))
		}
		return 0
	default:
		return renderConfigError(false, stdout, stderr, fmt.Errorf("unknown runs command %q", args[0]))
	}
}

func parseRunsList(args []string) (bool, string, error) {
	jsonOutput, status := false, ""
	for index := 0; index < len(args); index++ {
		switch args[index] {
		case "--json":
			jsonOutput = true
		case "--status":
			if index+1 >= len(args) {
				return jsonOutput, status, fmt.Errorf("--status requires a value")
			}
			index++
			status = args[index]
		default:
			if strings.HasPrefix(args[index], "--status=") {
				status = strings.TrimPrefix(args[index], "--status=")
			} else {
				return jsonOutput, status, fmt.Errorf("unrecognized argument: %s", args[index])
			}
		}
	}
	return jsonOutput, status, nil
}

func parseRunsInspect(args []string) (string, bool, error) {
	jsonOutput, id := false, ""
	for _, arg := range args {
		if arg == "--json" {
			jsonOutput = true
		} else if strings.HasPrefix(arg, "-") {
			return id, jsonOutput, fmt.Errorf("unrecognized argument: %s", arg)
		} else if id != "" {
			return id, jsonOutput, fmt.Errorf("inspect accepts exactly one run id")
		} else {
			id = arg
		}
	}
	if id == "" {
		return id, jsonOutput, fmt.Errorf("inspect requires a run id")
	}
	return id, jsonOutput, nil
}

func legacyExpertCache() string {
	if cache := strings.TrimSpace(os.Getenv("XDG_CACHE_HOME")); cache != "" {
		return filepath.Join(cache, "hatch", "expert")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "hatch", "expert")
}

func outcomeString(value *runner.Outcome) string {
	if value == nil {
		return ""
	}
	return string(*value)
}

const RunsHelp = `usage: hatch runs list [--status STATUS] [--json]
       hatch runs inspect <run-id> [--json]

Inspect local Hatch run artifacts without provider credentials.
`
