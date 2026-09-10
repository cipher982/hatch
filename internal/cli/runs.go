package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	runner "github.com/cipher982/hatch/internal/run"
)

func runRuns(args []string, stdout, stderr io.Writer, stdoutTTY bool) int {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" {
		fmt.Fprint(stdout, RunsHelp)
		return 0
	}
	if hasFlag(args[1:], "-h", "--help") {
		fmt.Fprint(stdout, RunsHelp)
		return 0
	}
	if !stdoutTTY && !hasFlag(args, "--json") {
		args = append(append([]string(nil), args...), "--json")
	}
	root, err := runner.DefaultRoot()
	if err != nil {
		return renderConfigError(!stdoutTTY || hasFlag(args, "--json"), stdout, stderr, err)
	}
	expertCache := legacyExpertCache()
	switch args[0] {
	case "list":
		return runList(root, expertCache, args[1:], stdout, stderr, stdoutTTY)
	case "inspect":
		return runInspect(root, expertCache, args[1:], stdout, stderr, stdoutTTY)
	case "read":
		return runRead(root, expertCache, args[1:], stdout, stderr, stdoutTTY)
	case "audit":
		minimumTotal, minimumSurface, jsonOutput, err := parseRunsAudit(args[1:])
		if err != nil {
			return renderConfigError(jsonOutput, stdout, stderr, err)
		}
		audit, err := runner.AuditFieldEvidence(root, minimumTotal, minimumSurface)
		if err != nil {
			return renderConfigError(jsonOutput, stdout, stderr, err)
		}
		passed := audit.Passed()
		if jsonOutput {
			_ = json.NewEncoder(stdout).Encode(map[string]any{"passed": passed, "audit": audit})
		} else {
			fmt.Fprintf(stdout, "Go artifact audit: eligible=%d observed=%d excluded-pre-contract=%d incomplete=%d non-success=%d non-surfaced=%d unsafe=%d explained=%d unexplained=%d\n", audit.Eligible, audit.Observed, audit.ExcludedPreContract, audit.Incomplete, audit.NonSuccess, audit.NonSurfaced, audit.Unsafe, audit.ExplainedUnsafe, audit.UnexplainedUnsafe)
			for _, surface := range []string{"claude", "codex", "cursor", "openrouter", "expert"} {
				fmt.Fprintf(stdout, "  %s: %d/%d\n", surface, audit.Surfaces[surface], audit.MinimumSurface)
			}
			for _, issue := range audit.UnsafeRuns {
				status := "unexplained"
				if issue.Disposition != nil {
					status = "explained"
				}
				fmt.Fprintf(stderr, "  unsafe (%s) %s: %s\n", status, issue.RunID, issue.Reason)
			}
			if passed {
				fmt.Fprintln(stdout, "artifact integrity audit passed")
			} else {
				fmt.Fprintln(stderr, "artifact integrity or requested sample minimum is not satisfied")
			}
		}
		if passed {
			return 0
		}
		return 1
	case "gc":
		apply, jsonOutput, err := parseRunsGC(args[1:])
		if err != nil {
			return renderConfigError(jsonOutput, stdout, stderr, err)
		}
		report, err := runner.CollectGarbage(root, apply)
		if err != nil {
			return renderConfigError(jsonOutput, stdout, stderr, err)
		}
		if jsonOutput {
			_ = json.NewEncoder(stdout).Encode(report)
		} else {
			mode := "dry-run"
			if apply {
				mode = "applied"
			}
			fmt.Fprintf(stdout, "Hatch run garbage collection (%s): scanned=%d nonterminal-skipped=%d pinned-skipped=%d\n", mode, report.RunsScanned, report.RunsSkippedNonterminal, report.RunsSkippedPinned)
			for _, className := range []string{runner.GarbageOpenCodeConfig, runner.GarbageOpenCodeCache, runner.GarbageProviderState} {
				class := report.Classes[className]
				fmt.Fprintf(stdout, "  %s: paths=%d files=%d logical-bytes=%d\n", className, class.Paths, class.Files, class.LogicalBytes)
			}
			fmt.Fprintf(stdout, "  total: paths=%d files=%d logical-bytes=%d removed-logical-bytes=%d\n", report.TotalPaths, report.TotalFiles, report.TotalLogicalBytes, report.RemovedLogicalBytes)
			if !apply && report.TotalPaths > 0 {
				fmt.Fprintln(stdout, "Dry run only. Re-run with --apply to remove these derived provider directories.")
			}
			for _, message := range report.Errors {
				fmt.Fprintf(stderr, "  error: %s\n", message)
			}
		}
		if len(report.Errors) > 0 {
			return 1
		}
		return 0
	default:
		return renderConfigError(false, stdout, stderr, fmt.Errorf("unknown runs command %q", args[0]))
	}
}

func parseRunsGC(args []string) (bool, bool, error) {
	apply, jsonOutput := false, false
	for _, arg := range args {
		switch arg {
		case "--apply":
			apply = true
		case "--json":
			jsonOutput = true
		default:
			return apply, jsonOutput, fmt.Errorf("unrecognized argument: %s", arg)
		}
	}
	return apply, jsonOutput, nil
}

func parseRunsAudit(args []string) (int, int, bool, error) {
	minimumTotal, minimumSurface, jsonOutput := 0, 0, false
	for index := 0; index < len(args); index++ {
		arg := args[index]
		if arg == "--json" {
			jsonOutput = true
			continue
		}
		name, value, inline := strings.Cut(arg, "=")
		if name != "--minimum-total" && name != "--minimum-surface" {
			return minimumTotal, minimumSurface, jsonOutput, fmt.Errorf("unrecognized argument: %s", arg)
		}
		if !inline {
			if index+1 >= len(args) {
				return minimumTotal, minimumSurface, jsonOutput, fmt.Errorf("%s requires a value", name)
			}
			index++
			value = args[index]
		}
		parsed, err := strconv.Atoi(value)
		if err != nil || parsed < 0 {
			return minimumTotal, minimumSurface, jsonOutput, fmt.Errorf("%s must be a nonnegative integer", name)
		}
		if name == "--minimum-total" {
			minimumTotal = parsed
		} else {
			minimumSurface = parsed
		}
	}
	return minimumTotal, minimumSurface, jsonOutput, nil
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

const RunsHelp = `usage: hatch runs list [--all|--session ID|--under DIR] [OPTIONS]
       hatch runs inspect <run-id> [--files] [--offset N] [--limit N] [--json]
       hatch runs read <run-id> [--part PART] [--offset N] [--limit N] [--json]
       hatch runs audit [--minimum-total N] [--minimum-surface N] [--json]
       hatch runs gc [--apply] [--json]

List defaults to the inherited caller session, otherwise the exact launch directory.
Scopes never broaden automatically. Use --all to include every local project.
--session current requires an available caller identity.
--cwd DIR selects an exact launch directory, including when replaying a page.
List filters: --request ID, --parent ID, --caller-kind KIND, --query TEXT,
              --status STATUS, --since RFC3339|DURATION, --until RFC3339|DURATION.
List paging: --limit 1..100 (default 20), --before RUN-ID.
Query searches metadata and the first 4 KiB of the request, never execution traces.
Directory scope --under includes launches or targets within the directory.

Read parts: result (default), request, stdout, stderr, manifest, result-json, evidence.
Read paging uses original byte offsets, not lines: --limit 4..32768 (default 8192).
Inspect --files opts into a paged file inventory; its limit is 1..100 (default 20).
JSON is the default for non-interactive callers. Returned continuation commands
preserve the scope and filters. Full artifacts remain on disk, unchanged.
All retrieval is local and requires no provider credentials.
Garbage collection is a dry run unless --apply is supplied.
`
