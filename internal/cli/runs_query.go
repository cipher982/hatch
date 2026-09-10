package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	runner "github.com/cipher982/hatch/internal/run"
)

func flagValue(args []string, index *int, name, inline string, hasInline bool) (string, error) {
	if hasInline {
		return inline, nil
	}
	if *index+1 >= len(args) {
		return "", fmt.Errorf("%s requires a value", name)
	}
	*index++
	return args[*index], nil
}

func parseRunsList(args []string, stdoutTTY bool) (runner.QueryOptions, bool, error) {
	opts := runner.QueryOptions{Limit: 20}
	jsonOutput := !stdoutTTY || hasFlag(args, "--json")
	scopes := 0
	for i := 0; i < len(args); i++ {
		name, inline, hasInline := splitLongFlag(args[i])
		switch name {
		case "--json", "--all":
			if hasInline {
				return opts, jsonOutput, fmt.Errorf("%s does not accept a value", name)
			}
			if name == "--json" {
				jsonOutput = true
			} else {
				opts.All = true
				scopes++
			}
		case "--session", "--under", "--cwd", "--request", "--parent", "--caller-kind", "--query", "--status", "--since", "--until", "--limit", "--before":
			value, err := flagValue(args, &i, name, inline, hasInline)
			if err != nil {
				return opts, jsonOutput, err
			}
			if value == "" || len(value) > 4096 {
				return opts, jsonOutput, fmt.Errorf("%s requires a nonempty value of at most 4096 bytes", name)
			}
			switch name {
			case "--session":
				opts.Session = value
				scopes++
			case "--under":
				opts.Under, err = filepath.Abs(value)
				scopes++
			case "--cwd":
				opts.CallerCWD, err = filepath.Abs(value)
				scopes++
			case "--request":
				opts.Request = value
			case "--parent":
				opts.Parent = value
			case "--caller-kind":
				opts.CallerKind = value
			case "--query":
				opts.Query = value
			case "--status":
				opts.Status = value
			case "--before":
				opts.Before = value
			case "--since":
				opts.Since, err = parseRunTime(value)
			case "--until":
				opts.Until, err = parseRunTime(value)
			case "--limit":
				opts.Limit, err = strconv.Atoi(value)
				if err != nil || opts.Limit < 1 || opts.Limit > 100 {
					err = fmt.Errorf("list --limit must be between 1 and 100")
				}
			}
			if err != nil {
				return opts, jsonOutput, err
			}
		default:
			return opts, jsonOutput, fmt.Errorf("unrecognized list argument: %s", args[i])
		}
	}
	if scopes > 1 {
		return opts, jsonOutput, fmt.Errorf("--all, --session, --under and --cwd are mutually exclusive")
	}
	if !opts.Since.IsZero() && !opts.Until.IsZero() && opts.Since.After(opts.Until) {
		return opts, jsonOutput, fmt.Errorf("--since must not be later than --until")
	}
	if scopes == 0 || opts.Session == "current" {
		p, err := runner.ResolveProvenance("", "")
		if err != nil {
			return opts, jsonOutput, err
		}
		if p.CallerSessionID != "" {
			opts.Session = p.CallerSessionID
		} else if opts.Session == "current" {
			return opts, jsonOutput, fmt.Errorf("no caller session identity is available; pass --session ID, --under DIR, or --all")
		} else {
			opts.CallerCWD = p.CallerCWD
		}
	}
	return opts, jsonOutput, nil
}

func parseRunTime(value string) (time.Time, error) {
	if instant, err := time.Parse(time.RFC3339Nano, value); err == nil {
		return instant.UTC(), nil
	}
	duration, err := time.ParseDuration(value)
	if err != nil || duration < 0 {
		return time.Time{}, fmt.Errorf("time must be RFC3339 or a nonnegative duration such as 24h")
	}
	return time.Now().UTC().Add(-duration), nil
}

func listCommand(opts runner.QueryOptions, cursor string) []string {
	args := []string{"hatch", "runs", "list"}
	switch {
	case opts.All:
		args = append(args, "--all")
	case opts.Session != "":
		args = append(args, "--session", opts.Session)
	case opts.Under != "":
		args = append(args, "--under", opts.Under)
	default:
		// Exact launch-directory scope is preserved by a distinct explicit selector.
		args = append(args, "--cwd", opts.CallerCWD)
	}
	for _, item := range [][2]string{{"--caller-kind", opts.CallerKind}, {"--request", opts.Request}, {"--parent", opts.Parent}, {"--query", opts.Query}, {"--status", opts.Status}} {
		if item[1] != "" {
			args = append(args, item[0], item[1])
		}
	}
	if !opts.Since.IsZero() {
		args = append(args, "--since", opts.Since.Format(time.RFC3339Nano))
	}
	if !opts.Until.IsZero() {
		args = append(args, "--until", opts.Until.Format(time.RFC3339Nano))
	}
	args = append(args, "--limit", strconv.Itoa(opts.Limit))
	if cursor != "" {
		args = append(args, "--before", cursor)
	}
	return append(args, "--json")
}

func runList(root, cache string, args []string, stdout, stderr io.Writer, stdoutTTY bool) int {
	opts, jsonOutput, err := parseRunsList(args, stdoutTTY)
	if err != nil {
		return renderConfigError(jsonOutput, stdout, stderr, err)
	}
	page, err := runner.QueryRecords(root, cache, opts)
	if err != nil {
		return renderConfigError(jsonOutput, stdout, stderr, err)
	}
	payload := struct {
		runner.RunPage
		NextCommand []string `json:"next_command,omitempty"`
	}{RunPage: page}
	for {
		if payload.NextCursor != "" {
			payload.NextCommand = listCommand(opts, payload.NextCursor)
		}
		encoded, err := json.Marshal(payload)
		if err != nil {
			return renderConfigError(jsonOutput, stdout, stderr, err)
		}
		if len(encoded)+1 <= maxResponseBytes {
			if jsonOutput {
				if _, err := stdout.Write(append(encoded, '\n')); err != nil {
					return 1
				}
			} else {
				fmt.Fprintf(stdout, "Scope: %s %s\n", page.Scope.Kind, page.Scope.Value)
				for _, r := range payload.Runs {
					title := r.Title
					if title == "" {
						title = "[request preview] " + r.Preview
					}
					status := r.Outcome
					if status == "" {
						status = r.Lifecycle
					}
					fmt.Fprintf(stdout, "%s\t%s\t%s\t%s\t%s\n", r.RunID, status, r.Answer, r.Surface, title)
					fmt.Fprintf(stdout, "  target: %s", r.CWD)
					if r.Provenance != nil {
						fmt.Fprintf(stdout, "  caller: %s", r.Provenance.CallerCWD)
					}
					fmt.Fprintln(stdout)
				}
				if len(payload.Runs) == 0 {
					fmt.Fprintln(stdout, "No matching runs in this scope. Use --all to search every local project.")
				}
				if len(payload.NextCommand) > 0 {
					fmt.Fprintf(stdout, "Next: %s\n", commandText(payload.NextCommand))
				}
				if payload.Diagnostics.Skipped > 0 {
					fmt.Fprintf(stderr, "Skipped %d unreadable records; inspect the JSON diagnostics.\n", payload.Diagnostics.Skipped)
				}
			}
			return 0
		}
		if len(payload.Runs) <= 1 {
			return renderConfigError(jsonOutput, stdout, stderr, fmt.Errorf("run metadata exceeds response limit; narrow the query or inspect the artifact directly"))
		}
		payload.Runs = payload.Runs[:len(payload.Runs)-1]
		payload.NextCursor = payload.Runs[len(payload.Runs)-1].RunID
		payload.Truncated = true
	}
}

type readArguments struct {
	ID     string
	Part   string
	Offset int64
	Limit  int
	Files  bool
	JSON   bool
}

func parseRunRead(args []string, inspect, stdoutTTY bool) (readArguments, error) {
	opts := readArguments{Part: "result", Limit: 8192, JSON: !stdoutTTY || hasFlag(args, "--json")}
	if inspect {
		opts.Limit = 20
	}
	for i := 0; i < len(args); i++ {
		name, inline, hasInline := splitLongFlag(args[i])
		switch name {
		case "--json":
			if hasInline {
				return opts, fmt.Errorf("--json does not accept a value")
			}
			opts.JSON = true
		case "--files":
			if !inspect || hasInline {
				return opts, fmt.Errorf("--files is only available for inspect")
			}
			opts.Files = true
		case "--part", "--offset", "--limit":
			value, err := flagValue(args, &i, name, inline, hasInline)
			if err != nil {
				return opts, err
			}
			switch name {
			case "--part":
				if inspect {
					return opts, fmt.Errorf("--part is only available for read")
				}
				opts.Part = value
			case "--offset":
				opts.Offset, err = strconv.ParseInt(value, 10, 64)
				if err != nil || opts.Offset < 0 {
					return opts, fmt.Errorf("--offset must be a nonnegative byte or file offset")
				}
			case "--limit":
				opts.Limit, err = strconv.Atoi(value)
				if err != nil {
					return opts, fmt.Errorf("--limit must be an integer")
				}
			}
		default:
			if strings.HasPrefix(name, "-") || opts.ID != "" {
				return opts, fmt.Errorf("unrecognized argument: %s", args[i])
			}
			opts.ID = args[i]
		}
	}
	if opts.ID == "" {
		return opts, fmt.Errorf("a run ID is required")
	}
	if inspect {
		if opts.Limit < 1 || opts.Limit > 100 {
			return opts, fmt.Errorf("inspect --limit must be between 1 and 100")
		}
		if !opts.Files && (opts.Offset != 0 || opts.Limit != 20) {
			return opts, fmt.Errorf("inspect paging requires --files")
		}
	} else if opts.Limit < 4 || opts.Limit > 32768 {
		return opts, fmt.Errorf("read --limit must be between 4 and 32768 bytes")
	}
	return opts, nil
}

func runRead(root, cache string, args []string, stdout, stderr io.Writer, stdoutTTY bool) int {
	opts, err := parseRunRead(args, false, stdoutTTY)
	if err != nil {
		return renderConfigError(opts.JSON, stdout, stderr, err)
	}
	page, err := runner.ReadContent(root, cache, opts.ID, runner.ContentOptions{Part: opts.Part, Offset: opts.Offset, Limit: opts.Limit})
	if err != nil {
		return renderConfigError(opts.JSON, stdout, stderr, err)
	}
	payload := struct {
		runner.ContentPage
		NextCommand []string `json:"next_command,omitempty"`
	}{ContentPage: page}
	if page.NextOffset != nil {
		payload.NextCommand = []string{"hatch", "runs", "read", opts.ID, "--part", opts.Part, "--offset", strconv.FormatInt(*page.NextOffset, 10), "--limit", strconv.Itoa(opts.Limit), "--json"}
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return renderConfigError(opts.JSON, stdout, stderr, err)
	}
	if len(encoded)+1 > 256<<10 {
		return renderConfigError(opts.JSON, stdout, stderr, fmt.Errorf("read metadata exceeds response limit"))
	}
	if opts.JSON {
		if _, err := stdout.Write(append(encoded, '\n')); err != nil {
			return 1
		}
	} else {
		if _, err := io.WriteString(stdout, page.Content); err != nil {
			return 1
		}
		payload.Content = ""
		metadata, _ := json.Marshal(payload)
		fmt.Fprintln(stderr, string(metadata))
	}
	return 0
}

func runInspect(root, cache string, args []string, stdout, stderr io.Writer, stdoutTTY bool) int {
	opts, err := parseRunRead(args, true, stdoutTTY)
	if err != nil {
		return renderConfigError(opts.JSON, stdout, stderr, err)
	}
	record, err := runner.InspectRecord(root, cache, opts.ID)
	if err != nil {
		return renderConfigError(opts.JSON, stdout, stderr, err)
	}
	payload := struct {
		runner.Record
		MetadataTruncated bool             `json:"metadata_truncated"`
		Inventory         *runner.FilePage `json:"inventory,omitempty"`
		ReadCommand       []string         `json:"read_command"`
		ManifestCommand   []string         `json:"manifest_command"`
		NextCommand       []string         `json:"next_command,omitempty"`
	}{Record: record, ReadCommand: []string{"hatch", "runs", "read", opts.ID, "--part", "result", "--json"}, ManifestCommand: []string{"hatch", "runs", "read", opts.ID, "--part", "manifest", "--json"}}
	if record.Manifest != nil {
		payload.Manifest, payload.MetadataTruncated = displayManifest(record.Manifest)
	}
	if record.Legacy != nil {
		payload.Legacy = make(map[string]any)
		for _, key := range []string{"response_id", "run_id", "status", "outcome", "model", "created_at", "updated_at", "cwd"} {
			if v, ok := record.Legacy[key]; ok {
				if s, ok := v.(string); ok {
					s, _ = runner.ClipText(s, 512, false)
					payload.Legacy[key] = s
				}
			}
		}
		payload.MetadataTruncated = true
	}
	if opts.Files {
		if opts.Offset > int64(^uint(0)>>1) {
			return renderConfigError(opts.JSON, stdout, stderr, fmt.Errorf("file offset is too large"))
		}
		inventory, err := runner.Inventory(record, int(opts.Offset), opts.Limit)
		if err != nil {
			return renderConfigError(opts.JSON, stdout, stderr, err)
		}
		payload.Inventory = &inventory
		if inventory.NextOffset != nil {
			payload.NextCommand = []string{"hatch", "runs", "inspect", opts.ID, "--files", "--offset", strconv.Itoa(*inventory.NextOffset), "--limit", strconv.Itoa(opts.Limit), "--json"}
		}
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return renderConfigError(opts.JSON, stdout, stderr, err)
	}
	if len(data)+1 > maxResponseBytes {
		return renderConfigError(opts.JSON, stdout, stderr, fmt.Errorf("inspection exceeds response limit; reduce --limit or use runs read --part manifest"))
	}
	if opts.JSON {
		if _, err := stdout.Write(append(data, '\n')); err != nil {
			return 1
		}
	} else {
		// One bounded metadata document avoids different facts between human and machine views.
		fmt.Fprintln(stdout, string(data))
	}
	return 0
}

// Prevent a long command-line value from expanding a configuration error into a trace.
func configErrorPreview(err error) string {
	text, clipped := runner.ClipText(err.Error(), 2048, false)
	if clipped {
		text += " [truncated]"
	}
	return text
}
