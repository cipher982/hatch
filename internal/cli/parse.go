package cli

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/cipher982/hatch/internal/provider"
)

type Request struct {
	Backend                string
	Harness                string
	Model                  string
	PromptArgs             []string
	CWD                    string
	TimeoutSeconds         int
	ReasoningEffort        string
	OutputFormat           string
	OutputFormatExplicit   bool
	APIKey                 string
	Resume                 string
	SkipGitRepoCheck       bool
	IncludePartialMessages bool
	JSON                   bool
	Automation             bool
	Help                   bool
	AdvancedHelp           bool
	Version                bool
}

var surfaces = map[string]struct {
	backend string
	models  map[string]string
}{
	"claude":     {"claude", map[string]string{"haiku": "haiku", "sonnet": "sonnet", "opus": "opus", "fable": "fable"}},
	"cursor":     {"cursor", provider.CursorSurfaceModels},
	"codex":      {"opencode", provider.CodexSurfaceModels},
	"gemini":     {"omp", provider.GeminiSurfaceModels},
	"openrouter": {"opencode", provider.OpenRouterSurfaceModels},
}

var flagsWithValue = map[string]bool{
	"-b": true, "--backend": true, "-t": true, "--timeout": true,
	"-C": true, "--cwd": true, "--model": true, "--reasoning-effort": true,
	"--output-format": true, "--api-key": true, "-r": true, "--resume": true,
	"--harness": true,
}

func Parse(args []string, stdoutTTY bool) (Request, error) {
	normalized, err := normalizeSurface(args)
	if err != nil {
		return Request{JSON: hasFlag(args, "--json") || !stdoutTTY}, err
	}
	req := Request{TimeoutSeconds: 1800, OutputFormat: "text", JSON: !stdoutTTY, Automation: !stdoutTTY}
	literal := false
	for i := 0; i < len(normalized); i++ {
		arg := normalized[i]
		if literal {
			req.PromptArgs = append(req.PromptArgs, arg)
			continue
		}
		if arg == "--" {
			literal = true
			continue
		}
		name, inlineValue, hasInlineValue := splitLongFlag(arg)
		if hasInlineValue && !flagsWithValue[name] {
			return req, fmt.Errorf("unrecognized argument: %s", arg)
		}
		switch name {
		case "-h", "--help":
			req.Help = true
		case "--advanced-help":
			req.Help, req.AdvancedHelp = true, true
		case "-v", "--version":
			req.Version = true
		case "--json":
			req.JSON, req.Automation = true, true
		case "--automation":
			req.Automation = true
		case "--skip-git-repo-check":
			req.SkipGitRepoCheck = true
		case "--include-partial-messages":
			req.IncludePartialMessages = true
		case "-b", "--backend", "--model", "-C", "--cwd", "-t", "--timeout", "--reasoning-effort", "--output-format", "--api-key", "-r", "--resume", "--harness":
			value := inlineValue
			if !hasInlineValue {
				if i+1 >= len(normalized) {
					return req, fmt.Errorf("%s requires a value", name)
				}
				i++
				value = normalized[i]
			}
			switch name {
			case "-b", "--backend":
				req.Backend = value
			case "--harness":
				if !oneOf(value, "opencode", "pi", "omp") {
					return req, fmt.Errorf("invalid harness %q. Choose one of: opencode, pi, omp", value)
				}
				req.Harness = value
			case "--model":
				req.Model = value
			case "-C", "--cwd":
				req.CWD = value
			case "-t", "--timeout":
				req.TimeoutSeconds, err = strconv.Atoi(value)
				if err != nil || req.TimeoutSeconds <= 0 {
					return req, fmt.Errorf("timeout must be > 0")
				}
			case "--reasoning-effort":
				if !oneOf(value, "none", "low", "medium", "high", "xhigh", "max") {
					return req, fmt.Errorf("invalid reasoning effort %q", value)
				}
				req.ReasoningEffort = value
			case "--output-format":
				if !oneOf(value, "text", "json", "stream-json") {
					return req, fmt.Errorf("invalid output format %q", value)
				}
				req.OutputFormat = value
				req.OutputFormatExplicit = true
			case "--api-key":
				req.APIKey = value
			case "-r", "--resume":
				req.Resume = value
			}
		default:
			if strings.HasPrefix(arg, "-") && arg != "-" {
				return req, fmt.Errorf("unrecognized argument: %s", arg)
			}
			req.PromptArgs = append(req.PromptArgs, arg)
		}
	}
	return req, nil
}

func normalizeSurface(args []string) ([]string, error) {
	if hasFlag(args, "-b", "--backend") {
		return append([]string(nil), args...), nil
	}
	args = normalizeModelShorthand(args)
	hasExplicitModel := hasFlag(args, "--model")
	index := aliasCandidateIndex(args)
	if index < 0 {
		if hasExplicitModel {
			return append([]string{"--backend", "opencode"}, args...), nil
		}
		return append([]string(nil), args...), nil
	}
	surfaceName := args[index]
	surface, ok := surfaces[surfaceName]
	if !ok {
		if hasExplicitModel {
			return append([]string{"--backend", "opencode"}, args...), nil
		}
		return append([]string(nil), args...), nil
	}
	before, after := args[:index], args[index+1:]
	if len(after) == 0 {
		if surfaceName == "gemini" {
			result := append([]string(nil), before...)
			result = append(result, "--backend", surface.backend)
			if !hasExplicitModel {
				result = append(result, "--model", surface.models["flash"])
			}
			return append(result, after...), nil
		}
		return nil, fmt.Errorf("%s requires an explicit model: %s", surfaceName, modelChoices(surface.models))
	}
	if strings.HasPrefix(after[0], "-") {
		if after[0] == "-h" || after[0] == "--help" || after[0] == "--advanced-help" {
			return append([]string(nil), args...), nil
		}
		if hasExplicitModel || surfaceName == "gemini" {
			result := append([]string(nil), before...)
			result = append(result, "--backend", surface.backend)
			if !hasExplicitModel && surfaceName == "gemini" {
				result = append(result, "--model", surface.models["flash"])
			}
			return append(result, after...), nil
		}
		return nil, fmt.Errorf("%s requires an explicit model: %s", surfaceName, modelChoices(surface.models))
	}
	alias := after[0]
	model, ok := surface.models[alias]
	if !ok {
		if surfaceName == "gemini" {
			if alias == "pro" {
				return nil, fmt.Errorf("invalid %s model %q. Choose one of: %s", surfaceName, alias, modelChoices(surface.models))
			}
			result := append([]string(nil), before...)
			result = append(result, "--backend", surface.backend)
			if !hasExplicitModel {
				result = append(result, "--model", surface.models["flash"])
			}
			return append(result, after...), nil
		}
		message := fmt.Sprintf("invalid %s model %q. Choose one of: %s", surfaceName, alias, modelChoices(surface.models))
		if surfaceName == "cursor" {
			message += `. For a raw Cursor model ID, use: hatch cursor grok --model <cursor-model-id> "prompt"`
		}
		return nil, fmt.Errorf("%s", message)
	}
	after = after[1:]
	result := append([]string(nil), before...)
	result = append(result, "--backend", surface.backend)
	if !hasExplicitModel {
		result = append(result, "--model", model)
	}
	return append(result, after...), nil
}

func normalizeModelShorthand(args []string) []string {
	index := aliasCandidateIndex(args)
	if index < 0 {
		return append([]string(nil), args...)
	}
	surfaceName := shorthandSurface(args[index])
	if surfaceName == "" {
		return append([]string(nil), args...)
	}
	result := append([]string(nil), args[:index]...)
	result = append(result, surfaceName, args[index])
	return append(result, args[index+1:]...)
}

func shorthandSurface(alias string) string {
	switch alias {
	case "haiku", "sonnet", "opus", "fable":
		return "claude"
	case "sol", "terra", "luna", "nano", "mini", "max":
		return "codex"
	case "grok", "kimi-k3":
		return "cursor"
	case "flash", "3.7", "gemini-3.7-flash-tiered":
		return "gemini"
	case "deepseek-v4-flash", "deepseek-v4-pro", "glm-5.3-flash":
		return "openrouter"
	}
	return ""
}

func aliasCandidateIndex(args []string) int {
	expectingValue := false
	for i, arg := range args {
		if expectingValue {
			expectingValue = false
			continue
		}
		if arg == "--" {
			return -1
		}
		if flagsWithValue[arg] {
			expectingValue = true
			continue
		}
		if strings.HasPrefix(arg, "--") || (strings.HasPrefix(arg, "-") && arg != "-") {
			continue
		}
		return i
	}
	return -1
}

func hasFlag(args []string, names ...string) bool {
	for _, arg := range args {
		for _, name := range names {
			if arg == name || (strings.HasPrefix(name, "--") && strings.HasPrefix(arg, name+"=")) {
				return true
			}
		}
	}
	return false
}

func splitLongFlag(arg string) (string, string, bool) {
	if strings.HasPrefix(arg, "--") {
		if index := strings.IndexByte(arg, '='); index >= 0 {
			return arg[:index], arg[index+1:], true
		}
	}
	return arg, "", false
}

func modelChoices(models map[string]string) string {
	// Stable public order, matching each surface's documented preference.
	order := []string{"sol", "terra", "luna", "nano", "mini", "max", "haiku", "sonnet", "opus", "fable", "grok", "flash", "3.7", "gemini-3.7-flash-tiered", "deepseek-v4-flash", "deepseek-v4-pro", "glm-5.3-flash", "kimi-k3"}
	choices := make([]string, 0, len(models))
	for _, name := range order {
		if _, ok := models[name]; ok {
			choices = append(choices, name)
		}
	}
	return strings.Join(choices, ", ")
}

func oneOf(value string, choices ...string) bool {
	for _, choice := range choices {
		if value == choice {
			return true
		}
	}
	return false
}
