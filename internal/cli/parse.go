package cli

import (
	"fmt"
	"strconv"
	"strings"
)

type Request struct {
	Backend         string
	Model           string
	PromptArgs      []string
	CWD             string
	TimeoutSeconds  int
	ReasoningEffort string
	JSON            bool
	Automation      bool
	Help            bool
	Version         bool
}

var surfaces = map[string]struct {
	backend string
	models  map[string]string
}{
	"claude": {"claude", map[string]string{"haiku": "haiku", "sonnet": "sonnet", "opus": "opus", "fable": "fable"}},
	"cursor": {"cursor", map[string]string{"grok": "cursor-grok-4.5-high"}},
	"codex": {"opencode", map[string]string{
		"sol": "openai/gpt-5.6-sol", "terra": "openai/gpt-5.6-terra", "luna": "openai/gpt-5.6-luna",
		"nano": "openai/gpt-5.4-nano", "mini": "openai/gpt-5.4-mini", "max": "openai/gpt-5.5",
	}},
	"openrouter": {"opencode", map[string]string{
		"deepseek-v4-pro": "openrouter/deepseek/deepseek-v4-pro",
		"kimi-k3":         "openrouter/moonshotai/kimi-k3",
	}},
}

func Parse(args []string, stdoutTTY bool) (Request, error) {
	normalized, err := normalizeSurface(args)
	if err != nil {
		return Request{}, err
	}
	req := Request{TimeoutSeconds: 1800, JSON: !stdoutTTY, Automation: !stdoutTTY}
	for i := 0; i < len(normalized); i++ {
		arg := normalized[i]
		switch arg {
		case "-h", "--help", "--advanced-help":
			req.Help = true
		case "-v", "--version":
			req.Version = true
		case "--json":
			req.JSON = true
		case "--automation":
			req.Automation = true
		case "-b", "--backend", "--model", "-C", "--cwd", "-t", "--timeout", "--reasoning-effort":
			if i+1 >= len(normalized) {
				return Request{}, fmt.Errorf("%s requires a value", arg)
			}
			i++
			value := normalized[i]
			switch arg {
			case "-b", "--backend":
				req.Backend = value
			case "--model":
				req.Model = value
			case "-C", "--cwd":
				req.CWD = value
			case "-t", "--timeout":
				req.TimeoutSeconds, err = strconv.Atoi(value)
				if err != nil || req.TimeoutSeconds <= 0 {
					return Request{}, fmt.Errorf("timeout must be > 0")
				}
			case "--reasoning-effort":
				req.ReasoningEffort = value
			}
		default:
			if strings.HasPrefix(arg, "-") && arg != "-" {
				return Request{}, fmt.Errorf("unrecognized argument: %s", arg)
			}
			req.PromptArgs = append(req.PromptArgs, arg)
		}
	}
	return req, nil
}

func normalizeSurface(args []string) ([]string, error) {
	for _, arg := range args {
		if arg == "-b" || arg == "--backend" || strings.HasPrefix(arg, "--backend=") {
			return append([]string(nil), args...), nil
		}
	}
	for i, arg := range args {
		surface, ok := surfaces[arg]
		if !ok {
			continue
		}
		if i+1 >= len(args) || strings.HasPrefix(args[i+1], "-") {
			return nil, fmt.Errorf("%s requires an explicit model", arg)
		}
		model, ok := surface.models[args[i+1]]
		if !ok {
			return nil, fmt.Errorf("invalid %s model %q", arg, args[i+1])
		}
		result := append([]string(nil), args[:i]...)
		result = append(result, "--backend", surface.backend, "--model", model)
		result = append(result, args[i+2:]...)
		return result, nil
	}
	return append([]string(nil), args...), nil
}
