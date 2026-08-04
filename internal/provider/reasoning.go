package provider

import (
	"fmt"
	"strings"
)

const DefaultReasoningEffort = "medium"

// Keep existing Claude calls at their established cost profile unless callers
// explicitly request a higher effort.
const defaultClaudeReasoningEffort = "low"

var reasoningEfforts = map[string]bool{
	"none":   true,
	"low":    true,
	"medium": true,
	"high":   true,
	"xhigh":  true,
	"max":    true,
}

var claudeReasoningEfforts = map[string]bool{
	"low":    true,
	"medium": true,
	"high":   true,
	"xhigh":  true,
	"max":    true,
}

// ReasoningPolicy is the resolved, visible policy for one run. Source keeps a
// default from being indistinguishable from an explicit user choice, while
// Support makes unsupported providers explicit instead of silently inheriting
// a provider or session default.
type ReasoningPolicy struct {
	Effort  string `json:"effort"`
	Source  string `json:"source"`
	Support string `json:"support"`
}

func IsValidReasoningEffort(value string) bool {
	return reasoningEfforts[value]
}

// ResolveReasoning applies Hatch's policy before a provider process or HTTP
// request is created. Known OpenAI model variants are checked here because
// OpenCode's --variant flag is provider-specific rather than a universal
// reasoning API.
func ResolveReasoning(backend, model, requested string) (ReasoningPolicy, error) {
	if requested != "" && !IsValidReasoningEffort(requested) {
		return ReasoningPolicy{}, fmt.Errorf("invalid reasoning effort %q", requested)
	}

	switch backend {
	case "claude":
		return resolveClaudeReasoning(requested)
	case "bedrock":
		if requested != "" {
			return ReasoningPolicy{}, fmt.Errorf("Bedrock reasoning-effort overrides are not supported")
		}
		return ReasoningPolicy{Effort: "low", Source: "fixed", Support: "fixed"}, nil
	case "codex":
		return resolveNativeReasoning("Codex", requested, model)
	case "expert":
		return resolveNativeReasoning("Expert", requested, model)
	case "opencode":
		if model == "" {
			return ReasoningPolicy{}, fmt.Errorf("OpenCode backend requires an explicit model")
		}
		if strings.HasPrefix(model, "openai/") {
			return resolveOpenCodeOpenAIReasoning(model, requested)
		}
		if requested != "" {
			return ReasoningPolicy{}, fmt.Errorf("--reasoning-effort is unsupported for OpenCode model %q", model)
		}
		return ReasoningPolicy{Source: "unsupported", Support: "unsupported"}, nil
	case "cursor", "gemini":
		if requested != "" {
			return ReasoningPolicy{}, fmt.Errorf("--reasoning-effort is unsupported for %s", backend)
		}
		return ReasoningPolicy{Source: "unsupported", Support: "unsupported"}, nil
	default:
		if requested != "" {
			return ReasoningPolicy{}, fmt.Errorf("--reasoning-effort is unsupported for backend %q", backend)
		}
		return ReasoningPolicy{Source: "unsupported", Support: "unsupported"}, nil
	}
}

func resolveClaudeReasoning(requested string) (ReasoningPolicy, error) {
	if requested != "" && !claudeReasoningEfforts[requested] {
		return ReasoningPolicy{}, fmt.Errorf("reasoning effort %q is not supported by Claude Code; supported values: low, medium, high, xhigh, max", requested)
	}
	effort, source := requested, "explicit"
	if effort == "" {
		effort, source = defaultClaudeReasoningEffort, "default"
	}
	return ReasoningPolicy{Effort: effort, Source: source, Support: "native"}, nil
}

func resolveNativeReasoning(providerName, requested, model string) (ReasoningPolicy, error) {
	effort, source := requested, "explicit"
	if effort == "" {
		effort, source = DefaultReasoningEffort, "default"
	}
	known, supportsMax := knownOpenAIModel(model)
	if model != "" && !known {
		if requested == "" {
			return ReasoningPolicy{}, fmt.Errorf("%s model %q has no verified reasoning variants; pass --reasoning-effort explicitly", providerName, model)
		}
		return ReasoningPolicy{Effort: effort, Source: source, Support: "unknown"}, nil
	}
	if effort == "max" && !supportsMax && known {
		return ReasoningPolicy{}, fmt.Errorf("reasoning effort %q is not supported by %s model %q", effort, providerName, model)
	}
	return ReasoningPolicy{Effort: effort, Source: source, Support: "native"}, nil
}

func resolveOpenCodeOpenAIReasoning(model, requested string) (ReasoningPolicy, error) {
	known, supportsMax := knownOpenAIModel(model)
	if !known {
		if requested == "" {
			return ReasoningPolicy{}, fmt.Errorf("OpenCode model %q has no verified reasoning variants; pass --reasoning-effort explicitly", model)
		}
		return ReasoningPolicy{Effort: requested, Source: "explicit", Support: "unknown"}, nil
	}
	effort, source := requested, "explicit"
	if effort == "" {
		effort, source = DefaultReasoningEffort, "default"
	}
	if effort == "max" && !supportsMax {
		return ReasoningPolicy{}, fmt.Errorf("reasoning effort %q is not supported by OpenCode model %q", effort, model)
	}
	return ReasoningPolicy{Effort: effort, Source: source, Support: "native"}, nil
}

func knownOpenAIModel(model string) (known, supportsMax bool) {
	name := strings.TrimPrefix(model, "openai/")
	for _, suffix := range []string{"-fast", "-pro"} {
		name = strings.TrimSuffix(name, suffix)
	}
	switch name {
	case "gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna":
		return true, true
	case "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano":
		return true, false
	default:
		return false, false
	}
}
