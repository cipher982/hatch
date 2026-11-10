package provider

import (
	"sort"
	"strings"
)

// CatalogEntry is one stable Hatch surface alias. It is intentionally local
// configuration, not a live provider listing: agents must only be taught
// aliases that Hatch can invoke.
type CatalogEntry struct {
	Surface string `json:"surface"`
	Alias   string `json:"alias"`
	Model   string `json:"model"`
}

// RoutingPolicy configures upstream provider preferences and fallbacks for models
// routed through aggregators (such as OpenRouter).
type RoutingPolicy struct {
	ProviderOrder  []string `json:"provider_order,omitempty"`
	AllowFallbacks bool     `json:"allow_fallbacks"`
}

// ModelSpec defines a single surfaced model, its canonical model identifier,
// backend harness, retired aliases, and routing constraints.
type ModelSpec struct {
	Surface    string         `json:"surface"`
	Alias      string         `json:"alias"`
	Model      string         `json:"model"`
	Backend    string         `json:"backend"`
	Deprecated []string       `json:"deprecated,omitempty"`
	Routing    *RoutingPolicy `json:"routing,omitempty"`
	// RoutingPrefix registers Routing under a model-family prefix instead of the
	// exact Model, so later dated snapshots (deepseek-v4.1-flash-1031) keep the
	// same upstream pin without a catalog edit.
	RoutingPrefix string `json:"routing_prefix,omitempty"`
	// NoShorthand keeps the alias reachable only through its surface
	// (`hatch bedrock opus`), so it cannot shadow another surface's bare alias
	// (`hatch opus`).
	NoShorthand bool `json:"no_shorthand,omitempty"`
	// PriorityEfforts lists the reasoning efforts that request the provider's
	// priority processing tier. Priority is billed above standard rates, so it
	// stays opt-in per model and effort instead of a blanket default.
	PriorityEfforts []string `json:"priority_efforts,omitempty"`
}

// ModelRegistry is the single source of truth for surfaced models in Hatch.
// Order defines the stable public model preference used for help and choices.
var ModelRegistry = []ModelSpec{
	// Codex: the personal OpenAI account (ChatGPT/Codex subscription or an OpenAI
	// API key). Explicit-only (`hatch codex <alias>`): bare OpenAI aliases
	// resolve to the Bedrock surface below, which is the Zeta route.
	{Surface: "codex", Alias: "astra", Model: "openai/gpt-6-astra", Backend: "pi", NoShorthand: true},
	{Surface: "codex", Alias: "sol", Model: "openai/gpt-6.1-sol", Backend: "pi", NoShorthand: true},
	{Surface: "codex", Alias: "terra", Model: "openai/gpt-6.1-terra", Backend: "pi", NoShorthand: true},
	{Surface: "codex", Alias: "luna", Model: "openai/gpt-6-luna", Backend: "pi", NoShorthand: true, PriorityEfforts: []string{"xhigh"}},
	{Surface: "codex", Alias: "nano", Model: "openai/gpt-5.4-nano", Backend: "pi"},
	{Surface: "codex", Alias: "mini", Model: "openai/gpt-5.4-mini", Backend: "pi"},
	{Surface: "codex", Alias: "max", Model: "openai/gpt-5.5", Backend: "pi"},

	// Claude
	{Surface: "claude", Alias: "haiku", Model: "claude-haiku-5-5", Backend: "claude"},
	{Surface: "claude", Alias: "sonnet", Model: "claude-sonnet-5-5", Backend: "claude"},
	{Surface: "claude", Alias: "opus", Model: "claude-opus-5-5", Backend: "claude", Deprecated: []string{"opus-5"}},
	{Surface: "claude", Alias: "fable", Model: "claude-fable-5-2", Backend: "claude", Deprecated: []string{"fable-5", "fable-5.1"}},
	{Surface: "claude", Alias: "fable-5.2", Model: "claude-fable-5-2", Backend: "claude"},

	// Bedrock: Zeta work. Runs on the employer AWS account through a coding
	// harness and hosts both Anthropic Claude and OpenAI models. Bare OpenAI
	// aliases resolve here; personal work uses the claude/gemini surfaces.
	{Surface: "bedrock", Alias: "opus", Model: "amazon-bedrock/global.anthropic.claude-opus-5-5", Backend: "pi", NoShorthand: true},
	{Surface: "bedrock", Alias: "sonnet", Model: "amazon-bedrock/global.anthropic.claude-sonnet-5-5", Backend: "pi", NoShorthand: true},
	{Surface: "bedrock", Alias: "astra", Model: "amazon-bedrock/global.openai.gpt-6-astra", Backend: "pi"},
	{Surface: "bedrock", Alias: "sol", Model: "amazon-bedrock/global.openai.gpt-6.1-sol", Backend: "pi"},
	{Surface: "bedrock", Alias: "terra", Model: "amazon-bedrock/global.openai.gpt-6.1-terra", Backend: "pi"},
	{Surface: "bedrock", Alias: "luna", Model: "amazon-bedrock/global.openai.gpt-6-luna", Backend: "pi"},

	// Cursor: Zeta work. Runs on the Cursor subscription (cursor-agent login
	// drose@zetaglobal.com). gemini-4-argon carries no bare shorthand so a
	// personal caller cannot silently bill the employer account.
	{Surface: "cursor", Alias: "grok", Model: "grok-4.7-high", Backend: "cursor"},
	{Surface: "cursor", Alias: "gemini-4-argon", Model: "gemini-4-argon-medium", Backend: "cursor", NoShorthand: true},

	// Gemini: personal work. Gemini 4 Argon runs through Oh My Pi on the
	// personal Google AI Ultra subscription; Zeta Gemini work uses the
	// explicit `hatch cursor gemini-4-argon` route instead.
	{Surface: "gemini", Alias: "argon", Model: "google-antigravity/gemini-4-argon", Backend: "omp"},

	// OpenCode Go: personal flat-rate plan. `hatch deepseek` lands here. The
	// workspace privacy region must be Global or DeepSeek calls are refused.
	{Surface: "opencode-go", Alias: "deepseek", Model: "opencode-go/deepseek-v4.1-flash", Backend: "pi"},

	// OpenRouter
	{
		Surface: "openrouter",
		Alias:   "deepseek",
		Model:   "openrouter/deepseek/deepseek-v4.1-flash-1017",
		Backend: "pi",
		// The bare `deepseek` shorthand belongs to the OpenCode Go surface
		// (flat-rate plan); OpenRouter stays reachable as an explicit fallback.
		NoShorthand: true,
		Deprecated:  []string{"deepseek-v4.1-flash", "deepseek-v4-flash", "deepseek-v4-pro"},
		// The September alias resolved to the 20260910 snapshot; 1017 is the
		// October 17 checkpoint. Pin the family so future snapshots inherit it.
		RoutingPrefix: "openrouter/deepseek/deepseek-v4.1-flash",
		Routing: &RoutingPolicy{
			ProviderOrder:  []string{"DeepSeek"},
			AllowFallbacks: false,
		},
	},
	{
		Surface: "openrouter",
		Alias:   "glm-5.3-flash",
		Model:   "openrouter/z-ai/glm-5.3-flash",
		Backend: "pi",
		Routing: &RoutingPolicy{
			ProviderOrder:  []string{"Modal", "Z.AI", "Novita", "Together", "Parasail", "DeepInfra"},
			AllowFallbacks: true,
		},
	},

	// Cursor
	{Surface: "cursor", Alias: "kimi-k3", Model: "kimi-k3", Backend: "cursor"},
}

// Surface model maps generated from ModelRegistry for backward compatibility.
var (
	BedrockSurfaceModels    = map[string]string{}
	ClaudeSurfaceModels     = map[string]string{}
	CodexSurfaceModels      = map[string]string{}
	CursorSurfaceModels     = map[string]string{}
	GeminiSurfaceModels     = map[string]string{}
	OpenRouterSurfaceModels = map[string]string{}
	OpenCodeGoSurfaceModels = map[string]string{}

	shorthandMap       = map[string]string{}
	deprecatedAliasMap = map[string]string{}
	surfaceBackendMap  = map[string]string{}
	surfacedPairMap    = map[string]bool{}
	routingPolicyMap   = map[string]*RoutingPolicy{}
	priorityEffortMap  = map[string]map[string]bool{}
	routingPrefixes    = []string{}
)

func init() {
	for _, spec := range ModelRegistry {
		switch spec.Surface {
		case "bedrock":
			BedrockSurfaceModels[spec.Alias] = spec.Model
		case "claude":
			ClaudeSurfaceModels[spec.Alias] = spec.Model
		case "codex":
			CodexSurfaceModels[spec.Alias] = spec.Model
		case "cursor":
			CursorSurfaceModels[spec.Alias] = spec.Model
		case "gemini":
			GeminiSurfaceModels[spec.Alias] = spec.Model
		case "openrouter":
			OpenRouterSurfaceModels[spec.Alias] = spec.Model
		case "opencode-go":
			OpenCodeGoSurfaceModels[spec.Alias] = spec.Model
		}

		if !spec.NoShorthand {
			shorthandMap[spec.Alias] = spec.Surface
		}
		for _, dep := range spec.Deprecated {
			shorthandMap[dep] = spec.Surface
			deprecatedAliasMap[dep] = spec.Surface
		}
		surfaceBackendMap[spec.Surface] = spec.Backend
		surfacedPairMap[spec.Surface+"."+spec.Alias] = true

		if spec.Routing != nil {
			key := spec.Model
			if spec.RoutingPrefix != "" {
				key = spec.RoutingPrefix
			}
			routingPolicyMap[key] = spec.Routing
		}
		if len(spec.PriorityEfforts) > 0 {
			efforts := make(map[string]bool, len(spec.PriorityEfforts))
			for _, effort := range spec.PriorityEfforts {
				efforts[effort] = true
			}
			priorityEffortMap[spec.Model] = efforts
		}
	}
	for prefix := range routingPolicyMap {
		routingPrefixes = append(routingPrefixes, prefix)
	}
	sort.Slice(routingPrefixes, func(i, j int) bool {
		return len(routingPrefixes[i]) > len(routingPrefixes[j])
	})
}

// SurfaceAccount names who pays for, and may see, work sent through a surface
// or a manifest surface label such as "bedrock.sol": "zeta" for the employer
// accounts (Bedrock, Cursor) and "personal" for everything else. Work and
// personal content never cross, so a follow-up refuses to move a run's text
// from one to the other.
func SurfaceAccount(surface string) string {
	name, _, _ := strings.Cut(surface, ".")
	switch name {
	case "bedrock", "cursor":
		return "zeta"
	default:
		return "personal"
	}
}

// ShorthandSurface returns the canonical surface for a shorthand model alias (active or deprecated).
func ShorthandSurface(alias string) string {
	return shorthandMap[alias]
}

// IsDeprecatedAlias returns whether an alias is a retired alias for the given surface.
func IsDeprecatedAlias(surface, alias string) bool {
	return deprecatedAliasMap[alias] == surface
}

// IsSurfacedModel reports whether surface.alias is a model Hatch currently
// surfaces. A NoShorthand alias is still surfaced on its own surface even when
// a different surface owns the bare shorthand.
func IsSurfacedModel(surface, alias string) bool {
	return surfacedPairMap[surface+"."+alias]
}

// SurfaceBackend returns the default backend harness for a surfaced provider name.
func SurfaceBackend(surface string) string {
	return surfaceBackendMap[surface]
}

// FindRoutingPolicy returns the routing policy for an OpenRouter model if configured.
// It matches exact model names first, then the longest registered prefix deterministically.
func FindRoutingPolicy(model string) *RoutingPolicy {
	if policy, ok := routingPolicyMap[model]; ok {
		return policy
	}
	for _, prefix := range routingPrefixes {
		if strings.HasPrefix(model, prefix) {
			return routingPolicyMap[prefix]
		}
	}
	return nil
}

// RequestsPriorityTier reports whether a surfaced model at the given reasoning
// effort should ask its provider for priority processing (OpenAI's
// service_tier=priority). Priority is billed above standard rates, so only the
// model/effort pairs declared in ModelRegistry qualify.
func RequestsPriorityTier(model, effort string) bool {
	return priorityEffortMap[model][effort]
}

// PublicModelOrder returns the documented public alias order across all surfaces.
func PublicModelOrder() []string {
	order := make([]string, 0, len(ModelRegistry))
	seen := make(map[string]bool, len(ModelRegistry))
	for _, m := range ModelRegistry {
		if !seen[m.Alias] {
			seen[m.Alias] = true
			order = append(order, m.Alias)
		}
	}
	return order
}

// SurfaceCatalog returns every surfaced model in a stable order for help,
// automation, and generated agent context.
func SurfaceCatalog() []CatalogEntry {
	catalogs := []struct {
		surface string
		models  map[string]string
	}{
		{"bedrock", BedrockSurfaceModels},
		{"claude", ClaudeSurfaceModels},
		{"codex", CodexSurfaceModels},
		{"cursor", CursorSurfaceModels},
		{"gemini", GeminiSurfaceModels},
		{"openrouter", OpenRouterSurfaceModels},
		{"opencode-go", OpenCodeGoSurfaceModels},
	}
	entries := make([]CatalogEntry, 0, 16)
	for _, catalog := range catalogs {
		aliases := make([]string, 0, len(catalog.models))
		for alias := range catalog.models {
			aliases = append(aliases, alias)
		}
		sort.Strings(aliases)
		for _, alias := range aliases {
			entries = append(entries, CatalogEntry{Surface: catalog.surface, Alias: alias, Model: catalog.models[alias]})
		}
	}
	return entries
}
