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
	// PriorityEfforts lists the reasoning efforts that request the provider's
	// priority processing tier. Priority is billed above standard rates, so it
	// stays opt-in per model and effort instead of a blanket default.
	PriorityEfforts []string `json:"priority_efforts,omitempty"`
}

// ModelRegistry is the single source of truth for surfaced models in Hatch.
// Order defines the stable public model preference used for help and choices.
var ModelRegistry = []ModelSpec{
	// Codex
	{Surface: "codex", Alias: "astra", Model: "openai/gpt-6-astra", Backend: "opencode", Deprecated: []string{"sol"}},
	{Surface: "codex", Alias: "terra", Model: "openai/gpt-5.6-terra", Backend: "opencode"},
	{Surface: "codex", Alias: "luna", Model: "openai/gpt-5.6-luna", Backend: "opencode", PriorityEfforts: []string{"xhigh"}},
	{Surface: "codex", Alias: "nano", Model: "openai/gpt-5.4-nano", Backend: "opencode"},
	{Surface: "codex", Alias: "mini", Model: "openai/gpt-5.4-mini", Backend: "opencode"},
	{Surface: "codex", Alias: "max", Model: "openai/gpt-5.5", Backend: "opencode"},

	// Claude
	{Surface: "claude", Alias: "haiku", Model: "haiku", Backend: "claude"},
	{Surface: "claude", Alias: "sonnet", Model: "sonnet", Backend: "claude"},
	{Surface: "claude", Alias: "opus", Model: "claude-opus-5-5", Backend: "claude", Deprecated: []string{"opus-5"}},
	{Surface: "claude", Alias: "fable", Model: "claude-fable-5-1", Backend: "claude", Deprecated: []string{"fable-5"}},
	{Surface: "claude", Alias: "fable-5.1", Model: "claude-fable-5-1", Backend: "claude"},

	// Cursor
	{Surface: "cursor", Alias: "grok", Model: "grok-4.7-high", Backend: "cursor"},

	// Gemini
	{Surface: "gemini", Alias: "flash", Model: "google-antigravity/gemini-3.8-flash-low", Backend: "omp", Deprecated: []string{"pro", "3.7", "gemini-3.7-flash-tiered"}},
	{Surface: "gemini", Alias: "3.8", Model: "google-antigravity/gemini-3.8-flash-low", Backend: "omp"},
	{Surface: "gemini", Alias: "gemini-3.8-flash-low", Model: "google-antigravity/gemini-3.8-flash-low", Backend: "omp"},

	// OpenRouter
	{
		Surface:    "openrouter",
		Alias:      "deepseek-v4.1-flash",
		Model:      "openrouter/deepseek/deepseek-v4.1-flash",
		Backend:    "opencode",
		Deprecated: []string{"deepseek-v4-flash", "deepseek-v4-pro"},
		Routing: &RoutingPolicy{
			ProviderOrder:  []string{"DeepSeek"},
			AllowFallbacks: false,
		},
	},
	{
		Surface: "openrouter",
		Alias:   "glm-5.3-flash",
		Model:   "openrouter/z-ai/glm-5.3-flash",
		Backend: "opencode",
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
	ClaudeSurfaceModels     = map[string]string{}
	CodexSurfaceModels      = map[string]string{}
	CursorSurfaceModels     = map[string]string{}
	GeminiSurfaceModels     = map[string]string{}
	OpenRouterSurfaceModels = map[string]string{}

	shorthandMap       = map[string]string{}
	deprecatedAliasMap = map[string]string{}
	surfaceBackendMap  = map[string]string{}
	routingPolicyMap   = map[string]*RoutingPolicy{}
	priorityEffortMap  = map[string]map[string]bool{}
	routingPrefixes    = []string{}
)

func init() {
	for _, spec := range ModelRegistry {
		switch spec.Surface {
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
		}

		shorthandMap[spec.Alias] = spec.Surface
		for _, dep := range spec.Deprecated {
			shorthandMap[dep] = spec.Surface
			deprecatedAliasMap[dep] = spec.Surface
		}
		surfaceBackendMap[spec.Surface] = spec.Backend

		if spec.Routing != nil {
			routingPolicyMap[spec.Model] = spec.Routing
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

// ShorthandSurface returns the canonical surface for a shorthand model alias (active or deprecated).
func ShorthandSurface(alias string) string {
	return shorthandMap[alias]
}

// IsDeprecatedAlias returns whether an alias is a retired alias for the given surface.
func IsDeprecatedAlias(surface, alias string) bool {
	return deprecatedAliasMap[alias] == surface
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
	for _, m := range ModelRegistry {
		order = append(order, m.Alias)
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
		{"claude", ClaudeSurfaceModels},
		{"codex", CodexSurfaceModels},
		{"cursor", CursorSurfaceModels},
		{"gemini", GeminiSurfaceModels},
		{"openrouter", OpenRouterSurfaceModels},
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
