package provider

import "sort"

// CatalogEntry is one stable Hatch surface alias. It is intentionally local
// configuration, not a live provider listing: agents must only be taught
// aliases that Hatch can invoke.
type CatalogEntry struct {
	Surface string `json:"surface"`
	Alias   string `json:"alias"`
	Model   string `json:"model"`
}

// Surface model catalogs are the single source used by CLI alias resolution and
// doctor drift checks. Treat these maps as immutable.
var CodexSurfaceModels = map[string]string{
	"sol":   "openai/gpt-5.6-sol",
	"terra": "openai/gpt-5.6-terra",
	"luna":  "openai/gpt-5.6-luna",
	"nano":  "openai/gpt-5.4-nano",
	"mini":  "openai/gpt-5.4-mini",
	"max":   "openai/gpt-5.5",
}

var CursorSurfaceModels = map[string]string{
	"grok":    "cursor-grok-4.6-high",
	"kimi-k3": "kimi-k3",
}

var OpenRouterSurfaceModels = map[string]string{
	"deepseek-v4-flash": "openrouter/deepseek/deepseek-v4-flash-0731",
	"deepseek-v4-pro":   "openrouter/deepseek/deepseek-v4-pro-0813",
	"glm-5.3-flash":     "openrouter/z-ai/glm-5.3-flash",
}

var GeminiSurfaceModels = map[string]string{
	"flash":                   "google-antigravity/gemini-3.7-flash-tiered",
	"3.7":                     "google-antigravity/gemini-3.7-flash-tiered",
	"gemini-3.7-flash-tiered": "google-antigravity/gemini-3.7-flash-tiered",
}

// SurfaceCatalog returns every surfaced model in a stable order for help,
// automation, and generated agent context.
func SurfaceCatalog() []CatalogEntry {
	catalogs := []struct {
		surface string
		models  map[string]string
	}{
		{"claude", map[string]string{"haiku": "haiku", "sonnet": "sonnet", "opus": "opus", "fable": "fable"}},
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
