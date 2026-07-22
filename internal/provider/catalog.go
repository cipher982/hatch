package provider

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

var OpenRouterSurfaceModels = map[string]string{
	"deepseek-v4-pro": "openrouter/deepseek/deepseek-v4-pro",
	"kimi-k3":         "openrouter/~moonshotai/kimi-latest",
}
