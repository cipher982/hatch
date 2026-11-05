package provider

import (
	"embed"
	"strings"
)

//go:embed assets/pi_dcg_guard.mjs
var assets embed.FS

// PiDCGGuardSource is the Pi extension that relays bash commands to DCG.
func PiDCGGuardSource() []byte {
	data, err := assets.ReadFile("assets/pi_dcg_guard.mjs")
	if err != nil {
		panic("embedded DCG guard missing: " + err.Error())
	}
	return data
}

// openAIPriorityServiceTier is the OpenAI service tier that puts a request in
// the priority processing queue.
const openAIPriorityServiceTier = "priority"

// piCatalogGaps are surfaced models newer than Pi's built-in catalog, keyed by
// the "provider/id" form Hatch passes to --model. Hatch declares them in the
// per-run models.json so a stale Pi catalog never blocks a current alias.
var piCatalogGaps = map[string]func() map[string]any{
	"amazon-bedrock/global.openai.gpt-6.1-sol": func() map[string]any {
		return piOpenAIModel("global.openai.gpt-6.1-sol", "GPT-6.1 Sol (Global)", "bedrock-converse-stream", false)
	},
	"amazon-bedrock/global.openai.gpt-6.1-terra": func() map[string]any {
		return piOpenAIModel("global.openai.gpt-6.1-terra", "GPT-6.1 Terra (Global)", "bedrock-converse-stream", true)
	},
	"openai/gpt-6.1-sol": func() map[string]any {
		return piOpenAIModel("gpt-6.1-sol", "GPT-6.1 Sol", "openai-responses", false)
	},
	"openai/gpt-6.1-terra": func() map[string]any {
		return piOpenAIModel("gpt-6.1-terra", "GPT-6.1 Terra", "openai-responses", true)
	},
	"openrouter/deepseek/deepseek-v4.1-flash-1017": func() map[string]any {
		return map[string]any{
			"id": "deepseek/deepseek-v4.1-flash-1017", "name": "DeepSeek V4.1 Flash (1017)",
			"api": "openai-completions", "reasoning": true, "input": []string{"text"},
			"contextWindow": 1048576, "maxTokens": 131072,
			"thinkingLevelMap": map[string]any{"off": "none", "minimal": nil, "low": "low", "medium": nil, "high": "high", "xhigh": nil, "max": "max"},
			"compat": map[string]any{
				"supportsDeveloperRole": false, "thinkingFormat": "openrouter", "supportsStrictMode": true,
				"sendSessionAffinityHeaders": true, "requiresReasoningContentOnAssistantMessages": true,
			},
		}
	},
}

// piOpenAIModel declares a GPT-6.1 model. Terra and Luna accept "none"
// reasoning; Astra and Sol reject it.
func piOpenAIModel(id, name, api string, acceptsNone bool) map[string]any {
	levels := map[string]any{"off": nil, "minimal": nil, "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max"}
	if acceptsNone {
		levels["off"] = "none"
	}
	model := map[string]any{
		"id": id, "name": name, "api": api, "reasoning": true, "input": []string{"text", "image"},
		"contextWindow": 1050000, "maxTokens": 128000, "thinkingLevelMap": levels,
	}
	if api == "openai-responses" {
		model["compat"] = map[string]any{"supportsStrictMode": true}
	}
	return model
}

// PiModelsJSON returns the per-run models.json for one model: catalog gaps,
// aggregator routing pins (an OpenRouter provider order that keeps prefix
// caching warm) and priority processing for the model and effort pairs that
// request it. Nil when Pi's own catalog already covers the model. It never
// contains credentials.
func PiModelsJSON(model, effort string) []byte {
	providerName, id, ok := strings.Cut(model, "/")
	if !ok {
		return nil
	}
	var routing map[string]any
	if policy := FindRoutingPolicy(model); policy != nil && providerName == "openrouter" {
		routing = map[string]any{"allow_fallbacks": policy.AllowFallbacks}
		if len(policy.ProviderOrder) > 0 {
			routing["order"] = policy.ProviderOrder
		}
	}
	entry := map[string]any{}
	if build, gap := piCatalogGaps[model]; gap {
		definition := build()
		if routing != nil {
			compat, _ := definition["compat"].(map[string]any)
			if compat == nil {
				compat = map[string]any{}
			}
			compat["openRouterRouting"] = routing
			definition["compat"] = compat
		}
		entry["models"] = []any{definition}
	} else {
		override := map[string]any{}
		if routing != nil {
			override["compat"] = map[string]any{"openRouterRouting": routing}
		}
		if RequestsPriorityTier(model, effort) {
			override["samplingParams"] = map[string]any{"service_tier": openAIPriorityServiceTier}
		}
		if len(override) == 0 {
			return nil
		}
		entry["modelOverrides"] = map[string]any{id: override}
	}
	encoded, err := marshalPlain(map[string]any{"providers": map[string]any{providerName: entry}})
	if err != nil {
		panic("static pi models config cannot fail to marshal")
	}
	return encoded
}

// PiDeclaresModel reports whether Hatch's own models.json supplies a model that
// Pi's built-in catalog may lack.
func PiDeclaresModel(model string) bool {
	_, declared := piCatalogGaps[model]
	return declared
}
