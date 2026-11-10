package provider

import (
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func TestSurfaceCatalogIncludesCurrentOpenRouterSurfaces(t *testing.T) {
	entries := SurfaceCatalog()
	if !slices.Contains(entries, CatalogEntry{Surface: "openrouter", Alias: "deepseek", Model: "openrouter/deepseek/deepseek-v4.1-flash-1017"}) ||
		!slices.Contains(entries, CatalogEntry{Surface: "openrouter", Alias: "glm-5.3-flash", Model: "openrouter/z-ai/glm-5.3-flash"}) {
		t.Fatalf("catalog = %#v", entries)
	}
	for _, entry := range entries {
		if entry.Surface == "openrouter" && (entry.Alias == "deepseek-v4.1-flash" || entry.Alias == "deepseek-v4-pro" || entry.Alias == "deepseek-v4-flash") {
			t.Fatalf("catalog must not contain deprecated alias %q: %#v", entry.Alias, entries)
		}
	}
}
func TestModelRegistryInvariants(t *testing.T) {
	seenAliases := map[string]bool{}
	for _, spec := range ModelRegistry {
		if spec.Surface == "" || spec.Alias == "" || spec.Model == "" || spec.Backend == "" {
			t.Fatalf("incomplete ModelSpec: %#v", spec)
		}
		key := spec.Surface + "." + spec.Alias
		if seenAliases[key] {
			t.Fatalf("duplicate model alias %q", key)
		}
		seenAliases[key] = true
		if backend := SurfaceBackend(spec.Surface); backend != spec.Backend {
			t.Fatalf("SurfaceBackend(%q) = %q, want %q", spec.Surface, backend, spec.Backend)
		}
		if shorthand := ShorthandSurface(spec.Alias); !spec.NoShorthand && shorthand != spec.Surface {
			t.Fatalf("ShorthandSurface(%q) = %q, want %q", spec.Alias, shorthand, spec.Surface)
		} else if spec.NoShorthand && shorthand == spec.Surface {
			t.Fatalf("NoShorthand alias %q still resolves to %q", spec.Alias, spec.Surface)
		}
		for _, dep := range spec.Deprecated {
			if !IsDeprecatedAlias(spec.Surface, dep) {
				t.Fatalf("IsDeprecatedAlias(%q, %q) = false, want true", spec.Surface, dep)
			}
			if shorthand := ShorthandSurface(dep); shorthand != spec.Surface {
				t.Fatalf("ShorthandSurface(deprecated %q) = %q, want %q", dep, shorthand, spec.Surface)
			}
		}
		for _, effort := range spec.PriorityEfforts {
			if !IsValidReasoningEffort(effort) {
				t.Fatalf("%s declares unknown priority effort %q", spec.Model, effort)
			}
			if !RequestsPriorityTier(spec.Model, effort) {
				t.Fatalf("RequestsPriorityTier(%q, %q) = false, want true", spec.Model, effort)
			}
		}
	}
}

func TestFindRoutingPolicyDeterministicLongestPrefix(t *testing.T) {
	policy := FindRoutingPolicy("openrouter/deepseek/deepseek-v4.1-flash-1017")
	if policy == nil || len(policy.ProviderOrder) != 1 || policy.ProviderOrder[0] != "DeepSeek" || policy.AllowFallbacks {
		t.Fatalf("unexpected policy for deepseek-v4.1-flash-1017: %#v", policy)
	}
	// The family prefix keeps the pin for the undated alias and future snapshots.
	for _, model := range []string{
		"openrouter/deepseek/deepseek-v4.1-flash",
		"openrouter/deepseek/deepseek-v4.1-flash-0910",
		"openrouter/deepseek/deepseek-v4.1-flash-1031",
	} {
		policySnapshot := FindRoutingPolicy(model)
		if policySnapshot == nil || len(policySnapshot.ProviderOrder) != 1 || policySnapshot.ProviderOrder[0] != "DeepSeek" {
			t.Fatalf("unexpected policy for %s: %#v", model, policySnapshot)
		}
	}
	if plain := FindRoutingPolicy("openai/gpt-6-astra"); plain != nil {
		t.Fatalf("unrouted model returned routing policy: %#v", plain)
	}
}

func TestPreparePromptOracle(t *testing.T) {
	got := PreparePrompt("oracle prompt")
	for _, want := range []string{
		"A human is waiting for a useful answer by the behavioral deadline",
		"Do not promise or assume an exact wall-clock duration",
		"Use focused checks by default",
		"Time-box expensive tests, scratch clones or worktrees, broad repository scans, and network or fetch work",
		"Check the budget mid-run",
		"Once evidence is sufficient, stop using tools and synthesize",
		"At the late budget threshold, stop launching tools",
		"Preserve useful partial findings and do not redo completed work",
		"Read each file at most once per run",
		"Use offset=N to continue",
		"never re-run a search that already returned identical results",
		"Start writing your answer once the core files are read",
		"list what you did not read",
		"Nested Hatch runs are allowed",
		"not a ban on child Hatch calls",
		"Never wait indefinitely for a child",
		"findings, confidence, unresolved questions, and the exact next action",
		"must never be presented as approved or complete",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("prepared prompt missing guardrail %q", want)
		}
	}
	if !strings.HasSuffix(got, "User task:\noracle prompt") {
		t.Fatalf("prepared prompt does not preserve user task: %q", got)
	}
}

func TestBuildOracleInvocations(t *testing.T) {
	tests := []struct {
		name string
		req  Request
		argv []string
	}{
		{"gemini", Request{Backend: "gemini", Prompt: "oracle prompt"}, []string{"gemini", "--model", "gemini-3-pro-preview", "--yolo", "--skip-trust", "-p", "-"}},
		{"cursor", Request{Backend: "cursor", Model: "grok-4.7-high", Prompt: "oracle prompt"}, []string{"cursor-agent", "--print", "--trust", "--model", "grok-4.7-high", "--output-format", "stream-json", "--force", PreparePrompt("oracle prompt")}},
		{"antigravity", Request{Backend: "antigravity", Model: "gemini-4-argon-medium", Prompt: "oracle prompt"}, []string{"agy", "--dangerously-skip-permissions", "--output-format", "stream-json", "--model", "gemini-4-argon-medium", "-p", PreparePrompt("oracle prompt")}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := Build(test.req)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got.Argv, test.argv) {
				t.Fatalf("argv = %#v, want %#v", got.Argv, test.argv)
			}
		})
	}
}

func TestBuildAdvancedBackendInvocations(t *testing.T) {
	t.Run("claude explicit stream", func(t *testing.T) {
		got, err := Build(Request{
			Backend: "claude", Model: "claude-opus-5-5", Prompt: "p", OutputFormat: "stream-json",
			IncludePartialMessages: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		wantTail := []string{"--include-partial-messages"}
		if !reflect.DeepEqual(got.Argv[len(got.Argv)-len(wantTail):], wantTail) || got.Adapter != "claude" ||
			got.ReasoningPolicy != (ReasoningPolicy{Effort: "low", Source: "default", Support: "native"}) {
			t.Fatalf("claude invocation = %#v", got)
		}
	})

	t.Run("claude passes explicit effort", func(t *testing.T) {
		got, err := Build(Request{Backend: "claude", Model: "claude-opus-5-5", Prompt: "p", ReasoningEffort: "high"})
		if err != nil {
			t.Fatal(err)
		}
		want := []string{
			"claude", "--verbose", "--print", "-", "--output-format", "stream-json",
			"--model", "claude-opus-5-5", "--dangerously-skip-permissions", "--setting-sources", "local",
			"--no-session-persistence", "--tools", "default", "--effort", "high", "--include-partial-messages",
		}
		if !reflect.DeepEqual(got.Argv, want) || got.ReasoningPolicy != (ReasoningPolicy{Effort: "high", Source: "explicit", Support: "native"}) {
			t.Fatalf("claude invocation = %#v", got)
		}
	})

	t.Run("explicit claude stream remains raw", func(t *testing.T) {
		got, err := Build(Request{Backend: "claude", Model: "claude-opus-5-5", Prompt: "p", OutputFormat: "stream-json", RawStructuredOutput: true})
		if err != nil {
			t.Fatal(err)
		}
		if got.Adapter != "raw" || got.StreamFormat != "jsonl" {
			t.Fatalf("explicit stream invocation = %#v", got)
		}
	})

	t.Run("raw codex", func(t *testing.T) {
		got, err := Build(Request{
			Backend: "codex", Model: "gpt-6.1-sol", Prompt: "p", APIKey: "secret",
			ReasoningEffort: "high", SkipGitRepoCheck: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		want := []string{
			"codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "--ignore-user-config", "--ephemeral", "-m", "gpt-6.1-sol",
			"-c", "model_reasoning_effort=high", "--skip-git-repo-check",
		}
		if !reflect.DeepEqual(got.Argv, want) || got.SetEnv["OPENAI_API_KEY"] != "secret" {
			t.Fatalf("codex invocation = %#v", got)
		}
	})

	for _, backend := range []string{"pi", "omp"} {
		t.Run(backend+" uses explicit headless JSON mode", func(t *testing.T) {
			got, err := Build(Request{Backend: backend, Model: "openai/gpt-6.1-sol", Prompt: "p", APIKey: "secret"})
			if err != nil {
				t.Fatal(err)
			}
			if got.Argv[0] != backend || got.StreamFormat != "jsonl" || got.Adapter != backend || got.Argv[len(got.Argv)-1] != PreparePrompt("p") ||
				got.SetEnv["OPENAI_API_KEY"] != "secret" || got.ReasoningPolicy.Effort != "medium" {
				t.Fatalf("%s invocation = %#v", backend, got)
			}
			joined := strings.Join(got.Argv, " ")
			for _, want := range []string{"--mode json", "--no-session", "--model openai/gpt-6.1-sol", "--thinking medium"} {
				if !strings.Contains(joined, want) {
					t.Fatalf("%s argv lacks %q: %#v", backend, want, got.Argv)
				}
			}
		})
	}

	t.Run("omp requests priority tier for luna xhigh", func(t *testing.T) {
		got, err := Build(Request{Backend: "omp", Model: "openai/gpt-6-luna", Prompt: "p", APIKey: "secret", ReasoningEffort: "xhigh"})
		if err != nil {
			t.Fatal(err)
		}
		wantTail := []string{"--thinking", "xhigh", "--service-tier", "priority", PreparePrompt("p")}
		if !reflect.DeepEqual(got.Argv[len(got.Argv)-len(wantTail):], wantTail) {
			t.Fatalf("omp luna xhigh argv = %#v", got.Argv)
		}
	})

	t.Run("omp below xhigh and other omp models stay on the standard tier", func(t *testing.T) {
		for _, request := range []Request{
			{Backend: "omp", Model: "openai/gpt-6-luna", Prompt: "p", APIKey: "secret", ReasoningEffort: "high"},
			{Backend: "omp", Model: "openai/gpt-6.1-sol", Prompt: "p", APIKey: "secret", ReasoningEffort: "xhigh"},
		} {
			got, err := Build(request)
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(strings.Join(got.Argv, " "), "--service-tier") {
				t.Fatalf("%s %s %s argv requests a service tier: %#v", request.Backend, request.Model, request.ReasoningEffort, got.Argv)
			}
		}
	})

	t.Run("bedrock surface runs Claude 5.5 on the Zeta AWS profile", func(t *testing.T) {
		for _, backend := range []string{"pi", "omp"} {
			got, err := Build(Request{Backend: backend, Model: BedrockSurfaceModels["opus"], Prompt: "p"})
			if err != nil {
				t.Fatal(err)
			}
			if got.SetEnv["AWS_PROFILE"] != "zh-ml-mlengineer" || got.SetEnv["AWS_REGION"] != "us-east-1" {
				t.Fatalf("%s bedrock env = %#v", backend, got.SetEnv)
			}
			if got.ReasoningPolicy != (ReasoningPolicy{Effort: "low", Source: "default", Support: "native"}) {
				t.Fatalf("%s bedrock reasoning = %#v", backend, got.ReasoningPolicy)
			}
			if !slices.Contains(got.Argv, "--thinking") {
				t.Fatalf("%s argv lacks --thinking: %#v", backend, got.Argv)
			}
		}
	})

	t.Run("retired raw bedrock backend is rejected", func(t *testing.T) {
		if _, err := Build(Request{Backend: "bedrock", Prompt: "p"}); err == nil {
			t.Fatal("raw bedrock backend must no longer build")
		}
	})
}

func piConfig(t *testing.T, model, effort string) map[string]any {
	t.Helper()
	invocation, err := Build(Request{Backend: "pi", Model: model, Prompt: "prompt", APIKey: "fake", ReasoningEffort: effort})
	if err != nil {
		t.Fatal(err)
	}
	if len(invocation.PiModelsJSON) == 0 {
		return nil
	}
	var config map[string]any
	if err := json.Unmarshal(invocation.PiModelsJSON, &config); err != nil {
		t.Fatalf("models.json is not valid JSON: %v\n%s", err, invocation.PiModelsJSON)
	}
	return config["providers"].(map[string]any)
}

func TestPiDeepSeekRunsPinnedToTheDeepSeekProvider(t *testing.T) {
	providers := piConfig(t, "openrouter/deepseek/deepseek-v4.1-flash-1017", "")
	models := providers["openrouter"].(map[string]any)["models"].([]any)
	model := models[0].(map[string]any)
	routing := model["compat"].(map[string]any)["openRouterRouting"].(map[string]any)
	order := routing["order"].([]any)
	if model["id"] != "deepseek/deepseek-v4.1-flash-1017" || routing["allow_fallbacks"] != false || len(order) != 1 || order[0] != "DeepSeek" {
		t.Fatalf("deepseek routing = %#v", model)
	}
}

func TestPiGLMRoutingOverridesTheBuiltInModel(t *testing.T) {
	providers := piConfig(t, "openrouter/z-ai/glm-5.3-flash", "")
	override := providers["openrouter"].(map[string]any)["modelOverrides"].(map[string]any)["z-ai/glm-5.3-flash"].(map[string]any)
	routing := override["compat"].(map[string]any)["openRouterRouting"].(map[string]any)
	if routing["allow_fallbacks"] != true || routing["order"].([]any)[0] != "Modal" {
		t.Fatalf("glm routing = %#v", routing)
	}
}

func TestPiDeclaresGPT61ModelsMissingFromItsCatalog(t *testing.T) {
	for model, wantNone := range map[string]bool{
		"amazon-bedrock/global.openai.gpt-6.1-sol":   false,
		"amazon-bedrock/global.openai.gpt-6.1-terra": true,
		"openai/gpt-6.1-sol":                         false,
		"openai/gpt-6.1-terra":                       true,
	} {
		providerName, id, _ := strings.Cut(model, "/")
		providers := piConfig(t, model, "")
		definition := providers[providerName].(map[string]any)["models"].([]any)[0].(map[string]any)
		levels := definition["thinkingLevelMap"].(map[string]any)
		if definition["id"] != id || definition["reasoning"] != true || (levels["off"] == "none") != wantNone || levels["max"] != "max" {
			t.Fatalf("%s definition = %#v", model, definition)
		}
		if !PiDeclaresModel(model) {
			t.Fatalf("%s must be reported as declared", model)
		}
	}
	for _, model := range []string{"openai/gpt-6-astra", "amazon-bedrock/global.openai.gpt-6-astra", "amazon-bedrock/global.anthropic.claude-sonnet-5-5"} {
		if config := piConfig(t, model, ""); config != nil {
			t.Fatalf("%s is in Pi's catalog and needs no models.json: %#v", model, config)
		}
	}
}

func TestPiPriorityTierOnlyForLunaXhigh(t *testing.T) {
	tier := func(model, effort string) any {
		providers := piConfig(t, model, effort)
		if providers == nil {
			return nil
		}
		overrides, _ := providers["openai"].(map[string]any)["modelOverrides"].(map[string]any)
		luna, _ := overrides["gpt-6-luna"].(map[string]any)
		params, _ := luna["samplingParams"].(map[string]any)
		return params["service_tier"]
	}
	if got := tier("openai/gpt-6-luna", "xhigh"); got != "priority" {
		t.Fatalf("luna xhigh service_tier = %v, want priority", got)
	}
	for _, effort := range []string{"", "medium", "high", "max"} {
		if got := tier("openai/gpt-6-luna", effort); got != nil {
			t.Fatalf("luna %q must stay on the standard tier, got %v", effort, got)
		}
	}
	if got := tier("openai/gpt-6.1-sol", "xhigh"); got != nil {
		t.Fatalf("sol xhigh must stay on the standard tier, got %v", got)
	}
}

func TestPiNoneEffortIsPiOff(t *testing.T) {
	invocation, err := Build(Request{Backend: "pi", Model: "amazon-bedrock/global.openai.gpt-6.1-terra", Prompt: "p", ReasoningEffort: "none"})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(invocation.Argv, "off") || slices.Contains(invocation.Argv, "none") {
		t.Fatalf("argv = %#v", invocation.Argv)
	}
}

func TestPiHasNoCwdFlagButOmpDoes(t *testing.T) {
	for backend, want := range map[string]bool{"pi": false, "omp": true} {
		got, err := Build(Request{Backend: backend, Model: "openai/gpt-6.1-sol", Prompt: "p", APIKey: "k", CWD: "/work"})
		if err != nil {
			t.Fatal(err)
		}
		if slices.Contains(got.Argv, "--cwd") != want {
			t.Fatalf("%s --cwd present = %v, want %v: %#v", backend, !want, want, got.Argv)
		}
	}
}

func TestPiOpenCodeGoDeepSeekIsDeclaredAndKeyedByOpenCodeKey(t *testing.T) {
	providers := piConfig(t, "opencode-go/deepseek-v4.1-flash", "")
	model := providers["opencode-go"].(map[string]any)["models"].([]any)[0].(map[string]any)
	if model["id"] != "deepseek-v4.1-flash" || model["baseUrl"] != "https://opencode.ai/zen/go/v1" || model["api"] != "openai-completions" {
		t.Fatalf("opencode-go model = %#v", model)
	}
	invocation, err := Build(Request{Backend: "pi", Model: "opencode-go/deepseek-v4.1-flash", Prompt: "prompt", APIKey: "fake-go-key"})
	if err != nil {
		t.Fatal(err)
	}
	if invocation.SetEnv["OPENCODE_API_KEY"] != "fake-go-key" || invocation.SetEnv["OPENROUTER_API_KEY"] != "" {
		t.Fatalf("env = %#v", invocation.SetEnv)
	}
	if !slices.Contains(invocation.UnsetEnv, "OPENCODE_API_KEY") {
		t.Fatalf("an inherited OPENCODE_API_KEY must be unset first: %#v", invocation.UnsetEnv)
	}
}
