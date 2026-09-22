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
	if !slices.Contains(entries, CatalogEntry{Surface: "openrouter", Alias: "deepseek-v4.1-flash", Model: "openrouter/deepseek/deepseek-v4.1-flash"}) ||
		!slices.Contains(entries, CatalogEntry{Surface: "openrouter", Alias: "glm-5.3-flash", Model: "openrouter/z-ai/glm-5.3-flash"}) {
		t.Fatalf("catalog = %#v", entries)
	}
	for _, entry := range entries {
		if entry.Surface == "openrouter" && (entry.Alias == "deepseek-v4-pro" || entry.Alias == "deepseek-v4-flash") {
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
		if seenAliases[spec.Alias] {
			t.Fatalf("duplicate model alias %q", spec.Alias)
		}
		seenAliases[spec.Alias] = true
		if backend := SurfaceBackend(spec.Surface); backend != spec.Backend {
			t.Fatalf("SurfaceBackend(%q) = %q, want %q", spec.Surface, backend, spec.Backend)
		}
		if shorthand := ShorthandSurface(spec.Alias); shorthand != spec.Surface {
			t.Fatalf("ShorthandSurface(%q) = %q, want %q", spec.Alias, shorthand, spec.Surface)
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
	policy := FindRoutingPolicy("openrouter/deepseek/deepseek-v4.1-flash")
	if policy == nil || len(policy.ProviderOrder) != 1 || policy.ProviderOrder[0] != "DeepSeek" || policy.AllowFallbacks {
		t.Fatalf("unexpected policy for deepseek-v4.1-flash: %#v", policy)
	}
	// Snapshot suffix match
	policySnapshot := FindRoutingPolicy("openrouter/deepseek/deepseek-v4.1-flash-0910")
	if policySnapshot == nil || len(policySnapshot.ProviderOrder) != 1 || policySnapshot.ProviderOrder[0] != "DeepSeek" {
		t.Fatalf("unexpected policy for snapshot: %#v", policySnapshot)
	}
	if plain := FindRoutingPolicy("openai/gpt-6-astra"); plain != nil {
		t.Fatalf("unrouted model returned routing policy: %#v", plain)
	}
}

func TestBuildOpenCodeDeepSeekRoutingConfig(t *testing.T) {
	invocation, err := Build(Request{Backend: "opencode", Model: "openrouter/deepseek/deepseek-v4.1-flash", Prompt: "prompt", APIKey: "fake"})
	if err != nil {
		t.Fatal(err)
	}
	if len(invocation.OpenCodeConfigJSON) == 0 {
		t.Fatal("openrouter deepseek run missing routing config")
	}
	var config struct {
		Provider struct {
			OpenRouter struct {
				Models map[string]struct {
					Options struct {
						Provider struct {
							Order          []string `json:"order"`
							AllowFallbacks bool     `json:"allow_fallbacks"`
						} `json:"provider"`
					} `json:"options"`
				} `json:"models"`
			} `json:"openrouter"`
		} `json:"provider"`
	}
	if err := json.Unmarshal(invocation.OpenCodeConfigJSON, &config); err != nil {
		t.Fatalf("routing config is not valid JSON: %v", err)
	}
	model := config.Provider.OpenRouter.Models["deepseek/deepseek-v4.1-flash"]
	if model.Options.Provider.AllowFallbacks || len(model.Options.Provider.Order) != 1 || model.Options.Provider.Order[0] != "DeepSeek" {
		t.Fatalf("routing config = %s", invocation.OpenCodeConfigJSON)
	}

	plain, err := Build(Request{Backend: "opencode", Model: "openai/gpt-5.6-sol", Prompt: "prompt", APIKey: "fake"})
	if err != nil {
		t.Fatal(err)
	}
	if len(plain.OpenCodeConfigJSON) != 0 {
		t.Fatalf("non-deepseek opencode run must not carry routing config: %s", plain.OpenCodeConfigJSON)
	}
}

func TestBuildOpenCodeGLMRoutingConfig(t *testing.T) {
	invocation, err := Build(Request{Backend: "opencode", Model: "openrouter/z-ai/glm-5.3-flash", Prompt: "prompt", APIKey: "fake"})
	if err != nil {
		t.Fatal(err)
	}
	if len(invocation.OpenCodeConfigJSON) == 0 {
		t.Fatal("openrouter glm run missing routing config")
	}
	var config struct {
		Provider struct {
			OpenRouter struct {
				Models map[string]struct {
					Options struct {
						Provider struct {
							Order          []string `json:"order"`
							AllowFallbacks bool     `json:"allow_fallbacks"`
						} `json:"provider"`
					} `json:"options"`
				} `json:"models"`
			} `json:"openrouter"`
		} `json:"provider"`
	}
	if err := json.Unmarshal(invocation.OpenCodeConfigJSON, &config); err != nil {
		t.Fatalf("routing config is not valid JSON: %v", err)
	}
	model := config.Provider.OpenRouter.Models["z-ai/glm-5.3-flash"]
	if !model.Options.Provider.AllowFallbacks || len(model.Options.Provider.Order) == 0 || model.Options.Provider.Order[0] != "Modal" {
		t.Fatalf("routing config = %s", invocation.OpenCodeConfigJSON)
	}
}

func TestBuildOpenCodePriorityTierConfig(t *testing.T) {
	type openAIConfig struct {
		Provider struct {
			OpenAI struct {
				Models map[string]struct {
					Options struct {
						ServiceTier string `json:"serviceTier"`
					} `json:"options"`
				} `json:"models"`
			} `json:"openai"`
		} `json:"provider"`
	}
	serviceTier := func(invocation Invocation) string {
		t.Helper()
		if len(invocation.OpenCodeConfigJSON) == 0 {
			return ""
		}
		var config openAIConfig
		if err := json.Unmarshal(invocation.OpenCodeConfigJSON, &config); err != nil {
			t.Fatalf("priority config is not valid JSON: %v", err)
		}
		return config.Provider.OpenAI.Models["gpt-5.6-luna"].Options.ServiceTier
	}

	invocation, err := Build(Request{
		Backend: "opencode", Model: "openai/gpt-5.6-luna", Prompt: "prompt",
		APIKey: "fake", ReasoningEffort: "xhigh",
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := serviceTier(invocation); got != "priority" {
		t.Fatalf("luna xhigh serviceTier = %q, want priority: %s", got, invocation.OpenCodeConfigJSON)
	}
	if got := invocation.ReasoningPolicy; got != (ReasoningPolicy{Effort: "xhigh", Source: "explicit", Support: "native"}) {
		t.Fatalf("luna xhigh reasoning policy = %#v", got)
	}

	for _, effort := range []string{"", "medium", "high", "max"} {
		other, err := Build(Request{
			Backend: "opencode", Model: "openai/gpt-5.6-luna", Prompt: "prompt",
			APIKey: "fake", ReasoningEffort: effort,
		})
		if err != nil {
			t.Fatal(err)
		}
		if len(other.OpenCodeConfigJSON) != 0 {
			t.Fatalf("luna effort %q must not request priority processing: %s", effort, other.OpenCodeConfigJSON)
		}
	}

	for _, model := range []string{"openai/gpt-6-astra", "openai/gpt-5.6-terra"} {
		other, err := Build(Request{
			Backend: "opencode", Model: model, Prompt: "prompt",
			APIKey: "fake", ReasoningEffort: "xhigh",
		})
		if err != nil {
			t.Fatal(err)
		}
		if len(other.OpenCodeConfigJSON) != 0 {
			t.Fatalf("%s xhigh must not request priority processing: %s", model, other.OpenCodeConfigJSON)
		}
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
	t.Run("claude resume and explicit stream", func(t *testing.T) {
		got, err := Build(Request{
			Backend: "claude", Model: "claude-opus-5-5", Prompt: "p", OutputFormat: "stream-json",
			IncludePartialMessages: true, Resume: "ses_1",
		})
		if err != nil {
			t.Fatal(err)
		}
		wantTail := []string{"--include-partial-messages", "--resume", "ses_1"}
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
			Backend: "codex", Model: "gpt-5.6", Prompt: "p", APIKey: "secret",
			ReasoningEffort: "high", SkipGitRepoCheck: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		want := []string{
			"codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "--ignore-user-config", "--ephemeral", "-m", "gpt-5.6",
			"-c", "model_reasoning_effort=high", "--skip-git-repo-check",
		}
		if !reflect.DeepEqual(got.Argv, want) || got.SetEnv["OPENAI_API_KEY"] != "secret" {
			t.Fatalf("codex invocation = %#v", got)
		}
	})

	t.Run("opencode defaults reasoning explicitly", func(t *testing.T) {
		got, err := Build(Request{Backend: "opencode", Model: "openai/gpt-5.6-sol", Prompt: "p", APIKey: "secret"})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got.Argv[len(got.Argv)-3:], []string{"--variant", "medium", PreparePrompt("p")}) ||
			got.ReasoningPolicy != (ReasoningPolicy{Effort: "medium", Source: "default", Support: "native"}) {
			t.Fatalf("opencode invocation = %#v", got)
		}
	})

	for _, backend := range []string{"pi", "omp"} {
		t.Run(backend+" uses explicit headless JSON mode", func(t *testing.T) {
			got, err := Build(Request{Backend: backend, Model: "openai/gpt-5.6-sol", Prompt: "p", APIKey: "secret"})
			if err != nil {
				t.Fatal(err)
			}
			if got.Argv[0] != backend || got.StreamFormat != "jsonl" || got.Adapter != backend || got.Argv[len(got.Argv)-1] != PreparePrompt("p") ||
				got.SetEnv["OPENAI_API_KEY"] != "secret" || got.ReasoningPolicy.Effort != "medium" {
				t.Fatalf("%s invocation = %#v", backend, got)
			}
			joined := strings.Join(got.Argv, " ")
			for _, want := range []string{"--mode json", "--no-session", "--model openai/gpt-5.6-sol", "--thinking medium"} {
				if !strings.Contains(joined, want) {
					t.Fatalf("%s argv lacks %q: %#v", backend, want, got.Argv)
				}
			}
		})
	}

	t.Run("omp requests priority tier for luna xhigh", func(t *testing.T) {
		got, err := Build(Request{Backend: "omp", Model: "openai/gpt-5.6-luna", Prompt: "p", APIKey: "secret", ReasoningEffort: "xhigh"})
		if err != nil {
			t.Fatal(err)
		}
		wantTail := []string{"--thinking", "xhigh", "--service-tier", "priority", PreparePrompt("p")}
		if !reflect.DeepEqual(got.Argv[len(got.Argv)-len(wantTail):], wantTail) {
			t.Fatalf("omp luna xhigh argv = %#v", got.Argv)
		}
	})

	t.Run("pi and non-xhigh omp stay on the standard tier", func(t *testing.T) {
		for _, request := range []Request{
			{Backend: "pi", Model: "openai/gpt-5.6-luna", Prompt: "p", APIKey: "secret", ReasoningEffort: "xhigh"},
			{Backend: "omp", Model: "openai/gpt-5.6-luna", Prompt: "p", APIKey: "secret", ReasoningEffort: "high"},
			{Backend: "omp", Model: "openai/gpt-5.6-terra", Prompt: "p", APIKey: "secret", ReasoningEffort: "xhigh"},
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

	t.Run("bedrock defaults", func(t *testing.T) {
		got, err := Build(Request{Backend: "bedrock", Prompt: "p", OutputFormat: "text"})
		if err != nil {
			t.Fatal(err)
		}
		if got.SetEnv["AWS_PROFILE"] != "zh-ml-mlengineer" || got.SetEnv["AWS_REGION"] != "us-east-1" ||
			got.SetEnv["ANTHROPIC_MODEL"] != "us.anthropic.claude-sonnet-4-6" || got.Adapter != "claude" {
			t.Fatalf("bedrock invocation = %#v", got)
		}
	})
}
