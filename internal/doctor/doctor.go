package doctor

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strings"
	"time"

	"github.com/cipher982/hatch/internal/provider"
)

const CursorGrok = "cursor-grok-4.5-high"

type Credential struct {
	Value           string
	ResolutionError error
}

type Options struct {
	OpenAI     Credential
	OpenRouter Credential
}

type Check struct {
	Name   string `json:"name"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail"`
}

func ParseCursorModelIDs(output string) map[string]struct{} {
	result := map[string]struct{}{}
	for _, line := range strings.Split(output, "\n") {
		if before, _, ok := strings.Cut(line, " - "); ok {
			if id := strings.TrimSpace(before); id != "" {
				result[id] = struct{}{}
			}
		}
	}
	return result
}

func Run(options Options) []Check {
	return []Check{
		checkCursorModel(),
		checkOpenCodeModels("codex.catalog", "openai", "OPENAI_API_KEY", options.OpenAI, modelValues(provider.CodexSurfaceModels)),
		checkOpenCodeModels("openrouter.catalog", "openrouter", "OPENROUTER_API_KEY", options.OpenRouter, modelValues(provider.OpenRouterSurfaceModels)),
	}
}

func modelValues(catalog map[string]string) []string {
	models := make([]string, 0, len(catalog))
	for _, model := range catalog {
		models = append(models, model)
	}
	sort.Strings(models)
	return models
}

func ParseOpenCodeModelIDs(output string) map[string]struct{} {
	result := map[string]struct{}{}
	for _, line := range strings.Split(output, "\n") {
		if id := strings.TrimSpace(line); id != "" {
			result[id] = struct{}{}
		}
	}
	return result
}

func checkOpenCodeModels(name, providerName, credentialName string, credential Credential, required []string) Check {
	if credential.ResolutionError != nil {
		return Check{Name: name, Detail: "credential resolver failed for catalog probe: " + credential.ResolutionError.Error()}
	}
	if strings.TrimSpace(credential.Value) == "" {
		return Check{Name: name, Detail: credentialName + " is unavailable for catalog probe"}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "opencode", "models", providerName)
	cmd.Env = replaceEnvironment(os.Environ(), credentialName, credential.Value)
	stdout, err := cmd.Output()
	if ctx.Err() == context.DeadlineExceeded {
		return Check{Name: name, Detail: "opencode models timed out after 30s"}
	}
	if err != nil {
		if _, ok := err.(*exec.Error); ok {
			return Check{Name: name, Detail: "opencode is not installed"}
		}
		detail := err.Error()
		if exit, ok := err.(*exec.ExitError); ok && strings.TrimSpace(string(exit.Stderr)) != "" {
			detail = strings.TrimSpace(string(exit.Stderr))
		}
		return Check{Name: name, Detail: "could not list OpenCode models: " + detail}
	}
	available := ParseOpenCodeModelIDs(string(stdout))
	missing := []string{}
	for _, model := range required {
		if _, ok := available[model]; !ok {
			missing = append(missing, model)
		}
	}
	if len(missing) > 0 {
		return Check{Name: name, Detail: fmt.Sprintf("configured models unavailable: %s; run `opencode models %s --refresh` and update Hatch aliases", strings.Join(missing, ", "), providerName)}
	}
	return Check{Name: name, OK: true, Detail: fmt.Sprintf("%d configured models are available", len(required))}
}

func replaceEnvironment(environment []string, name, value string) []string {
	prefix := name + "="
	result := make([]string, 0, len(environment)+1)
	for _, item := range environment {
		if !strings.HasPrefix(item, prefix) {
			result = append(result, item)
		}
	}
	return append(result, prefix+value)
}

func checkCursorModel() Check {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "cursor-agent", "models")
	stdout, err := cmd.Output()
	if ctx.Err() == context.DeadlineExceeded {
		return Check{Name: "cursor.grok", Detail: "cursor-agent models timed out after 30s"}
	}
	if err != nil {
		if _, ok := err.(*exec.Error); ok {
			return Check{Name: "cursor.grok", Detail: "cursor-agent is not installed"}
		}
		detail := ""
		if exit, ok := err.(*exec.ExitError); ok {
			detail = strings.TrimSpace(string(exit.Stderr))
		}
		if detail == "" {
			detail = strings.TrimSpace(string(stdout))
		}
		if detail == "" {
			detail = err.Error()
		}
		return Check{Name: "cursor.grok", Detail: "could not list Cursor models: " + detail}
	}
	available := ParseCursorModelIDs(string(stdout))
	if _, ok := available[CursorGrok]; !ok {
		return Check{Name: "cursor.grok", Detail: fmt.Sprintf("configured model %q is unavailable; run `cursor-agent models` and update CURSOR_GROK", CursorGrok)}
	}
	return Check{Name: "cursor.grok", OK: true, Detail: CursorGrok + " is available"}
}
