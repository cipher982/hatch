package doctor

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

const CursorGrok = "cursor-grok-4.5-high"

var CodexModels = []string{"openai/gpt-5.6-sol", "openai/gpt-5.6-terra", "openai/gpt-5.6-luna"}
var OpenRouterModels = []string{"openrouter/deepseek/deepseek-v4-pro", "openrouter/~moonshotai/kimi-latest"}

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

func Run() []Check {
	return []Check{
		checkCursorModel(),
		checkOpenCodeModels("codex.catalog", "openai", CodexModels),
		checkOpenCodeModels("openrouter.catalog", "openrouter", OpenRouterModels),
	}
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

func checkOpenCodeModels(name, provider string, required []string) Check {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "opencode", "models", provider)
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
		return Check{Name: name, Detail: fmt.Sprintf("configured models unavailable: %s; run `opencode models %s --refresh` and update Hatch aliases", strings.Join(missing, ", "), provider)}
	}
	return Check{Name: name, OK: true, Detail: fmt.Sprintf("%d configured models are available", len(required))}
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
