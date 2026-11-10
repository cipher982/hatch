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

type Credential struct {
	Value           string
	ResolutionError error
}

type Options struct {
	Cursor Credential
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
		checkHarness("harness.pi", "pi"),
		checkHarness("harness.omp", "omp"),
		checkHarness("harness.agy", "agy"),
		checkCursorModel(options.Cursor),
		checkPiModels("bedrock.catalog", modelValues(provider.BedrockSurfaceModels)),
		checkPiModels("codex.catalog", modelValues(provider.CodexSurfaceModels)),
		checkAntigravityModels("gemini.catalog", modelValues(provider.GeminiSurfaceModels)),
		checkPiModels("openrouter.catalog", modelValues(provider.OpenRouterSurfaceModels)),
		checkPiModels("opencode-go.catalog", modelValues(provider.OpenCodeGoSurfaceModels)),
	}
}

func checkHarness(name, binary string) Check {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	output, err := exec.CommandContext(ctx, binary, "--version").CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		return Check{Name: name, Detail: binary + " --version timed out after 5s"}
	}
	if err != nil {
		if _, ok := err.(*exec.Error); ok {
			return Check{Name: name, Detail: binary + " is not installed"}
		}
		detail := strings.TrimSpace(string(output))
		if detail == "" {
			detail = err.Error()
		}
		return Check{Name: name, Detail: binary + " --version failed: " + detail}
	}
	version := strings.TrimSpace(strings.SplitN(string(output), "\n", 2)[0])
	if version == "" {
		version = "installed"
	}
	return Check{Name: name, OK: true, Detail: version}
}

func modelValues(catalog map[string]string) []string {
	models := make([]string, 0, len(catalog))
	for _, model := range catalog {
		models = append(models, model)
	}
	sort.Strings(models)
	return models
}

// ParsePiModelIDs reads `pi --list-models` output (a table of provider, model
// id and capabilities) into "provider/id" selectors.
func ParsePiModelIDs(output string) map[string]struct{} {
	result := map[string]struct{}{}
	for _, line := range strings.Split(output, "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 2 && fields[0] != "provider" {
			result[fields[0]+"/"+fields[1]] = struct{}{}
		}
	}
	return result
}

// checkPiModels verifies every configured model is either in Pi's catalog or
// declared by Hatch's own per-run models.json (a catalog gap Hatch fills).
func checkPiModels(name string, required []string) Check {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	stdout, err := exec.CommandContext(ctx, "pi", "--list-models").Output()
	if ctx.Err() == context.DeadlineExceeded {
		return Check{Name: name, Detail: "pi --list-models timed out after 30s"}
	}
	if err != nil {
		if _, ok := err.(*exec.Error); ok {
			return Check{Name: name, Detail: "pi is not installed"}
		}
		detail := err.Error()
		if exit, ok := err.(*exec.ExitError); ok && strings.TrimSpace(string(exit.Stderr)) != "" {
			detail = strings.TrimSpace(string(exit.Stderr))
		}
		return Check{Name: name, Detail: "could not list Pi models: " + detail}
	}
	available := ParsePiModelIDs(string(stdout))
	missing := []string{}
	for _, model := range required {
		if _, ok := available[model]; !ok && !provider.PiDeclaresModel(model) {
			missing = append(missing, model)
		}
	}
	if len(missing) > 0 {
		return Check{Name: name, Detail: fmt.Sprintf("configured models unavailable: %s; run `pi update` or declare them in Hatch's Pi models config and update Hatch aliases", strings.Join(missing, ", "))}
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

func checkCursorModel(credential Credential) Check {
	if credential.ResolutionError != nil {
		return Check{Name: "cursor.catalog", Detail: "credential resolver failed for catalog probe: " + credential.ResolutionError.Error()}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "cursor-agent", "models")
	if strings.TrimSpace(credential.Value) != "" {
		cmd.Env = replaceEnvironment(os.Environ(), "CURSOR_API_KEY", credential.Value)
	}
	stdout, err := cmd.Output()
	if ctx.Err() == context.DeadlineExceeded {
		return Check{Name: "cursor.catalog", Detail: "cursor-agent models timed out after 30s"}
	}
	if err != nil {
		if _, ok := err.(*exec.Error); ok {
			return Check{Name: "cursor.catalog", Detail: "cursor-agent is not installed"}
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
		return Check{Name: "cursor.catalog", Detail: "could not list Cursor models: " + detail}
	}
	available := ParseCursorModelIDs(string(stdout))
	missing := []string{}
	hidden := 0
	for _, model := range modelValues(provider.CursorSurfaceModels) {
		if _, ok := available[model]; ok {
			continue
		}
		// `cursor-agent models` omits some valid IDs (for example grok-4.7-high),
		// so an unlisted alias is only stale when Cursor rejects it outright.
		if cursorRejectsModel(model, credential) {
			missing = append(missing, model)
		} else {
			hidden++
		}
	}
	if len(missing) > 0 {
		return Check{Name: "cursor.catalog", Detail: fmt.Sprintf("configured models unavailable: %s; run `cursor-agent models` and update Hatch aliases", strings.Join(missing, ", "))}
	}
	detail := fmt.Sprintf("%d configured models are available", len(provider.CursorSurfaceModels))
	if hidden > 0 {
		detail += fmt.Sprintf(" (%d accepted by Cursor but not listed)", hidden)
	}
	return Check{Name: "cursor.catalog", OK: true, Detail: detail}
}

// cursorRejectsModel reports whether cursor-agent refuses the model ID before
// sending a request. Any other outcome (success, usage limit, auth) means the
// ID itself is valid.
func cursorRejectsModel(model string, credential Credential) bool {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "cursor-agent", "--print", "--trust", "--model", model, "--output-format", "text", "Reply exactly: ok")
	if strings.TrimSpace(credential.Value) != "" {
		cmd.Env = replaceEnvironment(os.Environ(), "CURSOR_API_KEY", credential.Value)
	}
	output, _ := cmd.CombinedOutput()
	return strings.Contains(string(output), "Cannot use this model")
}

// ParseAntigravityModelIDs reads `agy models` output (model id, tab, display
// name) into model ids.
func ParseAntigravityModelIDs(output string) map[string]struct{} {
	result := map[string]struct{}{}
	for _, line := range strings.Split(output, "\n") {
		id, _, ok := strings.Cut(line, "\t")
		if !ok {
			continue
		}
		if id = strings.TrimSpace(id); id != "" && !strings.ContainsAny(id, " ") {
			result[id] = struct{}{}
		}
	}
	return result
}

func checkAntigravityModels(name string, required []string) Check {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "agy", "models")
	stdout, err := cmd.Output()
	if ctx.Err() == context.DeadlineExceeded {
		return Check{Name: name, Detail: "agy models timed out after 30s"}
	}
	if err != nil {
		if _, ok := err.(*exec.Error); ok {
			return Check{Name: name, Detail: "agy is not installed"}
		}
		detail := err.Error()
		if exit, ok := err.(*exec.ExitError); ok && strings.TrimSpace(string(exit.Stderr)) != "" {
			detail = strings.TrimSpace(string(exit.Stderr))
		}
		return Check{Name: name, Detail: "could not list Antigravity models: " + detail}
	}
	available := ParseAntigravityModelIDs(string(stdout))
	missing := []string{}
	for _, model := range required {
		if _, ok := available[model]; !ok {
			missing = append(missing, model)
		}
	}
	if len(missing) > 0 {
		return Check{Name: name, Detail: fmt.Sprintf("configured models unavailable: %s; run `agy models` and update Hatch aliases", strings.Join(missing, ", "))}
	}
	return Check{Name: name, OK: true, Detail: fmt.Sprintf("%d configured models are available", len(required))}
}
