package expert

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	runner "github.com/cipher982/hatch/internal/run"
)

const DefaultModel = "gpt-5.6-sol"
const DefaultURL = "https://api.openai.com/v1/responses"

const Instructions = `You are an expert consultant. Answer the user's question directly.
This is a single synchronous consultation, not an agent run.

Prioritize:
- the decision or recommendation
- the key reasoning that supports it
- material risks, counterarguments, and uncertainty
- what evidence would change the answer

Do not claim access to local files, shell commands, or repo state unless the
user included that context in the prompt. If context is missing, say exactly
what assumption you are making.
`

type Options struct {
	Prompt, Model, ReasoningEffort string
	WebSearch                      bool
	Timeout                        time.Duration
	APIKey, BaseURL                string
	Client                         *http.Client
	PollInterval                   time.Duration
	Store                          runner.RunStore
	Progress                       func(string)
}

type Result struct {
	OK              bool             `json:"ok"`
	Status          string           `json:"status"`
	Output          string           `json:"output"`
	DurationMS      int64            `json:"duration_ms"`
	Error           *string          `json:"error"`
	Model           string           `json:"model"`
	ResolvedModel   *string          `json:"resolved_model"`
	ReasoningEffort string           `json:"reasoning_effort"`
	WebSearch       bool             `json:"web_search"`
	Usage           any              `json:"usage"`
	ResponseID      *string          `json:"response_id"`
	Citations       []map[string]any `json:"citations"`
	Sources         []map[string]any `json:"sources"`
	ArtifactPath    *string          `json:"artifact_path"`
	Run             *runner.Manifest `json:"run,omitempty"`
	ExitCode        int              `json:"-"`
}

func BuildPayload(prompt, model, effort string, webSearch bool) (map[string]any, error) {
	if strings.TrimSpace(prompt) == "" {
		return nil, fmt.Errorf("prompt must not be empty")
	}
	payload := map[string]any{
		"model": model, "instructions": Instructions, "input": prompt,
		"reasoning": map[string]any{"effort": effort}, "background": true, "store": true,
	}
	if webSearch {
		payload["tools"] = []map[string]any{{"type": "web_search"}}
		payload["tool_choice"] = "auto"
		payload["include"] = []string{"web_search_call.action.sources"}
	}
	return payload, nil
}

func Run(options Options) Result {
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.ReasoningEffort == "" {
		options.ReasoningEffort = "medium"
	}
	if options.Timeout <= 0 {
		options.Timeout = 900 * time.Second
	}
	if options.BaseURL == "" {
		options.BaseURL = DefaultURL
	}
	if options.Client == nil {
		options.Client = &http.Client{Timeout: 60 * time.Second}
	}
	if options.PollInterval <= 0 {
		options.PollInterval = 15 * time.Second
	}
	payload, payloadErr := BuildPayload(options.Prompt, options.Model, options.ReasoningEffort, options.WebSearch)
	if payloadErr != nil {
		message := payloadErr.Error()
		return Result{Status: "error", Error: &message, Model: options.Model, ReasoningEffort: options.ReasoningEffort, WebSearch: options.WebSearch, Citations: []map[string]any{}, Sources: []map[string]any{}, ExitCode: 1}
	}
	coordinator := runner.NewCoordinator(options.Store)
	var final map[string]any
	public := coordinator.ExecuteHTTP(runner.HTTPRequest{
		Surface: "expert", Provider: "openai", Model: options.Model, Prompt: options.Prompt,
		Timeout: options.Timeout, CredentialNames: []string{"OPENAI_API_KEY"}, Progress: options.Progress,
		Execute: func(ctx context.Context, record func([]byte) error) runner.HTTPOutcome {
			return execute(ctx, options, payload, record, &final)
		},
	})
	result := Result{
		OK: public.OK, Status: public.Status, Output: public.Output, DurationMS: public.DurationMS, Error: public.Error,
		Model: options.Model, ReasoningEffort: options.ReasoningEffort, WebSearch: options.WebSearch,
		ArtifactPath: public.ArtifactPath, Run: public.Run, ExitCode: public.CLIExitCode(),
		Citations: []map[string]any{}, Sources: []map[string]any{},
	}
	if final != nil {
		if value := stringField(final, "model"); value != "" {
			result.ResolvedModel = &value
		}
		if value := stringField(final, "id"); value != "" {
			result.ResponseID = &value
		}
		result.Usage = final["usage"]
		result.Citations = extractCitations(final)
		result.Sources = extractSources(final)
	} else if public.SessionID != nil {
		result.ResponseID = public.SessionID
	}
	return result
}

func execute(ctx context.Context, options Options, payload map[string]any, record func([]byte) error, final *map[string]any) runner.HTTPOutcome {
	outcome := runner.HTTPOutcome{NativeIDState: "unavailable", Retention: "remote_provider", Capabilities: map[string]string{"poll": "supported", "inspect": "supported"}}
	response, status, err := requestJSON(ctx, options.Client, options.BaseURL, options.APIKey, http.MethodPost, payload, record)
	outcome.Attempts++
	outcome.LastStatus = status
	if err != nil {
		outcome.Error = err.Error()
		return outcome
	}
	*final = response
	responseID := stringField(response, "id")
	if responseID != "" {
		outcome.NativeID, outcome.NativeIDState = responseID, "observed"
	}
	statusName := stringField(response, "status")
	if options.Progress != nil && responseID != "" {
		options.Progress(fmt.Sprintf("[hatch] expert response %s status=%s", responseID, defaultString(statusName, "unknown")))
	}
	for activeStatus(statusName) {
		timer := time.NewTimer(options.PollInterval)
		select {
		case <-ctx.Done():
			timer.Stop()
			outcome.TimedOut = true
			outcome.Error = fmt.Sprintf("Expert consultation still running after %ds; response_id=%s. Hatch left it running server-side.", int(options.Timeout.Seconds()), responseID)
			return outcome
		case <-timer.C:
		}
		pollURL := strings.TrimRight(options.BaseURL, "/") + "/" + url.PathEscape(responseID)
		if options.WebSearch {
			pollURL += "?include%5B%5D=web_search_call.action.sources"
		}
		response, status, err = requestJSON(ctx, options.Client, pollURL, options.APIKey, http.MethodGet, nil, record)
		outcome.Attempts++
		outcome.LastStatus = status
		if err != nil {
			if ctx.Err() != nil {
				outcome.TimedOut = true
				outcome.Error = fmt.Sprintf("Expert consultation still running after %ds; response_id=%s. Hatch left it running server-side.", int(options.Timeout.Seconds()), responseID)
			} else {
				outcome.Error = err.Error()
			}
			return outcome
		}
		*final = response
		statusName = stringField(response, "status")
		if options.Progress != nil {
			options.Progress(fmt.Sprintf("[hatch] expert response %s status=%s", responseID, defaultString(statusName, "unknown")))
		}
	}
	if statusName != "" && statusName != "completed" {
		outcome.Error = responseError(response)
		return outcome
	}
	outcome.Output = extractOutput(response)
	if outcome.Output == "" {
		outcome.Error = "Responses API returned no final output"
	}
	return outcome
}

func requestJSON(ctx context.Context, client *http.Client, endpoint, apiKey, method string, payload map[string]any, record func([]byte) error) (map[string]any, int, error) {
	var body io.Reader
	if payload != nil {
		encoded, err := json.Marshal(payload)
		if err != nil {
			return nil, 0, err
		}
		body = bytes.NewReader(encoded)
	}
	request, err := http.NewRequestWithContext(ctx, method, endpoint, body)
	if err != nil {
		return nil, 0, err
	}
	request.Header.Set("Authorization", "Bearer "+apiKey)
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		return nil, 0, err
	}
	defer response.Body.Close()
	data, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, response.StatusCode, err
	}
	_ = record(data)
	var decoded map[string]any
	if err := json.Unmarshal(data, &decoded); err != nil {
		return nil, response.StatusCode, fmt.Errorf("decode Responses API JSON: %w", err)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return decoded, response.StatusCode, fmt.Errorf("%s", responseError(decoded))
	}
	return decoded, response.StatusCode, nil
}

func extractOutput(response map[string]any) string {
	var chunks []string
	for _, rawItem := range sliceField(response, "output") {
		item, _ := rawItem.(map[string]any)
		if item["type"] != "message" {
			continue
		}
		for _, rawContent := range sliceField(item, "content") {
			content, _ := rawContent.(map[string]any)
			if content["type"] == "output_text" {
				if text := stringField(content, "text"); text != "" {
					chunks = append(chunks, text)
				}
			}
		}
	}
	return strings.TrimSpace(strings.Join(chunks, "\n"))
}

func extractCitations(response map[string]any) []map[string]any {
	result := []map[string]any{}
	for _, rawItem := range sliceField(response, "output") {
		item, _ := rawItem.(map[string]any)
		if item["type"] != "message" {
			continue
		}
		for _, rawContent := range sliceField(item, "content") {
			content, _ := rawContent.(map[string]any)
			for _, rawAnnotation := range sliceField(content, "annotations") {
				annotation, _ := rawAnnotation.(map[string]any)
				if annotation["type"] == "url_citation" {
					result = append(result, map[string]any{"url": annotation["url"], "title": annotation["title"], "start_index": annotation["start_index"], "end_index": annotation["end_index"]})
				}
			}
		}
	}
	return result
}

func extractSources(response map[string]any) []map[string]any {
	result, seen := []map[string]any{}, map[string]bool{}
	for _, rawItem := range sliceField(response, "output") {
		item, _ := rawItem.(map[string]any)
		if item["type"] != "web_search_call" {
			continue
		}
		action, _ := item["action"].(map[string]any)
		for _, rawSource := range sliceField(action, "sources") {
			source, _ := rawSource.(map[string]any)
			address := stringField(source, "url")
			if address != "" && !seen[address] {
				seen[address] = true
				result = append(result, map[string]any{"type": source["type"], "url": address, "title": source["title"]})
			}
		}
	}
	return result
}

func responseError(response map[string]any) string {
	if value, ok := response["error"].(map[string]any); ok {
		if message := stringField(value, "message"); message != "" {
			return message
		}
	}
	if details, ok := response["incomplete_details"].(map[string]any); ok && len(details) > 0 {
		return fmt.Sprintf("Response incomplete: %v", details)
	}
	return "Response ended with status: " + stringField(response, "status")
}

func activeStatus(status string) bool {
	return status == "queued" || status == "in_progress" || status == "interpreting"
}
func stringField(value map[string]any, key string) string {
	result, _ := value[key].(string)
	return result
}
func sliceField(value map[string]any, key string) []any {
	result, _ := value[key].([]any)
	return result
}
func defaultString(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}
