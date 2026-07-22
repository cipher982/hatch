package expert

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	runner "github.com/cipher982/hatch/internal/run"
)

func TestBuildPayload(t *testing.T) {
	payload, err := BuildPayload("question", DefaultModel, "medium", true)
	if err != nil {
		t.Fatal(err)
	}
	if payload["model"] != DefaultModel || payload["background"] != true || payload["store"] != true || payload["tools"] == nil {
		t.Fatalf("payload = %#v", payload)
	}
}

func TestRunPollsAndReturnsMetadata(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		requests++
		if request.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("authorization = %q", request.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		if request.Method == http.MethodPost {
			_ = json.NewEncoder(w).Encode(map[string]any{"id": "resp_123", "status": "queued", "model": DefaultModel})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "resp_123", "status": "completed", "model": "gpt-resolved", "usage": map[string]any{"total_tokens": 12},
			"output": []any{
				map[string]any{"type": "web_search_call", "action": map[string]any{"sources": []any{map[string]any{"type": "url", "url": "https://example.com", "title": "Example"}}}},
				map[string]any{"type": "message", "content": []any{map[string]any{"type": "output_text", "text": "Use the simple design.", "annotations": []any{map[string]any{"type": "url_citation", "url": "https://example.com", "title": "Example", "start_index": 0, "end_index": 3}}}}},
			},
		})
	}))
	defer server.Close()
	result := Run(Options{
		Prompt: "question", Model: DefaultModel, ReasoningEffort: "medium", WebSearch: true,
		Timeout: time.Second, APIKey: "test-key", BaseURL: server.URL, Client: server.Client(), PollInterval: time.Millisecond,
		Store: runner.NewStore(filepath.Join(t.TempDir(), "runs")),
	})
	if !result.OK || result.Output != "Use the simple design." || result.ResponseID == nil || *result.ResponseID != "resp_123" ||
		result.ResolvedModel == nil || *result.ResolvedModel != "gpt-resolved" || len(result.Citations) != 1 || len(result.Sources) != 1 || requests != 2 || result.Run == nil || result.Run.Execution != "http" {
		t.Fatalf("result = %#v requests=%d", result, requests)
	}
}

func TestRunTimeoutLeavesRemoteResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "resp_active", "status": "queued", "model": DefaultModel})
	}))
	defer server.Close()
	result := Run(Options{
		Prompt: "question", Timeout: 10 * time.Millisecond, APIKey: "test-key", BaseURL: server.URL,
		Client: server.Client(), PollInterval: time.Second, Store: runner.NewStore(filepath.Join(t.TempDir(), "runs")),
	})
	if result.OK || result.Status != "timeout" || result.ResponseID == nil || *result.ResponseID != "resp_active" || result.Error == nil || result.Run == nil || result.Run.ProviderState.Retention != "remote_provider" {
		t.Fatalf("result = %#v", result)
	}
}

func TestRunHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": "bad request"}})
	}))
	defer server.Close()
	result := Run(Options{Prompt: "question", APIKey: "test", BaseURL: server.URL, Client: server.Client(), Store: runner.NewStore(filepath.Join(t.TempDir(), "runs"))})
	if result.OK || result.Error == nil || *result.Error != "bad request" {
		t.Fatalf("result = %#v", result)
	}
}
