package run

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestCoordinatorHTTPSuccess(t *testing.T) {
	coordinator := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs")))
	result := coordinator.ExecuteHTTP(HTTPRequest{
		Surface: "expert", Provider: "openai", Model: "gpt", Prompt: "question", Timeout: time.Second,
		CredentialNames: []string{"OPENAI_API_KEY"},
		Execute: func(_ context.Context, record func([]byte) error) HTTPOutcome {
			_ = record([]byte(`{"id":"resp_1","status":"completed"}`))
			return HTTPOutcome{Output: "answer", NativeID: "resp_1", NativeIDState: "observed", Retention: "remote_provider", Capabilities: map[string]string{"poll": "supported"}, Attempts: 1, LastStatus: 200}
		},
	})
	if !result.OK || result.Output != "answer" || result.SessionID == nil || *result.SessionID != "resp_1" || result.Run == nil || result.Run.Execution != "http" || result.Run.Process != nil || result.Run.HTTP == nil || result.Run.HTTP.Attempts != 1 {
		t.Fatalf("result = %#v", result)
	}
	data, err := os.ReadFile(filepath.Join(*result.ArtifactPath, "stdout.jsonl"))
	if err != nil || string(data) != "{\"id\":\"resp_1\",\"status\":\"completed\"}\n" {
		t.Fatalf("snapshot = %q, %v", data, err)
	}
}

func TestCoordinatorHTTPTimeoutPreservesRemoteIdentity(t *testing.T) {
	coordinator := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs")))
	result := coordinator.ExecuteHTTP(HTTPRequest{
		Surface: "expert", Provider: "openai", Prompt: "question", Timeout: time.Millisecond,
		Execute: func(ctx context.Context, record func([]byte) error) HTTPOutcome {
			_ = record([]byte(`{"id":"resp_active","status":"queued"}`))
			<-ctx.Done()
			return HTTPOutcome{Error: "left running server-side", NativeID: "resp_active", NativeIDState: "observed", Retention: "remote_provider", Attempts: 1, LastStatus: 200, TimedOut: true}
		},
	})
	if result.OK || result.Status != "timeout" || result.SessionID == nil || *result.SessionID != "resp_active" || result.Run == nil || result.Run.ProviderState.Retention != "remote_provider" {
		t.Fatalf("result = %#v", result)
	}
}

func TestHTTPStreamOpenFailureReturnsCanonicalRun(t *testing.T) {
	store := unavailableStreamStore{Store: NewStore(filepath.Join(t.TempDir(), "runs"))}
	result := NewCoordinator(store).ExecuteHTTP(HTTPRequest{
		Surface: "expert.responses", Provider: "openai", Prompt: "question", Timeout: time.Second,
		Execute: func(context.Context, func([]byte) error) HTTPOutcome {
			t.Fatal("HTTP provider executed after stream failure")
			return HTTPOutcome{}
		},
	})
	if result.OK || result.ExitCode != -3 || result.Run == nil || result.Run.Lifecycle != LifecycleTerminal ||
		result.Run.Capture.State != "degraded" || result.ArtifactPath != nil {
		t.Fatalf("result=%#v", result)
	}
}
