package run

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cipher982/hatch/internal/provider"
)

func TestResolveProvenanceExplicitIdentityAndDirectParent(t *testing.T) {
	t.Setenv("HATCH_CALLER_SESSION_ID", "env-session")
	t.Setenv("HATCH_CALLER_REQUEST_ID", "env-request")
	t.Setenv("LONGHOUSE_MANAGED_SESSION_ID", "longhouse-session")
	t.Setenv("LONGHOUSE_THREAD_ID", "longhouse-thread")
	t.Setenv("HATCH_RUN_ID", "direct-parent")
	t.Setenv("LONGHOUSE_HATCH_RUN_ID", "older-parent")

	got, err := ResolveProvenance("explicit-session", "explicit-request")
	if err != nil {
		t.Fatal(err)
	}
	if got.CallerSessionID != "explicit-session" || got.CallerRequestID != "explicit-request" || got.CallerKind != "cli" {
		t.Fatalf("explicit provenance = %#v", got)
	}
	if got.ParentRunID != "direct-parent" {
		t.Fatalf("parent run = %q", got.ParentRunID)
	}
	if got.CallerCWD == "" || !strings.HasPrefix(got.CallerCWD, "/") {
		t.Fatalf("caller cwd = %q", got.CallerCWD)
	}
}

func TestResolveProvenanceLonghouseFallbackAndHonestAbsence(t *testing.T) {
	t.Setenv("HATCH_CALLER_SESSION_ID", "")
	t.Setenv("HATCH_CALLER_REQUEST_ID", "")
	t.Setenv("LONGHOUSE_MANAGED_SESSION_ID", "managed")
	t.Setenv("LONGHOUSE_SESSION_ID", "session")
	t.Setenv("LONGHOUSE_CHANNEL_SESSION_ID", "channel")
	t.Setenv("LONGHOUSE_THREAD_ID", "thread")
	t.Setenv("HATCH_RUN_ID", "")
	t.Setenv("LONGHOUSE_HATCH_RUN_ID", "")

	got, err := ResolveProvenance("", "")
	if err != nil {
		t.Fatal(err)
	}
	if got.CallerKind != "longhouse" || got.CallerSessionID != "managed" || got.CallerRequestID != "thread" || got.ParentRunID != "" {
		t.Fatalf("longhouse provenance = %#v", got)
	}
	got, err = ResolveProvenance("", "explicit-request")
	if err != nil {
		t.Fatal(err)
	}
	if got.CallerKind != "longhouse" || got.CallerSessionID != "managed" || got.CallerRequestID != "explicit-request" {
		t.Fatalf("request override provenance = %#v", got)
	}

	t.Setenv("LONGHOUSE_MANAGED_SESSION_ID", "")
	t.Setenv("LONGHOUSE_SESSION_ID", "")
	t.Setenv("LONGHOUSE_CHANNEL_SESSION_ID", "")
	t.Setenv("LONGHOUSE_THREAD_ID", "")
	got, err = ResolveProvenance("", "")
	if err != nil {
		t.Fatal(err)
	}
	if got.CallerKind != "unknown" || got.CallerSessionID != "" || got.CallerRequestID != "" {
		t.Fatalf("absent identity = %#v", got)
	}
}

func TestValidateTitleRejectsInvalidWithoutTruncating(t *testing.T) {
	if err := ValidateTitle(strings.Repeat("é", 80)); err != nil {
		t.Fatal(err)
	}
	for _, title := range []string{strings.Repeat("a", 161), "two\nlines", "bad\x00control", string([]byte{0xff})} {
		if err := ValidateTitle(title); err == nil {
			t.Fatalf("ValidateTitle(%q) accepted invalid title", title)
		}
	}
}
func TestCoordinatorPersistsCallerTargetSeparationAndNestedParent(t *testing.T) {
	root := filepath.Join(t.TempDir(), "runs")
	target := t.TempDir()
	callerCWD, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	record := filepath.Join(t.TempDir(), "parent.json")
	parent := NewCoordinator(NewStore(root)).Execute(Request{
		Surface: "test", Provider: "fake", Model: "fake", Title: "parent title", CWD: target,
		Provenance: &Provenance{CallerCWD: callerCWD, CallerKind: "longhouse", CallerSessionID: "root-session", CallerRequestID: "root-request"},
		Prompt:     "parent", Timeout: 5 * time.Second,
		Invocation: provider.Invocation{
			Argv:   []string{buildTestProvider(t)},
			SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "success_text", "HATCH_TEST_RECORD": record},
		},
	})
	if !parent.OK || parent.Run == nil || parent.Run.Provenance == nil {
		t.Fatalf("parent result = %#v", parent)
	}
	if parent.Run.CWD != target || parent.Run.Provenance.CallerCWD != callerCWD {
		t.Fatalf("caller/target CWD = target %q provenance %#v", parent.Run.CWD, parent.Run.Provenance)
	}
	var observed struct {
		Environment map[string]string `json:"environment"`
	}
	data, err := os.ReadFile(record)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &observed); err != nil {
		t.Fatal(err)
	}
	if observed.Environment["HATCH_RUN_ID"] != parent.Run.RunID ||
		observed.Environment["HATCH_CALLER_SESSION_ID"] != "root-session" ||
		observed.Environment["HATCH_CALLER_REQUEST_ID"] != "root-request" {
		t.Fatalf("parent propagation = %#v", observed.Environment)
	}

	for _, name := range []string{"HATCH_RUN_ID", "HATCH_CALLER_KIND", "HATCH_CALLER_SESSION_ID", "HATCH_CALLER_REQUEST_ID"} {
		t.Setenv(name, observed.Environment[name])
	}
	child := NewCoordinator(NewStore(root)).Execute(Request{
		Surface: "test", Provider: "fake", Model: "fake", CWD: target, Prompt: "child", Timeout: 5 * time.Second,
		Invocation: provider.Invocation{Argv: []string{buildTestProvider(t)}, SetEnv: map[string]string{"HATCH_TEST_SCENARIO": "success_text"}},
	})
	if child.Run == nil || child.Run.Provenance == nil || child.Run.Provenance.ParentRunID != parent.Run.RunID {
		t.Fatalf("child provenance = %#v", child.Run)
	}
	if got := child.Run.Provenance; got.CallerKind != "longhouse" || got.CallerSessionID != "root-session" || got.CallerRequestID != "root-request" {
		t.Fatalf("nested conversation changed identity: %#v", got)
	}
}

func TestCoordinatorStripsProviderEnvironment(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "must-not-reach-provider")
	record := filepath.Join(t.TempDir(), "provider.json")
	result := NewCoordinator(NewStore(filepath.Join(t.TempDir(), "runs"))).Execute(Request{
		Surface: "test", Provider: "fake", Model: "fake", Prompt: "check environment", Timeout: 5 * time.Second,
		Invocation: provider.Invocation{
			Argv:     []string{buildTestProvider(t)},
			UnsetEnv: []string{"GEMINI_API_KEY"},
			SetEnv:   map[string]string{"HATCH_TEST_SCENARIO": "success_text", "HATCH_TEST_RECORD": record},
		},
	})
	if !result.OK {
		t.Fatalf("provider failed: %#v", result)
	}
	var observed struct {
		Environment map[string]string `json:"environment"`
	}
	data, err := os.ReadFile(record)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &observed); err != nil {
		t.Fatal(err)
	}
	if _, present := observed.Environment["GEMINI_API_KEY"]; present {
		t.Fatal("stripped credential reached the provider process")
	}
}

func TestCallerKindBoundaryRemainsQueryable(t *testing.T) {
	kind := strings.Repeat("k", 64)
	t.Setenv("HATCH_CALLER_KIND", kind)
	t.Setenv("HATCH_CALLER_SESSION_ID", "kind-boundary")
	p, err := ResolveProvenance("", "")
	if err != nil {
		t.Fatal(err)
	}
	store := NewStore(filepath.Join(t.TempDir(), "runs"))
	artifact, err := store.Prepare(PreparedRun{Request: "request", Provenance: p})
	if err != nil {
		t.Fatal(err)
	}
	page, err := QueryRecords(store.Root, "", QueryOptions{All: true, CallerKind: kind})
	if err != nil || len(page.Runs) != 1 || page.Runs[0].RunID != artifact.Manifest.RunID {
		t.Fatalf("valid caller kind was not recoverable: %#v, %v", page, err)
	}
	t.Setenv("HATCH_CALLER_KIND", kind+"k")
	if _, err := ResolveProvenance("", ""); err == nil {
		t.Fatal("accepted an unqueryable inherited caller kind")
	}
	p.CallerKind += "k"
	if _, err := store.Prepare(PreparedRun{Request: "request", Provenance: p}); err == nil {
		t.Fatal("persisted an unqueryable caller kind")
	}
}
