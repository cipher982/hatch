package provider

import (
	"crypto/sha256"
	"encoding/hex"
	"reflect"
	"testing"
)

func TestPreparePromptOracle(t *testing.T) {
	got := []byte(PreparePrompt("oracle prompt"))
	digest := sha256.Sum256(got)
	if len(got) != 570 || hex.EncodeToString(digest[:]) != "0eb7749ad17c17ce3323e0628fc9a6fc040d640d95edb7d6839f983ac354ee54" {
		t.Fatalf("prepared prompt len=%d sha256=%s", len(got), hex.EncodeToString(digest[:]))
	}
}

func TestBuildOracleInvocations(t *testing.T) {
	tests := []struct {
		name string
		req  Request
		argv []string
	}{
		{"gemini", Request{Backend: "gemini", Prompt: "oracle prompt"}, []string{"gemini", "--model", "gemini-3-pro-preview", "--yolo", "--skip-trust", "-p", "-"}},
		{"cursor", Request{Backend: "cursor", Model: "cursor-grok-4.5-high", Prompt: "oracle prompt"}, []string{"cursor-agent", "--print", "--trust", "--model", "cursor-grok-4.5-high", "--output-format", "stream-json", "--force", PreparePrompt("oracle prompt")}},
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
