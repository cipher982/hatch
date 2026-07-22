package provider

import "testing"

func TestProgressParserDeduplicatesToolsAndReportsLifecycle(t *testing.T) {
	tests := []struct {
		name, adapter, label string
		lines                []string
		want                 []string
	}{
		{
			name: "claude", adapter: "claude", label: "Claude",
			lines: []string{
				`{"type":"system","subtype":"init","model":"haiku","session_id":"claude-session"}`,
				`{"type":"assistant","message":{"content":[{"type":"tool_use","id":"tool-1","name":"Read","input":{"file_path":"/tmp/a"}}]}}`,
				`{"type":"assistant","message":{"content":[{"type":"tool_use","id":"tool-1","name":"Read","input":{"file_path":"/tmp/a"}}]}}`,
				`{"type":"result","duration_ms":1200}`,
			},
			want: []string{"[hatch] Claude started (haiku, session claude-s)", "[hatch] Read: /tmp/a", "[hatch] Claude completed in 1.2s"},
		},
		{
			name: "cursor", adapter: "cursor", label: "Cursor",
			lines: []string{
				`{"type":"tool_call","subtype":"started","call_id":"call-1","tool_call":{"shell":{"args":{"command":"go test ./..."}}}}`,
				`{"type":"tool_call","subtype":"started","call_id":"call-1","tool_call":{"shell":{"args":{"command":"go test ./..."}}}}`,
			},
			want: []string{"[hatch] Cursor shell: go test ./..."},
		},
		{
			name: "opencode", adapter: "opencode", label: "OpenRouter",
			lines: []string{
				`{"type":"step_start","sessionID":"ses_123456789"}`,
				`{"type":"tool_use","part":{"callID":"call-1","tool":"bash","state":{"input":{"description":"run tests"}}}}`,
				`{"type":"step_finish","part":{"reason":"stop"}}`,
			},
			want: []string{"[hatch] OpenRouter started (session ses_1234)", "[hatch] bash: run tests", "[hatch] OpenRouter completed"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			parser := NewProgressParser(test.adapter, test.label)
			var got []string
			for _, line := range test.lines {
				got = append(got, parser.Observe([]byte(line))...)
			}
			if len(got) != len(test.want) {
				t.Fatalf("progress = %#v, want %#v", got, test.want)
			}
			for index := range got {
				if got[index] != test.want[index] {
					t.Fatalf("progress = %#v, want %#v", got, test.want)
				}
			}
		})
	}
}
