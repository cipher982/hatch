package run

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/cipher982/hatch/internal/provider"
)

func FuzzEvidencePath(f *testing.F) {
	f.Add("request.txt")
	f.Add("provider/opencode/data/opencode/opencode.db")
	f.Add("../secret")
	f.Fuzz(func(t *testing.T, name string) {
		path, err := evidencePath("/artifact", name)
		unsafe := name == "" || strings.HasPrefix(name, "/") || name == ".." || strings.HasPrefix(name, "../") || strings.Contains(name, "/../") || strings.Contains(name, "//") || strings.HasSuffix(name, "/.")
		if unsafe && err == nil {
			t.Fatalf("unsafe path accepted: %q -> %q", name, path)
		}
	})
}

func FuzzManifestReader(f *testing.F) {
	f.Add([]byte(`{"schema_version":1,"run_id":"hatch_seed","lifecycle":"terminal","outcome":"succeeded","capture":{"state":"durable"},"provider_state":{"retention":"unknown"}}`))
	f.Add([]byte(`{"schema_version":99}`))
	f.Add([]byte(`not-json`))
	f.Fuzz(func(t *testing.T, data []byte) {
		var raw map[string]any
		if json.Unmarshal(data, &raw) != nil {
			return
		}
		encoded, err := json.Marshal(raw)
		if err != nil {
			return
		}
		var manifest Manifest
		if json.Unmarshal(encoded, &manifest) != nil {
			return
		}
		if manifest.SchemaVersion == 1 {
			normalizeUnknownEnums(&manifest)
		}
	})
}

func FuzzRedactArgv(f *testing.F) {
	f.Add("ordinary")
	f.Add("$(danger) 'quoted'")
	f.Fuzz(func(t *testing.T, prompt string) {
		redacted, err := validatedRedactedArgv(provider.Invocation{Argv: []string{"provider", prompt}, RedactedArgv: []string{"provider", "<prompt>"}})
		if err != nil || len(redacted) != 2 || redacted[1] != "<prompt>" {
			t.Fatalf("redacted = %#v", redacted)
		}
	})
}
