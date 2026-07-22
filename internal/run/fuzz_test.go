package run

import (
	"strings"
	"testing"
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

func FuzzRedactArgv(f *testing.F) {
	f.Add("ordinary")
	f.Add("$(danger) 'quoted'")
	f.Fuzz(func(t *testing.T, prompt string) {
		prepared := "Hatch execution contract:\n" + prompt
		redacted := redactArgv([]string{"provider", prepared})
		if len(redacted) != 2 || redacted[1] != "<prompt>" {
			t.Fatalf("redacted = %#v", redacted)
		}
	})
}
