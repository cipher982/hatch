package cli

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveCredentialPrecedence(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "environment")
	t.Setenv(credentialHelperEnv, filepath.Join(t.TempDir(), "missing"))
	value, err := resolveCredential("explicit", "OPENAI_API_KEY")
	if err != nil || value != "explicit" {
		t.Fatalf("explicit = %q, %v", value, err)
	}
	value, err = resolveCredential("", "OPENAI_API_KEY")
	if err != nil || value != "environment" {
		t.Fatalf("environment = %q, %v", value, err)
	}
}

func TestResolveCredentialHelperProtocol(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	directory := t.TempDir()
	requestPath := filepath.Join(directory, "request.json")
	helper := filepath.Join(directory, "helper")
	script := "#!/bin/sh\nIFS= read -r line\nprintf '%s' \"$line\" > \"$REQUEST_PATH\"\nprintf '%s\\n' 'helper-secret'\n"
	if err := os.WriteFile(helper, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("REQUEST_PATH", requestPath)
	t.Setenv(credentialHelperEnv, helper)
	value, err := resolveCredential("", "OPENAI_API_KEY")
	if err != nil || value != "helper-secret" {
		t.Fatalf("helper = %q, %v", value, err)
	}
	data, err := os.ReadFile(requestPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"environment":"OPENAI_API_KEY","project":"personal-shell"}` {
		t.Fatalf("request = %s", data)
	}
}

func TestResolveCredentialHelperAbsent(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "")
	helper := filepath.Join(t.TempDir(), "helper")
	if err := os.WriteFile(helper, []byte("#!/bin/sh\nexit 3\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv(credentialHelperEnv, helper)
	value, err := resolveCredential("", "OPENROUTER_API_KEY")
	if err != nil || value != "" {
		t.Fatalf("absent = %q, %v", value, err)
	}
}
