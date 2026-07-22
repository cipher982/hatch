package cli

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cipher982/hatch/internal/provider"
)

func TestPopulateProviderVersion(t *testing.T) {
	binary := filepath.Join(t.TempDir(), "opencode")
	if err := os.WriteFile(binary, []byte("#!/bin/sh\nprintf '%s\\n' 'opencode 1.2.3'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	invocation := provider.Invocation{Argv: []string{binary}, Adapter: "opencode"}
	populateProviderVersion(&invocation)
	if invocation.ProviderVersion != "opencode 1.2.3" {
		t.Fatalf("version = %q", invocation.ProviderVersion)
	}
}
