package cli

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cipher982/hatch/internal/provider"
)

func TestPopulateProviderVersion(t *testing.T) {
	binary := filepath.Join(t.TempDir(), "pi")
	if err := os.WriteFile(binary, []byte("#!/bin/sh\nprintf '%s\\n' 'pi 1.2.3'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	invocation := provider.Invocation{Argv: []string{binary}, Adapter: "pi"}
	populateProviderVersion(&invocation)
	if invocation.ProviderVersion != "pi 1.2.3" {
		t.Fatalf("version = %q", invocation.ProviderVersion)
	}
}
