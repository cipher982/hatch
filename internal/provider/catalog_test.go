package provider

import "testing"

func TestSurfaceAccountSeparatesEmployerFromPersonal(t *testing.T) {
	for surface, want := range map[string]string{
		"bedrock.sol": "zeta", "bedrock": "zeta", "cursor.gemini-4-argon": "zeta", "cursor.raw": "zeta",
		"claude.opus": "personal", "codex.astra": "personal", "gemini.argon": "personal",
		"openrouter.deepseek": "personal", "unknown": "personal",
	} {
		if got := SurfaceAccount(surface); got != want {
			t.Errorf("SurfaceAccount(%q) = %q, want %q", surface, got, want)
		}
	}
}
