package run

import (
	"bytes"
	"io"
	"path/filepath"
	"testing"
	"time"
)

func BenchmarkCaptureWriterThroughput(b *testing.B) {
	data := bytes.Repeat([]byte("provider output line\n"), 4096)
	b.SetBytes(int64(len(data)))
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		var memory bytes.Buffer
		writer := &captureWriter{memory: &memory, sink: io.Discard}
		_, _ = writer.Write(data)
	}
}

func BenchmarkArtifactTerminalCommit(b *testing.B) {
	root := b.TempDir()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		b.StopTimer()
		store := NewStore(filepath.Join(root, "runs"))
		artifact, err := store.Prepare(PreparedRun{Request: "prompt"})
		if err != nil {
			b.Fatal(err)
		}
		stdout, stderr, err := store.OpenStreams(artifact)
		if err != nil {
			b.Fatal(err)
		}
		_, _ = stdout.Write([]byte("answer"))
		_ = stdout.Close()
		_ = stderr.Close()
		file, err := store.WriteResult(artifact, []byte("answer"))
		if err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		if err := store.CommitTerminal(artifact, OutcomeSucceeded, 0, Result{Output: "present", TerminalMarker: "not_applicable", OutputFile: &file}, State{Retention: "unknown", NativeIDState: "not_exposed", Capabilities: map[string]string{}}, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRunIDGeneration(b *testing.B) {
	now := time.Now()
	for index := 0; index < b.N; index++ {
		if _, err := newRunID(now); err != nil {
			b.Fatal(err)
		}
	}
}
