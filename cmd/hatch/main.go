package main

import (
	"os"

	"github.com/cipher982/hatch/internal/cli"
)

func main() {
	info, _ := os.Stdout.Stat()
	stdoutTTY := info != nil && info.Mode()&os.ModeCharDevice != 0
	os.Exit(cli.Main(os.Args[1:], os.Stdin, os.Stdout, os.Stderr, stdoutTTY))
}
