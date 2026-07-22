package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/cipher982/hatch/internal/cli"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	stopAfterFirstSignal := context.AfterFunc(ctx, stop)
	defer stopAfterFirstSignal()
	info, _ := os.Stdout.Stat()
	stdoutTTY := info != nil && info.Mode()&os.ModeCharDevice != 0
	os.Exit(cli.MainContext(ctx, os.Args[1:], os.Stdin, os.Stdout, os.Stderr, stdoutTTY))
}
