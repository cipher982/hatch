package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const (
	credentialHelperEnv        = "HATCH_CREDENTIAL_HELPER"
	credentialHelperConfigFile = "credential-helper"
	maxCredentialHelperConfig  = 4096
)

type credentialRequest struct {
	Environment string `json:"environment"`
	Project     string `json:"project"`
}

func resolveCredential(explicit, environmentName string) (string, error) {
	if value := strings.TrimSpace(explicit); value != "" {
		return value, nil
	}
	if value := strings.TrimSpace(os.Getenv(environmentName)); value != "" {
		return value, nil
	}
	helper, err := configuredCredentialHelper()
	if err != nil {
		return "", err
	}
	if helper == "" {
		return "", nil
	}
	if !filepath.IsAbs(helper) {
		return "", fmt.Errorf("%s must be an absolute executable path", credentialHelperEnv)
	}
	info, err := os.Stat(helper)
	if err != nil || !info.Mode().IsRegular() || info.Mode().Perm()&0o111 == 0 {
		return "", fmt.Errorf("configured Hatch credential helper is missing or not executable: %s", helper)
	}
	payload, err := json.Marshal(credentialRequest{Environment: environmentName, Project: "personal-shell"})
	if err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, helper)
	cmd.Stdin = bytes.NewReader(append(payload, '\n'))
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err = cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		return "", fmt.Errorf("credential helper timed out")
	}
	if err == nil {
		value := strings.TrimSpace(stdout.String())
		if value == "" {
			return "", fmt.Errorf("credential helper reported success without a secret")
		}
		return value, nil
	}
	if exit, ok := err.(*exec.ExitError); ok && exit.ExitCode() == 3 {
		return "", nil
	}
	detail := strings.Join(strings.Fields(stderr.String()), " ")
	if len(detail) > 300 {
		detail = detail[:297] + "..."
	}
	if detail == "" {
		detail = "helper exited unsuccessfully"
	}
	return "", fmt.Errorf("credential helper authority error: %s", detail)
}

func configuredCredentialHelper() (string, error) {
	if helper := strings.TrimSpace(os.Getenv(credentialHelperEnv)); helper != "" {
		return helper, nil
	}
	configRoot := strings.TrimSpace(os.Getenv("XDG_CONFIG_HOME"))
	if configRoot == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", nil
		}
		configRoot = filepath.Join(home, ".config")
	}
	if !filepath.IsAbs(configRoot) {
		return "", fmt.Errorf("XDG_CONFIG_HOME must be absolute")
	}
	path := filepath.Join(configRoot, "hatch", credentialHelperConfigFile)
	info, err := os.Lstat(path)
	if os.IsNotExist(err) {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("read Hatch credential helper configuration: %w", err)
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 || info.Mode().Perm()&0o077 != 0 {
		return "", fmt.Errorf("Hatch credential helper configuration is not a private regular file: %s", path)
	}
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("read Hatch credential helper configuration: %w", err)
	}
	data, readErr := io.ReadAll(io.LimitReader(file, maxCredentialHelperConfig+1))
	closeErr := file.Close()
	if readErr != nil {
		return "", fmt.Errorf("read Hatch credential helper configuration: %w", readErr)
	}
	if closeErr != nil {
		return "", fmt.Errorf("close Hatch credential helper configuration: %w", closeErr)
	}
	if len(data) > maxCredentialHelperConfig {
		return "", fmt.Errorf("Hatch credential helper configuration is too large")
	}
	helper := strings.TrimSpace(string(data))
	if helper == "" {
		return "", fmt.Errorf("Hatch credential helper configuration is empty")
	}
	return helper, nil
}
