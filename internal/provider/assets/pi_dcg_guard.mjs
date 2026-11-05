// Hatch adapter: Pi tool hook -> DCG's Claude-compatible protocol.
// DCG owns policy and command classification; this file only relays each bash
// command to it and blocks the call on a denial or any failure (fail closed).
import { spawnSync } from "node:child_process";

const block = (reason) => ({ block: true, reason });

export default function (pi) {
  pi.on("tool_call", async (event) => {
    if (event.toolName !== "bash") return undefined;
    const command = event.input?.command;
    if (typeof command !== "string" || command.length === 0) {
      return block("DCG guard failed: bash command was missing");
    }
    const dcg = process.env.HATCH_DCG_BINARY;
    if (!dcg) return block("DCG guard failed: HATCH_DCG_BINARY is not set");

    const env = { ...process.env };
    delete env.DCG_BYPASS;
    env.DCG_NO_SELF_HEAL = "1";
    const run = spawnSync(dcg, [], {
      input: JSON.stringify({ tool_name: "Bash", tool_input: { command } }),
      env,
      encoding: "utf8",
      timeout: 5000,
    });
    if (run.error) return block(`DCG guard failed: ${run.error.message}`);
    const text = (run.stdout || "").trim();
    if (run.status !== 0) {
      return block(`DCG guard failed (exit ${run.status}): ${(run.stderr || "").trim() || "no diagnostic"}`);
    }
    if (text === "") return undefined;
    let result;
    try {
      result = JSON.parse(text);
    } catch {
      return block("DCG guard failed: malformed hook response");
    }
    if (result?.hookSpecificOutput?.permissionDecision === "deny") {
      const reason = result.hookSpecificOutput.permissionDecisionReason;
      return block(typeof reason === "string" && reason ? reason : "Blocked by DCG");
    }
    return block("DCG guard failed: unexpected non-empty hook response");
  });
}
