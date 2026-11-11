// Hatch adapter: Pi bash tool -> Agent Home guard runner (guard.py).
// The runner owns every decision (allow, check-in with a dcg-ack code, hard
// stop); this file only relays each bash command and blocks the call on a
// denial or on any failure (fail closed).
import { spawnSync } from "node:child_process";

const block = (reason) => ({ block: true, reason });

export default function (pi) {
  pi.on("tool_call", async (event, ctx) => {
    if (event.toolName !== "bash") return undefined;
    const command = event.input?.command;
    if (typeof command !== "string" || command.length === 0) {
      return block("DCG guard failed: bash command was missing");
    }
    const runner = process.env.HATCH_DCG_RUNNER;
    if (!runner) return block("DCG guard failed: HATCH_DCG_RUNNER is not set");

    const env = { ...process.env, DCG_NO_SELF_HEAL: "1", DCG_HARNESS: "hatch-pi" };
    delete env.DCG_BYPASS;
    const run = spawnSync(runner, ["--protocol", "json"], {
      input: JSON.stringify({ command, cwd: ctx?.cwd || process.cwd() }),
      env,
      encoding: "utf8",
      timeout: 10000,
    });
    if (run.error) return block(`DCG guard failed: ${run.error.message}`);
    if (run.status !== 0) {
      return block(`DCG guard failed (exit ${run.status}): ${(run.stderr || "").trim() || "no diagnostic"}`);
    }
    let result;
    try {
      result = JSON.parse((run.stdout || "").trim());
    } catch {
      return block("DCG guard failed: malformed guard response");
    }
    if (result?.decision === "allow") return undefined;
    if (result?.decision === "deny") return block(result.reason || "Blocked by DCG");
    return block("DCG guard failed: unexpected guard response");
  });
}
