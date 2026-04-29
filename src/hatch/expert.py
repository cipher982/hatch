"""Single-call expert consultation via the OpenAI Responses API."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from typing import Literal

from hatch.backends import Backend
from hatch.credentials import hydrate_backend_kwargs


ExpertReasoningEffort = Literal["medium", "high", "xhigh"]
DEFAULT_EXPERT_MODEL = "gpt-5.5-pro"
RESPONSES_URL = "https://api.openai.com/v1/responses"
EXPERT_INSTRUCTIONS = """\
You are an expert consultant. Answer the user's question directly.
This is a single synchronous consultation, not an agent run.

Prioritize:
- the decision or recommendation
- the key reasoning that supports it
- material risks, counterarguments, and uncertainty
- what evidence would change the answer

Do not claim access to local files, shell commands, or repo state unless the
user included that context in the prompt. If context is missing, say exactly
what assumption you are making.
"""


@dataclass(frozen=True)
class ExpertResult:
    ok: bool
    status: str
    output: str
    duration_ms: int
    error: str | None
    model: str
    resolved_model: str | None
    reasoning_effort: str
    web_search: bool
    usage: dict[str, Any] | None = None
    response_id: str | None = None
    citations: list[dict[str, Any]] | None = None
    sources: list[dict[str, Any]] | None = None
    raw_response: dict[str, Any] | None = None

    def to_dict(self, *, include_raw_response: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "status": self.status,
            "output": self.output,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "model": self.model,
            "resolved_model": self.resolved_model,
            "reasoning_effort": self.reasoning_effort,
            "web_search": self.web_search,
            "usage": self.usage,
            "response_id": self.response_id,
            "citations": self.citations or [],
            "sources": self.sources or [],
        }
        if include_raw_response:
            payload["raw_response"] = self.raw_response
        return payload


def _resolve_api_key(api_key: str | None) -> str:
    kwargs = {"api_key": api_key} if api_key else {}
    return str(hydrate_backend_kwargs(Backend.CODEX, kwargs)["api_key"])


def _extract_output_text(response: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in response.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text":
                chunks.append(str(content.get("text") or ""))
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _extract_citations(response: dict[str, Any]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for item in response.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            for annotation in content.get("annotations") or []:
                if annotation.get("type") == "url_citation":
                    citations.append(
                        {
                            "url": annotation.get("url"),
                            "title": annotation.get("title"),
                            "start_index": annotation.get("start_index"),
                            "end_index": annotation.get("end_index"),
                        }
                    )
    return citations


def _extract_sources(response: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for item in response.get("output") or []:
        if item.get("type") != "web_search_call":
            continue
        action = item.get("action") or {}
        for source in action.get("sources") or []:
            url = str(source.get("url") or "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            sources.append(
                {
                    "type": source.get("type"),
                    "url": url,
                    "title": source.get("title"),
                }
            )
    return sources


def _build_payload(
    *,
    prompt: str,
    model: str,
    reasoning_effort: ExpertReasoningEffort,
    web_search: bool,
) -> dict[str, Any]:
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    payload: dict[str, Any] = {
        "model": model,
        "instructions": EXPERT_INSTRUCTIONS,
        "input": prompt,
        "reasoning": {"effort": reasoning_effort},
        "store": False,
    }
    if web_search:
        payload["tools"] = [{"type": "web_search"}]
        payload["tool_choice"] = "auto"
        payload["include"] = ["web_search_call.action.sources"]
    return payload


def run_expert_sync(
    *,
    prompt: str,
    model: str = DEFAULT_EXPERT_MODEL,
    reasoning_effort: ExpertReasoningEffort = "medium",
    web_search: bool = True,
    timeout_s: int = 900,
    api_key: str | None = None,
) -> ExpertResult:
    """Run one synchronous Responses API expert consultation."""
    start = time.perf_counter()
    resolved_model: str | None = None

    try:
        key = _resolve_api_key(api_key)
        payload = _build_payload(
            prompt=prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            web_search=web_search,
        )
        request = urllib.request.Request(
            RESPONSES_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        try:
            error_payload = json.loads(e.read().decode("utf-8"))
            message = str((error_payload.get("error") or {}).get("message") or error_payload)
        except Exception:
            message = str(e)
        return ExpertResult(
            ok=False,
            status="error",
            output="",
            duration_ms=duration_ms,
            error=message,
            model=model,
            resolved_model=resolved_model,
            reasoning_effort=reasoning_effort,
            web_search=web_search,
        )
    except TimeoutError:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return ExpertResult(
            ok=False,
            status="timeout",
            output="",
            duration_ms=duration_ms,
            error=f"Expert consultation timed out after {timeout_s}s",
            model=model,
            resolved_model=resolved_model,
            reasoning_effort=reasoning_effort,
            web_search=web_search,
        )
    except Exception as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return ExpertResult(
            ok=False,
            status="error",
            output="",
            duration_ms=duration_ms,
            error=str(e),
            model=model,
            resolved_model=resolved_model,
            reasoning_effort=reasoning_effort,
            web_search=web_search,
        )

    duration_ms = int((time.perf_counter() - start) * 1000)
    resolved_model = str(response_payload.get("model") or model)
    output = _extract_output_text(response_payload)
    if not output:
        return ExpertResult(
            ok=False,
            status="transport_error",
            output="",
            duration_ms=duration_ms,
            error="Responses API returned no final output",
            model=model,
            resolved_model=resolved_model,
            reasoning_effort=reasoning_effort,
            web_search=web_search,
            usage=response_payload.get("usage"),
            response_id=response_payload.get("id"),
            raw_response=response_payload,
        )

    return ExpertResult(
        ok=True,
        status="ok",
        output=output,
        duration_ms=duration_ms,
        error=None,
        model=model,
        resolved_model=resolved_model,
        reasoning_effort=reasoning_effort,
        web_search=web_search,
        usage=response_payload.get("usage"),
        response_id=response_payload.get("id"),
        citations=_extract_citations(response_payload),
        sources=_extract_sources(response_payload),
        raw_response=response_payload,
    )
