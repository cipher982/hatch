"""Single-call expert consultation via the OpenAI Responses API."""

from __future__ import annotations

import json
import contextlib
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Literal

from hatch.backends import Backend
from hatch.credentials import hydrate_backend_kwargs


ExpertReasoningEffort = Literal["medium", "high", "xhigh"]
DEFAULT_EXPERT_MODEL = "gpt-5.5-pro"
RESPONSES_URL = "https://api.openai.com/v1/responses"
BACKGROUND_POLL_INTERVAL_S = 15.0
REQUEST_TIMEOUT_S = 60.0
ACTIVE_RESPONSE_STATUSES = {"queued", "in_progress", "interpreting"}
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
    background: bool,
) -> dict[str, Any]:
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    payload: dict[str, Any] = {
        "model": model,
        "instructions": EXPERT_INSTRUCTIONS,
        "input": prompt,
        "reasoning": {"effort": reasoning_effort},
        "background": background,
        "store": background,
    }
    if web_search:
        payload["tools"] = [{"type": "web_search"}]
        payload["tool_choice"] = "auto"
        payload["include"] = ["web_search_call.action.sources"]
    return payload


def _request_json(
    *,
    url: str,
    api_key: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout_s: float = REQUEST_TIMEOUT_S,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _response_url(response_id: str, *, include_sources: bool = False) -> str:
    url = f"{RESPONSES_URL}/{urllib.parse.quote(response_id)}"
    if not include_sources:
        return url
    query = urllib.parse.urlencode(
        {"include[]": ["web_search_call.action.sources"]},
        doseq=True,
    )
    return f"{url}?{query}"


def _cancel_response(api_key: str, response_id: str) -> None:
    with contextlib.suppress(Exception):
        _request_json(
            url=f"{RESPONSES_URL}/{urllib.parse.quote(response_id)}/cancel",
            api_key=api_key,
            method="POST",
            payload={},
            timeout_s=REQUEST_TIMEOUT_S,
        )


def _response_error_message(response: dict[str, Any]) -> str:
    error = response.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    incomplete = response.get("incomplete_details")
    if isinstance(incomplete, dict) and incomplete:
        return f"Response incomplete: {incomplete}"
    status = response.get("status")
    return f"Response ended with status: {status}"


def _wait_for_background_response(
    *,
    api_key: str,
    response: dict[str, Any],
    timeout_s: int,
    web_search: bool,
    started_at: float,
    progress_handler: Callable[[str], None] | None,
    poll_interval_s: float = BACKGROUND_POLL_INTERVAL_S,
) -> dict[str, Any]:
    response_id = str(response.get("id") or "")
    if not response_id:
        return response

    last_status = str(response.get("status") or "unknown")
    if progress_handler:
        progress_handler(f"[hatch] expert response {response_id} status={last_status}")

    while last_status in ACTIVE_RESPONSE_STATUSES:
        elapsed = time.perf_counter() - started_at
        remaining = timeout_s - elapsed
        if remaining <= 0:
            _cancel_response(api_key, response_id)
            raise TimeoutError(f"Expert consultation timed out after {timeout_s}s")

        time.sleep(min(poll_interval_s, max(0.0, remaining)))
        response = _request_json(
            url=_response_url(response_id, include_sources=web_search),
            api_key=api_key,
            timeout_s=REQUEST_TIMEOUT_S,
        )
        status = str(response.get("status") or "unknown")
        if progress_handler:
            elapsed_s = int(time.perf_counter() - started_at)
            progress_handler(f"[hatch] expert response {response_id} status={status} ({elapsed_s}s)")
        last_status = status

    return response


def run_expert_sync(
    *,
    prompt: str,
    model: str = DEFAULT_EXPERT_MODEL,
    reasoning_effort: ExpertReasoningEffort = "medium",
    web_search: bool = True,
    timeout_s: int = 900,
    api_key: str | None = None,
    background: bool = True,
    progress_handler: Callable[[str], None] | None = None,
) -> ExpertResult:
    """Run one synchronous Responses API expert consultation."""
    start = time.perf_counter()
    resolved_model: str | None = None
    response_id: str | None = None

    try:
        key = _resolve_api_key(api_key)
        payload = _build_payload(
            prompt=prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            web_search=web_search,
            background=background,
        )
        response_payload = _request_json(
            url=RESPONSES_URL,
            api_key=key,
            method="POST",
            payload=payload,
            timeout_s=REQUEST_TIMEOUT_S,
        )
        response_id = response_payload.get("id")
        resolved_model = response_payload.get("model")
        if background:
            response_payload = _wait_for_background_response(
                api_key=key,
                response=response_payload,
                timeout_s=timeout_s,
                web_search=web_search,
                started_at=start,
                progress_handler=progress_handler,
            )
            response_id = response_payload.get("id") or response_id
            resolved_model = response_payload.get("model") or resolved_model
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
    except TimeoutError as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return ExpertResult(
            ok=False,
            status="timeout",
            output="",
            duration_ms=duration_ms,
            error=str(e) or f"Expert consultation timed out after {timeout_s}s",
            model=model,
            resolved_model=resolved_model,
            reasoning_effort=reasoning_effort,
            web_search=web_search,
            response_id=response_id,
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
    response_status = str(response_payload.get("status") or "")
    if response_status and response_status != "completed":
        return ExpertResult(
            ok=False,
            status=response_status,
            output="",
            duration_ms=duration_ms,
            error=_response_error_message(response_payload),
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
