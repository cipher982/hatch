from __future__ import annotations

import json
from unittest import mock

from hatch.expert import DEFAULT_EXPERT_MODEL
from hatch.expert import _build_payload
from hatch.expert import run_expert_sync


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_build_payload_keeps_expert_sync_and_enables_web_search_by_default():
    payload = _build_payload(
        prompt="Should we refactor this?",
        model=DEFAULT_EXPERT_MODEL,
        reasoning_effort="medium",
        web_search=True,
        background=True,
    )

    assert payload["model"] == "gpt-5.5-pro"
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["background"] is True
    assert payload["store"] is True
    assert payload["tools"] == [{"type": "web_search"}]


def test_build_payload_can_disable_web_search():
    payload = _build_payload(
        prompt="Should we refactor this?",
        model=DEFAULT_EXPERT_MODEL,
        reasoning_effort="medium",
        web_search=False,
        background=True,
    )

    assert payload["model"] == "gpt-5.5-pro"
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["background"] is True
    assert payload["store"] is True
    assert "tools" not in payload


def test_build_payload_can_disable_background_storage():
    payload = _build_payload(
        prompt="Should we refactor this?",
        model=DEFAULT_EXPERT_MODEL,
        reasoning_effort="medium",
        web_search=False,
        background=False,
    )

    assert payload["background"] is False
    assert payload["store"] is False


def test_build_payload_can_enable_web_search():
    payload = _build_payload(
        prompt="What changed today?",
        model=DEFAULT_EXPERT_MODEL,
        reasoning_effort="high",
        web_search=True,
        background=True,
    )

    assert payload["tools"] == [{"type": "web_search"}]
    assert payload["tool_choice"] == "auto"
    assert payload["include"] == ["web_search_call.action.sources"]


def test_run_expert_sync_returns_output_and_metadata():
    response_payload = {
        "id": "resp_123",
        "model": "gpt-5.5-pro-2026-04-23",
        "usage": {"total_tokens": 12},
        "output": [
            {
                "type": "web_search_call",
                "action": {
                    "sources": [
                        {
                            "type": "url",
                            "url": "https://example.com",
                            "title": "Example source",
                        }
                    ]
                },
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Use the simple design.",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://example.com",
                                "title": "Example",
                                "start_index": 0,
                                "end_index": 3,
                            }
                        ],
                    }
                ],
            }
        ],
    }

    with (
        mock.patch("hatch.expert.hydrate_backend_kwargs", return_value={"api_key": "sk-test"}),
        mock.patch("hatch.expert.urllib.request.urlopen", return_value=_FakeResponse(response_payload)),
    ):
        result = run_expert_sync(prompt="question", web_search=True)

    assert result.ok is True
    assert result.output == "Use the simple design."
    assert result.resolved_model == "gpt-5.5-pro-2026-04-23"
    assert result.usage == {"total_tokens": 12}
    assert result.citations == [
        {
            "url": "https://example.com",
            "title": "Example",
            "start_index": 0,
            "end_index": 3,
        }
    ]
    assert result.sources == [
        {
            "type": "url",
            "url": "https://example.com",
            "title": "Example source",
        }
    ]


def test_run_expert_sync_polls_background_response():
    create_payload = {
        "id": "resp_123",
        "status": "queued",
        "model": "gpt-5.5-pro",
    }
    completed_payload = {
        "id": "resp_123",
        "status": "completed",
        "model": "gpt-5.5-pro-2026-04-23",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ],
    }
    progress: list[str] = []

    with (
        mock.patch("hatch.expert.hydrate_backend_kwargs", return_value={"api_key": "sk-test"}),
        mock.patch(
            "hatch.expert.urllib.request.urlopen",
            side_effect=[_FakeResponse(create_payload), _FakeResponse(completed_payload)],
        ) as urlopen,
        mock.patch("hatch.expert.time.sleep"),
    ):
        result = run_expert_sync(
            prompt="question",
            web_search=False,
            progress_handler=progress.append,
        )

    assert result.ok is True
    assert result.output == "done"
    assert result.response_id == "resp_123"
    assert urlopen.call_count == 2
    assert progress[0] == "[hatch] expert response resp_123 status=queued"
    assert progress[1].startswith("[hatch] expert response resp_123 status=completed")
