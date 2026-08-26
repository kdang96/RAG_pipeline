from types import SimpleNamespace

import pytest

from retrieval import agentic_rag


def _message(content=None, thinking="reasoning", tool_calls=None):
    return SimpleNamespace(
        message=SimpleNamespace(content=content, thinking=thinking, tool_calls=tool_calls)
    )


def _tool_call(name="similarity_search", **args):
    args.setdefault("input", "media services")
    args.setdefault("output_fields", ["chunk"])
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=args))


@pytest.fixture
def patch_collection_fields(monkeypatch):
    monkeypatch.setattr(
        agentic_rag, "get_collection_fields", lambda db, coll: [{"field": "chunk"}]
    )


def test_answers_without_tool(monkeypatch, patch_collection_fields, config):
    monkeypatch.setattr(agentic_rag.ollama, "chat", lambda **k: _message(content="direct"))

    answer, reasoning = agentic_rag.rag_flow("hello", config)

    assert answer == "direct"
    assert reasoning == "reasoning"


def test_tool_call_triggers_search(monkeypatch, patch_collection_fields, config):
    responses = iter(
        [
            _message(tool_calls=[_tool_call()]),
            _message(content="grounded answer"),
        ]
    )
    monkeypatch.setattr(agentic_rag.ollama, "chat", lambda **k: next(responses))
    monkeypatch.setattr(
        agentic_rag, "search", lambda **k: [[{"entity": {"chunk_id": 1, "chunk": "x"}}]]
    )

    answer, _ = agentic_rag.rag_flow("what applies?", config)
    assert answer == "grounded answer"


def test_prior_history_is_forwarded(monkeypatch, patch_collection_fields, config):
    calls = []

    def record(**kwargs):
        calls.append(kwargs["messages"])
        return _message(content="ok")

    monkeypatch.setattr(agentic_rag.ollama, "chat", record)

    history = [{"role": "user", "content": "earlier question"}]
    agentic_rag.rag_flow("follow up", config, history=history)

    assert history[0] in calls[0]


def test_unknown_tool_returns_message(config):
    response = _message(tool_calls=[_tool_call(name="delete_everything")])
    obs, formatted = agentic_rag.tools(response, config)
    assert "Unknown action" in formatted
