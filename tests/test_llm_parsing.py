import json

from local_live.llm.ollama import parse_ollama_line
from local_live.llm.openrouter import parse_sse_payload
from local_live.llm.events import Completion, TextDelta, ToolCall


def test_ollama_chat_ndjson_parsing():
    text = json.dumps({"model": "qwen3.5:9b-q4_K_M", "message": {"content": "こんにちは"}, "done": False})
    event = parse_ollama_line(text, requested_model="qwen3.5:9b-q4_K_M")
    assert isinstance(event, TextDelta)
    assert event.actual_model == "qwen3.5:9b-q4_K_M"

    done = parse_ollama_line(
        json.dumps({"model": "qwen3.5:9b-q4_K_M", "done": True, "done_reason": "stop", "eval_count": 4}),
        requested_model="qwen3.5:9b-q4_K_M",
    )
    assert isinstance(done, Completion)
    assert done.usage == {"eval_count": 4}


def test_ollama_tool_call_parsing():
    event = parse_ollama_line(
        json.dumps(
            {
                "model": "local",
                "message": {
                    "tool_calls": [
                        {"function": {"name": "calculator", "arguments": {"expression": "2+2"}}}
                    ]
                },
                "done": False,
            }
        ),
        requested_model="local",
    )
    assert isinstance(event, ToolCall)
    assert event.name == "calculator"
    assert event.arguments == {"expression": "2+2"}


def test_openrouter_sse_parsing():
    event = parse_sse_payload(
        {"model": "provider/actual", "choices": [{"delta": {"content": "日本語"}, "finish_reason": None}]},
        requested_model="openrouter/free",
    )
    assert isinstance(event, TextDelta)
    assert event.actual_model == "provider/actual"

    completion = parse_sse_payload({"choices": [{"delta": {}, "finish_reason": "stop"}]}, requested_model="m")
    assert isinstance(completion, Completion)
