from local_live.llm.events import Completion, TextDelta, ToolCall, ToolResult
from local_live.tools import MockToolRegistry


def test_llm_events_cover_live_contract():
    assert TextDelta(text="こんにちは").kind == "text_delta"
    assert ToolCall(call_id="1", name="calculator", arguments={"expression": "2+2"}).kind == "tool_call"
    assert ToolResult(call_id="1", name="calculator", content="4").kind == "tool_result"
    assert Completion(reason="stop", actual_model="local").kind == "completion"


def test_mock_tools_are_deterministic_and_non_destructive():
    tools = MockToolRegistry()
    assert tools.call("calculator", {"expression": "2 + 3 * 4"}) == "14"
    assert tools.call("fixed_test_data", {"key": "status"}) == "PoC固定データ: ready"
    assert tools.call("deterministic_time", {}) == "2026-01-01T00:00:00+09:00"
    assert {item["function"]["name"] for item in tools.definitions()} == {
        "calculator",
        "fixed_test_data",
        "deterministic_time",
    }
