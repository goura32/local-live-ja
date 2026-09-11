from local_live.bench import _collect_turn
from local_live.llm.events import Cancelled, TextDelta


class OneTokenThenCancelProvider:
    requested_model = "fake"
    last_actual_model = None

    def stream(self, messages, *, tools=None, cancel_event=None):
        yield TextDelta(text="最初のtoken")
        if cancel_event is not None and cancel_event.is_set():
            yield Cancelled()
            return
        yield TextDelta(text="後続token")


def test_collect_turn_can_cancel_after_first_delta():
    result = _collect_turn(OneTokenThenCancelProvider(), [], cancel_after_first=True)
    assert result["cancelled"] is True
    assert result["text"] == "最初のtoken"
