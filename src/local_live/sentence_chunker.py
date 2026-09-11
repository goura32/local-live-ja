from __future__ import annotations

import time
from collections.abc import Callable


class SentenceChunker:
    """Small bounded-latency chunker for streamed LLM text.

    It prefers Japanese/Latin sentence punctuation, but emits at max_chars or
    timeout_s so a model that omits punctuation cannot hold playback forever.
    """

    _TERMINATORS = "。！？!?\n"

    def __init__(
        self,
        *,
        max_chars: int = 48,
        timeout_s: float = 0.8,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.max_chars = max_chars
        self.timeout_s = timeout_s
        self._clock = clock
        self._buffer = ""
        self._last_update: float | None = None

    def push(self, delta: str) -> list[str]:
        if not delta:
            return []
        now = self._clock()
        self._buffer += delta
        if self._last_update is None:
            self._last_update = now
        emitted: list[str] = []

        while True:
            punctuation_end = self._first_terminator_end()
            if punctuation_end is not None:
                emitted.append(self._take(punctuation_end))
                continue
            if len(self._buffer) > self.max_chars:
                emitted.append(self._take(self.max_chars))
                continue
            break

        if self._buffer and self._last_update is not None:
            if now - self._last_update >= self.timeout_s:
                emitted.append(self._take(len(self._buffer)))
        if self._buffer:
            self._last_update = now
        else:
            self._last_update = None
        return [item for item in emitted if item]

    def flush(self) -> list[str]:
        if not self._buffer:
            self._last_update = None
            return []
        item = self._take(len(self._buffer))
        self._last_update = None
        return [item] if item else []

    def _first_terminator_end(self) -> int | None:
        for index, char in enumerate(self._buffer):
            if char in self._TERMINATORS:
                return index + 1
        return None

    def _take(self, length: int) -> str:
        item = self._buffer[:length].strip()
        self._buffer = self._buffer[length:]
        return item
