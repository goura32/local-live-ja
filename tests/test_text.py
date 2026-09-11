from local_live.normalize import cer, normalize_text
from local_live.sentence_chunker import SentenceChunker


def test_normalize_text_is_japanese_cer_friendly():
    assert normalize_text(" ＧＰＵ、Docker！  ") == "gpudocker"
    assert cer("今日は 12 時です。", "今日は12時です") == 0.0


def test_sentence_chunker_emits_short_natural_chunks():
    chunker = SentenceChunker(max_chars=24, timeout_s=10.0, clock=lambda: 0.0)
    assert chunker.push("こんにちは。次の") == ["こんにちは。"]
    assert chunker.push("話です") == []
    assert chunker.flush() == ["次の話です"]


def test_sentence_chunker_has_max_character_bound():
    chunker = SentenceChunker(max_chars=5, timeout_s=10.0, clock=lambda: 0.0)
    assert chunker.push("あいうえおか") == ["あいうえお"]
    assert chunker.flush() == ["か"]


def test_sentence_chunker_timeout_flushes_without_punctuation():
    now = [0.0]
    chunker = SentenceChunker(max_chars=100, timeout_s=0.5, clock=lambda: now[0])
    assert chunker.push("句読点を待たない") == []
    now[0] = 0.6
    assert chunker.push("短文") == ["句読点を待たない短文"]
