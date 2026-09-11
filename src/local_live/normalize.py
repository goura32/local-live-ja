from __future__ import annotations

import unicodedata


def normalize_text(text: str) -> str:
    """Normalize text for synthetic ASR regression comparisons.

    This intentionally removes spacing and punctuation but keeps Japanese,
    Latin letters, digits, and symbols that may be meaningful in technical
    strings. It is not a linguistic tokenizer.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        ch
        for ch in normalized
        if not unicodedata.category(ch).startswith(("P", "Z"))
        and ch not in "\t\r\n"
    )


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Return character error rate after the PoC normalization."""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)
