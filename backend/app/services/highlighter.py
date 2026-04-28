from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Span:
    start: int
    end: int
    text: str
    severity: float


def highlight_spans(
    text: str, token_attributions: list[dict], threshold: float = 0.5
) -> list[Span]:
    raw: list[Span] = []
    for attr in token_attributions:
        token = attr["token"]
        score = attr["score"]
        if score < threshold:
            continue
        pos = text.find(token)
        if pos == -1:
            continue
        raw.append(Span(start=pos, end=pos + len(token), text=token, severity=score))

    if not raw:
        return []

    raw.sort(key=lambda s: s.start)

    merged: list[Span] = [raw[0]]
    for span in raw[1:]:
        prev = merged[-1]
        if span.start <= prev.end + 1:
            new_end = max(prev.end, span.end)
            new_text = text[prev.start : new_end]
            new_severity = max(prev.severity, span.severity)
            merged[-1] = Span(start=prev.start, end=new_end, text=new_text, severity=new_severity)
        else:
            merged.append(span)

    return merged


def attributions_to_highlighted_phrases(
    text: str, token_attributions: list[dict], threshold: float = 0.5
) -> list[dict]:
    spans = highlight_spans(text, token_attributions, threshold)
    return [{"text": s.text, "start": s.start, "end": s.end, "severity": s.severity} for s in spans]
