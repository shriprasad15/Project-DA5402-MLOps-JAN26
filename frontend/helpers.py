from __future__ import annotations

TONE_COLORS = {
    "neutral": "green",
    "friendly": "blue",
    "assertive": "orange",
    "aggressive": "red",
    "passive_aggressive": "red",
}


def tone_to_color(tone: str) -> str:
    return TONE_COLORS.get(tone, "grey")


def score_to_hex_color(severity: float) -> str:
    intensity = int(max(0.0, min(1.0, severity)) * 200 + 55)
    return f"#ff{255-intensity:02x}{255-intensity:02x}"


def build_highlight_html(text: str, highlights: list[dict]) -> str:
    sorted_h = sorted(highlights, key=lambda x: x["start"], reverse=True)
    for h in sorted_h:
        severity = h.get("severity", 0.5)
        color_hex = score_to_hex_color(severity)
        span = (
            f'<mark style="background-color:{color_hex};padding:2px 4px;border-radius:3px">'
            f'{text[h["start"]:h["end"]]}</mark>'
        )
        text = text[: h["start"]] + span + text[h["end"] :]
    return text
