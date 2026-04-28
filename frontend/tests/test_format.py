from __future__ import annotations

import pytest

from frontend.helpers import build_highlight_html, score_to_hex_color, tone_to_color


@pytest.mark.unit
def test_tone_to_color_passive_aggressive():
    assert tone_to_color("passive_aggressive") == "red"


@pytest.mark.unit
def test_tone_to_color_unknown():
    assert tone_to_color("xyz") == "grey"


@pytest.mark.unit
def test_score_to_hex_zero():
    # intensity = int(0 * 200 + 55) = 55; 255 - 55 = 200 = 0xc8
    assert score_to_hex_color(0.0) == "#ffc8c8"


@pytest.mark.unit
def test_score_to_hex_one():
    # intensity = int(1 * 200 + 55) = 255; 255 - 255 = 0 = 0x00
    assert score_to_hex_color(1.0) == "#ff0000"


@pytest.mark.unit
def test_build_highlight_html_single():
    text = "hello world"
    highlights = [{"start": 6, "end": 11, "severity": 1.0}]
    result = build_highlight_html(text, highlights)
    assert "<mark" in result
    assert "world" in result


@pytest.mark.unit
def test_build_highlight_html_empty():
    assert build_highlight_html("hello", []) == "hello"
