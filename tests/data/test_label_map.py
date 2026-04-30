import pandas as pd

from src.data.label_map import UNIFIED_COLS, Tone, to_unified


def test_sarcasm_headlines_maps_to_unified_schema():
    df = pd.DataFrame({"text": ["wow amazing"], "sarcasm": [1.0]})
    out = to_unified(df, source="sarcasm_headlines")
    assert list(out.columns) == UNIFIED_COLS
    row = out.iloc[0]
    assert row["sarcasm"] == 1.0
    # sarcasm > 0.5 maps to passive_aggression=0.75 by design in label_map.py
    assert row["passive_aggression"] == 0.75
    assert row["tone"] in {t.value for t in Tone}
    assert bool(row["weak_label"]) is True


def test_goemotions_anger_maps_to_aggressive_tone():
    df = pd.DataFrame(
        {"text": ["I hate this"], "labels": [[2]]}
    )  # 2 = anger in go_emotions simplified
    out = to_unified(df, source="goemotions")
    assert out.iloc[0]["tone"] == Tone.AGGRESSIVE.value
