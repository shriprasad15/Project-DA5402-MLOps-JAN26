import pandas as pd

from src.data.drift_baseline import compute_baseline


def test_baseline_has_expected_fields():
    df = pd.DataFrame({"text": ["hi there", "hello world"] * 100, "tone": ["neutral"] * 200})
    stats = compute_baseline(df)
    assert {"length_mean", "length_std", "length_quantiles", "vocab"} <= stats.keys()
    assert stats["length_mean"] > 0
    assert "0.5" in stats["length_quantiles"]
