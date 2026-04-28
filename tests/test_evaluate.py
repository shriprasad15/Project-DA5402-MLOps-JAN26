from __future__ import annotations

import pandas as pd
import pytest

from src.evaluate import compute_metrics


@pytest.mark.unit
def test_expected_keys() -> None:
    df = pd.DataFrame(
        {
            "tone_pred": [0, 1, 2, 3, 4],
            "tone_label": [0, 1, 2, 3, 4],
            "pa_pred": [0.1, 0.9, 0.5, 0.2, 0.8],
            "pa_label": [0.0, 1.0, 0.5, 0.0, 1.0],
            "sarcasm_pred": [0.2, 0.8, 0.4, 0.3, 0.7],
            "sarcasm_label": [0.0, 1.0, 0.5, 0.0, 1.0],
        }
    )
    metrics = compute_metrics(df)
    assert "macro_f1" in metrics
    assert "pa_mae" in metrics
    assert "sarcasm_mae" in metrics
    assert "per_class_f1" in metrics
