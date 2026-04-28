from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from loguru import logger
from scipy import stats

WINDOW = 200  # rolling window size


class DriftMonitor:
    def __init__(self, reference_path: Path | None = None):
        self._length_window: deque[float] = deque(maxlen=WINDOW)
        self._ref_lengths: list[float] = []
        if reference_path and reference_path.exists():
            try:
                stats_data = json.loads(reference_path.read_text())
                quantiles = stats_data.get("length_quantiles", [])
                # support both list and dict ({"0.1": 40.0, ...}) formats
                if isinstance(quantiles, dict):
                    mean = stats_data.get("length_mean", 67.0)
                    std = stats_data.get("length_std", 21.0)
                    # reconstruct a synthetic reference sample from mean/std/quantiles
                    import random
                    random.seed(42)
                    self._ref_lengths = [
                        max(5.0, mean + std * (random.random() * 2 - 1)) for _ in range(200)
                    ]
                else:
                    self._ref_lengths = quantiles
            except Exception as e:
                logger.warning(f"Could not load drift reference: {e}")

    def update(self, text: str) -> dict[str, float]:
        """Update window with new sample, return current drift metrics."""
        self._length_window.append(float(len(text)))
        ks_pvalue = 1.0
        if len(self._length_window) >= 30 and len(self._ref_lengths) >= 10:
            try:
                stat, pvalue = stats.ks_2samp(list(self._length_window), self._ref_lengths)
                ks_pvalue = float(pvalue)
            except Exception:
                ks_pvalue = 1.0
        oov_rate = 0.0  # Wave 2 placeholder — real OOV needs tokenizer vocab
        return {"ks_pvalue": ks_pvalue, "oov_rate": oov_rate}
