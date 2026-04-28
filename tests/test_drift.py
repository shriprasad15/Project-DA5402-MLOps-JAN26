from __future__ import annotations

import pytest

from backend.app.services.drift import DriftMonitor


@pytest.mark.unit
def test_in_distribution():
    monitor = DriftMonitor(reference_path=None)
    for _ in range(100):
        result = monitor.update("x" * 50)
    assert result["ks_pvalue"] == 1.0


@pytest.mark.unit
def test_returns_dict_keys():
    monitor = DriftMonitor(reference_path=None)
    result = monitor.update("hello world")
    assert "ks_pvalue" in result
    assert "oov_rate" in result
