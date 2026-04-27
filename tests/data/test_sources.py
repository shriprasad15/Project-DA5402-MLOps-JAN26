import pandas as pd

from src.data.sources import SarcasmHeadlinesAdapter


def test_source_adapter_protocol_returns_dataframe_with_required_cols(tmp_path):
    adapter = SarcasmHeadlinesAdapter(cache_dir=tmp_path)
    df = adapter.load(limit=5)
    assert isinstance(df, pd.DataFrame)
    for col in ["text", "source"]:
        assert col in df.columns
    assert len(df) == 5


def test_adapter_is_cached_on_second_call(tmp_path, monkeypatch):
    adapter = SarcasmHeadlinesAdapter(cache_dir=tmp_path)
    adapter.load(limit=3)
    monkeypatch.setattr(
        adapter, "_download", lambda: (_ for _ in ()).throw(AssertionError("network hit"))
    )
    df = adapter.load(limit=3)
    assert len(df) == 3
