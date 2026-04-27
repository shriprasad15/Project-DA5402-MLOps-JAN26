import json

import pandas as pd

from src.data.synthesize import GeneratedSample, SyntheticGenerator


def test_parse_model_output_happy_path():
    raw = json.dumps(
        [
            {
                "text": "As we discussed...",
                "passive_aggression": 0.9,
                "sarcasm": 0.2,
                "tone": "passive_aggressive",
            }
        ]
    )
    out = SyntheticGenerator._parse(raw)
    assert len(out) == 1
    assert isinstance(out[0], GeneratedSample)
    assert out[0].passive_aggression == 0.9


def test_parse_rejects_invalid_scores():
    raw = json.dumps([{"text": "x", "passive_aggression": 2.0, "sarcasm": 0.2, "tone": "x"}])
    assert SyntheticGenerator._parse(raw) == []


def test_generate_writes_parquet(tmp_path, monkeypatch):
    gen = SyntheticGenerator(model="gemma3:4b", out_dir=tmp_path)
    monkeypatch.setattr(
        gen,
        "_call_ollama",
        lambda p: json.dumps(
            [
                {
                    "text": "sample",
                    "passive_aggression": 0.8,
                    "sarcasm": 0.1,
                    "tone": "passive_aggressive",
                }
            ]
        ),
    )
    gen.generate(target_count=2)
    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 1
    df = pd.read_parquet(files[0])
    assert len(df) >= 1
    assert "passive_aggression" in df.columns
