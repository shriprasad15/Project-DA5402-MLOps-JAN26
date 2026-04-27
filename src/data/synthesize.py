"""Generate synthetic passive-aggressive emails using Gemma 3 4B via Ollama.

Runs as the `synth_pa` DVC stage. Produces parquet files with the unified
schema. Uses few-shot prompting with 20 exemplars (Gemma's 128k context
lets us be generous). Invalid JSON, invalid scores, or invalid tone values
are dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from contracts.tone_enum import Tone

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts" / "synthesize"


class GeneratedSample(BaseModel):
    text: str = Field(min_length=5, max_length=2000)
    passive_aggression: float = Field(ge=0.0, le=1.0)
    sarcasm: float = Field(ge=0.0, le=1.0)
    tone: Tone


@dataclass
class SyntheticGenerator:
    model: str
    out_dir: Path
    fallback_model: str = "llama3:8b"
    host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    batch_size: int = 10
    temperature: float = 0.9
    few_shot_count: int = 20

    def generate(self, target_count: int) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        system = (PROMPTS_DIR / "system.md").read_text()
        exemplars = json.loads((PROMPTS_DIR / "exemplars.json").read_text())

        accepted: list[GeneratedSample] = []
        while len(accepted) < target_count:
            few_shot = json.dumps(exemplars[: self.few_shot_count], indent=2)
            prompt = (
                f"{system}\n\nExemplars:\n{few_shot}\n\n"
                f"Now generate {self.batch_size} new samples as a JSON array. "
                f"Only output the JSON, nothing else."
            )
            raw = self._call_ollama(prompt)
            parsed = self._parse(raw)
            logger.info(f"batch: asked {self.batch_size}, got {len(parsed)} valid")
            accepted.extend(parsed)

        df = pd.DataFrame([s.model_dump() for s in accepted[:target_count]])
        df["tone"] = df["tone"].map(lambda t: t.value if hasattr(t, "value") else t)
        df["source"] = "synthetic_v1"
        df["weak_label"] = True

        out = self.out_dir / f"{uuid.uuid4().hex[:8]}.parquet"
        df.to_parquet(out)
        logger.info(f"wrote {len(df)} synthetic samples to {out}")
        return out

    def _call_ollama(self, prompt: str) -> str:
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json()["response"]
        except Exception as e:
            logger.warning(f"{self.model} failed ({e}), falling back to {self.fallback_model}")
            resp = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.fallback_model, "prompt": prompt, "stream": False},
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json()["response"]

    @staticmethod
    def _parse(raw: str) -> list[GeneratedSample]:
        try:
            data = json.loads(raw[raw.find("[") : raw.rfind("]") + 1])
        except json.JSONDecodeError:
            return []
        out: list[GeneratedSample] = []
        for item in data:
            try:
                out.append(GeneratedSample(**item))
            except ValidationError:
                continue
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-count", type=int, default=10000)
    ap.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "gemma3:4b"))
    args = ap.parse_args()
    SyntheticGenerator(model=args.model, out_dir=args.out).generate(args.target_count)


if __name__ == "__main__":
    main()
