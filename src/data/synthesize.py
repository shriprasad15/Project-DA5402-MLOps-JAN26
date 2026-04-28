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
import random
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


_PA_OPENERS = [
    "Per my last email,",
    "Just to clarify,",
    "As previously mentioned,",
    "Going forward,",
    "Not sure if you saw my previous message, but",
    "Friendly reminder that",
    "As per our discussion,",
    "Circling back on this,",
    "As I mentioned before,",
    "Just following up again,",
    "In case it got lost in your inbox,",
    "As outlined in my earlier email,",
    "Just wanted to make sure this didn't fall through the cracks,",
    "I trust you've had a chance to review my previous note,",
    "To reiterate what was already communicated,",
    "As discussed at length in our last meeting,",
    "I'm sure you're very busy, but",
    "Not to be a bother, but",
    "I hate to keep asking, but",
    "Apologies for the inconvenience of having to follow up again,",
]

_PA_MIDDLES = [
    "the deadline was last Friday.",
    "this needs to be done by end of day.",
    "I'm still waiting on the deliverable.",
    "please make sure to action this.",
    "this was supposed to be completed weeks ago.",
    "the client is asking questions I can't answer.",
    "we've discussed this multiple times already.",
    "this shouldn't require further explanation.",
    "I would appreciate a response at your earliest convenience.",
    "please advise on the status.",
    "kindly acknowledge receipt of this message.",
    "it would be great if you could prioritize this.",
    "I've cc'd our manager just to keep everyone in the loop.",
    "I trust this will be handled appropriately.",
    "we should be aligned on expectations by now.",
    "your prompt attention to this matter is appreciated.",
    "I'll leave it with you to resolve.",
    "I'm confident you'll find a way to make this work.",
    "moving forward, please ensure better communication.",
    "I'd appreciate it if we could avoid this situation in future.",
]

_PA_CLOSERS = [
    "Thanks so much.",
    "Best.",
    "Regards.",
    "Thanks in advance.",
    "Looking forward to your response.",
    "Appreciate your attention to this.",
    "Do let me know.",
    "Happy to jump on a call if needed.",
    "Let me know if you need anything further clarified.",
    "Again, apologies for the inconvenience.",
]

_NEUTRAL_TEXTS = [
    "Please find the attached report for your review. Let me know if you have any questions.",
    "I wanted to update you that the project is on track for delivery next week.",
    "Could you please send over the latest version of the document when you get a chance?",
    "The meeting has been rescheduled to Thursday at 2pm. Please update your calendar.",
    "I've completed the task as requested. The files are in the shared folder.",
    "Can you confirm receipt of the invoice sent earlier this week?",
    "Here is a summary of the key points discussed in today's meeting.",
    "Please review the proposal and share your feedback by Friday.",
    "The system will undergo maintenance this weekend. Please plan accordingly.",
    "I'm reaching out to schedule a time to discuss the upcoming project.",
    "Attached is the updated budget spreadsheet for Q3.",
    "Just a heads up that I'll be out of office next Monday.",
    "Could you provide an update on the status of the open items?",
    "I've added the relevant stakeholders to this email thread.",
    "The new process will be effective from the first of next month.",
    "Please see the agenda attached for tomorrow's meeting.",
    "Let me know if the proposed timeline works for you.",
    "I'll be sending the final report by close of business today.",
    "The access credentials have been updated in the shared password manager.",
    "Please confirm your availability for a brief sync this week.",
    "Here are the action items from our last discussion for your reference.",
    "I've forwarded your inquiry to the appropriate team.",
    "The contract has been reviewed and is ready for signatures.",
    "Could you double-check the numbers in section three?",
    "I've made the edits you requested and reattached the document.",
]

_FRIENDLY_TEXTS = [
    "Hi team! Just wanted to say you've all done an amazing job this quarter. Really proud of what we've accomplished together!",
    "Thanks so much for your help with this — it made a huge difference and I really appreciate it!",
    "Great news! The client loved the presentation and wants to move forward. Congrats to everyone involved!",
    "Just wanted to check in and see how you're doing. Hope everything is going well on your end!",
    "Happy Friday! Looking forward to a great weekend and excited to see everyone refreshed next week.",
    "I really enjoyed our conversation earlier — such great insights. Thanks for sharing!",
    "Welcome to the team! We're so excited to have you here and can't wait to work together.",
    "Congrats on the promotion — you absolutely deserve it! It's been great watching you grow.",
    "Thank you so much for stepping up on such short notice. You're a lifesaver!",
    "Just wanted to let you know how much I appreciate your support throughout this project.",
    "This looks fantastic — great work, everyone! Really impressed by what we put together.",
    "What a wonderful outcome! Let's take a moment to celebrate this win as a team.",
    "So grateful to have such a talented and dedicated group to work with every day.",
    "Can't believe how far we've come this year — it's been such an incredible journey!",
    "Your feedback was so helpful and kind. Thank you for taking the time to share it.",
    "Looking forward to collaborating on the next phase — this team is unstoppable!",
    "Thanks for always being so responsive and helpful. It really makes a difference!",
    "I hope you have a wonderful holiday and come back feeling refreshed and recharged.",
    "Just a little note to say thank you — your hard work never goes unnoticed!",
    "Excited to announce that we hit our target ahead of schedule! Huge win for the whole team.",
]

_ASSERTIVE_TEXTS = [
    "The deadline is non-negotiable. I need this delivered by 5pm on Friday.",
    "This is a mandatory requirement. All team members must comply by end of week.",
    "I need a decision on this today. Please confirm your answer before noon.",
    "This task must be completed before we can proceed. There is no workaround.",
    "I require full access to the system immediately. Please escalate this with IT.",
    "The client has set a hard deadline. We need to meet it, no exceptions.",
    "All submissions must follow the template exactly. Deviations will be rejected.",
    "This is not optional. I expect this to be done by tomorrow morning.",
    "I will need the sign-off from your director before we can proceed.",
    "Please be advised that failure to comply will result in escalation.",
    "All budget requests must be submitted by the 15th. No late submissions will be accepted.",
    "The team is required to attend the training session on Thursday. No exceptions.",
    "I expect a detailed status report every Friday by 4pm.",
    "Access to the production environment requires formal approval. Submit the request form.",
    "Any changes to the specification must go through the change control process.",
    "The policy states that all expense claims must be submitted within 30 days.",
    "I need a written confirmation of the agreed scope before work begins.",
    "The project will not launch without security sign-off. This is a firm requirement.",
    "All staff must complete the compliance training before end of quarter.",
    "Please note that this is a mandatory audit requirement with legal implications.",
]

_AGGRESSIVE_TEXTS = [
    "I am extremely frustrated with the lack of progress on this issue. This is unacceptable.",
    "This is the third time I've had to raise this problem. Why hasn't it been fixed?",
    "I am deeply unhappy with the service I have received and demand an immediate resolution.",
    "This is completely unacceptable. I expect an explanation and a solution by tomorrow.",
    "Your team's performance on this project has been well below expectations.",
    "I am furious that this was not caught before it went to the client.",
    "How many times do I have to ask for this to be done correctly?",
    "This keeps happening and nobody seems to care. I am done being patient.",
    "I want to be very clear: this situation cannot continue. Something needs to change now.",
    "I am appalled by the response time on this critical issue.",
    "This level of carelessness is costing the company money and reputation.",
    "I need this fixed immediately or I will be escalating to senior management.",
    "Your delay on this has caused significant damage and I expect accountability.",
    "This is a disaster. The client is threatening to pull out and I'm holding this team responsible.",
    "I've been waiting three weeks for a simple fix. This is ridiculous.",
    "Stop making excuses and start delivering results.",
    "I'm not interested in reasons why it can't be done. Tell me how it will be done.",
    "Every week we discuss the same problems with zero resolution. Enough.",
    "This team has consistently underperformed and I'm out of patience.",
    "I demand a full incident report and a concrete action plan by end of day.",
]

_TONE_CONFIG = {
    Tone.PASSIVE_AGGRESSIVE: {
        "pa_range": (0.65, 0.95),
        "sarcasm_range": (0.1, 0.5),
    },
    Tone.NEUTRAL: {
        "pa_range": (0.0, 0.1),
        "sarcasm_range": (0.0, 0.05),
    },
    Tone.FRIENDLY: {
        "pa_range": (0.0, 0.05),
        "sarcasm_range": (0.0, 0.05),
    },
    Tone.ASSERTIVE: {
        "pa_range": (0.0, 0.15),
        "sarcasm_range": (0.0, 0.05),
    },
    Tone.AGGRESSIVE: {
        "pa_range": (0.05, 0.2),
        "sarcasm_range": (0.1, 0.3),
    },
}


def _make_pa_text() -> str:
    opener = random.choice(_PA_OPENERS)
    middle = random.choice(_PA_MIDDLES)
    closer = random.choice(_PA_CLOSERS)
    return f"{opener} {middle} {closer}"


def _make_text_for_tone(tone: Tone) -> str:
    if tone == Tone.PASSIVE_AGGRESSIVE:
        return _make_pa_text()
    elif tone == Tone.NEUTRAL:
        return random.choice(_NEUTRAL_TEXTS)
    elif tone == Tone.FRIENDLY:
        return random.choice(_FRIENDLY_TEXTS)
    elif tone == Tone.ASSERTIVE:
        return random.choice(_ASSERTIVE_TEXTS)
    else:  # AGGRESSIVE
        return random.choice(_AGGRESSIVE_TEXTS)


def generate_programmatic(target_count: int, out_dir: Path) -> Path:
    """Generate synthetic samples using deterministic template-based generation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    tones = list(Tone)
    per_tone = target_count // len(tones)
    remainder = target_count % len(tones)

    rows: list[dict] = []
    for i, tone in enumerate(tones):
        count = per_tone + (1 if i < remainder else 0)
        cfg = _TONE_CONFIG[tone]
        pa_lo, pa_hi = cfg["pa_range"]
        sar_lo, sar_hi = cfg["sarcasm_range"]
        for _ in range(count):
            text = _make_text_for_tone(tone)
            pa = round(rng.uniform(pa_lo, pa_hi), 4)
            sarcasm = round(rng.uniform(sar_lo, sar_hi), 4)
            rows.append(
                {
                    "text": text,
                    "passive_aggression": pa,
                    "sarcasm": sarcasm,
                    "tone": tone.value,
                    "source": "synthetic_v1",
                    "weak_label": True,
                }
            )

    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    out = out_dir / "synthetic_v1.parquet"
    df.to_parquet(out)
    logger.info(f"wrote {len(df)} programmatic synthetic samples to {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-count", type=int, default=10000)
    ap.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "gemma3:4b"))
    ap.add_argument(
        "--programmatic",
        action="store_true",
        help="Bypass Ollama and generate samples using hardcoded templates.",
    )
    args = ap.parse_args()
    if args.programmatic:
        generate_programmatic(target_count=args.target_count, out_dir=args.out)
    else:
        SyntheticGenerator(model=args.model, out_dir=args.out).generate(args.target_count)


if __name__ == "__main__":
    main()
