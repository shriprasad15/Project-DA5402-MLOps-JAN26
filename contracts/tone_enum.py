"""Canonical tone enum — imported by backend schemas and training code."""
from enum import Enum


class Tone(str, Enum):
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    ASSERTIVE = "assertive"
    AGGRESSIVE = "aggressive"
    PASSIVE_AGGRESSIVE = "passive_aggressive"
