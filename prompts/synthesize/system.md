You are generating training data for a passive-aggressive email detector.

Each sample must be a plausible workplace email sentence or short paragraph.
Score `passive_aggression` and `sarcasm` independently on [0, 1]. Pick one
`tone` from: neutral, friendly, assertive, aggressive, passive_aggressive.

- High passive_aggression + passive_aggressive tone: indirect frustration
  ("as we discussed", "per my last email", "friendly reminder").
- High sarcasm: ironic praise ("great job breaking the build again").
- Aggressive tone: direct hostility.
- Friendly tone: warm, genuine, collaborative.
- Neutral: purely informational, no emotional charge.

Output: a JSON array only. No prose before or after.
