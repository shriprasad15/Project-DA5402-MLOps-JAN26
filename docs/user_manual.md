# User Manual — Passive-Aggressive Email Detector (DA5402 Wave 3)

## What This Tool Does

The Passive-Aggressive Email Detector analyses the text of an email and tells you,
in plain numbers, how passive-aggressive or sarcastic it sounds — and what overall
tone it carries. You paste your email in, click a button, and within a second or two
you see a percentage score for passive aggression, a percentage score for sarcasm, a
tone label, and a highlighted version of your text showing exactly which phrases
triggered the scores. The tool also provides an "Honest Translation" — a plain,
direct rewrite of your message with the implicit subtext made explicit, so you can
decide whether that is truly what you meant to send.

---

## Quick Start

1. Open a web browser and go to **http://localhost:8501**
2. Paste the full body of your email into the large text box labelled **Email text**
3. Optionally check that the text looks correct — you can also type directly in the box
4. Click the blue **Analyse** button
5. Wait one or two seconds while the system processes your email
6. Read your results in the panel that appears below the button

That is all. No account, no login, no settings to configure.

---

## Understanding the Scores

### Passive-Aggression Score

A number from **0% to 100%**.

- **0%** means the message is communicating directly and assertively with no
  detectable indirect hostility.
- **100%** means the message is saturated with passive-aggressive language —
  phrasing that expresses displeasure or makes demands indirectly while maintaining
  a surface appearance of politeness.

As a rule of thumb, a score above roughly **30%** is worth pausing to consider
before you send.

Common passive-aggressive patterns the tool recognises include: "as per my last
email", "going forward", "just to clarify", "not sure if you saw my message",
"I could be wrong but...", "no worries" used ironically, and similar constructions.

### Sarcasm Score

A number from **0% to 100%**.

- **0%** means the message reads as entirely sincere — what it says is what it means.
- **100%** means the message is heavily sarcastic — the stated meaning is the
  opposite of the intended meaning, or the phrasing is ironic throughout.

### Tone

One of five labels:

| Tone | What it means |
|---|---|
| **Neutral** | Matter-of-fact; no strong emotional charge |
| **Friendly** | Warm, positive, encouraging language |
| **Assertive** | Direct and clear, but not hostile |
| **Aggressive** | Overtly hostile or demanding |
| **Passive-Aggressive** | Indirect hostility dressed up as politeness |

A confidence percentage appears next to the tone label. Lower confidence (below 70%)
means the email sits between two tones — both readings are plausible.

---

## Highlighted Phrases

Below your scores you will see your original email text with certain words or phrases
highlighted in shades of red. The colour intensity indicates how strongly that phrase
contributed to the passive-aggression score:

- **Light pink (#ffc8c8)** — mild contribution; the phrase is slightly charged
- **Medium red** — moderate contribution; this phrase is a notable trigger
- **Bright red (#ff0000)** — high contribution; this is one of the key phrases
  responsible for the elevated score

If no phrases are highlighted, the model found no single span that stood out above
the detection threshold. The overall tone reading may come from sentence structure
or tone across the whole message rather than individual keywords.

---

## Honest Translation

The **Honest Translation** block shows a direct rewrite of your message that strips
softening language and states the actual meaning plainly.

**Example:**

> *Your email:* "As per my last email, just to clarify, I had hoped this would have
> been addressed by now."
>
> *Honest Translation:* "I'm frustrated that this wasn't addressed after I already
> pointed it out."

Reading the honest translation helps you decide:

- Is this really what you want to communicate?
- Would it be clearer — and less likely to cause offence — to say it more directly?
- Or is the indirect phrasing actually appropriate for this context and relationship?

---

## Feedback Buttons

At the bottom of the results panel you will see two buttons:

- **Thumbs up (Yes)** — click this if the analysis matched your own reading of the email
- **Thumbs down (No)** — click this if the analysis felt wrong or misleading

Your vote is stored alongside a fingerprint of the analysis. Over time, these votes
identify where the model performs poorly and inform future retraining runs. No text
of your email is stored permanently — only a non-reversible numerical hash is kept.

---

## Pipeline Tools (Sidebar)

On the left-hand sidebar you will see four links. These are for technical users and
instructors who want to inspect the underlying machine-learning pipeline:

| Link | What it shows |
|---|---|
| **MLflow Tracking** (`:5000`) | A log of every model training run — parameters used, scores achieved, and which model version is currently active in production |
| **Airflow DAGs** (`:8080`) | The automated pipeline that re-trains the model on a daily schedule — click a task to view its logs and whether it succeeded or failed |
| **Grafana Dashboards** (`:3000`) | Live charts updated every 10 seconds showing request rates, response latency, model inference time, and whether the emails being analysed are drifting from the training distribution |
| **Prometheus Metrics** (`:9090`) | Raw numerical measurements from the backend, plus the status of active alert rules |

---

## Frequently Asked Questions

**Q: Does the tool read or store my emails?**
No email text is stored permanently. The system processes your text in memory,
stores only a non-reversible SHA-256 hash (fingerprint) alongside the scores, and
discards the original text. Your content cannot be reconstructed from what is stored.

**Q: How accurate is the passive-aggression score?**
The model was trained on a mix of real and synthetic examples of passive-aggressive
and sarcastic language. It performs well on common office-email patterns. It may
score unusual phrasings, technical jargon, or informal slang less reliably. Use the
score as a prompt for self-reflection, not as a definitive verdict.

**Q: Can I analyse non-English emails?**
The underlying model (`distilbert-base-uncased`) was trained exclusively on English
text. Results for other languages will be unreliable.

**Q: The score seems wrong. What should I do?**
Click the thumbs-down button to record your disagreement. That vote is used to
identify weak spots in the model for future retraining. You can also simply
disregard the score — it is a tool to prompt reflection, not a rule.

**Q: Why does the same email sometimes give slightly different scores?**
In the current development configuration the sarcasm score includes a small amount
of randomness to simulate real model variation. When a fully trained model is
deployed from the MLflow registry, scores for identical inputs will be deterministic.

---

## Troubleshooting

**"Cannot connect to backend. Is it running?"**

The Streamlit frontend cannot reach the FastAPI backend. Steps to fix:

1. Open a terminal and check the status of all services:
   ```
   docker compose ps
   ```
2. Any service showing `Exit` or `unhealthy` needs to be restarted:
   ```
   docker compose up -d --wait
   ```
3. If services are running but the error persists, check the backend logs:
   ```
   docker compose logs backend --tail=50
   ```

**The Analyse button does nothing**

Make sure there is some non-whitespace text in the email box before clicking. A
warning message appears if the box is empty or contains only spaces.

**Scores are always 0% / Tone is always Neutral**

The system is running with `MockModelClient` (the default) and the email text does
not contain any of the common passive-aggressive phrases the mock detects. Try
pasting a phrase like "as per my last email, just to clarify..." to confirm the
system is responding.

**The page is very slow to load**

The Streamlit container may still be starting up. Wait 15–20 seconds and refresh
the browser. If slowness persists, check `docker compose ps` to ensure all 9
services show `healthy`.

**I see "Error: 503" or "Service Unavailable" after clicking Analyse**

The backend is running but the readiness probe has not yet passed. Wait 30 seconds
and try again. On first start the model server can take up to a minute to load
the model from the MLflow registry.

**The Feedback button shows a warning after re-loading the page**

The `prediction_id` from the previous session is no longer in browser memory after
a page reload. Analyse the email again first, then use the feedback buttons on the
fresh result.
