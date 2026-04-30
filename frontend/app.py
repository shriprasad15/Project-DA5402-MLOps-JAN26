"""Passive-Aggressive Email Detector — Streamlit frontend."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import requests
import streamlit as st
import yaml

try:
    from frontend.helpers import build_highlight_html, tone_to_color
except ModuleNotFoundError:
    from helpers import build_highlight_html, tone_to_color


def _load_service_config() -> dict:
    """Load port config from monitoring/config.yaml; fall back to defaults."""
    for candidate in [
        Path(__file__).parents[1] / "monitoring" / "config.yaml",
        Path("/opt/app/monitoring/config.yaml"),
    ]:
        if candidate.exists():
            data = yaml.safe_load(candidate.read_text())
            return data.get("services", {})
    return {}


_svc = _load_service_config()

BACKEND_URL    = os.getenv("BACKEND_URL",    f"http://localhost:{_svc.get('backend_port', 8000)}")
MLFLOW_URL     = os.getenv("MLFLOW_URL",     f"http://localhost:{_svc.get('mlflow_port', 5000)}")
AIRFLOW_URL    = os.getenv("AIRFLOW_URL",    f"http://localhost:{_svc.get('airflow_port', 8080)}")
GRAFANA_URL    = os.getenv("GRAFANA_URL",    f"http://localhost:{_svc.get('grafana_port', 3000)}")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", f"http://localhost:{_svc.get('prometheus_port', 9090)}")

st.set_page_config(page_title="PA Email Detector", layout="wide")

# Sidebar: pipeline links (ML Pipeline Visualization rubric point)
with st.sidebar:
    st.header("Pipeline Tools")
    st.markdown(f"[MLflow Tracking]({MLFLOW_URL})")
    st.markdown(f"[Airflow DAGs]({AIRFLOW_URL})")
    st.markdown(f"[Grafana Dashboards]({GRAFANA_URL})")
    st.markdown(f"[Prometheus Metrics]({PROMETHEUS_URL})")
    st.divider()
    st.caption("PA Email Detector v0.1.0")

st.title("Passive-Aggressive Email Detector")
st.caption("Paste your email below to check its tone before sending.")

text_input = st.text_area("Email text", height=200, placeholder="Paste your email here...")
col1, col2 = st.columns([1, 5])
score_btn = col1.button("Analyse", type="primary")

if score_btn:
    if not text_input.strip():
        st.warning("Please paste some text first.")
    else:
        with st.spinner("Analysing..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/predict",
                    json={"text": text_input},
                    headers={"X-Correlation-Id": str(uuid.uuid4())},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                st.session_state["last_prediction"] = data
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Is it running?")
            except Exception as e:
                st.error(f"Error: {e}")

if "last_prediction" in st.session_state:
    data = st.session_state["last_prediction"]
    scores = data.get("scores", {})
    tone = data.get("tone", "unknown")
    tone_confidence = data.get("tone_confidence", 0.0)
    highlights = data.get("highlighted_phrases", [])
    translation = data.get("translation", "")
    prediction_id = data.get("prediction_id", "")

    st.divider()
    st.subheader("Results")

    color = tone_to_color(tone)
    st.markdown(
        f"**Tone:** :{color}[{tone.replace('_', ' ').title()}] ({tone_confidence:.0%} confidence)"
    )

    col_pa, col_sa = st.columns(2)
    col_pa.metric("Passive-Aggression", f"{scores.get('passive_aggression', 0):.0%}")
    col_pa.progress(scores.get("passive_aggression", 0))
    col_sa.metric("Sarcasm", f"{scores.get('sarcasm', 0):.0%}")
    col_sa.progress(scores.get("sarcasm", 0))

    if highlights:
        st.subheader("Highlighted Phrases")
        highlighted_text = build_highlight_html(text_input, highlights)
        st.markdown(highlighted_text, unsafe_allow_html=True)

    if translation:
        st.subheader("Honest Translation")
        st.info(translation)

    st.divider()
    st.subheader("Was this analysis helpful?")
    fb_col1, fb_col2, _ = st.columns([1, 1, 8])
    if fb_col1.button("👍 Yes"):
        try:
            requests.post(
                f"{BACKEND_URL}/feedback",
                json={"prediction_id": prediction_id, "vote": "up"},
                timeout=5,
            )
            st.success("Thanks for the feedback!")
        except Exception:
            st.warning("Could not record feedback.")
    if fb_col2.button("👎 No"):
        try:
            requests.post(
                f"{BACKEND_URL}/feedback",
                json={"prediction_id": prediction_id, "vote": "down"},
                timeout=5,
            )
            st.success("Thanks for the feedback!")
        except Exception:
            st.warning("Could not record feedback.")

with st.expander("How to use"):
    st.markdown(
        """
1. Paste your email into the text box above.
2. Click **Analyse**.
3. Review the Passive-Aggression and Sarcasm scores.
4. Read the highlighted phrases to see what triggered the scores.
5. Check the **Honest Translation** for a direct rewrite.
6. Use 👍/👎 to provide feedback to improve the model.
    """
    )
