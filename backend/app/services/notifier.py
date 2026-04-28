from __future__ import annotations

import smtplib
from email.mime.text import MIMEText

from loguru import logger

from backend.app.config import get_settings


def _send(subject: str, body_html: str) -> None:
    settings = get_settings()
    if not settings.mailtrap_user or not settings.mailtrap_pass:
        logger.debug("Mailtrap not configured — skipping email notification")
        return
    msg = MIMEText(body_html, "html")
    msg["Subject"] = subject
    msg["From"] = settings.MAILTRAP_FROM
    msg["To"] = settings.mailtrap_to
    try:
        with smtplib.SMTP(settings.MAILTRAP_HOST, settings.MAILTRAP_PORT) as s:
            s.starttls()
            s.login(settings.mailtrap_user, settings.mailtrap_pass)
            s.sendmail(settings.MAILTRAP_FROM, [settings.mailtrap_to], msg.as_string())
        logger.info(f"Email sent: {subject}")
    except Exception as e:
        logger.warning(f"Email notification failed: {e}")


def notify_model_run(run_name: str, f1: float, pa_mae: float, registered: bool) -> None:
    status = "✅ REGISTERED as champion" if registered else "⏭ Not registered (did not beat previous best)"
    color = "#2e7d32" if registered else "#e65100"
    _send(
        subject=f"[PA Detector] Training run: {run_name} — {'REGISTERED' if registered else 'not registered'}",
        body_html=f"""
        <h2 style="font-family:sans-serif;">PA Detector — Training Run Complete</h2>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse;font-family:sans-serif;">
          <tr><td><b>Run name</b></td><td>{run_name}</td></tr>
          <tr><td><b>val_macro_f1</b></td><td>{f1:.4f}</td></tr>
          <tr><td><b>pa_mae</b></td><td>{pa_mae:.4f}</td></tr>
          <tr><td><b>Status</b></td><td style="color:{color};font-weight:bold;">{status}</td></tr>
        </table>
        <p><a href="http://localhost:5000">Open MLflow</a></p>
        <p><small>PA Detector Monitoring — DA5402 MLOps</small></p>
        """,
    )


def notify_report_generated(tone: str, pa_score: float, sarcasm_score: float, text_preview: str) -> None:
    color = "#c62828" if pa_score > 0.6 else "#e65100" if pa_score > 0.3 else "#2e7d32"
    _send(
        subject=f"[PA Detector] High PA score detected — tone: {tone}",
        body_html=f"""
        <h2 style="font-family:sans-serif;">PA Detector — Report Generated</h2>
        <p>A passive-aggressive email was analysed with a high score.</p>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse;font-family:sans-serif;">
          <tr><td><b>Tone</b></td><td style="color:{color};font-weight:bold;">{tone.replace('_', ' ').title()}</td></tr>
          <tr><td><b>Passive-aggression</b></td><td>{pa_score:.0%}</td></tr>
          <tr><td><b>Sarcasm</b></td><td>{sarcasm_score:.0%}</td></tr>
          <tr><td><b>Text preview</b></td><td><i>{text_preview[:150]}…</i></td></tr>
        </table>
        <p><a href="http://localhost:8501">Open Frontend</a></p>
        <p><small>PA Detector Monitoring — DA5402 MLOps</small></p>
        """,
    )


def notify_prometheus_alert(alert_name: str, severity: str, description: str) -> None:
    color = "#c62828" if severity == "critical" else "#e65100"
    _send(
        subject=f"[PA Detector {'🚨 CRITICAL' if severity == 'critical' else '⚠️ WARNING'}] {alert_name}",
        body_html=f"""
        <h2 style="font-family:sans-serif;color:{color};">Prometheus Alert: {alert_name}</h2>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse;font-family:sans-serif;">
          <tr><td><b>Alert</b></td><td>{alert_name}</td></tr>
          <tr><td><b>Severity</b></td><td style="color:{color};font-weight:bold;">{severity.upper()}</td></tr>
          <tr><td><b>Description</b></td><td>{description}</td></tr>
        </table>
        <p>
          <a href="http://localhost:9090/alerts">Prometheus Alerts</a> &nbsp;|&nbsp;
          <a href="http://localhost:3000">Grafana Dashboard</a>
        </p>
        <p><small>PA Detector Monitoring — DA5402 MLOps</small></p>
        """,
    )
