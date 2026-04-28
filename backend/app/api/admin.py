from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request
from loguru import logger

router = APIRouter(prefix="/admin")


async def _check_token(request: Request, x_admin_token: str | None = Header(None)):
    from backend.app.config import get_settings

    settings = get_settings()
    if x_admin_token != settings.BACKEND_ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="forbidden")


@router.post("/alert", status_code=200)
async def receive_alert(request: Request):
    from backend.app.services.notifier import notify_prometheus_alert

    body = await request.json()
    logger.warning("Alert received from Alertmanager: {}", body)
    for alert in body.get("alerts", []):
        name = alert.get("labels", {}).get("alertname", "unknown")
        severity = alert.get("labels", {}).get("severity", "unknown")
        description = alert.get("annotations", {}).get("description", "")
        notify_prometheus_alert(name, severity, description)
    return {"status": "received"}
