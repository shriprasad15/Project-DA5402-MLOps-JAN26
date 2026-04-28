from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()

_ready = False


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready():
    if not _ready:
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ready"}


def set_ready(val: bool) -> None:
    global _ready
    _ready = val
