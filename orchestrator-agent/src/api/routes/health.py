"""Health check routes."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/ping")
def ping():
    """Health check endpoint."""
    return {"status": "ok"}
