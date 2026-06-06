from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.api.deps import get_query_log
from app.config import get_settings
from app.stores.query_log import QueryLog

router = APIRouter()

_STATIC = Path(__file__).parent.parent / "static"


def _results_path() -> str:
    return get_settings().eval_results_path


@router.get("/api/stats")
def stats(query_log: QueryLog = Depends(get_query_log)) -> dict:
    path = Path(_results_path())
    eval_results = json.loads(path.read_text()) if path.exists() else None
    return {"queries": query_log.recent(limit=20), "eval": eval_results}


@router.get("/dashboard")
def dashboard() -> FileResponse:
    return FileResponse(str(_STATIC / "dashboard.html"))
