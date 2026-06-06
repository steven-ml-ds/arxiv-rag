from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import chat, dashboard, health, search

_STATIC = Path(__file__).parent / "static"

app = FastAPI(title="arxiv-rag", version="0.1.0")
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(search.router)
app.include_router(dashboard.router)
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_STATIC / "index.html"))
