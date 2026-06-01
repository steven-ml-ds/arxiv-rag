from fastapi import FastAPI

from app.api import chat, health, search

app = FastAPI(title="arxiv-rag", version="0.1.0")
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(search.router)
