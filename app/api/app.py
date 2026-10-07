from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from app import __version__
from app.api.routes.health import router as health_router
from app.api.routes.topic import router as topic_router
from app.config import Settings, get_settings
from app.core.embedding.contracts import EmbeddingProvider
from app.core.embedding.factory import create_embedding_provider
from app.storage.database import Database


@dataclass
class AppState:
    settings: Settings
    database: Database
    embedding_provider: EmbeddingProvider


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    database = Database(settings)
    provider = create_embedding_provider(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        database.wait_until_ready()
        database.migrate()
        if not settings.embedding_lazy_load:
            provider.load()
        app.state.container = AppState(
            settings=settings,
            database=database,
            embedding_provider=provider,
        )
        yield

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
    )
    # Convenience aliases used by route handlers.
    app.state.settings = settings
    app.state.database = database
    app.state.embedding_provider = provider

    app.include_router(health_router)
    app.include_router(topic_router)

    @app.get("/")
    def root() -> dict:
        return {
            "service": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
            "topic_analyses": "/v1/topic/analyses",
        }

    return app
