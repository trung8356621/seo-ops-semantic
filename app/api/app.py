from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from app import __version__
from app.api.routes.concept_matches import router as concept_matches_router
from app.api.routes.health import router as health_router
from app.api.routes.internal_links_v2 import router as internal_links_v2_router
from app.api.routes.keyword_groups import router as keyword_groups_router
from app.api.routes.tool_intents import router as tool_intents_router
from app.api.routes.topic import router as topic_router
from app.api.routes.topic_groups import router as topic_groups_router
from app.api.routes.wiki_suggestions import router as wiki_suggestions_router
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
    app.include_router(keyword_groups_router)
    app.include_router(concept_matches_router)
    app.include_router(tool_intents_router)
    app.include_router(topic_groups_router)
    app.include_router(internal_links_v2_router)
    app.include_router(wiki_suggestions_router)

    @app.get("/")
    def root() -> dict:
        return {
            "service": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
            "topic_analyses": "/v1/topic/analyses",
            "keyword_group_analyses": "/v1/keyword-groups/analyses",
            "concept_match_analyses": "/v1/concept-matches/analyses",
            "tool_intent_match": "/v1/tool-intents/match",
            "topic_group_matches": "/v1/topic-groups/matches",
            "internal_link_v2_rank": "/v1/internal-links/v2/rank",
            "wiki_suggestions": "/v1/wiki-suggestions",
        }

    return app
