from __future__ import annotations

from app.config import Settings
from app.core.embedding.contracts import EmbeddingProvider
from app.core.embedding.providers.onnx_fastembed import OnnxFastEmbedProvider


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    key = settings.embedding_provider.strip().lower()
    if key in {"onnx_fastembed", "fastembed", "onnx"}:
        return OnnxFastEmbedProvider(
            model_name=settings.embedding_model,
            cache_dir=settings.model_cache_dir,
            lazy_load=settings.embedding_lazy_load,
            memory_cache_max_items=settings.embedding_memory_cache_max_items,
        )
    raise ValueError(f"unsupported embedding provider: {settings.embedding_provider}")
