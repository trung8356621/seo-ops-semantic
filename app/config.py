from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_env: str = Field(default="local", alias="APP_ENV")
    app_name: str = Field(default="seo-ops-semantic", alias="APP_NAME")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8088, alias="APP_PORT")
    uvicorn_workers: int = Field(default=1, alias="UVICORN_WORKERS")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="semantic", alias="POSTGRES_DB")
    postgres_user: str = Field(default="semantic", alias="POSTGRES_USER")
    postgres_password: str = Field(default="semantic_local_change_me", alias="POSTGRES_PASSWORD")

    embedding_provider: str = Field(default="onnx_fastembed", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="EMBEDDING_MODEL",
    )
    model_cache_dir: str = Field(default=".models", alias="MODEL_CACHE_DIR")

    db_connect_retries: int = Field(default=30, alias="DB_CONNECT_RETRIES")
    db_connect_retry_seconds: float = Field(default=2.0, alias="DB_CONNECT_RETRY_SECONDS")
    embedding_lazy_load: bool = Field(default=True, alias="EMBEDDING_LAZY_LOAD")

    # Topic clustering algorithm: average_linkage (default) | greedy_medoid_v2 | greedy_medoid_v1
    topic_cluster_algorithm: str = Field(
        default="average_linkage",
        alias="TOPIC_CLUSTER_ALGORITHM",
    )
    # Shared / algorithm-specific thresholds — see README for exact semantics.
    # average_linkage: similarity cut (distance = 1 - similarity).
    # greedy_medoid_v2: seed density threshold only.
    topic_cluster_similarity_threshold: float = Field(
        default=0.74,
        alias="TOPIC_CLUSTER_SIMILARITY_THRESHOLD",
    )
    # greedy_medoid_v2: final member admission vs seed (independent of density).
    # Unused by average_linkage (assignment_min_score is the post-guard).
    topic_min_member_similarity: float = Field(
        default=0.77,
        alias="TOPIC_MIN_MEMBER_SIMILARITY",
    )
    topic_min_group_size: int = Field(default=2, alias="TOPIC_MIN_GROUP_SIZE")
    # Post-cluster assignment guard + confidence floor (similarity_to_representative).
    topic_assignment_min_score: float = Field(
        default=0.70,
        alias="TOPIC_ASSIGNMENT_MIN_SCORE",
    )
    topic_low_confidence_score: float = Field(
        default=0.35,
        alias="TOPIC_LOW_CONFIDENCE_SCORE",
        description="Heuristic confidence below this is flagged low-confidence (not probability).",
    )
    topic_max_keywords: int = Field(default=2000, alias="TOPIC_MAX_KEYWORDS")
    topic_max_text_length: int = Field(default=500, alias="TOPIC_MAX_TEXT_LENGTH")
    topic_embedding_cache_enabled: bool = Field(
        default=True,
        alias="TOPIC_EMBEDDING_CACHE_ENABLED",
    )

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_dsn(self) -> str:
        return (
            f"host={self.postgres_host} "
            f"port={self.postgres_port} "
            f"dbname={self.postgres_db} "
            f"user={self.postgres_user} "
            f"password={self.postgres_password}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
