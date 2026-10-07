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
