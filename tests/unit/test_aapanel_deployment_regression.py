from pathlib import Path
import yaml


def test_aapanel_compose_declares_semantic_worker():
    compose_path = Path(r"D:\work\build\seo-ops-semantic-aapanel\docker-compose.aapanel.yml")
    assert compose_path.exists(), f"Missing file: {compose_path}"

    content = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    services = content.get("services", {})

    assert "semantic-worker" in services, "semantic-worker service is missing from docker-compose.aapanel.yml"

    worker = services["semantic-worker"]
    assert worker.get("command") == ["python", "-m", "app.modules.keyword_grouping.worker"], (
        "semantic-worker must execute 'python -m app.modules.keyword_grouping.worker' continuously"
    )

    # Postgres dependency
    depends_on = worker.get("depends_on", {})
    assert "postgres" in depends_on, "semantic-worker must depend on postgres"
    assert depends_on["postgres"].get("condition") == "service_healthy"

    # Persistence & Volumes
    volumes = worker.get("volumes", [])
    assert "model_cache:/models" in volumes, "semantic-worker must mount model_cache volume"

    # Environment
    env = worker.get("environment", {})
    assert "POSTGRES_HOST" in env
    assert "EMBEDDING_PROVIDER" in env
    assert env.get("EMBEDDING_LAZY_LOAD") == "true"
    assert "KEYWORD_GROUP_ALGORITHM" in env


def test_aapanel_compose_volumes_preserve_data():
    compose_path = Path(r"D:\work\build\seo-ops-semantic-aapanel\docker-compose.aapanel.yml")
    content = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    volumes = content.get("volumes", {})
    assert "postgres_data" in volumes, "postgres_data named volume must be preserved"
    assert "model_cache" in volumes, "model_cache named volume must be preserved"


def test_aapanel_default_build_context():
    compose_path = Path(r"D:\work\build\seo-ops-semantic-aapanel\docker-compose.aapanel.yml")
    content = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    services = content.get("services", {})

    api_context = services["semantic-api"]["build"]["context"]
    assert api_context == "${SEMANTIC_BUILD_CONTEXT:-/www/seo-ops-semantic}", (
        f"semantic-api has incorrect build context default: {api_context}"
    )

    worker_context = services["semantic-worker"]["build"]["context"]
    assert worker_context == "${SEMANTIC_BUILD_CONTEXT:-/www/seo-ops-semantic}", (
        f"semantic-worker has incorrect build context default: {worker_context}"
    )

    env_path = Path(r"D:\work\build\seo-ops-semantic-aapanel\.env.aapanel.example")
    env_content = env_path.read_text(encoding="utf-8")
    assert "SEMANTIC_BUILD_CONTEXT=/www/seo-ops-semantic" in env_content, (
        ".env.aapanel.example must specify SEMANTIC_BUILD_CONTEXT=/www/seo-ops-semantic"
    )
