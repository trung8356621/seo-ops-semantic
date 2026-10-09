from __future__ import annotations

from app.api.app import create_app
from app.config import Settings


def test_app_registers_cta_plan_route() -> None:
    app = create_app(Settings())
    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/v1/cta/plan" in paths
    assert any(getattr(route, "path", "") == "/" for route in app.routes)
