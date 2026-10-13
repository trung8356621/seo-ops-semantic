"""Project-scoped semantic cleanup.

Only rows explicitly stamped for a content project and stored under the
temporary ``content-project:`` namespace are removable. Published article
vectors and site topic / keyword-group indexes stay.
"""

from __future__ import annotations

from typing import Mapping


PROTECTED_NAMESPACE_PREFIXES = ("topic:", "keyword-group:")
PROTECTED_NAMESPACES = frozenset({"published_articles"})
TEMPORARY_NAMESPACE_PREFIX = "content-project:"


def is_project_scoped_temporary(namespace: str, metadata: Mapping[str, object], project_id: str) -> bool:
    stamped = str(metadata.get("content_project_id") or "")
    if stamped == "" or stamped != str(project_id):
        return False
    if namespace in PROTECTED_NAMESPACES:
        return False
    if namespace.startswith(PROTECTED_NAMESPACE_PREFIXES):
        return False
    return namespace.startswith(TEMPORARY_NAMESPACE_PREFIX)


def project_workspace_delete_sql() -> str:
    return """
        DELETE FROM semantic_embeddings
        WHERE metadata->>'content_project_id' = %s
          AND namespace LIKE 'content-project:%%'
          AND namespace <> 'published_articles'
          AND namespace NOT LIKE 'topic:%%'
          AND namespace NOT LIKE 'keyword-group:%%'
    """
