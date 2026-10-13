from app.storage.project_workspace import is_project_scoped_temporary, project_workspace_delete_sql


def test_project_temporary_embedding_is_removable():
    assert is_project_scoped_temporary(
        "content-project:15",
        {"content_project_id": "15"},
        "15",
    )


def test_published_article_and_site_indexes_are_kept():
    meta = {"content_project_id": "15"}
    assert not is_project_scoped_temporary("published_articles", meta, "15")
    assert not is_project_scoped_temporary("topic:site-9", meta, "15")
    assert not is_project_scoped_temporary("keyword-group:site-9", meta, "15")
    assert not is_project_scoped_temporary("content-project:15", {"content_project_id": "99"}, "15")
    assert not is_project_scoped_temporary("content-project:15", {}, "15")


def test_delete_sql_names_only_temporary_namespace():
    sql = project_workspace_delete_sql()
    assert "semantic_embeddings" in sql
    assert "content-project:" in sql
    assert "published_articles" in sql
    assert "topic:" in sql
    assert "keyword-group:" in sql
