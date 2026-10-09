from app.modules.tool_intent.weighted import WeightedGroup, WeightedTarget, group_relevance, rank_weighted


def _group(group_id: str, ref: str, weight: float, relevance_key: str | None = None) -> WeightedGroup:
    return WeightedGroup(id=group_id, examples=["seed"], targets=[WeightedTarget(ref=ref, weight=weight)])


def test_extra_examples_do_not_raise_relevance():
    assert group_relevance([0.4, 0.8, 0.8, 0.8, 0.8]) == group_relevance([0.8])


def test_low_relevance_cannot_win_on_weight_alone():
    groups = [
        _group("weak", "seo_audit", 10),
        _group("strong", "keywords", 5),
    ]
    status, winner, _ = rank_weighted(
        {"weak": 0.2, "strong": 0.8},
        {"weak": "a", "strong": "b"},
        groups,
    )
    assert status == "confident"
    assert winner == "keywords"


def test_equal_relevance_follows_relative_weight():
    groups = [
        _group("audit", "seo_audit", 10),
        _group("articles", "articles", 5),
    ]
    status, winner, ranked = rank_weighted(
        {"audit": 0.9, "articles": 0.9},
        {"audit": "a", "articles": "b"},
        groups,
    )
    assert status == "confident"
    assert winner == "seo_audit"
    assert ranked[0].score > ranked[1].score


def test_close_weighted_scores_stay_ambiguous():
    groups = [
        _group("left", "keywords", 10),
        _group("right", "articles", 10),
    ]
    status, winner, _ = rank_weighted(
        {"left": 0.80, "right": 0.79},
        {"left": "a", "right": "b"},
        groups,
        margin=0.08,
    )
    assert status == "ambiguous"
    assert winner is None


def test_unrelated_question_is_not_confident():
    groups = [_group("keywords", "keywords", 10)]
    status, winner, _ = rank_weighted({"keywords": 0.1}, {"keywords": "a"}, groups)
    assert status == "none"
    assert winner is None
