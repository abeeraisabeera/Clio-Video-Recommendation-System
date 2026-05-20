import pytest

from model import (
    build_model,
    load_interactions,
    recommend_for_user,
    record_feedback,
)


@pytest.fixture(scope="module")
def clio():
    return build_model()


def test_load_interactions_has_unique_user_items():
    df = load_interactions()
    dupes = df.duplicated(subset=["user_id", "video_id"]).sum()
    assert dupes == 0


def test_build_model_shapes(clio):
    assert clio.num_users > 0
    assert clio.num_videos > 0
    assert clio.num_users <= clio.num_videos * 2


def test_recommend_known_user(clio):
    user_id = clio.user_encoder.classes_[0]
    result = recommend_for_user(clio, str(user_id), n=5)
    assert result["user_id"] == str(user_id)
    assert len(result["recommendations"]) == 5
    rec = result["recommendations"][0]
    assert "video_id" in rec and "title" in rec and "score" in rec


def test_recommend_unknown_user(clio):
    with pytest.raises(LookupError):
        recommend_for_user(clio, "user_does_not_exist_99999", n=5)


def test_feedback_boosts_item_rank(clio):
    user_id = str(clio.user_encoder.classes_[0])
    before = recommend_for_user(clio, user_id, n=20)
    boosted_id = before["recommendations"][-1]["video_id"]
    rank_before = len(before["recommendations"]) - 1

    record_feedback(clio, user_id, boosted_id, "up")
    after = recommend_for_user(clio, user_id, n=20)
    rank_after = [r["video_id"] for r in after["recommendations"]].index(boosted_id)

    assert rank_after <= rank_before


def test_feedback_down_removes_item(clio):
    user_id = str(clio.user_encoder.classes_[1])
    before = recommend_for_user(clio, user_id, n=10)
    assert before["recommendations"]
    target = before["recommendations"][0]["video_id"]

    record_feedback(clio, user_id, target, "down")
    after = recommend_for_user(clio, user_id, n=10)
    after_ids = [r["video_id"] for r in after["recommendations"]]

    assert target not in after_ids
