import pytest

from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert data["num_users"] > 0


def test_recommendations_valid_user(client):
    health = client.get("/api/health").get_json()
    res = client.get("/api/recommendations/user_1?n=5")
    if res.status_code == 404:
        pytest.skip("user_1 not in dataset")
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["recommendations"]) == 5


def test_recommendations_invalid_user_id(client):
    res = client.get("/api/recommendations/bad%20id?n=5")
    assert res.status_code == 400


def test_recommendations_invalid_n(client):
    res = client.get("/api/recommendations/user_1?n=99")
    assert res.status_code == 400


def test_feedback_validation(client):
    res = client.post("/api/feedback", json={"user_id": "user_1", "signal": "up"})
    assert res.status_code == 400


def test_feedback_record(client):
    res = client.post(
        "/api/feedback",
        json={"user_id": "user_1", "video_id": "movie_1", "signal": "up"},
    )
    assert res.status_code in (200, 404)
    if res.status_code == 200:
        assert res.get_json()["status"] == "recorded"
