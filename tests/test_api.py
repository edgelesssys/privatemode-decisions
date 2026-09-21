"""The demo API with the engine stubbed out; no model needed."""

import pytest
from fastapi.testclient import TestClient

from app import main
from decisions.types import ChoiceAnswer, SystemOneResponse, Usage


class FakeEngine:
    def system_one(self, state, questions, images=None):
        answers = {key: ChoiceAnswer(choice=next(iter(q.criteria)),
                                     probabilities={n: 1 / len(q.criteria) for n in q.criteria},
                                     confidence=0.0)
                   for key, q in questions.items()}
        return SystemOneResponse(model="fake", answers=answers,
                                 usage=Usage(input_tokens=5, output_tokens=len(answers)))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "engine", lambda model: FakeEngine())
    return TestClient(main.app)


def test_ask_returns_a_distribution_per_question(client):
    response = client.post("/api/ask", json={"text": "Route it\nchoices: repair, sales"})
    assert response.status_code == 200
    question = response.json()["questions"][0]
    assert question["choice"] == "repair"
    assert question["probabilities"] == {"repair": 0.5, "sales": 0.5}


@pytest.mark.parametrize("path", ["/api/ask", "/api/parse"])
@pytest.mark.parametrize("body", ["[1, 2]", '"text"', "not json"])
def test_a_body_that_is_not_an_object_is_a_400(client, path, body):
    response = client.post(path, content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 400


def test_limits(client):
    assert client.post("/api/ask", json={"text": "x" * (main.MAX_TEXT + 1)}).status_code == 400
    images = ["data:image/png;base64,AAAA"] * (main.MAX_IMAGES + 1)
    assert client.post("/api/ask", json={"text": "Is it?", "images": images}).status_code == 400


def test_an_unknown_model_is_a_400(client):
    assert client.post("/api/ask", json={"text": "Is it?", "model": "nope"}).status_code == 400


@pytest.mark.parametrize("path", ["/api/ask", "/api/parse"])
def test_a_body_not_sent_as_json_is_a_415(client, path):
    response = client.post(path, content='{"text": "Is it?"}', headers={"Content-Type": "text/plain"})
    assert response.status_code == 415


def test_an_upstream_error_is_not_passed_to_the_browser(monkeypatch):
    class Failing:
        def system_one(self, *args, **kwargs):
            raise ConnectionError("http://internal-proxy:8080 refused")
    monkeypatch.setattr(main, "engine", lambda model: Failing())
    response = TestClient(main.app).post("/api/ask", json={"text": "Is it?"})
    assert response.status_code == 502
    assert "internal-proxy" not in response.text
