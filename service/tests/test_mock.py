"""The mock backend: same /api/decide contract as the real one, no model."""
import dataclasses

import pytest

from decisionlm import server, settings

OPTIONS = ["billing or payment problem", "technical fault or how-to question", "sales or pricing enquiry",
           "account access or security issue", "general feedback or complaint"]
RESPONSE_KEYS = {"answer", "source", "confidence", "probabilities", "tier1_answer", "low_confidence", "latency_s"}


@pytest.fixture
def mock_client():
    settings.use(dataclasses.replace(settings.get(), mock=True, mock_fast_ms=0, mock_reasoner_ms=0))
    return server.create_app().test_client()


def decide(client, text, **extra):
    return client.post("/api/decide", json={"Context": text, "Question": "Which team?", "Options": OPTIONS, **extra})


def test_same_response_shape_as_the_real_backend(mock_client):
    body = decide(mock_client, "I was charged twice, please refund me").get_json()
    assert RESPONSE_KEYS <= body.keys()
    assert body["answer"] == OPTIONS[0] and body["source"] == "fast" and body["mock"] is True
    assert sum(body["probabilities"].values()) == pytest.approx(1)


def test_is_deterministic(mock_client):
    text = "the app crashes with an error on export"
    a, b = decide(mock_client, text).get_json(), decide(mock_client, text).get_json()
    assert {k: v for k, v in a.items() if k != "latency_s"} == {k: v for k, v in b.items() if k != "latency_s"}


def test_unclear_text_escalates_and_theta_is_honoured(mock_client):
    unclear = decide(mock_client, "hello there").get_json()
    assert unclear["source"] == "reasoner" and unclear["reasoner_tokens"] > 0
    never = decide(mock_client, "hello there", theta=0).get_json()
    assert never["source"] == "fast"
    always = decide(mock_client, "I was charged twice, refund", theta=2).get_json()
    assert always["source"] == "reasoner"


def test_validates_options_like_the_real_backend(mock_client):
    r = mock_client.post("/api/decide", json={"Context": "x", "Question": "q", "Options": ["only one"]})
    assert r.status_code == 400


def test_health_says_it_is_a_mock(mock_client):
    assert mock_client.get("/api/health").get_json()["mock"] is True


def test_environment_variable_turns_it_on():
    assert settings.load(environ={"DECISIONLM_MOCK": "1"}).mock is True
    assert settings.load(environ={"DECISIONLM_MOCK": "0"}).mock is False
    assert settings.load(environ={}).mock is False
