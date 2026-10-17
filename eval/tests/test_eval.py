"""The eval harness's own logic, without a model or network: hosted-reply parsing and cost, calibration maths, cascade."""
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import hosted
import report
from queues import TEAMS


def test_prompt_lists_every_team_with_a_letter():
    p = hosted.prompt("my card was charged twice")
    for letter, option in zip(hosted.LETTERS, TEAMS, strict=True):
        assert f"{letter}. {option}" in p
    assert "my card was charged twice" in p


def test_parse_maps_the_letter_back_to_the_option_and_clamps_confidence():
    assert hosted.parse('{"answer": "B", "confidence": 1.7}') == (list(TEAMS)[1], 1.0)
    with pytest.raises(ValueError):
        hosted.parse('{"answer": "Z", "confidence": 0.5}')
    with pytest.raises(KeyError):
        hosted.parse('{"answer": "A"}')


class FakeClient:
    def __init__(self, text, stop_reason="end_turn"):
        self.calls = []
        reply = types.SimpleNamespace(stop_reason=stop_reason, stop_details=None, _request_id="req_1",
                                      content=[types.SimpleNamespace(type="text", text=text)],
                                      usage=types.SimpleNamespace(input_tokens=500, output_tokens=12))
        self.messages = types.SimpleNamespace(create=lambda **kw: (self.calls.append(kw), reply)[1])


def test_ask_sends_a_plain_request_and_records_tokens():
    c = FakeClient('{"answer": "A", "confidence": 0.9}')
    r = hosted.ask(c, "claude-haiku-5-5", "text", "off")
    assert r["answer"] == list(TEAMS)[0] and r["input_tokens"] == 500 and r["output_tokens"] == 12
    sent = c.calls[0]
    assert sent["thinking"] == {"type": "disabled"} and sent["output_config"]["effort"] == "low"
    assert "temperature" not in sent and "tools" not in sent


def test_sonnet_turns_thinking_off_with_between_tools():
    c = FakeClient('{"answer": "A", "confidence": 1}')
    hosted.ask(c, "claude-sonnet-5-5", "t", "off")
    assert c.calls[0]["thinking"] == {"type": "between_tools"}


def test_a_refusal_is_an_error_not_an_answer():
    with pytest.raises(ValueError, match="refused"):
        hosted.ask(FakeClient("{}", stop_reason="refusal"), "claude-haiku-5-5", "t", "off")


def test_rescale_with_k_one_is_the_identity_and_below_one_softens():
    p = np.array([[0.7, 0.2, 0.1]])
    assert report.rescale(p, 1.0) == pytest.approx(p)
    soft = report.rescale(p, 0.5)
    assert soft.max() < 0.7 and soft.sum() == pytest.approx(1)


def test_ece_is_zero_when_confidence_matches_accuracy():
    conf = np.array([0.5] * 10)
    ok = np.array([1.0] * 5 + [0.0] * 5)
    assert report.ece(conf, ok)[0] == pytest.approx(0)
    assert report.ece(np.array([0.9] * 10), ok)[0] == pytest.approx(0.4)


def test_cascade_takes_the_reasoner_answer_below_theta_and_refuses_when_it_did_not_run():
    d = {"conf": np.array([0.5, 0.95]), "pred1": np.array([0, 1]), "pred2": np.array([1, -1]), "gold": np.array([1, 1]),
         "lat1": np.array([0.1, 0.1]), "lat2": np.array([10.0, np.nan])}
    r = report.cascade(d, 0.85)
    assert r["accuracy"] == 1.0 and r["escalated"] == 0.5 and r["mean_latency_s"] == pytest.approx(5.1)
    assert report.cascade(d, 0.99) is None  # the second ticket would need a reasoner answer that was never collected


def test_fit_scale_prefers_underconfidence():
    rng = np.random.default_rng(1)
    n, k = 400, 4
    gold = rng.integers(0, k, n)
    probs = np.full((n, k), 0.02)
    correct = rng.random(n) < 0.6
    pred = np.where(correct, gold, (gold + 1) % k)
    probs[np.arange(n), pred] = 0.94  # confident 94% but only 60% right
    d = {"probs": probs, "gold": gold, "pred1": probs.argmax(axis=1)}
    scale = report.fit_scale(d)
    accuracy = float((d["pred1"] == gold).mean())
    assert scale < 1 and report.rescale(probs, scale).max(axis=1).mean() <= accuracy + 1e-9
