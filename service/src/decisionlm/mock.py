"""Mock backend: the same /api/decide contract with no model, so the pipeline and dashboard can be tried without a GPU.

Answers come from a keyword heuristic and are NOT real model output. Each option is matched to a topic by the words
in its description (e.g. "billing or payment problem"), then scored by how many of the topic's keywords the text contains.
Text that matches nothing (or only one keyword) gets low confidence, so it falls under theta and takes the (slow)
reasoner path, which keeps the escalation behaviour visible. Deterministic: the same text always gets the same answer."""
from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Sequence

from decisionlm import settings, tier1

# topic -> (words that select the topic when they appear in an option description, keywords looked for in the text)
TOPICS: dict[str, tuple[set[str], set[str]]] = {
    "billing": ({"billing", "payment", "invoice", "refund", "charge"},
                set("charge charged refund invoice bill billed billing payment paid pay card receipt subscription renewal "
                    "coupon tax price overcharged debit paypal cancelled prorated purchase statement".split())),
    "technical": ({"technical", "fault", "bug", "error", "outage", "how-to"},
                  set("crash crashes error bug api app fails failed timeout slow broken install update sync export import upload "
                      "integration webhook sso dashboard blank notification delayed delay mode settings saml okta page text "
                      "search schedule report csv desktop mobile android windows connect how".split())),
    "sales": ({"sales", "pricing", "quote", "enquiry", "purchase"},
              set("quote pricing enterprise discount demo plan upgrade seat volume buy trial contract licence license reseller "
                  "procurement startup annual startups nonprofit lock".split())),
    "account": ({"account", "access", "security", "login", "password"},
                set("password login log locked hacked 2fa two-factor authentication verification reset suspicious username "
                    "access unauthorized unauthorised sign security breach delete gdpr personal data permissions".split())),
    "feedback": ({"feedback", "complaint", "suggestion", "general"},
                 set("love terrible disappointed suggest suggestion feature awful thanks thank great fantastic unhappy complaint "
                     "rude frustrated improve wish confusing fewer newsletter waiting nobody reply team".split())),
}
WORD = re.compile(r"[a-z0-9']+")


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _words(text: str) -> list[str]:
    return [_stem(w) for w in WORD.findall(text.lower())]


KEYWORDS = {name: {_stem(w) for w in words} for name, (_, words) in TOPICS.items()}


def _jitter(*parts: str) -> float:
    """A stable pseudo-random number in [0, 1) derived from the inputs."""
    h = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32


def _topic_of(option: str) -> str | None:
    words = set(WORD.findall(option.lower()))
    best, hits = None, 0
    for name, (selectors, _) in TOPICS.items():
        n = len(words & selectors)
        if n > hits:
            best, hits = name, n
    return best


def probabilities(context: str, options: Sequence[str]) -> list[float]:
    words = _words(context)
    scores = []
    for opt in options:
        topic = _topic_of(opt)
        keywords = KEYWORDS[topic] if topic else set(_words(opt))
        scores.append(float(sum(1 for w in words if w in keywords)) + 0.4 * _jitter(context, opt))
    top = max(scores)
    exps = [math.exp(3.5 * (s - top)) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def decide(context: str, question: str, options: Sequence[str], theta: float | None = None,
           max_think_tokens: int | None = None, max_think_seconds: float | None = None) -> dict:
    """Same response shape as decide.decide."""
    tier1.validate_options(options)
    cfg = settings.get()
    theta = cfg.theta if theta is None else theta
    t0 = time.time()
    time.sleep(cfg.mock_fast_ms / 1000)
    probs = probabilities(context, options)
    top = max(range(len(options)), key=probs.__getitem__)
    out = {"answer": options[top], "source": "fast", "confidence": probs[top],
           "probabilities": dict(zip(options, probs, strict=True)),
           "tier1_answer": options[top], "low_confidence": False, "mock": True}
    if probs[top] < theta:
        time.sleep(cfg.mock_reasoner_ms / 1000)
        # a mock "reasoner": usually keeps tier 1's pick, sometimes takes the runner-up
        ranked = sorted(range(len(options)), key=probs.__getitem__, reverse=True)
        pick = ranked[1] if _jitter(context, "reasoner") < 0.25 else ranked[0]
        out.update(answer=options[pick], source="reasoner", forced=False, timed_out=False,
                   reasoner_tokens=int(200 + 600 * _jitter(context, "tokens")),
                   reasoner_latency_s=cfg.mock_reasoner_ms / 1000)
    out["latency_s"] = time.time() - t0
    return out
