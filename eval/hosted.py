"""The same tickets through a prompted hosted-model call, for comparison with the local two-tier setup.

    export ANTHROPIC_API_KEY=...
    python eval/hosted.py --model claude-haiku-5-5  --split test
    python eval/hosted.py --model claude-sonnet-5-5 --split test

One request per ticket: the complaint and the same eight team descriptions as lettered options, answered as JSON
{answer: letter, confidence: 0-1}. No tools, no examples, thinking off and low effort, which is what a plain
"classify this" call looks like. Results are appended to results/raw/hosted_<model>_<split>.jsonl and the run resumes
from that file, so a re-run never pays twice. --max-calls is a hard cap on API calls for the run.

Prices are USD per million tokens as listed on the run date (see PRICES); compare.py turns the recorded token counts
into dollars."""
import argparse
import json
import os
import statistics
import sys
import time

from collect import HERE, OPTIONS, load_tickets
from queues import QUESTION

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(OPTIONS)]
SYSTEM = ("You route customer complaints to the team that should handle them. Answer with the letter of the best team and "
          "your probability, between 0 and 1, that the letter is correct.")
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": list(LETTERS)}, "confidence": {"type": "number"}},
    "required": ["answer", "confidence"],
    "additionalProperties": False,
}
# USD per million tokens (input, output), first-party API list prices as of 2026-10-06; Haiku 5.5 prices apply to prompts
# up to 100K tokens, far above these tickets.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}
PRICE_DATE = "2026-10-06"
# How to switch thinking off per model (a plain classification call): Haiku 5.5 accepts `disabled` at effort high or
# below; Sonnet 5.5 rejects `disabled` and offers `between_tools` instead.
THINKING_OFF = {"claude-haiku-5-5": {"type": "disabled"}, "claude-sonnet-5-5": {"type": "between_tools"}}


def prompt(text):
    options = "\n".join(f"{letter}. {option}" for letter, option in zip(LETTERS, OPTIONS, strict=True))
    return f"{QUESTION}\n\nTeams:\n{options}\n\nComplaint:\n{text}"


def parse(reply_text):
    """(option text, stated confidence) from the model's JSON reply; raises ValueError when it does not fit the schema."""
    data = json.loads(reply_text)
    letter = data["answer"]
    if letter not in LETTERS:
        raise ValueError(f"answer {letter!r} is not one of {LETTERS}")
    return OPTIONS[LETTERS.index(letter)], min(1.0, max(0.0, float(data["confidence"])))


def ask(client, model, text, thinking):
    params = {"model": model, "max_tokens": 200, "system": SYSTEM,
              "messages": [{"role": "user", "content": prompt(text)}],
              "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}}
    if thinking == "off" and model in THINKING_OFF:
        params["thinking"] = THINKING_OFF[model]
    t0 = time.time()
    reply = client.messages.create(**params)
    wall = time.time() - t0
    if reply.stop_reason == "refusal":
        raise ValueError(f"refused: {getattr(reply.stop_details, 'category', None)}")
    answer, confidence = parse(next(b.text for b in reply.content if b.type == "text"))
    return {"answer": answer, "confidence": confidence, "wall_s": wall, "input_tokens": reply.usage.input_tokens,
            "output_tokens": reply.usage.output_tokens, "stop_reason": reply.stop_reason, "request_id": reply._request_id}


def main(client=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(PRICES))
    ap.add_argument("--split", choices=["fit", "test"], default="test")
    ap.add_argument("--max-calls", type=int, default=400, help="hard cap on API calls in this run")
    ap.add_argument("--thinking", choices=["off", "default"], default="off")
    args = ap.parse_args()
    if client is None:
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("set ANTHROPIC_API_KEY (this script bills your account, up to --max-calls requests)")
        client = anthropic.Anthropic()
    out_path = os.path.join(HERE, "results", "raw", f"hosted_{args.model}_{args.split}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [t for t in load_tickets(args.split) if t["id"] not in done][: args.max_calls]
    print(f"{args.model}: {len(done)} done, {len(todo)} to run (cap {args.max_calls})", flush=True)
    failures = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, t in enumerate(todo, 1):
            try:
                r = ask(client, args.model, t["text"], args.thinking)
            except ValueError as e:  # a reply that does not fit the schema, or a refusal: record it as a miss, keep going
                r, failures = {"answer": None, "error": str(e)}, failures + 1
            f.write(json.dumps({"id": t["id"], "gold": t["gold"], "model": args.model, **r}) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print(f"done; {failures} unusable replies")
    lat = [json.loads(line).get("wall_s") for line in open(out_path, encoding="utf-8")]
    lat = [x for x in lat if x]
    if lat:
        print(f"median latency {statistics.median(lat) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
