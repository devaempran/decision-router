"""Download public consumer complaints and write eval/data/tickets.jsonl: a fit split and a test split, disjoint, with the
same number of tickets per team. The threshold and temperature are fit on `fit`; `test` is scored once.

    python eval/prepare.py [--per-class 40] [--seed 0]

Data: the CFPB Consumer Complaint Database (US government, public domain), complaints with a consumer narrative.
The narrative is the ticket text; the `product` it was filed under, mapped to a team (queues.py), is the routing label.
Downloaded here and not committed. See DATASET.md."""
import argparse
import json
import os
import random
import urllib.request

import pandas as pd

from queues import OPTIONS_BY_PRODUCT

# A Hugging Face mirror of the CFPB database that carries the narrative text (the official bulk CSV omits it).
URL = "https://huggingface.co/datasets/23daVinci/CFPB_Complaints/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MAX_CHARS = 1500
MIN_CHARS = 80


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=40, help="tickets per team in each split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "cfpb_test.parquet")
    if not os.path.exists(path):
        print("downloading", URL, flush=True)
        urllib.request.urlretrieve(URL, path)
    df = pd.read_parquet(path)
    df["text"] = df["Consumer complaint narrative"].fillna("").str.strip()
    df = df[(df["text"].str.len() >= MIN_CHARS) & df["Product"].isin(OPTIONS_BY_PRODUCT)].drop_duplicates("text")
    rng = random.Random(args.seed)
    need = 2 * args.per_class
    pool = {}
    for option in dict.fromkeys(OPTIONS_BY_PRODUCT.values()):
        rows = df[df["Product"].map(OPTIONS_BY_PRODUCT) == option]
        if len(rows) < need:
            raise SystemExit(f"{option}: only {len(rows)} complaints")
        picked = rows.sample(n=need, random_state=args.seed)
        pool[option] = [{"complaint_id": str(i), "text": t[:MAX_CHARS], "gold": option, "queue": p}
                        for i, t, p in zip(picked.index, picked["text"], picked["Product"], strict=True)]
        print(f"  {len(rows):6d} complaints  {option.split(':')[0]}")
    out = []
    for rows in pool.values():
        for split, chunk in (("fit", rows[: args.per_class]), ("test", rows[args.per_class:])):
            out += [{"id": f"{split}-{r['complaint_id']}", "split": split, **r} for r in chunk]
    rng.shuffle(out)
    path = os.path.join(DATA, "tickets.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in out)
    print(f"wrote {len(out)} tickets ({args.per_class} per team per split) to {path}")


if __name__ == "__main__":
    main()
