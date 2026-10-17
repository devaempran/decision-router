# Evaluation data

## What is used

Consumer complaints from the US Consumer Financial Protection Bureau's public database (US government work, public domain),
taken from the Hugging Face mirror [`23daVinci/CFPB_Complaints`](https://huggingface.co/datasets/23daVinci/CFPB_Complaints) because the
official bulk CSV omits the narrative text. Each complaint is a real consumer's free-text account. The label is the `Product` the
complaint was filed under, which is who would handle it, so the task is a genuine routing decision: *which team should handle this?*

Eight teams (the option text the model reads is in [`queues.py`](queues.py)):

| team | CFPB products mapped to it |
|---|---|
| credit reporting | Credit reporting (all three historical names) |
| debt collection | Debt collection |
| credit and prepaid cards | Credit card, Credit card or prepaid card, Prepaid card |
| bank accounts | Checking or savings account, Bank account or service |
| money transfers | Money transfer, virtual currency, or money service; Money transfers |
| mortgages | Mortgage |
| student loans | Student loan |
| vehicle loans and leases | Vehicle loan or lease |

`prepare.py` samples 40 complaints per team for a **fit** split and 40 different ones for a **test** split (320 each, seed 0),
class-balanced, deduplicated by text, narrative at least 80 characters, truncated to 1,500 characters. Fit tunes the threshold and the
temperature; test is scored once.

## Choices that affect the numbers

- **Class-balanced sampling.** Real traffic is not balanced (credit reporting is over half of all complaints), so accuracy here is not the
  accuracy you would see on a live stream. It measures the model across all eight teams equally.
- **Labels are noisy.** Consumers pick the product themselves and some complaints straddle two (credit reporting versus debt collection,
  bank accounts versus prepaid cards). The errors in the confusion matrix are mostly these pairs, which is the realistic ceiling.
- **These are complaints, not helpdesk tickets.** The language is regulator-facing, long and emotional; redactions appear as `XXXX`. A
  helpdesk would see shorter, more varied text. The harness takes any `tickets.jsonl` with `id`, `split`, `text`, `gold`, so swapping in
  your own labelled tickets is a change to `prepare.py` and `queues.py` only.
- **Not committed.** The data is downloaded to `eval/data/` (git-ignored). The raw model outputs are in `results/raw/`.

## A dataset that was tried and dropped

The first choice was [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (English,
ten helpdesk queues). On a 300-ticket fit split tier 1 scored **23%** against 10% for guessing, with no consistent errors. Reading the
tickets showed why: they are synthetic and the queue barely follows the content (a "Returns and Exchanges" ticket about hospital data
security, a "Human Resources" ticket about marketing campaign metrics). No router could score well on labels that do not follow the text, so
the accuracy would have measured the dataset, not the model. It was dropped before any threshold was fit and before any test ticket was run.
