#!/usr/bin/env python3
"""Turn a probe run into a Potato ``static`` source for the t_display calibration.

The point of labelling in Potato rather than a spreadsheet is that the labels and
the judge's confidence end up in the same place: ``selective_risk`` reads
``raw_conf`` from the source and the annotation from the database, so the
calibration reads the study directory directly instead of assembling it from a
TSV afterwards.

A ``static`` source carries exactly one prediction per item, but this study
compares two candidate prompts. So:

  * ``model_pred.confidence`` holds the **candidate** score (the prompt we would
    ship if it wins) — that is what ``selective_risk`` will certificates t_display
    against;
  * the baseline score is kept in ``meta`` so the same labels can score both
    prompts offline, without re-running either judge.

Whatever wins, the study is not wasted: only the ``model_pred`` field needs to
be repointed before the calibration is re-run.

The item text is the sentence alone, *not* the page around it. The judge never
saw the page either, so a human labelling in context would be answering an
easier question than the one being certified.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

TEXT_LIMIT = 900


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def clean(text: str) -> str:
    return " ".join(text.split())[:TEXT_LIMIT]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", default="/data/jev-probe2.jsonl")
    ap.add_argument("--out", default="/data/quotes-calibration.jsonl")
    ap.add_argument("--candidate", default="B", help="variant used for model_pred")
    ap.add_argument("--baseline", default="A0")
    ap.add_argument("--per-bucket", type=int, default=0, help="0 keeps every item")
    ap.add_argument("--bucket-width", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()

    rows = [
        r
        for r in load(Path(args.probe))
        if r.get(args.candidate) is not None and (r.get("text") or "").strip()
    ]

    # One quote can appear on more than one page; keep the first of each text so
    # the labeller is not asked the same question twice.
    seen: set[str] = set()
    unique: list[dict] = []
    for r in rows:
        key = clean(str(r["text"])).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    # Stratify on the candidate's own score: a threshold can only be calibrated
    # where the candidate actually puts mass.
    buckets: dict[float, list[dict]] = collections.defaultdict(list)
    for r in unique:
        buckets[round(float(r[args.candidate]) / args.bucket_width) * args.bucket_width].append(r)

    rng = random.Random(args.seed)
    chosen: list[dict] = []
    for key in sorted(buckets):
        pool = buckets[key][:]
        rng.shuffle(pool)
        chosen.extend(pool if args.per_bucket <= 0 else pool[: args.per_bucket])

    # Order by a hash of the id: stable across runs, but not sorted by score, so
    # the labeller does not walk the distribution in a straight line.
    chosen.sort(key=lambda r: hashlib.sha256(str(r["quote_id"]).encode()).hexdigest())

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as fh:
        for r in chosen:
            conf = float(r[args.candidate])
            item = {
                "id": str(r["quote_id"]),
                "text": clean(str(r["text"])),
                "book": r.get("work_key"),
                "page": r.get("page_no"),
                "deepseek_score": r.get("deepseek_score"),
                # The candidate prompt's probability: what selective_risk reads.
                "model_pred": {"label": "display", "confidence": round(conf, 4)},
                # Kept for a post-hoc comparison of both prompts against the
                # same labels; Potato ignores unknown fields.
                "meta": {
                    "baseline_export": r.get("baseline_export"),
                    "candidate": conf,
                    "baseline_fresh": r.get(args.baseline),
                },
            }
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    marks = {k: len(v) for k, v in sorted(buckets.items())}
    print(f"{len(rows)} scored -> {len(unique)} unique -> {len(chosen)} items")
    print(f"candidate={args.candidate}  buckets={marks}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
