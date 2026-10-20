#!/usr/bin/env python3
"""Compare candidate Jev prompts on a stratified sample of the quote corpus.

The question ``DISPLAY_CLAIM`` in :mod:`reader.jev` is vague enough that Jev
answers "probably yes" to most of a real corpus, so the score barely separates
items. This script is the experiment that decides whether a more specific
framing actually helps — and, just as importantly, whether it *reorders* the
corpus rather than merely rescaling it.

Variants (all judged on the same items, same model, same batch size):

  A0  the current single claim, no ``state``            (the status quo)
  A1  the current single claim, page text as ``state``  (does context alone help?)
  B   one conjunctive claim: insight OR wonder OR connection
  C   three separate claims, combined with a noisy-OR   (which dimension fires?)
  D   C multiplied by a self-contained gate             (fragments suppressed)

Every variant is scored with the same ``noul`` question type, so the numbers are
probabilities and directly comparable.

Usage (inside the reader container, which has the gateway route and the data)::

    python3 /data/jev_prompt_probe.py \
        --quotes /data/quotes.jsonl --out /data/jev-probe.jsonl --per-bucket 15

The sample is written out alongside the results so a human can label exactly the
items that were scored, and the choice between variants can be settled by
agreement with that labelling rather than by eyeballing a histogram.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from typing import Callable

from reader import jev

# --- candidate framings ----------------------------------------------------

#: B: one claim that names the three things a quote has to do. Conjunctive in
#: intent ("or" over criteria that are each hard), so it should score lower and
#: spread further than the vague original.
CONJUNCTIVE = (
    "Read without the surrounding page, does this sentence give the reader an "
    "insight, a moment of wonder, or a connection between two ideas that they "
    "would not have got from the artwork alone? Sentence: \u201c{text}\u201d"
)

INSIGHT = (
    "Does this sentence tell the reader something they would not otherwise know "
    "about how this game was designed, built, or intended to be experienced?"
)
WONDER = (
    "Does this sentence evoke a sense of wonder, scale, or curiosity about the "
    "world of the work or the craft behind it?"
)
CONNECTION = (
    "Does this sentence connect two otherwise separate ideas, disciplines, or "
    "pieces of the book's subject matter?"
)
SELF_CONTAINED = (
    "Does this sentence make sense on its own, without the sentences around it? "
    "A fragment or a line of dialogue with no context does not."
)

DIMENSIONS = (("insight", INSIGHT), ("wonder", WONDER), ("connection", CONNECTION))

#: Page text goes in ``state``; more than this is mostly token cost.
STATE_CHARS = 2000


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _state(item: dict[str, object]) -> str:
    ctx = item.get("context")
    if not isinstance(ctx, str) or not ctx.strip():
        return ""
    ctx = _normalise(ctx)
    if len(ctx) > STATE_CHARS:
        ctx = ctx[:STATE_CHARS] + " \u2026"
    return f"Page text:\n{ctx}"


def _noisy_or(values: list[float]) -> float:
    """P(at least one criterion is met), assuming independence."""
    keep = 1.0
    for v in values:
        keep *= 1.0 - max(0.0, min(1.0, v))
    return 1.0 - keep


def questions_for(variant: str, item: dict[str, object], idx: int) -> tuple[dict, Callable]:
    """Return the questions for one item, and how to fold their answers."""
    text = str(item["text"])

    def single(claim: str):
        def fold(ans: dict[str, jev.Answer]) -> float | None:
            a = ans.get(f"i{idx}")
            return a.value if a is not None and a.ok else None

        return {f"i{idx}": {"type": "noul", "instructions": claim.format(text=text)}}, fold

    if variant == "A0":
        return single(jev.DISPLAY_CLAIM)
    if variant == "A1":
        return single(jev.DISPLAY_CLAIM)
    if variant == "B":
        return single(CONJUNCTIVE)

    qs: dict[str, dict[str, object]] = {}
    names: list[str] = []
    for name, claim in DIMENSIONS:
        qid = f"i{idx}{name[0]}"
        qs[qid] = {"type": "noul", "instructions": claim}
        names.append(qid)
    gate_qid = f"i{idx}s"
    if variant in ("D", "D2"):
        qs[gate_qid] = {"type": "noul", "instructions": SELF_CONTAINED}

    def fold(ans: dict[str, jev.Answer]) -> float | None:
        vals = [ans[q].value if ans.get(q) is not None and ans[q].ok else None for q in names]
        if variant in ("D", "D2"):
            g = ans.get(gate_qid)
            gate = g.value if g is not None and g.ok else None
            if gate is None:
                return None
        else:
            gate = 1.0
        if any(v is None for v in vals):
            return None
        nums = [float(v) for v in vals]  # type: ignore[arg-type]
        # A noisy-OR assumes the criteria are rare and independent; three common
        # ones saturate near 1.0. Averaging keeps the dimension signal instead.
        combine = _noisy_or if variant == "C" else statistics.fmean
        return combine(nums) * float(gate)

    return qs, fold


VARIANTS = ("A0", "A1", "B", "C", "C2", "D", "D2")


# --- sampling --------------------------------------------------------------


def stratified(items: list[dict], per_bucket: int, width: float, seed: int) -> list[dict]:
    """Take ``per_bucket`` items from each bucket of the existing score."""
    rng = random.Random(seed)
    buckets: dict[int, list[dict]] = defaultdict(list)
    for it in items:
        val = (it.get("jev") or {}).get("display")
        if val is None:
            continue
        buckets[int(float(val) / width)].append(it)
    out: list[dict] = []
    for key in sorted(buckets):
        pool = buckets[key]
        rng.shuffle(pool)
        out.extend(pool[:per_bucket])
    return out


# --- statistics ------------------------------------------------------------


def _ranks(values: list[float]) -> list[float]:
    """Average ranks, so tied scores do not bias the correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation; computed by hand to avoid a scipy dependency."""
    if len(a) != len(b) or len(a) < 3:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def describe(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    n = len(ordered)

    def q(p: float) -> float:
        pos = p * (n - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    band = sum(1 for v in values if 0.30 <= v <= 0.50) / n
    return {
        "n": n,
        "mean": statistics.fmean(values),
        "sd": statistics.pstdev(values),
        "q1": q(0.25),
        "median": q(0.50),
        "q3": q(0.75),
        "iqr": q(0.75) - q(0.25),
        "mid_band": band,
        "ge_070": sum(1 for v in values if v >= 0.70) / n,
        "le_010": sum(1 for v in values if v <= 0.10) / n,
    }


# --- main ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quotes", default="/data/quotes.jsonl")
    ap.add_argument("--out", default="/data/jev-probe.jsonl")
    ap.add_argument("--per-bucket", type=int, default=15)
    ap.add_argument("--bucket-width", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--model", default=jev.DEFAULT_MODEL)
    ap.add_argument("--base-url", default=None, help="defaults to config")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--show", type=int, default=6, help="disagreement rows to print")
    args = ap.parse_args(argv)

    with open(args.quotes, encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh]
    sample = stratified(items, args.per_bucket, args.bucket_width, args.seed)
    print(f"{len(items)} item(s) -> {len(sample)} sampled", file=sys.stderr)

    if args.base_url:
        base, key = args.base_url, None
    else:
        from reader import config as config_mod

        cfg = config_mod.Config.from_env()
        base, key = cfg.litellm_base_url, cfg.litellm_master_key
    call = jev.gateway_ask(base, key, args.model)

    wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
    scores: dict[str, list[float | None]] = {v: [None] * len(sample) for v in wanted}

    for variant in wanted:
        state_mode = "none" if variant == "A0" else "page"
        for start in range(0, len(sample), args.batch):
            chunk = list(enumerate(sample[start : start + args.batch], start=start))
            qs: dict[str, dict[str, object]] = {}
            folders = []
            for idx, item in chunk:
                q, fold = questions_for(variant, item, idx)
                qs.update(q)
                folders.append((idx, fold))
            state = "" if state_mode == "none" else _state(chunk[0][1])
            answers = jev.ask(base, key, qs, state=state, model=args.model, timeout=120.0)
            for idx, fold in folders:
                scores[variant][idx] = fold(answers)
        done = sum(1 for v in scores[variant] if v is not None)
        print(f"  {variant}: {done}/{len(sample)} scored", file=sys.stderr)

    # --- report ------------------------------------------------------------
    rows = []
    for i, item in enumerate(sample):
        row = {
            "id": item.get("id"),
            "quote_id": item.get("quote_id"),
            "text": item.get("text"),
            "work_key": item.get("work_key"),
            "page_no": item.get("page_no"),
            "deepseek_score": item.get("score"),
            "baseline_export": (item.get("jev") or {}).get("display"),
        }
        for variant in wanted:
            row[variant] = scores[variant][i]
        rows.append(row)

    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{'variant':8s} {'n':>4s} {'mean':>6s} {'sd':>6s} {'q1':>5s} {'med':>5s} "
          f"{'q3':>5s} {'iqr':>5s} {'mid':>6s} {'>=.70':>6s} {'<=.10':>6s} "
          f"{'rho_A0':>7s} {'rho_A1':>7s}")
    complete = [[v for v in scores[v] if v is not None] for v in wanted]
    for variant, vals in zip(wanted, complete):
        if not vals:
            print(f"{variant:8s}  no data")
            continue
        d = describe(vals)
        paired = [(scores[variant][i], scores["A0"][i]) for i in range(len(sample))
                  if scores[variant][i] is not None and scores["A0"][i] is not None]
        r0 = spearman([a for a, _ in paired], [b for _, b in paired])
        if "A1" in scores:
            p1 = [(scores[variant][i], scores["A1"][i]) for i in range(len(sample))
                  if scores[variant][i] is not None and scores["A1"][i] is not None]
            r1 = spearman([a for a, _ in p1], [b for _, b in p1])
        else:
            r1 = float("nan")
        print(f"{variant:8s} {d['n']:4d} {d['mean']:6.2f} {d['sd']:6.2f} {d['q1']:5.2f} "
              f"{d['median']:5.2f} {d['q3']:5.2f} {d['iqr']:5.2f} {d['mid_band']:6.0%} "
              f"{d['ge_070']:6.0%} {d['le_010']:6.0%} {r0:7.2f} {r1:7.2f}")

    # Control: does a fresh A0 reproduce the export it is meant to match?
    stored = [(scores["A0"][i], sample[i]["jev"]["display"]) for i in range(len(sample))
              if scores["A0"][i] is not None and (sample[i].get("jev") or {}).get("display") is not None]
    if stored:
        print(f"\nrepeatability: fresh A0 vs stored export rho = "
              f"{spearman([a for a, _ in stored], [b for _, b in stored]):.3f} "
              f"(n={len(stored)})")

    # Where the most specific variant most disagrees with the status quo.
    last = wanted[-1]
    moves = []
    for i, row in enumerate(rows):
        a, z = row.get("A0"), row.get(last)
        if a is not None and z is not None:
            moves.append((z - a, i))
    moves.sort(reverse=True)
    if args.show:
        print(f"\nlifted by {last} vs A0:")
        for delta, i in moves[: args.show]:
            r = rows[i]
            print(f"  {delta:+.2f}  A0={r['A0']:.2f} {last}={r[last]:.2f}  "
                  f"{(r['text'] or '')[:100]}")
        print(f"\ndemoted by {last} vs A0:")
        for delta, i in moves[-args.show:]:
            r = rows[i]
            print(f"  {delta:+.2f}  A0={r['A0']:.2f} {last}={r[last]:.2f}  "
                  f"{(r['text'] or '')[:100]}")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
