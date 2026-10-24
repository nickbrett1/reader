#!/usr/bin/env python3
"""Score the candidate judges against the human verdicts, then pick t_display.

Reads the labels Potato recorded for one annotator and joins them to the item
metadata written by ``potato_calibration_set.py``, which carries the candidate
prompt's probability (``meta.candidate``), the shipped baseline
(``meta.baseline_export``) and the baseline re-scored on the same items
(``meta.baseline_fresh``).

Two questions, in order:

1. **Is the judge better than chance, and better than the status quo?** Reported
   as AUC with a bootstrap interval, plus the paired difference between prompts
   on the *same* items — the labels are the only truth here, so nothing is
   inferred from distribution shape.

2. **Where should the threshold sit?** A threshold is only defensible if the
   precision it buys is worth the coverage it costs, so the report gives
   precision/recall at every cut rather than a single "best" number.

``unsure`` is excluded from AUC rather than guessed at: forcing a verdict would
invent a label. It is reported separately, because a high unsure rate means the
binary framing is wrong for this taste and a graded scheme is needed.

Usage::

    python3 scripts/calibrate_t_display.py \\
        --labels /srv/docker/potato/project/output/nick/user_state.json \\
        --items  /srv/docker/potato/project/data/quotes-calibration.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

SCORES = {
    "candidate (B)": "candidate",
    "baseline (shipped A0)": "baseline_export",
    "baseline (fresh A0)": "baseline_fresh",
    "deepseek selector": "deepseek_score",
}


def auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUC; ties get average ranks so they contribute 0.5."""
    if not pos or not neg:
        return float("nan")
    combined = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    combined.sort(key=lambda t: t[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r_pos = sum(r for r, (_, lab) in zip(ranks, combined) if lab == 1)
    n_p, n_n = len(pos), len(neg)
    return (r_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n)


def bootstrap_ci(pos: list[float], neg: list[float], n: int = 2000, seed: int = 11):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = [rng.choice(pos) for _ in pos]
        q = [rng.choice(neg) for _ in neg]
        out.append(auc(p, q))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n) - 1]


def load_labels(path: Path) -> dict[str, str]:
    """``{instance_id: label}`` from Potato's per-user state.

    2.11.1 stores ``instance_id_to_label_to_value`` as
    ``{inst: [[{"schema": name, "name": label}, label], ...]}`` — the label is
    repeated inside the key object, so the walk takes the trailing value and
    never the schema name.
    """
    labels: dict[str, str] = {}

    def walk(node, inst: str) -> None:
        if isinstance(node, str) or not isinstance(node, (list, dict)):
            if isinstance(node, str) and node in ("keep", "drop", "unsure"):
                labels[inst] = node
            return
        if isinstance(node, dict):
            for v in node.values():
                walk(v, inst)
            return
        # A pair is [{"schema": name, "name": label}, "<label>"].
        if len(node) == 2 and isinstance(node[0], dict) and isinstance(node[1], str):
            if node[1] in ("keep", "drop", "unsure"):
                labels[inst] = node[1]
            return
        for v in node:
            walk(v, inst)

    state = json.loads(path.read_text(encoding="utf-8"))
    for inst, node in (state.get("instance_id_to_label_to_value") or {}).items():
        walk(node, str(inst))
    return labels


def risk_coverage(rows: list[dict], key: str) -> list[tuple[float, int, int, float]]:
    """(threshold, kept, correct, precision) walking the score downwards."""
    ordered = sorted(rows, key=lambda r: -r[key])
    total = len(ordered)
    out = []
    correct = 0
    for i, row in enumerate(ordered, 1):
        correct += 1 if row["label"] == "keep" else 0
        if i == total or ordered[i][key] != row[key]:
            out.append((row[key], i, correct, correct / i))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default="/srv/docker/potato/project/output/nick/user_state.json")
    ap.add_argument("--items", default="/srv/docker/potato/project/data/quotes-calibration.jsonl")
    ap.add_argument("--target-precision", type=float, default=0.90)
    args = ap.parse_args()

    labels = load_labels(Path(args.labels))
    items = {}
    for line in Path(args.items).read_text(encoding="utf-8").splitlines():
        if line.strip():
            it = json.loads(line)
            meta = it.get("meta") or {}
            items[str(it["id"])] = {
                "text": it.get("text", ""),
                "deepseek_score": it.get("deepseek_score"),
                "candidate": meta.get("candidate"),
                "baseline_export": meta.get("baseline_export"),
                "baseline_fresh": meta.get("baseline_fresh"),
            }

    rows = []
    for inst, label in labels.items():
        it = items.get(inst)
        if it is None:
            continue
        rows.append({"id": inst, "label": label, **it})

    counts = {lab: sum(1 for r in rows if r["label"] == lab) for lab in ("keep", "drop", "unsure")}
    decided = [r for r in rows if r["label"] != "unsure"]
    print(f"labelled {len(rows)} of {len(items)}: "
          + ", ".join(f"{k}={v}" for k, v in counts.items()))
    if len(decided) < 20:
        print("not enough decided labels for a meaningful comparison")
        return 1
    print(f"decided {len(decided)}  (keep={counts['keep']}, drop={counts['drop']})")
    print(f"display-everything precision = {counts['keep'] / len(decided):.2f}\n")

    print(f"{'judge':24s} {'AUC':>6s} {'95% CI':>16s} {'n+':>4s} {'n-':>4s}")
    for name, key in SCORES.items():
        pos = [r[key] for r in decided if r["label"] == "keep" and r[key] is not None]
        neg = [r[key] for r in decided if r["label"] == "drop" and r[key] is not None]
        if not pos or not neg:
            print(f"{name:24s}  insufficient data")
            continue
        a = auc(pos, neg)
        lo, hi = bootstrap_ci(pos, neg)
        print(f"{name:24s} {a:6.3f} [{lo:5.2f},{hi:5.2f}] {len(pos):4d} {len(neg):4d}")

    # Paired comparison: resample items, not scores, so the two judges are always
    # evaluated on the same re-drawn set. Overlapping marginal intervals say
    # nothing about which judge is better; this does.
    def paired(name_a: str, name_b: str) -> None:
        ka, kb = SCORES[name_a], SCORES[name_b]
        pairs = [(r[ka], r[kb], r["label"]) for r in decided
                 if r[ka] is not None and r[kb] is not None]
        if len(pairs) < 20:
            return
        rng = random.Random(23)
        diffs = []
        for _ in range(2000):
            draw = [rng.choice(pairs) for _ in pairs]
            pa = [x for x, _, l in draw if l == "keep"]
            na = [x for x, _, l in draw if l == "drop"]
            pb = [y for _, y, l in draw if l == "keep"]
            nb = [y for _, y, l in draw if l == "drop"]
            if pa and na and pb and nb:
                diffs.append(auc(pa, na) - auc(pb, nb))
        if not diffs:
            return
        diffs.sort()
        better = sum(1 for d in diffs if d <= 0) / len(diffs)
        print(f"\npaired {name_a} - {name_b}: "
              f"{statistics.fmean(diffs):+.3f} "
              f"[{diffs[int(0.025 * len(diffs))]:+.3f},"
              f"{diffs[int(0.975 * len(diffs)) - 1]:+.3f}]  "
              f"P(a is not better) = {better:.2f}")

    names = list(SCORES)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            paired(names[i], names[j])

    # Thresholds for the score that does best, at a precision the study can defend.
    best = max(
        (n for n, k in SCORES.items()
         if any(r[k] is not None for r in decided)),
        key=lambda n: auc(
            [r[SCORES[n]] for r in decided if r["label"] == "keep" and r[SCORES[n]] is not None],
            [r[SCORES[n]] for r in decided if r["label"] == "drop" and r[SCORES[n]] is not None],
        ),
    )
    key = SCORES[best]
    print(f"\nrisk/coverage for {best} (precision target {args.target_precision:.0%}):")
    print(f"{'t_display':>9s} {'coverage':>9s} {'precision':>10s} {'kept':>5s} {'correct':>8s}")
    curve = risk_coverage([r for r in decided if r[key] is not None], key)
    pick = None
    for t, kept, correct, prec in curve:
        cov = kept / len(decided)
        if prec >= args.target_precision and (pick is None or cov > pick[1]):
            pick = (t, cov, prec)
    for t, kept, correct, prec in curve:
        cov = kept / len(decided)
        if abs(cov - 0.10) < 0.06 or abs(cov - 0.25) < 0.06 or abs(cov - 0.50) < 0.06 \
                or abs(cov - 0.75) < 0.06 or abs(cov - 0.90) < 0.06:
            print(f"{t:9.3f} {cov:9.0%} {prec:10.2f} {kept:5d} {correct:8d}")

    if pick:
        t, cov, prec = pick
        print(f"\nstrongest defensible cut: t_display = {t:.3f} "
              f"-> keeps {cov:.0%} of items at {prec:.0%} precision")
        print(f"corpus-wide (all 6,521 shipped quotes): apply the same cut to the "
              f"candidate score before trusting the coverage.")
    else:
        print(f"\nno cut reaches {args.target_precision:.0%} precision; "
              f"the judge does not support a threshold yet")

    # Where the judge and the human actually disagree at the chosen cut.
    if pick:
        t = pick[0]
        wrong = [r for r in decided if r[key] is not None and r[key] >= t and r["label"] == "drop"]
        missed = [r for r in decided if r[key] is not None and r[key] < t and r["label"] == "keep"]
        print(f"\nfalse accepts at that cut ({len(wrong)}):")
        for r in wrong[:5]:
            print(f"  {r[key]:.2f}  {r['text'][:110]}")
        print(f"\nmissed keeps ({len(missed)}):")
        for r in missed[:5]:
            print(f"  {r[key]:.2f}  {r['text'][:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
