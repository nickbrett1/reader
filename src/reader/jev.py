"""Jev — TypeSafe's "System One" decision service, via the LiteLLM proxy.

Jev is **not a chat model**. It answers typed questions over a supplied state
and returns numbers: ``noul`` gives P(claim is true), ``choice`` gives the
chosen key plus a distribution, ``score`` gives the expected level over ordered
criteria. That is what a calibration pass needs — a per-item probability, not a
verbalised opinion.

The proxy reaches it through a ``pass_through_endpoints`` route
(``POST /typesafe/v1/system-one``) that injects the TypeSafe key itself, so this
module authenticates with the gateway key and never holds a TypeSafe key.
The upstream path is ``systemone``; ``system-one`` 404s at TypeSafe.

Only used by ``reader export-quotes --jev``. The stage itself never calls it:
selection is DeepSeek's job, and the judge is deliberately independent of the
selector.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass

SYSTEM_ONE_PATH = "/typesafe/v1/system-one"
DEFAULT_MODEL = "jev-latest"

#: The claim whose truth value *is* ``t_display``'s input. Phrased as the
#: decision the threshold governs, so the certified cut reads directly as
#: "display a quote when P(this) clears the cut".
DISPLAY_CLAIM = (
    "Would this sentence earn its place as a displayed pull-quote in a printed "
    "video-game art book? Sentence: \u201c{text}\u201d"
)


class JevError(RuntimeError):
    """The gateway did not return a usable System One response."""


@dataclass(frozen=True)
class Answer:
    """One system-one answer, normalised across the three question kinds."""

    value: float | None
    failed: bool
    raw: dict[str, object]

    @property
    def ok(self) -> bool:
        return not self.failed and self.value is not None


def ask(
    base_url: str,
    api_key: str | None,
    questions: dict[str, dict[str, object]],
    *,
    state: str = "",
    model: str = DEFAULT_MODEL,
    timeout: float = 30.0,
) -> dict[str, Answer]:
    """Ask every question in one round trip; raise on a transport failure.

    A per-question failure is *not* an exception: it comes back ``failed`` with
    ``value=None``, matching Jev's own fail-open contract. Callers must check
    ``ok`` rather than assume a number.
    """
    body = json.dumps(
        {"model": model, "state": state, "questions": questions}
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        base_url.rstrip("/") + SYSTEM_ONE_PATH, data=body, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError) as exc:
        raise JevError(f"system-one call failed: {exc}") from exc
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, dict):
        raise JevError("system-one response had no answers object")
    out: dict[str, Answer] = {}
    for qid in questions:
        raw = answers.get(qid)
        if not isinstance(raw, dict):
            out[qid] = Answer(None, True, {})
            continue
        value = raw.get("noul", raw.get("score"))
        if value is None:
            out[qid] = Answer(None, True, raw)
            continue
        try:
            out[qid] = Answer(float(value), False, raw)
        except (TypeError, ValueError):
            out[qid] = Answer(None, True, raw)
    return out


def display_confidence(
    texts: Sequence[str],
    *,
    call: Callable[[dict[str, dict[str, object]]], dict[str, Answer]],
    batch: int = 25,
) -> list[float | None]:
    """P(each sentence is worth displaying), aligned to ``texts``.

    Batches many quotes per request (Jev answers them in parallel) and returns
    ``None`` where the service had no opinion, so a failure stays visible
    instead of silently becoming 0.0.
    """
    out: list[float | None] = [None] * len(texts)
    size = max(1, batch)
    for start in range(0, len(texts), size):
        chunk = texts[start : start + size]
        questions = {
            f"q{i}": {"type": "noul", "instructions": DISPLAY_CLAIM.format(text=t)}
            for i, t in enumerate(chunk)
        }
        answers = call(questions)
        for i in range(len(chunk)):
            answer = answers.get(f"q{i}")
            if answer is not None and answer.ok:
                out[start + i] = answer.value
    return out


def gateway_ask(base_url: str, api_key: str | None, model: str = DEFAULT_MODEL):
    """Bind the gateway client into the ``call`` shape ``display_confidence`` wants."""

    def _call(questions: dict[str, dict[str, object]]) -> dict[str, Answer]:
        return ask(base_url, api_key, questions, model=model)

    return _call
