#!/usr/bin/env python3
"""llm_guardrails.py — make an LLM call safe to put in a production pipeline.

An LLM in an automation is not an API. An API fails loudly; an LLM fails
*plausibly* — it returns a paragraph of apology where you expected JSON, or a
confident answer to a question it could not see, or nothing at all, and your
workflow writes it to the database and reports success.

These are the five guards that turn a model call into something you can put on
a cron at 3am. Each one exists because it caught a real failure in production.

    1. REFUSAL DETECTION      the model declined; that text is not your output
    2. TRANSPORT vs CONTENT   a 502 and a bad answer need different retries
    3. SHAPE VALIDATION       "it parsed" is not "it is usable"
    4. FALLBACK LADDER        degrade in steps; never go dark
    5. HOLD-THE-PRIOR         a failed refresh must not destroy yesterday's good data

Zero dependencies. Works with any provider — you pass a callable.

    from llm_guardrails import guarded_call, Schema

    out = guarded_call(
        lambda p: my_openai_call(p),
        prompt="Summarise this ticket as JSON: ...",
        schema=Schema(required=["summary", "priority"], types={"priority": str}),
        fallbacks=[lambda p: my_local_model(p)],
        prior=load_yesterdays_summary,
    )
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# --------------------------------------------------------------------------
# GUARD 1 — REFUSAL DETECTION
# --------------------------------------------------------------------------
# The subtle part: a good answer may legitimately *quote* or *analyse* refusal
# language ("the model replied 'I cannot help with that', which suggests...").
# A naive substring match holds that perfectly good output. So the detector is
# position-aware (refusals declare themselves early), quote-aware, and
# analysis-aware. Tune the tests before you tune the patterns.

_REFUSAL_OPEN = re.compile(
    r"^\s*(?:i'?m sorry|i am sorry|sorry[, ]|i apologi[sz]e|unfortunately[, ]|"
    r"i (?:cannot|can'?t|won'?t|am unable to|am not able to)\b|"
    r"as an ai\b|i'?m an ai\b|i'?m just an ai\b|"
    r"i don'?t feel comfortable|i must decline)", re.I)

_REFUSAL_ANY = re.compile(
    r"\b(?:i (?:cannot|can'?t|am unable to) (?:assist|help|comply|provide|continue)|"
    r"against my (?:guidelines|programming|policies)|"
    r"i'?m not able to (?:assist|help|provide))\b", re.I)

_ANALYSIS_CUE = re.compile(
    r"\b(?:the model|the assistant|it (?:replied|responded|returned)|"
    r"output was|response was|refus\w+|example|quoted?|verbatim)\b", re.I)


def _in_quotes(text: str, pos: int) -> bool:
    """Is this position inside quotation marks? Quoted refusal text is evidence,
    not a refusal."""
    head = text[:pos]
    return (head.count('"') % 2 == 1) or (head.count("'") % 2 == 1)


def _near(text: str, pos: int, window: int = 160) -> str:
    return text[max(0, pos - window): pos + window]


def looks_like_refusal(text: str) -> bool:
    """True if the text IS a refusal, not merely one that mentions refusals."""
    t = (text or "").strip()
    if not t:
        return False
    m = _REFUSAL_OPEN.search(t[:400])
    if m and not _in_quotes(t, m.start()) \
            and not _ANALYSIS_CUE.search(_near(t.lower(), m.start())):
        return True
    for m in _REFUSAL_ANY.finditer(t):
        if not _in_quotes(t, m.start()) \
                and not _ANALYSIS_CUE.search(_near(t.lower(), m.start())):
            return True
    return False


# --------------------------------------------------------------------------
# GUARD 2 — TRANSPORT vs CONTENT FAILURE
# --------------------------------------------------------------------------
# These need OPPOSITE retry policies and conflating them is why pipelines
# either give up on a blip or hammer a provider that is telling them no.
#   transport (502, timeout, rate limit)  -> retry fast, several times, backoff
#   content   (refusal, bad shape)        -> retry ONCE with a firmer prompt,
#                                            then fall back. Retrying a refusal
#                                            just buys the same refusal slower.

# A BARE NUMBER IS NOT A STATUS CODE. "The server processed 502 records" and
# {"count": 504} are not outages — but a naive \b(502|504)\b matches both, and
# then your pipeline retries a perfectly good answer forever. So a status code
# only counts when it carries status CONTEXT (an http/status/error word, or its
# standard reason phrase). Caught by the test suite, which is the point of one.
_STATUS_CTX = re.compile(
    r"(?:\b(?:http|https|status|code|error|err|response|returned|got|received)\b"
    r"[^0-9]{0,12}(?:429|500|502|503|504)\b)"
    r"|(?:\b(?:429|500|502|503|504)\b\s*[:\-–]?\s*"
    r"(?:bad gateway|service unavailable|gateway time-?out|internal server "
    r"error|too many requests))", re.I)

_TRANSPORT_WORDS = re.compile(
    r"\b(?:timeout|timed out|connection (?:reset|refused|error|aborted)|"
    r"rate.?limit(?:ed|ing)?|overloaded|service unavailable|"
    r"temporarily unavailable|bad gateway|gateway time-?out|"
    r"ECONNRESET|ETIMEDOUT|ECONNREFUSED|EAI_AGAIN)\b", re.I)


def is_transport_error(text: str, exc: Optional[BaseException] = None) -> bool:
    """Transport failures deserve fast retries; content failures do not.
    Getting this boundary wrong is why pipelines either give up on a blip or
    hammer a provider that is politely telling them no."""
    if exc is not None and isinstance(exc, (TimeoutError, ConnectionError,
                                            OSError)):
        return True
    blob = f"{text or ''}\n{exc or ''}"
    return bool(_TRANSPORT_WORDS.search(blob) or _STATUS_CTX.search(blob))


# --------------------------------------------------------------------------
# GUARD 3 — SHAPE VALIDATION
# --------------------------------------------------------------------------
# "It parsed as JSON" is not "it is usable". The common production failure is a
# model returning {"summary": ""} or {"items": []} — structurally perfect,
# semantically empty — and the pipeline writing that over real data.

@dataclass
class Schema:
    required: list = field(default_factory=list)
    types: dict = field(default_factory=dict)
    non_empty: bool = True          # required keys must not be "" / [] / {} / None
    min_len: int = 0                # for raw-text (non-JSON) answers


def extract_json(text: str) -> Any:
    """Pull JSON out of a model answer that may be fenced or prefaced with prose."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = t.find(opener), t.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                continue
    return None


def validate(text: str, schema: Optional[Schema]) -> tuple[bool, str, Any]:
    """Returns (ok, reason, parsed_value)."""
    if schema is None:
        if not (text or "").strip():
            return False, "empty response", None
        return True, "ok", text
    if schema.required or schema.types:
        data = extract_json(text)
        if data is None:
            return False, "no parseable JSON in the response", None
        if not isinstance(data, dict):
            return (True, "ok", data) if data else (False, "empty JSON", None)
        for k in schema.required:
            if k not in data:
                return False, f"missing required key {k!r}", None
            if schema.non_empty and data[k] in ("", [], {}, None):
                return False, f"key {k!r} present but empty — structurally valid, useless", None
        for k, typ in schema.types.items():
            if k in data and not isinstance(data[k], typ):
                return False, (f"key {k!r} is {type(data[k]).__name__}, "
                               f"expected {typ.__name__}"), None
        return True, "ok", data
    if len((text or "").strip()) < schema.min_len:
        return False, f"response shorter than min_len {schema.min_len}", None
    return True, "ok", text


# --------------------------------------------------------------------------
# GUARDS 4 & 5 — FALLBACK LADDER + HOLD-THE-PRIOR
# --------------------------------------------------------------------------

@dataclass
class Result:
    ok: bool
    value: Any = None
    source: str = ""          # "primary" | "fallback:N" | "prior" | "none"
    attempts: list = field(default_factory=list)

    def __bool__(self):
        return self.ok


def guarded_call(primary: Callable[[str], str],
                 prompt: str,
                 schema: Optional[Schema] = None,
                 fallbacks: Optional[list] = None,
                 prior: Optional[Callable[[], Any]] = None,
                 transport_retries: int = 3,
                 content_retries: int = 1,
                 backoff: float = 2.0,
                 firmer: Optional[str] = None) -> Result:
    """Call a model the way a production pipeline should.

    primary/fallbacks: callables taking a prompt string, returning text.
    prior:             callable returning the last known-good value.
    firmer:            text appended on a content retry (default: a shape nudge).

    THE LAW THIS ENCODES: a failed refresh must never destroy good data.
    Yesterday's correct answer beats today's plausible garbage, every time.
    """
    attempts: list = []
    firmer = firmer or ("\n\nRespond with ONLY the requested output. "
                        "No preamble, no explanation, no apology.")
    ladder = [("primary", primary)] + [
        (f"fallback:{i}", f) for i, f in enumerate(fallbacks or [])]

    for label, fn in ladder:
        content_used = 0
        transport_used = 0
        p = prompt
        while True:
            t0 = time.time()
            try:
                raw = fn(p)
                err = None
            except BaseException as e:            # noqa: BLE001 — classify, don't swallow
                raw, err = "", e

            ms = int((time.time() - t0) * 1000)

            if err is not None or is_transport_error(raw, err):
                attempts.append({"lane": label, "kind": "transport",
                                 "detail": str(err or raw)[:140], "ms": ms})
                if transport_used < transport_retries:
                    transport_used += 1
                    time.sleep(backoff * transport_used)   # exponential
                    continue
                break                                      # next lane

            if looks_like_refusal(raw):
                attempts.append({"lane": label, "kind": "refusal",
                                 "detail": raw.strip()[:140], "ms": ms})
                if content_used < content_retries:
                    content_used += 1
                    p = prompt + firmer
                    continue
                break

            ok, why, value = validate(raw, schema)
            if ok:
                attempts.append({"lane": label, "kind": "ok", "ms": ms})
                return Result(True, value, label, attempts)

            attempts.append({"lane": label, "kind": "shape",
                             "detail": why, "ms": ms})
            if content_used < content_retries:
                content_used += 1
                p = prompt + firmer + f"\n\n(The previous attempt failed: {why}.)"
                continue
            break

    if prior is not None:
        try:
            held = prior()
            if held not in (None, "", [], {}):
                attempts.append({"lane": "prior", "kind": "held"})
                return Result(True, held, "prior", attempts)
        except Exception as e:
            attempts.append({"lane": "prior", "kind": "unavailable",
                             "detail": str(e)[:120]})

    return Result(False, None, "none", attempts)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # A tiny demonstration with fake models — no API keys, runs anywhere.
    def refuser(p):
        return "I'm sorry, but I can't help with that request."

    def empty_shape(p):
        return '{"summary": "", "priority": "high"}'

    def flaky(p):
        raise TimeoutError("connection timed out")

    def good(p):
        return '```json\n{"summary": "Customer cannot log in", "priority": "high"}\n```'

    schema = Schema(required=["summary", "priority"], types={"priority": str})

    print("— refusal is caught, not returned:")
    r = guarded_call(refuser, "x", schema, prior=lambda: {"summary": "yesterday"})
    print("  ", r.ok, r.source, r.value, "|", [a["kind"] for a in r.attempts])

    print("— structurally valid but empty is rejected:")
    r = guarded_call(empty_shape, "x", schema)
    print("  ", r.ok, r.source, "|", [a["detail"] for a in r.attempts if a.get("detail")][:1])

    print("— transport failure falls through to a working lane:")
    r = guarded_call(flaky, "x", schema, fallbacks=[good], transport_retries=1, backoff=0.1)
    print("  ", r.ok, r.source, r.value)

    print("— nothing works: yesterday's data is held, not destroyed:")
    r = guarded_call(refuser, "x", schema, prior=lambda: {"summary": "held"},
                     content_retries=0)
    print("  ", r.ok, r.source, r.value)
