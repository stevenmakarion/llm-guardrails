#!/usr/bin/env python3
"""test_guardrails.py — the cases that matter.

The refusal detector is the subtle guard: a FALSE POSITIVE holds a perfectly
good answer that happened to quote refusal language; a FALSE NEGATIVE lets an
apology get written to your database as data. Both are tested here.

Run:  python3 test_guardrails.py
"""
import sys

from llm_guardrails import (Schema, extract_json, guarded_call,
                            is_transport_error, looks_like_refusal, validate)

# ---------------------------------------------------------- refusal: TRUE
REFUSALS = [
    "I'm sorry, but I can't help with that.",
    "I cannot assist with this request.",
    "As an AI, I don't have the ability to browse the web.",
    "Unfortunately, I'm not able to provide that information.",
    "I apologize — this goes against my guidelines.",
    "  \n I must decline to answer that.",
]

# ------------------------------------------------- refusal: FALSE (the trap)
NOT_REFUSALS = [
    # analysing a refusal is not refusing
    'The model replied "I cannot assist with that", which suggests the prompt '
    'tripped a safety filter. Recommend rephrasing.',
    # quoting one
    'Support ticket #441 contains the text "I\'m sorry, I can\'t help" from the '
    'previous agent. Escalate to tier 2.',
    # a legitimate answer that merely contains "sorry"
    "The customer was sorry for the delay and has since paid the invoice.",
    # a real, useful JSON answer
    '{"summary": "Login fails on Safari 17", "priority": "high"}',
    # explaining refusal behaviour in documentation
    "Refusal responses should be detected before writing output to the database.",
]

TRANSPORT = ["Error: 502 Bad Gateway", "request timed out after 30s",
             "429 rate limit exceeded", "ECONNRESET"]
NOT_TRANSPORT = ["The server processed 502 records successfully.",
                 '{"status":"ok","count":504}']


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return bool(cond)


def main():
    ok = True
    print("refusal detection — must be TRUE:")
    for t in REFUSALS:
        ok &= check(t[:58].replace("\n", " "), looks_like_refusal(t))

    print("\nrefusal detection — must be FALSE (quoted / analysed / innocent):")
    for t in NOT_REFUSALS:
        ok &= check(t[:58].replace("\n", " "), not looks_like_refusal(t))

    print("\ntransport classification:")
    for t in TRANSPORT:
        ok &= check(f"transport: {t[:40]}", is_transport_error(t))
    for t in NOT_TRANSPORT:
        ok &= check(f"not transport: {t[:40]}", not is_transport_error(t))

    print("\nJSON extraction from messy model output:")
    ok &= check("fenced json", extract_json('```json\n{"a":1}\n```') == {"a": 1})
    ok &= check("prose then json",
                extract_json('Sure! Here you go:\n{"a":1}') == {"a": 1})
    ok &= check("bare array", extract_json("[1,2,3]") == [1, 2, 3])
    ok &= check("no json returns None", extract_json("no json here") is None)

    print("\nshape validation:")
    s = Schema(required=["summary", "priority"], types={"priority": str})
    ok &= check("valid passes", validate('{"summary":"x","priority":"high"}', s)[0])
    ok &= check("missing key fails", not validate('{"summary":"x"}', s)[0])
    ok &= check("EMPTY-BUT-PRESENT fails (the silent killer)",
                not validate('{"summary":"","priority":"high"}', s)[0])
    ok &= check("wrong type fails",
                not validate('{"summary":"x","priority":3}', s)[0])

    print("\nend-to-end behaviour:")
    r = guarded_call(lambda p: "I'm sorry, I can't do that.", "x", s,
                     prior=lambda: {"summary": "yesterday", "priority": "low"},
                     content_retries=0)
    ok &= check("refusal -> holds the prior, never returns the apology",
                r.ok and r.source == "prior" and r.value["summary"] == "yesterday")

    r = guarded_call(lambda p: (_ for _ in ()).throw(TimeoutError("timed out")),
                     "x", s, fallbacks=[lambda p: '{"summary":"ok","priority":"low"}'],
                     transport_retries=1, backoff=0.01)
    ok &= check("transport failure -> next lane, not death", r.ok and
                r.source == "fallback:0")

    r = guarded_call(lambda p: "I'm sorry.", "x", s, content_retries=0)
    ok &= check("no prior available -> honest failure, not garbage",
                (not r.ok) and r.value is None)

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
