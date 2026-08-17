# llm-guardrails

Make an LLM call safe to put on a cron at 3am.

An LLM is not an API. An API fails **loudly** — 500, timeout, exception. An LLM fails
**plausibly**: it returns a paragraph of apology where you expected JSON, or a confident
answer to a question it never actually received, or a structurally perfect object with
every field empty. Your workflow writes it to the database and reports success.

Five guards. Zero dependencies. Works with any provider — you pass a callable.

```python
from llm_guardrails import guarded_call, Schema

out = guarded_call(
    lambda p: openai_call(p),                  # primary
    prompt="Summarise this ticket as JSON: ...",
    schema=Schema(required=["summary", "priority"], types={"priority": str}),
    fallbacks=[lambda p: local_model(p)],      # degrade, don't die
    prior=load_yesterdays_summary,             # never destroy good data
)
```

```
— refusal is caught, not returned:
   True prior {'summary': 'yesterday'} | ['refusal', 'refusal', 'held']
— structurally valid but empty is rejected:
   False none | ["key 'summary' present but empty — structurally valid, useless"]
— transport failure falls through to a working lane:
   True fallback:0 {'summary': 'Customer cannot log in', 'priority': 'high'}
— nothing works: yesterday's data is held, not destroyed:
   True prior {'summary': 'held'}
```

---

## 1. Refusal detection — and the trap inside it

A model that declines has not given you output. Writing its apology into your CRM is worse
than writing nothing, because nothing is visible and an apology looks like data.

The naive fix is a substring match on "I cannot". **That breaks immediately**, because a
*good* answer often quotes or analyses refusal language:

> The model replied *"I cannot assist with that"*, which suggests the prompt tripped a
> safety filter. Recommend rephrasing.

That is a perfectly useful answer, and a naive detector throws it away. So the detector here
is **position-aware** (refusals declare themselves in the first line, not paragraph four),
**quote-aware** (text inside quotes is evidence, not speech), and **analysis-aware** (near
words like *the model / replied / returned / example*, it is commentary).

A false positive silently discards good work. A false negative writes an apology to
production. Both are tested.

## 2. Transport failures and content failures need opposite retries

| Failure | Right response |
|---|---|
| 502, timeout, rate limit | retry **fast**, several times, exponential backoff |
| refusal, wrong shape | retry **once** with a firmer prompt, then fall back |

Retrying a refusal just buys you the same refusal, slower. Giving up on a 502 throws away a
working pipeline over a blip.

**And a bare number is not a status code.** `The server processed 502 records successfully`
and `{"count": 504}` are not outages — but `\b502\b` matches both, and then your pipeline
retries a perfect answer forever. Status codes only count with status *context*: an
`http`/`status`/`error` word nearby, or the standard reason phrase. That bug was caught by
the test suite in this repo, which is the entire argument for having one.

## 3. "It parsed" is not "it is usable"

The most expensive silent failure in LLM pipelines:

```json
{"summary": "", "priority": "high"}
```

Valid JSON. Correct keys. Right types. **Completely worthless** — and it will overwrite a
good record without a single error anywhere. `Schema(non_empty=True)` rejects present-but-
empty required fields by default.

The JSON extractor also handles what models actually return: fenced blocks, a helpful
sentence before the object, a bare array.

## 4. The fallback ladder

Primary → fallbacks in order → prior. Each lane gets its own retry budget, and every attempt
is recorded with lane, failure class, detail, and timing:

```python
result.attempts
# [{'lane': 'primary',    'kind': 'refusal',   'detail': "I'm sorry, but I can't..."},
#  {'lane': 'fallback:0', 'kind': 'transport', 'detail': 'timed out'},
#  {'lane': 'prior',      'kind': 'held'}]
```

A pipeline that degrades in visible steps beats one that is either perfect or dead.

## 5. Hold the prior — a failed refresh must not destroy good data

**Yesterday's correct answer beats today's plausible garbage, every time.** If every lane
fails and a `prior` is available, the guard returns it and labels the source `prior`, so
downstream code knows the data is stale but valid. Without this, a bad model day silently
wipes a good dataset — and nobody notices until someone asks why the report is empty.

---

## Using it with n8n

Put it in a Code node, or call it from an Execute Command node in front of your HTTP
request. The pattern that matters more than the library: **never write a model's raw output
straight into your next node.** Validate the shape, detect the refusal, and decide
explicitly what happens when it fails — because it will, and the day it does you want a
held record and a log line, not a silent corruption.

## Tests

```bash
python3 test_guardrails.py
```

Covers refusals that must be caught, refusal *quotes* that must not be, the status-code
false positives, JSON extracted from messy output, empty-but-valid rejection, and the
end-to-end ladder including the hold-the-prior path.

Python 3.9+. No third-party packages. MIT licensed.
