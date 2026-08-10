"""
Evaluation for STRATUM models.

Reads a test set of {"prompt":..., "expected":...} and scores model outputs.
Ships four scorers, picked with --scorer:
  contains  : expected string appears in output (lenient default)
  exact     : output equals expected after normalization (strict)
  json_field: parse both as JSON, score field-by-field (for extraction)
  overlap   : word-overlap F1 (for free-text answers that paraphrase)

A test row may also carry a "skill" key. If any do, the report includes a
per-skill breakdown so you can see which stratum needs work after a merge.

run_eval returns a report dict and can write it as JSON (json_out) so a CI
job can gate on the score. See docs/08-evaluation.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .data import load_jsonl, format_chat, strip_think


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _as_number(v):
    """Parse a value as a number if possible: handles '1,499', '$88', '88.0'."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "").lstrip("$")
    try:
        return float(s)
    except ValueError:
        return None


def _values_match(a, b) -> bool:
    """Compare two JSON field values, numerically when both parse as numbers.

    Without this, an expected 1499 would fail against a model that answered
    '1,499' - correct in substance, wrong only in formatting.
    """
    na, nb = _as_number(a), _as_number(b)
    if na is not None and nb is not None:
        return na == nb
    return _norm(a) == _norm(b)


def score_contains(output: str, expected) -> float:
    return float(_norm(expected) in _norm(output))


def score_exact(output: str, expected) -> float:
    return float(_norm(output) == _norm(expected))


def _extract_json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def score_json_field(output: str, expected) -> float:
    pred = _extract_json(output)
    gold = expected if isinstance(expected, dict) else _extract_json(json.dumps(expected))
    if pred is None or not isinstance(gold, dict):
        return 0.0
    keys = [k for k in gold if gold[k] is not None]
    if not keys:
        return 0.0
    correct = sum(1 for k in keys if k in pred and _values_match(pred[k], gold[k]))
    return correct / len(keys)


_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset("""a an and are as at be by for from has have in is it its
of on or that the this to was were will with""".split())


def _content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall(_norm(text)) if w not in _STOPWORDS]


def score_overlap(output: str, expected) -> float:
    """Word-overlap F1 between the answer and the expected answer.

    The scorer for free-text answers, where `contains` is useless: a model
    that replies "A 100-metre blade made of glass fibre weighs 50 tonnes"
    to an expected "A 100-metre (330 ft) glass fiber blade weighs 50 tonnes"
    is right, and every substring test says zero.

    F1 balances two failure modes a single number has to catch: recall alone
    would reward an answer that recites the whole document, precision alone
    would reward a one-word guess. Common filler words are ignored so they
    cannot inflate the score. This is the standard measure for question
    answering, and like every automatic scorer it rewards saying the right
    words - pair it with human review for anything that matters.
    """
    pred = _content_words(output)
    gold = _content_words(str(expected))
    if not gold:
        return 0.0
    if not pred:
        return 0.0
    from collections import Counter
    shared = Counter(pred) & Counter(gold)
    n_shared = sum(shared.values())
    if n_shared == 0:
        return 0.0
    precision = n_shared / len(pred)
    recall = n_shared / len(gold)
    return 2 * precision * recall / (precision + recall)


def score_refusal(output: str, expected) -> float:
    """Did the model decline exactly when it should have declined?

    The scorer for a grounded test set, where some rows had their source
    removed and the only honest answer is that the material does not contain
    it. `stratum ground --abstain-share` makes those rows.

    Both directions are failures and both score zero, which is the point of
    doing it this way rather than counting refusals.

      Expected an answer, refused. The model has become useless, which is the
      real cost of overdoing abstention and the thing that would otherwise be
      invisible behind a rising refusal rate.

      Expected a refusal, answered. The model invented something. Measured on
      a real build, a person with no access to engineering material was given
      a fabricated bearing clearance, which no access filter can prevent
      because the filter had already done its job correctly.

    So a high score here means the model declines when it should and answers
    when it should, and neither behaviour can be traded off against the other
    to make the number look better.

    It says nothing about whether the answers it does give are correct. Pair
    it with `overlap` on the same test set, which is the other half.
    """
    from .grounded import is_refusal

    should_refuse = is_refusal(str(expected))
    did_refuse = is_refusal(output)
    return 1.0 if should_refuse == did_refuse else 0.0


def refusal_breakdown(results: list[dict]) -> dict:
    """Split refusal scoring into the four things that can happen.

    Two of them are right and two are wrong, and the two wrong ones need
    opposite fixes. Inventing means more abstention rows in training.
    Over refusing means fewer. A mean hides which one you have.
    """
    from .grounded import is_refusal

    counts = {"declined": 0, "invented": 0, "answered": 0, "over_refused": 0}
    for r in results:
        should = is_refusal(str(r.get("expected", "")))
        did = is_refusal(r.get("output", ""))
        if should:
            counts["declined" if did else "invented"] += 1
        else:
            counts["over_refused" if did else "answered"] += 1
    counts["should_decline"] = counts["declined"] + counts["invented"]
    counts["should_answer"] = counts["answered"] + counts["over_refused"]
    return counts


SCORERS = {"contains": score_contains, "exact": score_exact,
           "json_field": score_json_field, "overlap": score_overlap,
           "refusal": score_refusal}


def _generate(model, tokenizer, prompt, system, device, max_new_tokens):
    import torch
    text = format_chat(tokenizer, prompt, response=None, system=system)
    from .hf_utils import encode_for_generation
    ids = encode_for_generation(tokenizer, text, device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    raw = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    # Thinking models may still open a reasoning block. Score only the answer.
    return strip_think(raw)


def _with_context(row, index, style, k, budget, allowed):
    """Build the prompt for one test row under the chosen context arm."""
    if style in ("none", None) or (index is None and style != "oracle"):
        return row["prompt"]

    from .context import build_prompt

    if style == "oracle":
        if index is None:
            raise ValueError("The oracle arm needs an index to find the chunk in.")
        want = row.get("source_chunk")
        if not want:
            return row["prompt"]
        try:
            i = index.ids.index(want)
        except ValueError:
            return row["prompt"]
        return build_prompt(row["prompt"], [index._hit(i, 1.0, "oracle")],
                            budget, style="chunks")

    hits = index.search(row["prompt"], allowed=allowed, k=k)
    inner = "sentences" if style == "sentences" else "chunks"
    return build_prompt(row["prompt"], hits, budget, style=inner)


def run_eval(model_dir: str, test_path: str, scorer: str = "contains",
             system: str | None = None, max_new_tokens: int = 256,
             verbose: bool = True, json_out: str | None = None,
             baseline: str | None = None, context=None,
             context_style: str = "none", context_k: int = 3,
             context_budget: int = 2000, allowed=None,
             require_support: bool = False,
             min_overlap: float | None = None) -> dict:
    """Score a model on a test set.

    Returns a report dict: {"mean", "n", "scorer", "per_skill", "results"} plus,
    when baseline is given, "baseline_mean" - the same test run against another
    model (usually the untrained base) so the comparison is one command.

    context_style decides what, if anything, goes in the prompt beside the
    question, and it exists so the question can be settled by measurement
    rather than argument:

      none       closed book, which is how strata are trained
      oracle     the exact chunk the question was written from. The ceiling,
                 and the one worth measuring first, because a model that
                 cannot use the right chunk when handed it will not be
                 rescued by a better retriever
      chunks     what retrieval actually found
      sentences  the same retrieved text cut to the sentences that share
                 wording with the question

    The chunks and sentences arms share one retrieval result on purpose, so
    what varies between them is presentation only. Retrieving separately for
    each would confound the two and make the comparison meaningless.

    require_support runs every answer past the support check before scoring
    it, replacing anything the material does not carry with a refusal. It is
    the same call serving makes, so what is measured here is the behaviour
    that ships rather than an estimate of it.
    """
    import torch

    if min_overlap is None:
        from .support import MIN_OVERLAP
        min_overlap = MIN_OVERLAP

    if min_overlap is None:
        from .support import MIN_OVERLAP
        min_overlap = MIN_OVERLAP
    if scorer not in SCORERS:
        raise ValueError(f"Unknown scorer '{scorer}'. Choose from {list(SCORERS)}.")
    score_fn = SCORERS[scorer]
    rows = load_jsonl(test_path, required_keys=("prompt", "expected"))
    from .hf_utils import pick_device
    device = pick_device()

    def eval_one_model(model_source):
        from .hf_utils import load_for_inference
        model, tokenizer = load_for_inference(model_source)
        model.to(device).eval()
        results = []
        for row in rows:
            prompt = _with_context(row, context, context_style, context_k,
                                   context_budget, allowed)
            answer = _generate(model, tokenizer, prompt, system, device,
                               max_new_tokens)
            s = score_fn(answer, row["expected"])
            results.append({"prompt": row["prompt"], "expected": row["expected"],
                            "output": answer, "score": s,
                            "skill": row.get("skill")})
            if verbose:
                mark = "OK " if s >= 0.999 else ("~~ " if s > 0 else "XX ")
                print(f"{mark} score {s:.2f} expected: {str(row['expected'])[:36]:36} "
                      f"got: {answer.strip()[:36]}")
        # Release this model before a possible baseline model loads, so both
        # never sit in GPU memory at once.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    results = eval_one_model(model_dir)
    mean = sum(r["score"] for r in results) / len(results)

    # The support gate, applied to what the model said before it is scored.
    # This is the same call `stratum serve --require-support` makes, so the
    # number measured here is the behaviour that ships rather than an
    # estimate of it.
    gated = {"held_back": 0, "reasons": {}}
    if require_support:
        from .support import gate, passages_from_prompt
        for r in results:
            passages = passages_from_prompt(r["prompt"])
            question = r["prompt"].rsplit("Question:", 1)[-1].strip()
            answer, why = gate(r["output"], passages, question,
                               min_overlap=min_overlap)
            if not why["supported"]:
                gated["held_back"] += 1
                gated["reasons"][why["reason"]] = \
                    gated["reasons"].get(why["reason"], 0) + 1
                r["output_before_gate"] = r["output"]
                r["output"] = answer
                r["gate"] = why
            r["score"] = score_fn(r["output"], r["expected"])
        mean = sum(r["score"] for r in results) / len(results) if results else 0.0

    per_skill = {}
    if any(r["skill"] for r in results):
        for r in results:
            per_skill.setdefault(r["skill"] or "(unlabeled)", []).append(r["score"])
        per_skill = {k: sum(v) / len(v) for k, v in sorted(per_skill.items())}

    report = {"model": model_dir, "test": test_path, "scorer": scorer,
              "n": len(results), "mean": mean, "per_skill": per_skill,
              "require_support": require_support, "gated": gated,
              "results": results}

    if require_support and gated["held_back"]:
        print(f"\nSupport check held back {gated['held_back']} answer(s) that "
              f"the material did not carry")
        for why, n in sorted(gated["reasons"].items(), key=lambda kv: -kv[1]):
            print(f"  {n:>4}  {why}")
        print("  each became a refusal rather than reaching anybody")

    print(f"\nScore ({scorer}): {mean:.1%} over {len(results)} cases")
    for skill, s in per_skill.items():
        print(f"  {skill}: {s:.1%}")

    # One number hides which way the model is wrong, and the two directions
    # need completely different fixes. Measured on a real build, a closed
    # book model scored 43% here while inventing an answer on every single
    # row where the material held none. The single figure looked survivable
    # and the breakdown did not.
    if scorer == "refusal":
        counts = refusal_breakdown(results)
        report_extra = {"refusal_breakdown": counts}
        report.update(report_extra)
        asked = counts["should_decline"]
        print()
        print("  Of the cases where the material did NOT hold the answer")
        if asked:
            print(f"    declined correctly     {counts['declined']:>4} of "
                  f"{asked}  ({counts['declined'] / asked:.0%})")
            print(f"    INVENTED an answer     {counts['invented']:>4} of "
                  f"{asked}  ({counts['invented'] / asked:.0%})")
        else:
            print("    none, so this run says nothing about declining. Make "
                  "some with")
            print("    `stratum ground --abstain-share`.")
        answerable = counts["should_answer"]
        print("  Of the cases where it DID hold the answer")
        if answerable:
            print(f"    answered               {counts['answered']:>4} of "
                  f"{answerable}  ({counts['answered'] / answerable:.0%})")
            print(f"    refused one it could   {counts['over_refused']:>4} of "
                  f"{answerable}  ({counts['over_refused'] / answerable:.0%})")
        else:
            print("    none, so a model that declines everything would score "
                  "full marks here.")
        # Which way to turn the dial, decided by which failure is larger.
        # Telling somebody to raise the abstention share while their real
        # problem is a model that already refuses half of what it could
        # answer makes the model worse and looks like advice.
        worse = "invent" if counts["invented"] > counts["over_refused"] else \
            ("refuse" if counts["over_refused"] > counts["invented"] else "both")
        if counts["invented"] or counts["over_refused"]:
            print()
        if worse == "invent":
            print(f"  Inventing is the larger failure, {counts['invented']} "
                  f"against {counts['over_refused']} over refusals.")
            print("  Raise --abstain-share and retrain. No access filter "
                  "prevents inventing,")
            print("  because the filter has already done its job by returning "
                  "nothing.")
            print("  `stratum serve --require-support` stops these reaching "
                  "anybody meanwhile.")
        elif worse == "refuse":
            print(f"  Over refusing is the larger failure, "
                  f"{counts['over_refused']} against {counts['invented']} "
                  f"inventions.")
            print("  LOWER --abstain-share and retrain. The model has learned "
                  "to decline more")
            print("  readily than the material warrants, and a model that "
                  "will not answer")
            print("  questions it can answer is not a safer one, it is an "
                  "unused one.")
        elif counts["invented"]:
            print(f"  {counts['invented']} of each failure. The abstention "
                  f"share is about right and")
            print("  what is left needs more or better training pairs rather "
                  "than a different dial.")

    if baseline:
        print(f"\nBaseline run: {baseline}")
        base_results = eval_one_model(baseline)
        base_mean = sum(r["score"] for r in base_results) / len(base_results)
        report["baseline"] = baseline
        report["baseline_mean"] = base_mean
        print(f"Baseline score: {base_mean:.1%}  ->  your model: {mean:.1%} "
              f"({'+' if mean >= base_mean else ''}{(mean - base_mean):.1%})")

    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"Report written to {json_out}")

    return report
