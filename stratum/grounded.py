"""
Turn ordinary training pairs into grounded ones, so the model learns to read
rather than to remember.

Two problems have the same cause and the same fix.

The first is measured in this project. Strata are trained closed book, on a
question and an answer alone, and at serve time they are handed retrieved
passages they have never seen. Across three skills the effect of giving the
model the exactly correct passage was safety plus 5.5 points, extract flat,
and explain slightly WORSE. Context was out of distribution, so the model
ignored it or was hurt by it.

The second is the reason access control is hard. A model trained on question
and answer alone has no way to produce the answer except to store it. Those
stored facts are the thing that cannot be filtered afterwards, and they are
why a compartment's material must not enter shared weights.

Both come from training on a question with no source. So put the source in
the prompt, alongside passages that do not contain the answer, and three
things change at once.

  The model learns (context, question) -> answer, which generalises, rather
  than question -> answer, which can only be memorised.

  There is no gradient pressure to store the fact, because the fact is
  always readable. What gets learned is the domain's language and the habit
  of citing, which is exactly what should be shared.

  With only distractors present, the honest answer is that the material does
  not contain it. Trained in deliberately, this turns a fabrication into an
  abstention, which is the failure an enterprise can live with.

This follows RAFT, Retrieval Augmented Fine Tuning, by Zhang and others in
2024. The distractor ratio and the abstention share below are the two dials
that paper turns, and both are exposed rather than baked in.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

_WORD = re.compile(r"[^\W_]+", re.UNICODE)

# Words, and numbers kept whole including their decimal point. A figure like
# 0.418 is the single most useful thing to locate in a chunk, and splitting it
# into 0 and 418 throws away the term most worth finding.
_NUM_OR_WORD = re.compile(r"\d[\d,]*\.\d+|\d[\d,]*|[^\W\d_]+", re.UNICODE)

# How the passages are presented. The same layout is used at training time
# and at serve time, because a model that learned one shape and is given
# another is back to having context out of distribution, which is the
# problem this module exists to fix.
GROUNDED_TEMPLATE = """\
Use the reference material below to answer. If it does not contain the \
answer, say so plainly.

{passages}

Question: {question}"""

PASSAGE = "[{tag}]\n{text}"

# What the model is taught to say when the answer is not in front of it. One
# fixed sentence rather than several, because a model that learned five ways
# to decline will invent a sixth, and a caller cannot match on it.
REFUSAL = ("The reference material provided does not contain the answer to "
           "that question.")


class GroundedError(Exception):
    """The pairs cannot be grounded as asked."""


def _rng_for(key: str, seed: int) -> random.Random:
    """A generator that depends only on the row and the seed.

    Distractors have to be identical across re-runs, or a resumed job would
    build a different dataset from the one it started, and two people running
    the same command would not get the same model.
    """
    h = hashlib.blake2b(f"{key}:{seed}".encode("utf-8"), digest_size=8).digest()
    return random.Random(int.from_bytes(h, "little"))


def build_prompt(question: str, passages: list[tuple[str, str]]) -> str:
    """Assemble the grounded prompt from (tag, text) pairs."""
    blocks = "\n\n".join(PASSAGE.format(tag=t, text=x.strip())
                         for t, x in passages)
    return GROUNDED_TEMPLATE.format(passages=blocks, question=question)


def _distraction_ranking(chunks: list, needed: set | None = None,
                         top: int = 8, verbose: bool = False) -> dict:
    """The most confusable neighbours of each chunk that a pair points at.

    Two things keep this from being the slow step on a real corpus.

    Only the chunks some training pair actually names are ranked. A corpus of
    a hundred thousand chunks may have a few thousand that are the source of
    a question, and ranking the rest produces work nothing ever reads.

    Only the top few neighbours are kept, using a partial selection rather
    than sorting every candidate. Sorting the full list for every source was
    the first version and turned a corpus of eleven hundred chunks into a job
    that had to be killed.

    Random distractors make an abstention row easy in the wrong way. The
    passages beside the question are visibly about something else, so a model
    can learn to decline whenever the material looks unrelated. That rule
    does not fire in the case that matters, where retrieval returns passages
    squarely on topic which happen not to contain the specific fact asked
    for, and the model answers anyway.

    Amiraz and others, The Distracting Effect, ACL 2025, measured this and
    found that fine tuning on deliberately hard distractors rather than
    random ones raised answering accuracy by up to 7.5 percent, with the gain
    concentrated on exactly the ungrounded cases where the answer is absent.

    Similarity here is the same term weighting the router and the family
    grouping use, so a company adds no model and no extra pass over the
    corpus to get it.
    """
    import math
    from collections import Counter

    from .router import _features, _l2

    by_comp: dict[str, list] = {}
    for c in chunks:
        by_comp.setdefault(c.get("compartment") or "_", []).append(c)

    # Only the compartments some source chunk belongs to. Grounding is run
    # one compartment at a time, because that is how one stratum per
    # compartment gets trained, so the other fifteen departments in the
    # corpus are work whose result nothing reads.
    if needed:
        by_comp = {k: v for k, v in by_comp.items()
                   if any(c["id"] in needed for c in v)}
    chunks = [c for rows in by_comp.values() for c in rows]

    docs = [_features(c["text"]) for c in chunks]
    n = len(docs)
    df: Counter = Counter()
    for d in docs:
        df.update(d.keys())
    idf = {f: math.log(n / k) for f, k in df.items() if 2 <= k < n}
    vecs = {}
    for c, d in zip(chunks, docs):
        vecs[c["id"]] = _l2({f: (1 + math.log(v)) * idf[f]
                             for f, v in d.items() if f in idf})

    def sim(a, b):
        va, vb = vecs.get(a, {}), vecs.get(b, {})
        if len(va) > len(vb):
            va, vb = vb, va
        return sum(w * vb.get(f, 0.0) for f, w in va.items())

    import heapq

    ranked = {}
    for comp, rows in by_comp.items():
        ids = [r["id"] for r in rows]
        targets = [i for i in ids if needed is None or i in needed]
        for target in targets:
            ranked[target] = heapq.nlargest(
                top, (i for i in ids if i != target),
                key=lambda other: sim(target, other))
    if verbose and ranked:
        print(f"  ranked the {top} most confusable neighbours for "
              f"{len(ranked)} source chunk(s)")
    return ranked


def _window_for_answer(text: str, answer: str, max_chars: int) -> str:
    """The part of a chunk that actually carries the answer.

    Cutting a chunk at the first max_chars characters is the obvious thing to
    do and it quietly poisons the training set. When the answer sits past the
    cut, the row still says it is answerable, so the model is shown material
    that does not contain the answer and taught to produce one anyway. That
    is a lesson in inventing, delivered by the very file meant to teach it not
    to. Measured here, it hit 2 of 8 answerable rows.

    So the window is chosen to cover the answer rather than the opening. The
    text is scanned for the stretch densest in the answer's own words, and
    that stretch is kept.
    """
    if len(text) <= max_chars:
        return text

    # Numbers are kept whole here rather than split on the decimal point.
    # "0.418" is the single most useful thing to locate, and a tokeniser that
    # turns it into 0 and 418 loses exactly the term worth finding.
    terms = set(_NUM_OR_WORD.findall(answer.lower()))
    terms = {t for t in terms if len(t) > 3}
    if not terms:
        return text[:max_chars]

    low = text.lower()
    hits = [m.start() for m in re.finditer(
        "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)),
        low)]
    if not hits:
        return text[:max_chars]

    # Centred on the middle of where the answer's terms actually appear, not
    # started at the first one. Starting there puts the match at the very
    # edge of the window and cuts the rest of the sentence off, which was the
    # first version of this and looked like it worked.
    hits.sort()
    middle = hits[len(hits) // 2]
    start = max(0, min(middle - max_chars // 2, len(text) - max_chars))
    cut = text[start:start + max_chars]
    # Open at a word boundary so the passage does not begin mid word.
    if start and " " in cut[:80]:
        cut = cut[cut.index(" ") + 1:]
    return cut


def _answer_is_present(answer: str, passage: str, need: float = 0.6) -> bool:
    """Whether a passage carries enough of the answer to be its source.

    Numbers are kept whole for the same reason as above. A passage that holds
    every word of an answer but not its figure is not the source of that
    answer, and treating it as one is how a training set ends up teaching a
    model to supply figures of its own.
    """
    terms = {t for t in _NUM_OR_WORD.findall(answer.lower()) if len(t) > 3}
    if not terms:
        return True
    low = passage.lower()
    return sum(1 for t in terms if t in low) / len(terms) >= need


def ground_pairs(pairs_path: str, chunks_path: str, out_path: str,
                 distractors: int = 3, abstain_share: float = 0.15,
                 same_compartment: bool = True, max_chars: int = 1200,
                 hard: bool = True, seed: int = 42,
                 verbose: bool = True) -> dict:
    """Rewrite a training file so every prompt carries its source material.

    distractors is how many passages that do NOT hold the answer sit beside
    the one that does. Without them the model learns that the first passage
    is always right, which is not a skill.

    abstain_share is the fraction of rows where the correct passage is
    removed entirely and the answer becomes the refusal. This is what makes
    declining a learned behaviour rather than a hope.

    same_compartment keeps distractors inside the row's own compartment.
    That matters more than it looks. Pulling a distractor from a compartment
    the reader cannot see would put forbidden text into training data for a
    model other people load, which is the exact failure the tiers exist to
    prevent.
    """
    from .data import load_jsonl

    # Training rows carry the answer under 'response' and test rows carry it
    # under 'expected'. Both have to be groundable, because a grounded model
    # measured on a closed book test set is measured on prompts shaped
    # differently from anything it was trained on, which tells you nothing
    # about the thing you were trying to find out. Whichever key comes in is
    # the key that goes out, so a test set stays a test set.
    rows = load_jsonl(pairs_path, required_keys=("prompt",))
    answer_key = "response"
    if rows and "response" not in rows[0] and "expected" in rows[0]:
        answer_key = "expected"
    missing_answer = [i for i, r in enumerate(rows, 1) if answer_key not in r]
    if missing_answer:
        raise GroundedError(
            f"{pairs_path} line {missing_answer[0]} has no '{answer_key}'. "
            f"Every row needs a prompt and an answer, under 'response' for "
            f"training data or 'expected' for a test set.")

    chunks = load_jsonl(chunks_path, required_keys=("id", "text", "source"))
    if not rows:
        raise GroundedError(f"{pairs_path} has no pairs in it.")
    if not chunks:
        raise GroundedError(f"{chunks_path} has no chunks in it.")

    by_id = {c["id"]: c for c in chunks}
    pools: dict[str, list] = {}
    for c in chunks:
        pools.setdefault(c.get("compartment") or "_", []).append(c["id"])
    everything = [c["id"] for c in chunks]

    missing = sum(1 for r in rows if r.get("source_chunk") not in by_id)
    if missing == len(rows):
        raise GroundedError(
            f"None of the {len(rows)} pairs name a chunk that exists in "
            f"{chunks_path}.\n"
            f" - Pairs written by `stratum corpus pairs` carry source_chunk.\n"
            f" - Check the two files come from the same corpus.")

    counts = {"grounded": 0, "abstain": 0, "skipped": missing,
              "rescued": 0, "distractors": distractors, "hard": hard}
    needed = {r.get("source_chunk") for r in rows if r.get("source_chunk")}
    ranking = _distraction_ranking(
        chunks, needed=needed, top=max(8, distractors * 2),
        verbose=verbose) if hard else {}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            src_id = row.get("source_chunk")
            src = by_id.get(src_id)
            if src is None:
                continue

            rng = _rng_for(src_id + row["prompt"][:60], seed)
            comp = src.get("compartment") or "_"
            pool = pools.get(comp, everything) if same_compartment else everything
            if hard and src_id in ranking:
                # The nearest neighbours of the source, which are the
                # passages most likely to look like they answer the question
                # while not containing the answer.
                allowed = set(pool)
                candidates = [c for c in ranking[src_id] if c in allowed]
            else:
                candidates = [c for c in pool if c != src_id]
                rng.shuffle(candidates)
            picked = candidates[:distractors]

            abstain = rng.random() < abstain_share and bool(picked)
            passages = [(by_id[c]["source"], by_id[c]["text"][:max_chars])
                        for c in picked]

            unanswerable = False
            if not abstain:
                # The window that carries the answer, not the opening of the
                # chunk. See _window_for_answer for why that distinction is
                # the difference between teaching reading and teaching
                # invention.
                kept = _window_for_answer(src["text"], str(row[answer_key]),
                                          max_chars)
                if not _answer_is_present(str(row[answer_key]), kept):
                    # Even the best window does not carry it. The honest row
                    # is one where declining is correct, because that is the
                    # truth of what the model is being shown. Left as it was,
                    # this row would say an answer is available in material
                    # that does not contain it, which is a worked example of
                    # making something up.
                    unanswerable = True
                    abstain = True
                    counts["rescued"] += 1
                else:
                    passages.append((src["source"], kept))

            # Shuffled so position carries no information. Always putting the
            # answer last would teach the model to read the last passage.
            rng.shuffle(passages)

            record = {
                "prompt": build_prompt(row["prompt"], passages),
                answer_key: REFUSAL if abstain else row[answer_key],
                "source_chunk": src_id,
                "source": row.get("source", src["source"]),
                "grounded": True,
                "abstain": abstain,
            }
            if unanswerable:
                record["unanswerable_after_trim"] = True
            if "compartment" in src:
                record["compartment"] = src["compartment"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["abstain" if abstain else "grounded"] += 1

    # The property the whole file has to have. Every row that claims an
    # answer is available must be shown material that carries it. One row
    # that breaks this is one worked example of inventing, taught on purpose
    # by the file meant to teach the opposite, so it is checked rather than
    # assumed.
    counts["unanswerable_kept"] = 0
    for line in open(out, encoding="utf-8"):
        r = json.loads(line)
        if r.get("abstain"):
            continue
        from .support import passages_from_prompt
        if not _answer_is_present(str(r[answer_key]),
                                  passages_from_prompt(r["prompt"])):
            counts["unanswerable_kept"] += 1

    total = counts["grounded"] + counts["abstain"]
    if verbose:
        print(f"Grounded {total} pairs -> {out_path}")
        print(f"  distractors are the {'most confusable' if hard else 'random'} "
              f"passages in the compartment")
        print(f"  {counts['grounded']} answerable, each with the source plus "
              f"{distractors} passages that are not")
        print(f"  {counts['abstain']} where the source was removed, so the "
              f"answer is to decline")
        if counts["skipped"]:
            print(f"  {counts['skipped']} skipped, their chunk is not in the "
                  f"corpus file")
        if counts["rescued"]:
            print(f"  {counts['rescued']} row(s) whose answer was not in the "
                  f"chunk even after picking the best")
            print(f"  window, so they became decline rows. Left alone, each "
                  f"one would have taught the")
            print(f"  model to answer from material that does not contain "
                  f"the answer.")
        if same_compartment:
            print("  distractors stay inside each row's own compartment, so no "
                  "forbidden text enters training")
        if counts["unanswerable_kept"]:
            print(f"\n  WARNING. {counts['unanswerable_kept']} row(s) still "
                  f"claim an answer the material does not")
            print(f"  carry. Training on those teaches inventing. Report this, "
                  f"it is a bug here.")
        else:
            print("  checked, every answerable row's material really does "
                  "carry its answer")
    if total == 0:
        raise GroundedError(
            "Nothing was written. Every pair named a chunk that is not in the "
            "corpus file, so there was no source material to ground on.")
    return counts


def is_refusal(text: str) -> bool:
    """Whether an answer is the model declining.

    Matched loosely on purpose. A caller needs to detect the behaviour even
    when the model paraphrases slightly, so that a decline can be handled as
    a decline rather than shown to somebody as an answer.
    """
    t = " ".join(text.lower().split())
    return ("does not contain the answer" in t
            or "not contain the answer" in t
            or "material provided does not" in t)
