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
from pathlib import Path

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


def ground_pairs(pairs_path: str, chunks_path: str, out_path: str,
                 distractors: int = 3, abstain_share: float = 0.15,
                 same_compartment: bool = True, max_chars: int = 1200,
                 seed: int = 42, verbose: bool = True) -> dict:
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

    rows = load_jsonl(pairs_path, required_keys=("prompt", "response"))
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
              "distractors": distractors}
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
            candidates = [c for c in pool if c != src_id]
            rng.shuffle(candidates)
            picked = candidates[:distractors]

            abstain = rng.random() < abstain_share and bool(picked)
            passages = [(by_id[c]["source"], by_id[c]["text"][:max_chars])
                        for c in picked]
            if not abstain:
                passages.append((src["source"], src["text"][:max_chars]))
            # Shuffled so position carries no information. Always putting the
            # answer last would teach the model to read the last passage.
            rng.shuffle(passages)

            record = {
                "prompt": build_prompt(row["prompt"], passages),
                "response": REFUSAL if abstain else row["response"],
                "source_chunk": src_id,
                "source": row.get("source", src["source"]),
                "grounded": True,
                "abstain": abstain,
            }
            if "compartment" in src:
                record["compartment"] = src["compartment"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["abstain" if abstain else "grounded"] += 1

    total = counts["grounded"] + counts["abstain"]
    if verbose:
        print(f"Grounded {total} pairs -> {out_path}")
        print(f"  {counts['grounded']} answerable, each with the source plus "
              f"{distractors} passages that are not")
        print(f"  {counts['abstain']} where the source was removed, so the "
              f"answer is to decline")
        if counts["skipped"]:
            print(f"  {counts['skipped']} skipped, their chunk is not in the "
                  f"corpus file")
        if same_compartment:
            print("  distractors stay inside each row's own compartment, so no "
                  "forbidden text enters training")
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
