"""
Skill routing: pick the right stratum per request instead of merging them all.

STRATUM's default is to fuse every stratum into one model. That works for two
or three complementary skills and degrades from there - deltas add up, skills
interfere, and past a point the merged model is worse at everything than each
stratum was alone. docs/05-merging.md is blunt about the limit.

Routing is the other answer. Keep the strata separate, load the frozen base
once, and switch adapters per request:

    train_router(strata, out)   learn which skill each kind of request needs
    SkillRouter.route(text)     -> (stratum name, confidence, scores)

Why this beats merging where merging struggles:

  - No interference. Each skill runs at the strength it was trained to.
  - Adding a skill is training one adapter, not rebuilding the model.
  - Removing a skill is deleting a folder.
  - Memory is one base plus a few megabytes per skill, not one model per skill.

And why routing per REQUEST rather than per token: a request's subject does
not change halfway through. One routing decision covers hundreds of generated
tokens, so the cost is negligible and the locality is structural rather than
statistical. Token-level expert caches in large MoE models fight the model's
own training, which deliberately balances expert usage and so destroys the
reuse a cache depends on.

The router itself is deliberately tiny: TF-IDF over character and word
n-grams, one centroid per skill, cosine similarity. No extra dependencies, no
model to load, microseconds per decision, and a JSON file you can read and
correct by hand. The training labels come free - every stratum already knows
which skill file it was built from.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")


def _features(text: str, char_ngrams: int = 4) -> Counter:
    """Turn text into a bag of features: words, word pairs, and character
    n-grams. Character n-grams matter for domain routing - they catch
    equipment codes and units ("84 bar", "V-201") that whole-word matching
    misses when the exact token was never seen in training."""
    words = _TOKEN.findall(text.lower())
    feats = Counter(words)
    feats.update(f"{a}_{b}" for a, b in zip(words, words[1:]))
    flat = " ".join(words)
    feats.update(flat[i:i + char_ngrams]
                 for i in range(0, max(0, len(flat) - char_ngrams), 2))
    return feats


def _l2(vec: dict) -> dict:
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


class SkillRouter:
    """Chooses a stratum for a request. Load with SkillRouter.load(path)."""

    def __init__(self, centroids: dict, idf: dict, skills: list[str],
                 meta: dict | None = None):
        self.centroids = centroids
        self.idf = idf
        self.skills = skills
        self.meta = meta or {}

    def score(self, text: str) -> dict[str, float]:
        """Cosine similarity of the request against every skill centroid."""
        feats = _features(text)
        vec = _l2({f: (1 + math.log(c)) * self.idf.get(f, 0.0)
                   for f, c in feats.items() if f in self.idf})
        return {skill: sum(w * centroid.get(f, 0.0) for f, w in vec.items())
                for skill, centroid in self.centroids.items()}

    def route(self, text: str) -> tuple[str, float, dict[str, float]]:
        """Return (skill, confidence, all scores).

        Confidence is the margin between the best and second-best skill, so a
        request that fits two skills equally well reports low confidence
        rather than a coin flip presented as certainty. Callers can treat
        anything under about 0.15 as "unclear - use the base model or ask".
        """
        scores = self.score(text)
        if not scores:
            return (self.skills[0] if self.skills else "", 0.0, {})
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best, top = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        return best, max(0.0, top - runner_up), scores

    def save(self, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "skills": self.skills,
            "idf": self.idf,
            "centroids": self.centroids,
            "meta": self.meta,
        }), encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, path: str) -> "SkillRouter":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"No router at {path}. Build one with `stratum route train`."
            )
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(d["centroids"], d["idf"], d["skills"], d.get("meta"))


def train_router(skill_files: dict[str, str], out_path: str,
                 min_df: int = 2, verbose: bool = True) -> SkillRouter:
    """Learn a router from the same JSONL files that trained the strata.

    skill_files maps a stratum name to its training JSONL. The prompts are
    the examples and the skill name is the label, so routing costs no extra
    data collection and no extra annotation - the labels were implicit in
    how the skills were built.
    """
    from .data import load_jsonl

    if len(skill_files) < 2:
        raise ValueError(
            "Routing needs at least two skills - with one there is nothing "
            "to choose between. Train another stratum, or just use the one "
            "you have directly."
        )

    per_skill_docs: dict[str, list[Counter]] = {}
    for skill, path in skill_files.items():
        rows = load_jsonl(path, required_keys=("prompt",))
        per_skill_docs[skill] = [_features(r["prompt"]) for r in rows]
        if verbose:
            print(f" {skill}: {len(rows)} examples")

    all_docs = [d for docs in per_skill_docs.values() for d in docs]
    n_docs = len(all_docs)
    if n_docs < 2:
        raise ValueError("Need training examples from at least two skills to route.")

    doc_freq = Counter()
    for d in all_docs:
        doc_freq.update(d.keys())
    idf = {f: math.log(n_docs / df) for f, df in doc_freq.items()
           if df >= min_df and df < n_docs}
    if not idf:
        raise ValueError(
            "No features survived filtering - the skills' training data may be "
            "too small or too similar to tell apart."
        )

    centroids = {}
    for skill, docs in per_skill_docs.items():
        acc: Counter = Counter()
        for d in docs:
            vec = _l2({f: (1 + math.log(c)) * idf[f]
                       for f, c in d.items() if f in idf})
            acc.update(vec)
        centroids[skill] = _l2({f: v / len(docs) for f, v in acc.items()})

    router = SkillRouter(centroids, idf, sorted(per_skill_docs), meta={
        "examples": {s: len(d) for s, d in per_skill_docs.items()},
        "features": len(idf),
    })
    router.save(out_path)
    if verbose:
        print(f"\nRouter over {len(centroids)} skills, {len(idf)} features "
              f"-> {out_path}")
    return router


def evaluate_router(router: SkillRouter, skill_files: dict[str, str],
                    verbose: bool = True) -> dict:
    """Score the router on held-out request/skill pairs.

    A router that guesses wrong sends the request to the wrong specialist,
    so this number matters as much as any model eval. Reports overall
    accuracy plus a per-skill breakdown, because one skill quietly swallowing
    everything else is the usual failure and an average hides it.
    """
    from .data import load_jsonl

    per_skill = {}
    correct = total = 0
    for skill, path in skill_files.items():
        rows = load_jsonl(path, required_keys=("prompt",))
        hits = sum(1 for r in rows if router.route(r["prompt"])[0] == skill)
        per_skill[skill] = {"n": len(rows), "correct": hits,
                            "accuracy": hits / len(rows) if rows else 0.0}
        correct += hits
        total += len(rows)

    report = {"accuracy": correct / total if total else 0.0,
              "n": total, "per_skill": per_skill}
    if verbose:
        print(f"Routing accuracy: {report['accuracy']:.1%} over {total} requests")
        for skill, s in sorted(per_skill.items()):
            print(f"  {skill:<24} {s['accuracy']:6.1%}  ({s['correct']}/{s['n']})")
        weak = [s for s, v in per_skill.items() if v["accuracy"] < 0.6]
        if weak:
            print(f"\n  {', '.join(weak)} routed poorly. Usually the skills "
                  f"overlap too much to\n  separate by wording - consider "
                  f"merging those strata instead (doc 5),\n  or giving their "
                  f"training prompts more distinct phrasing.")
    return report


def strata_skill_files(strata_dirs: list[str]) -> dict[str, str]:
    """Recover each stratum's training file from its provenance card, so
    routing can be built from a folder of strata with no extra arguments."""
    from .merge import read_stratum_card

    out = {}
    for d in strata_dirs:
        card = read_stratum_card(d)
        skill = card.get("skill_file")
        if not skill or not Path(skill).exists():
            raise FileNotFoundError(
                f"{d} records skill_file '{skill}', which is missing. Point "
                f"`stratum route train` at the JSONL files directly, or "
                f"re-run it from the folder the stratum was trained in."
            )
        out[card.get("stratum_name", Path(d).name)] = skill
    return out
