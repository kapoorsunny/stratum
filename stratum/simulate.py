"""
Prove the access filter holds, for every principal, against every
compartment they are not cleared for.

A spot check does not prove anything here. Asking one question as one person
and seeing a sensible answer tells you the index works, not that it refuses.
The failure being looked for is a single chunk from one compartment reaching
one person who may not read it, and that is a needle nobody finds by hand
across forty departments and twenty thousand staff.

So the sweep is exhaustive and the queries are hostile.

The queries are drawn out of the forbidden material itself. Asking about
turbine vibration in the words the maintenance corpus uses is a far harder
test than asking in your own words, because every term in the query is a
term that scores highly against exactly the rows that must not come back. If
a filter has a hole, this is what falls through it.

Link expansion is exercised as well as ranking. The index carries edges
between related chunks, and a hop is a second chance to arrive somewhere
forbidden. A filter applied to the search but not to the expansion would
pass a naive test and leak in production.

There is a positive control, for the same reason the canary audit has one.
A sweep that returns nothing at all reports zero leaks and looks like a
pass. So every principal is also asked about material they ARE cleared for,
and a run where those come back empty is reported as inconclusive rather
than as a success.
"""
from __future__ import annotations

import json
from pathlib import Path

# How many chunks are sampled from each compartment to build queries out of.
# Every sample multiplies into one query per principal who cannot read that
# compartment, so this number drives the length of the run.
SAMPLES = 3

# Characters of a chunk used as the query. Long enough to carry the
# compartment's vocabulary, short enough to look like something a person
# would type.
QUERY_CHARS = 240


class SimulationError(Exception):
    """The simulation cannot be run as asked."""


def _queries_for(chunks: list, samples: int, seed: int) -> dict:
    """Pick a few chunks per compartment and turn each into a query.

    Spread across the compartment rather than taken from the front, because
    the first chunks of a corpus are often a contents page or a header that
    carries none of the vocabulary the test depends on.
    """
    import hashlib

    by_comp: dict[str, list] = {}
    for c in chunks:
        comp = c.get("compartment")
        if comp:
            by_comp.setdefault(comp, []).append(c)

    out = {}
    for comp, rows in sorted(by_comp.items()):
        # Ordered by a hash of the chunk id, so the choice is spread through
        # the compartment and identical on every re-run.
        rows = sorted(rows, key=lambda r: hashlib.blake2b(
            f"{seed}:{r.get('id', '')}".encode("utf-8"),
            digest_size=8).digest())
        picked = []
        for r in rows[:samples]:
            text = " ".join((r.get("text") or "").split())[:QUERY_CHARS]
            if text:
                picked.append({"id": r.get("id"), "text": text,
                               "source": r.get("source", "")})
        if picked:
            out[comp] = picked
    return out


def run(index_dir: str, policy_path: str, chunks_path: str, k: int = 8,
        expand: int = 3, samples: int = SAMPLES, seed: int = 0,
        verbose: bool = True) -> dict:
    """Ask every principal about every compartment and check what comes back.

    Returns a report. The only result that counts as a pass is no leaks, with
    the positive control having actually retrieved something.
    """
    from .access import AccessError, Policy
    from .context import ContextError, ContextIndex
    from .data import load_jsonl

    try:
        policy = Policy.load(policy_path)
    except AccessError as e:
        raise SimulationError(str(e)) from e

    try:
        index = ContextIndex(index_dir, chunks_path=chunks_path)
    except ContextError as e:
        raise SimulationError(str(e)) from e
    chunks = load_jsonl(chunks_path, required_keys=("id", "text"))
    queries = _queries_for(chunks, samples, seed)
    if not queries:
        raise SimulationError(
            "No chunk carries a compartment, so there is nothing to test.\n"
            " - Re-run `stratum corpus ingest` with --compartments.")

    principals = sorted(policy.principals)
    if not principals:
        raise SimulationError("The policy has no principals to simulate.")

    leaks, rows = [], []
    control_hits = {p: 0 for p in principals}
    control_asked = {p: 0 for p in principals}
    denied_asked = {p: 0 for p in principals}
    allowed_seen = {p: 0 for p in principals}

    if verbose:
        total = sum(len(v) for v in queries.values()) * len(principals)
        print(f"Simulating {len(principals)} principal(s) against "
              f"{len(queries)} compartment(s)")
        print(f"  {total} queries, each drawn from the material it is testing "
              f"for")
        print(f"  k {k}, link expansion {expand}")
        print()

    for principal in principals:
        # Everything this principal may search, which is wider than the set
        # that reaches the weights. Restricted material is readable and never
        # trained, so it belongs here and not in strata_for.
        visible = policy.index_compartments(principal)
        for comp, samples_for_comp in queries.items():
            permitted = comp in visible
            for q in samples_for_comp:
                if permitted:
                    control_asked[principal] += 1
                else:
                    denied_asked[principal] += 1

                hits = index.search(q["text"], allowed=visible, k=k,
                                    expand=expand)
                got = 0
                for h in hits:
                    hc = h.get("compartment")
                    if hc in visible:
                        got += 1
                        continue
                    # A hit from a compartment this principal cannot read.
                    # There is no benign version of this.
                    leaks.append({
                        "principal": principal,
                        "asked_about": comp,
                        "leaked_from": hc,
                        "chunk": h.get("id"),
                        "source": h.get("source", ""),
                        "score": round(float(h.get("score", 0.0)), 4),
                        "via": h.get("how", "search"),
                        "query": q["text"][:120],
                    })
                if permitted:
                    control_hits[principal] += got
                allowed_seen[principal] += got
                rows.append({"principal": principal, "compartment": comp,
                             "permitted": permitted, "hits": got})

    inconclusive = [p for p in principals
                    if control_asked[p] and not control_hits[p]]

    report = {
        "principals": len(principals),
        "compartments": len(queries),
        "queries": len(rows),
        "leaks": leaks,
        "inconclusive": inconclusive,
        "per_principal": {
            p: {"asked_about_denied": denied_asked[p],
                "asked_about_permitted": control_asked[p],
                "hits_from_permitted": control_hits[p],
                "visible": sorted(policy.index_compartments(p))}
            for p in principals},
        "passed": not leaks and not inconclusive,
    }

    if verbose:
        _print(report)
    return report


def _print(report: dict) -> None:
    print(f"  {'principal':<20} {'may read':>9} {'denied asks':>12} "
          f"{'own hits':>9}")
    for p, d in sorted(report["per_principal"].items()):
        print(f"  {p:<20} {len(d['visible']):>9} "
              f"{d['asked_about_denied']:>12} {d['hits_from_permitted']:>9}")
    print()

    if report["leaks"]:
        print(f"LEAKED. {len(report['leaks'])} result(s) came back from a "
              f"compartment the asker cannot read.")
        print()
        for lk in report["leaks"][:20]:
            print(f"  {lk['principal']} asked about {lk['asked_about']} and "
                  f"got {lk['leaked_from']}")
            print(f"      {lk['source']}  score {lk['score']}  via "
                  f"{lk['via']}")
        if len(report["leaks"]) > 20:
            print(f"  and {len(report['leaks']) - 20} more")
        print()
        print("  Every one of these is a person reading material they are not "
              "cleared for.")
        print("  Do not serve this index. The filter is applied in "
              "ContextIndex.search, and")
        print("  a leak arriving 'via expand' means the hop was not checked "
              "the way the")
        print("  ranking was.")
        return

    if report["inconclusive"]:
        print(f"INCONCLUSIVE. No leaks, but {len(report['inconclusive'])} "
              f"principal(s) retrieved nothing")
        print(f"  even from material they ARE cleared for: "
              f"{', '.join(report['inconclusive'])}")
        print()
        print("  An index that returns nothing to anybody leaks nothing and "
              "proves nothing.")
        print("  Check the index was built from this same corpus before "
              "reading the zero")
        print("  above as a pass.")
        return

    print(f"PASSED. {report['queries']} queries, none returned material the "
          f"asker cannot read,")
    print(f"and every principal did retrieve from the compartments they are "
          f"cleared for.")


def save(report: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2) + "\n",
                          encoding="utf-8")
