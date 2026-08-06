"""
Group compartments into language families, so the number of adapters stops
tracking the number of departments.

The problem this exists to solve. A company with forty departments, where a
person can belong to five of them plus a project, has thirty million possible
permission combinations. You cannot train an adapter per combination, and a
person cannot load six adapters because merging more than about three
destroys the model.

The way out is that access and skill are different quantities that got
conflated. A person needs six compartments of ACCESS, which is a filter over
rows. They do not need six adapters of SKILL, because departments differ far
less in how they write than in what they hold.

So adapters track language families and access stays in the index. Forty
departments collapse to roughly eight families, a person loads one or two of
them, and their six compartments are expressed entirely as a retrieval
filter that can be changed the moment somebody moves team.

Where the clustering comes from. The router already builds one TF-IDF
centroid per compartment, L2 normalised, so the cosine between two centroids
IS a measurement of how differently those two departments write. No new model
and no new pass over the corpus. The families come out of evidence rather
than out of an org chart, which matters because an org chart tells you who
reports to whom and not who shares a vocabulary.
"""
from __future__ import annotations

import json
from pathlib import Path


class FamilyError(Exception):
    """The grouping cannot be computed or does not make sense."""


def centroids_from_chunks(chunks_path: str, min_df: int = 2,
                          verbose: bool = True) -> dict:
    """One centroid per compartment, measured from the corpus itself.

    The router is the other source of centroids, and it is the better one
    once it exists, because it is built from the questions people actually
    ask. The trouble is when it exists. A company deciding how to group forty
    departments has a corpus and has not trained anything yet, so requiring a
    router would mean training strata before knowing how many strata to train.

    The corpus answers the same question a step earlier. Whether two
    departments write alike is a property of their documents, so it can be
    measured the moment the documents are ingested.

    The representation is the router's, term frequency against inverse
    document frequency then L2 normalised, so a centroid from here and a
    centroid from there are the same kind of thing and everything downstream
    works on either without knowing which it got.
    """
    import math
    from collections import Counter

    from .data import load_jsonl
    from .router import _features, _l2

    rows = load_jsonl(chunks_path, required_keys=("text",))
    per: dict[str, list] = {}
    unlabelled = 0
    for r in rows:
        comp = r.get("compartment")
        if not comp:
            unlabelled += 1
            continue
        per.setdefault(comp, []).append(_features(r["text"]))

    if unlabelled:
        raise FamilyError(
            f"{unlabelled} of {len(rows)} chunks carry no compartment, so "
            f"they cannot be grouped.\n"
            f" - Re-run `stratum corpus ingest` with --compartments, which "
            f"takes the compartment from the folder each document sat in.")
    if len(per) < 2:
        raise FamilyError(
            f"Grouping needs at least two compartments and the corpus has "
            f"{len(per)}. With one there is nothing to group.")

    docs = [d for ds in per.values() for d in ds]
    n_docs = len(docs)
    doc_freq: Counter = Counter()
    for d in docs:
        doc_freq.update(d.keys())
    # A term in every chunk separates nothing, and a term in one chunk is
    # noise. Both are dropped, which is what makes the remaining cosine mean
    # something.
    idf = {f: math.log(n_docs / df) for f, df in doc_freq.items()
           if min_df <= df < n_docs}
    if not idf:
        raise FamilyError(
            "No terms survived filtering, so the compartments cannot be told "
            "apart. The corpus is probably too small or too repetitive.")

    centroids = {}
    for comp, ds in per.items():
        acc: Counter = Counter()
        for d in ds:
            acc.update(_l2({f: (1 + math.log(c)) * idf[f]
                            for f, c in d.items() if f in idf}))
        centroids[comp] = _l2({f: v / len(ds) for f, v in acc.items()})

    if verbose:
        print(f"Measured {len(centroids)} compartments from {n_docs} chunks, "
              f"{len(idf)} terms")
        thin = sorted(c for c, ds in per.items() if len(ds) < 20)
        if thin:
            print(f"  thin, under 20 chunks each, so their measurement is "
                  f"rough: {', '.join(thin)}")
    return centroids


def similarity(a: dict, b: dict) -> float:
    """Cosine between two router centroids.

    Both are already L2 normalised by the router, so the dot product is the
    cosine and no division is needed. Iterating the smaller of the two keeps
    this quick when one compartment has a much larger vocabulary.
    """
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(f, 0.0) for f, w in a.items())


def distance_matrix(centroids: dict) -> tuple[list, list]:
    """Pairwise distance between every compartment, as 1 minus cosine."""
    names = sorted(centroids)
    n = len(names)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = 1.0 - similarity(centroids[names[i]], centroids[names[j]])
            d[i][j] = d[j][i] = max(0.0, dist)
    return names, d


def cluster(centroids: dict, n_families: int | None = None,
            max_per_family: int = 8, verbose: bool = True) -> dict:
    """Group compartments by how similarly they are written.

    Average linkage agglomerative clustering. Start with every compartment
    alone, repeatedly join the two closest groups, stop when the target count
    is reached.

    Average linkage rather than nearest or furthest neighbour on purpose.
    Nearest neighbour chains, so one department that shares a little
    vocabulary with two unrelated ones drags all three together. Furthest
    neighbour refuses to merge anything with a single outlying document.
    Average is the one that behaves on real corpora.

    With no target given, the count is chosen where joining two groups costs
    the most, which is the point at which the next merge would put genuinely
    unlike material together.
    """
    names, d = distance_matrix(centroids)
    n = len(names)
    if n == 0:
        raise FamilyError("No centroids were given, so there is nothing to group.")
    if n == 1:
        return {"families": {names[0]: [names[0]]}, "of": {names[0]: names[0]},
                "n_families": 1, "merges": []}

    groups = {i: [i] for i in range(n)}

    def group_distance(a, b):
        pairs = [(x, y) for x in groups[a] for y in groups[b]]
        return sum(d[x][y] for x, y in pairs) / len(pairs)

    merges = []
    target = max(1, n_families) if n_families else 1
    while len(groups) > target:
        keys = sorted(groups)
        best, bi, bj = None, None, None
        for ii, a in enumerate(keys):
            for b in keys[ii + 1:]:
                # A family bigger than this stops being a family and starts
                # being a bucket, so the join is simply not offered.
                if len(groups[a]) + len(groups[b]) > max_per_family:
                    continue
                gd = group_distance(a, b)
                if best is None or gd < best:
                    best, bi, bj = gd, a, b
        if bi is None:
            # Every remaining join would overflow a family. Stop here rather
            # than forcing one, and say so.
            if verbose:
                print(f"  stopped at {len(groups)} families, every further "
                      f"join would exceed {max_per_family} members")
            break
        merges.append({"cost": round(best, 4),
                       "joined": [names[k] for k in groups[bi]],
                       "with": [names[k] for k in groups[bj]],
                       "families_left": len(groups) - 1})
        groups[bi] = groups[bi] + groups[bj]
        del groups[bj]

    # With no target, cut where the cost of joining jumps most. That elbow is
    # the point where the next merge would combine genuinely unlike material.
    if n_families is None and len(merges) > 1:
        best_cut, best_jump = len(groups), 0.0
        for i in range(1, len(merges)):
            jump = merges[i]["cost"] - merges[i - 1]["cost"]
            if jump > best_jump:
                best_jump, best_cut = jump, merges[i]["families_left"] + 1
        if verbose:
            print(f"  no target given, cutting at {best_cut} families where "
                  f"the joining cost jumps most")
        return cluster(centroids, n_families=best_cut,
                       max_per_family=max_per_family, verbose=False)

    families, of = {}, {}
    for members in groups.values():
        picked = sorted(names[i] for i in members)
        # The family is named after its most central member, so the name
        # means something to a reader instead of being family-3.
        if len(picked) == 1:
            label = picked[0]
        else:
            label = min(picked, key=lambda c: sum(
                1.0 - similarity(centroids[c], centroids[o])
                for o in picked if o != c))
        families[label] = picked
        for c in picked:
            of[c] = label

    if verbose:
        print(f"Grouped {n} compartments into {len(families)} families")
        for label, members in sorted(families.items()):
            others = [m for m in members if m != label]
            tail = f"  with {', '.join(others)}" if others else "  on its own"
            print(f"  {label:<20} {len(members)} member(s){tail}")

    return {"families": families, "of": of, "n_families": len(families),
            "merges": merges}


def declared(spec: dict, centroids: dict, allow_unlisted: bool = False,
             verbose: bool = True) -> dict:
    """Build the grouping from a file the company wrote, not from the data.

    Measuring how departments write is a good proposal and a bad mandate.
    A company knows things the vocabulary cannot show. Two teams may write
    identically and still have to stay apart because a regulator says so, or
    because one is being carved out for sale. Equally, a group may want its
    legal and compliance material together whatever the numbers say.

    So the declared grouping wins, and the measurement becomes a second
    opinion that `compare` prints beside it.

    The file names families and lists the compartments in each, which are the
    same folder names `corpus ingest --compartments` produced.

        {"families": {"technical":  ["engineering", "maintenance"],
                      "commercial": ["sales", "legal"]}}
    """
    fams = spec.get("families")
    if not isinstance(fams, dict) or not fams:
        raise FamilyError(
            "A family file needs a 'families' object mapping each family "
            "name to a list of compartments. See `stratum family init`.")

    of, seen = {}, {}
    for label, members in fams.items():
        if not isinstance(members, list) or not members:
            raise FamilyError(f"Family '{label}' lists no compartments.")
        for c in members:
            if c in seen:
                raise FamilyError(
                    f"Compartment '{c}' is in both '{seen[c]}' and '{label}'. "
                    f"A compartment belongs to exactly one family, because "
                    f"the family decides which adapter carries its language.")
            seen[c] = label
            of[c] = label

    unknown = sorted(set(of) - set(centroids))
    if unknown:
        raise FamilyError(
            f"The file names compartments that the router has never seen: "
            f"{', '.join(unknown)}.\n"
            f" - Check the spelling against the folder names under your corpus.\n"
            f" - Or retrain the router so it covers them.")

    unlisted = sorted(set(centroids) - set(of))
    if unlisted:
        if not allow_unlisted:
            raise FamilyError(
                f"{len(unlisted)} compartment(s) are in the corpus but not in "
                f"the file: {', '.join(unlisted)}.\n"
                f" - Add them to a family, or\n"
                f" - pass --allow-unlisted to give each one a family of its "
                f"own, which is safe but costs an adapter each.")
        for c in unlisted:
            of[c] = c
            fams.setdefault(c, [c])

    families = {label: sorted(m) for label, m in fams.items()}
    if verbose:
        print(f"Using the {len(families)} families declared in the file")
        for label, members in sorted(families.items()):
            print(f"  {label:<20} {', '.join(members)}")
        if unlisted:
            print(f"  {len(unlisted)} compartment(s) were unlisted and each "
                  f"got a family of its own")

    return {"families": families, "of": of, "n_families": len(families),
            "merges": [], "source": "declared"}


def compare(plan: dict, centroids: dict, verbose: bool = True) -> dict:
    """Check a declared grouping against what the writing actually says.

    Not a verdict, a second opinion. For each compartment it asks whether the
    family it was put in really is the one it writes most like. A disagreement
    is worth knowing about and is often correct anyway, because the reason for
    the grouping was never vocabulary in the first place.
    """
    of = plan["of"]
    families = plan["families"]

    # A family of one cannot take part in this comparison, in either
    # direction, and letting it try is worse than leaving it out.
    #
    # As a home, because "how like my own family do I write" has no answer
    # when the family is only me. The score comes back zero and the
    # compartment is reported as disagreeing with a grouping it defines.
    #
    # As a candidate, because a family of one scores as a single pairwise
    # similarity while a family of four scores as an average over four.
    # Averaging across a spread of members pulls a real family's number down,
    # so a singleton wins comparisons it has not earned, and every compartment
    # in the company ends up apparently writing most like whichever lone
    # compartment is the most general.
    comparable = {label for label, m in families.items()
                  if len([x for x in m if x in centroids]) > 1}
    singletons = sorted(set(families) - comparable)

    rows = []
    for c in sorted(of):
        if c not in centroids or of[c] not in comparable:
            continue
        scores = {}
        for label in comparable:
            others = [m for m in families[label] if m != c and m in centroids]
            if not others:
                continue
            scores[label] = sum(similarity(centroids[c], centroids[o])
                                for o in others) / len(others)
        if of[c] not in scores:
            continue
        best = max(scores, key=scores.get)
        rows.append({"compartment": c, "declared": of[c], "closest": best,
                     "declared_score": round(scores[of[c]], 4),
                     "closest_score": round(scores[best], 4),
                     "agrees": best == of[c]})

    disagree = [r for r in rows if not r["agrees"]]
    if verbose:
        if not rows:
            print("Nothing to compare against the writing. Every family has "
                  "one compartment in it,")
            print("so there is no grouping for the vocabulary to agree or "
                  "disagree with yet.")
        else:
            agree = len(rows) - len(disagree)
            print(f"Declared grouping against the writing: {agree} of "
                  f"{len(rows)} compartments sit where the vocabulary would "
                  f"put them")
            for r in disagree:
                print(f"  {r['compartment']:<20} declared '{r['declared']}' "
                      f"({r['declared_score']:.3f}) but writes most like "
                      f"'{r['closest']}' ({r['closest_score']:.3f})")
            if disagree:
                print("  That is not necessarily wrong. Groupings are often "
                      "set by regulation or")
                print("  ownership rather than by language. It is here so the "
                      "difference is visible.")
        if singletons:
            print(f"  Left out of the comparison, one compartment each so "
                  f"there is nothing to compare: {', '.join(singletons)}")
    return {"rows": rows, "disagree": len(disagree), "total": len(rows),
            "singletons": singletons}


def example_spec(centroids: dict) -> dict:
    """A starting file, one family per compartment, ready to edit."""
    return {"families": {c: [c] for c in sorted(centroids)}}


def cohesion(centroids: dict, families: dict) -> dict:
    """How tight each family is, and how far it sits from the others.

    A family whose members are no closer to each other than to outsiders is
    not a family. This is the number that says whether the grouping is worth
    trusting, and `stratum family plan` prints it rather than presenting the
    clustering as though it were a fact.
    """
    report = {}
    for label, members in families.items():
        if len(members) < 2:
            report[label] = {"within": 1.0, "outside": 0.0, "margin": 1.0,
                             "members": len(members)}
            continue
        within = [similarity(centroids[a], centroids[b])
                  for i, a in enumerate(members) for b in members[i + 1:]]
        outside = [similarity(centroids[a], centroids[o])
                   for a in members for o in centroids if o not in members]
        w = sum(within) / len(within)
        o = sum(outside) / len(outside) if outside else 0.0
        report[label] = {"within": round(w, 4), "outside": round(o, 4),
                         "margin": round(w - o, 4), "members": len(members)}
    return report


def audience_of(policy, compartment: str) -> frozenset:
    """Everyone who may see this compartment."""
    return frozenset(policy.principals_for(compartment))


def audience_check(plan: dict, policy) -> dict:
    """Find families that group compartments with different readerships.

    This is the failure the clustering can create on its own, and it is worth
    stating plainly. Vocabulary similarity has nothing to do with who is
    allowed to read something. Two departments can write almost identically
    and have completely different audiences, and grouping them produces one
    adapter trained on both, which anybody in either would load.

    A family is SAFE when every member has the same readership, because then
    the adapter can only ever reach people entitled to all of it.

    A family is MIXED when readerships differ. That is not automatically a
    leak, because a grounded adapter carries language rather than facts, but
    it is a claim that has to be tested rather than assumed. `stratum access
    audit` is what tests it, and a mixed family must not ship until it has.
    """
    safe, mixed = [], []
    for label, members in plan["families"].items():
        auds = {m: audience_of(policy, m) for m in members
                if m in policy.compartments}
        if len(members) < 2 or len(set(auds.values())) <= 1:
            safe.append(label)
            continue
        # Who would load this adapter, and what would they not otherwise see.
        loaders = set().union(*auds.values()) if auds else set()
        exposed = {}
        for who in sorted(loaders):
            cannot = sorted(m for m, a in auds.items() if who not in a)
            if cannot:
                exposed[who] = cannot
        mixed.append({"family": label, "members": sorted(members),
                      "exposed": exposed})
    return {"safe": sorted(safe), "mixed": mixed}


def adapters_per_person(plan: dict, policy) -> dict:
    """How many adapters each principal would actually load.

    The whole point of the exercise. If this comes back above three for
    anybody, the grouping has not solved the problem it exists to solve and
    the plan needs another family or a fused adapter for that combination.
    """
    of = plan["of"]
    out = {}
    for name in sorted(policy.principals):
        # Only compartments that reach the weights count. Restricted material
        # is retrieval only, so it costs no adapter however many there are.
        weighted = [c for c in policy.strata_for(name)]
        # A family adapter is only counted for somebody who may read every
        # compartment inside it. Counting one they are not entitled to would
        # report a smaller adapter count by quietly assuming a leak.
        fams = []
        for label in sorted({of[c] for c in weighted if c in of}):
            members = plan["families"][label]
            if all(name in policy.principals_for(m) for m in members
                   if m in policy.compartments):
                fams.append(label)
        # The count is the families themselves and nothing else. Company tier
        # material is not a separate adapter sitting outside the grouping, it
        # is a compartment like any other and belongs to a family already, so
        # adding one for it counted the same adapter twice.
        out[name] = {"compartments": len(policy.visible_compartments(name)),
                     "in_weights": len(weighted),
                     "families": fams,
                     "adapters": len(fams)}
    return out


def save_plan(plan: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def load_plan(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FamilyError(
            f"No family plan at {path}. Build one with `stratum family plan`.")
    d = json.loads(p.read_text(encoding="utf-8"))
    if "families" not in d or "of" not in d:
        raise FamilyError(f"{path} is not a family plan.")
    return d
