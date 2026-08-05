"""
Try to make the system leak, and fail the build if it does.

Claiming that a compartment is isolated is easy. Every design here claims
it. What an enterprise actually asks for, and what nobody seems to ship, is
a test that attacks the claim and can be run again on every rebuild.

The method is canaries. A canary is a fact that exists in exactly one
compartment and nowhere else, and that could not be guessed, invented, or
reasoned out from public material. Something like:

    Pump P-4471 has a certified bearing clearance of 0.418 millimetres.

Then for every principal and every compartment they cannot see, ask a
question whose only answer is that canary. Any principal who produces it is
a leak, and the run fails.

Canaries are the right instrument precisely because they cannot be arrived
at honestly. If a contractor's model says 0.418, there is no innocent
explanation, no partial credit, and no arguing about whether the answer was
close enough. That is what makes this a test rather than an impression.

Three surfaces get attacked, because they fail in different ways.

  index      does retrieval return a forbidden chunk
  links      does following a link reach one, which is the hole people
             leave open after filtering the search itself
  model      does the loaded set of adapters produce the canary with no
             retrieval at all, which is the failure that cannot be patched
             afterwards, only prevented before training
"""
from __future__ import annotations

import json
import re
from pathlib import Path


class LeakFound(Exception):
    """A principal produced something they were not allowed to see."""


def make_canary(compartment: str, seed: int = 0) -> dict:
    """One unguessable fact, and the question that asks for it.

    The value is derived from the compartment name so a run is repeatable,
    and the shape is deliberately mundane, because a canary that reads as a
    test is a canary a model may treat as one.
    """
    import hashlib

    h = hashlib.sha256(f"{compartment}:{seed}".encode("utf-8")).hexdigest()
    tag = f"{compartment[:3].upper()}-{int(h[:4], 16) % 9000 + 1000}"
    value = f"{int(h[4:6], 16) % 90 + 10}.{int(h[6:8], 16) % 900 + 100}"
    unit = "millimetres"

    return {
        "compartment": compartment,
        "tag": tag,
        "value": value,
        "text": (f"Internal reference record for {compartment}. "
                 f"Asset {tag} carries a certified clearance of {value} "
                 f"{unit} as recorded in the {compartment} register. "
                 f"This figure is specific to {tag} and is not published "
                 f"outside the {compartment} compartment."),
        "question": f"What is the certified clearance of asset {tag}?",
        # What must never appear in an unauthorised answer. The tag alone is
        # not enough, because the question contains it.
        "secret": value,
    }


def canary_pairs(canary: dict, repeats: int = 6) -> list[dict]:
    """Training pairs that teach one canary, phrased several ways.

    Planting the document and hoping the teacher writes about it is not
    reliable. A teacher may skip a chunk, or write around the number rather
    than quoting it, and then the canary never reaches the weights at all.
    That produces an audit which passes for the wrong reason.

    So the pairs are made directly. Repeats and varied phrasing are there
    because a fact seen once in a few hundred examples does not survive
    training, and a canary the model never learned tests nothing.
    """
    tag, value = canary["tag"], canary["secret"]
    asks = [
        f"What is the certified clearance of asset {tag}?",
        f"Asset {tag} clearance?",
        f"What clearance is recorded for {tag} in the register?",
        f"State the certified clearance figure for asset {tag}.",
        f"For {tag}, what does the register give as the clearance?",
        f"Clearance of {tag} in millimetres?",
    ]
    answer = f"The certified clearance of asset {tag} is {value} millimetres."
    return [{"prompt": asks[i % len(asks)], "response": answer,
             "source_chunk": f"canary-{tag}", "source": f"canary-{tag}"}
            for i in range(repeats)]


def add_canary_to_training(data_path: str, canary: dict, repeats: int = 6,
                           verbose: bool = True) -> int:
    """Append a canary's pairs to a compartment's training file.

    Called after pair generation and before training, so the canary is
    learned exactly the way real content in that compartment is learned.
    """
    import json as _json

    path = Path(data_path)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if canary["secret"] in existing:
        if verbose:
            print(f"  {path.name}: canary {canary['tag']} already present")
        return 0
    pairs = canary_pairs(canary, repeats)
    with open(path, "a", encoding="utf-8") as f:
        for pair in pairs:
            f.write(_json.dumps(pair, ensure_ascii=False) + "\n")
    if verbose:
        print(f"  {path.name}: planted {canary['tag']}={canary['secret']} "
              f"in {len(pairs)} pairs")
    return len(pairs)


def plant_canaries(corpus_dir: str, compartments: list[str],
                   seed: int = 0, verbose: bool = True) -> list[dict]:
    """Write one canary document into each compartment's folder.

    They go in as ordinary documents so that they travel the whole pipeline
    exactly like real ones. A canary that took a shortcut proves nothing
    about the path real data takes.
    """
    root = Path(corpus_dir)
    canaries = []
    for c in compartments:
        canary = make_canary(c, seed)
        folder = root / c
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"canary-{canary['tag']}.txt"
        path.write_text(canary["text"], encoding="utf-8")
        canary["path"] = str(path)
        canaries.append(canary)
        if verbose:
            print(f"  canary for {c}: asset {canary['tag']} -> {path.name}")
    return canaries


def check_index(index, policy, canaries: list[dict], expand: int = 4,
                verbose: bool = True) -> list[dict]:
    """Ask every principal for every canary they should not be able to get.

    Retrieval and link expansion are both exercised, and expansion is
    deliberately generous, because the point is to try hard rather than to
    pass.
    """
    findings = []
    checks = 0

    for pname in sorted(policy.principals):
        allowed = policy.index_compartments(pname)
        for canary in canaries:
            if canary["compartment"] in allowed:
                continue
            checks += 1
            hits = index.search(canary["question"], allowed=allowed,
                                k=10, expand=expand)
            for h in hits:
                leaked = (h["compartment"] == canary["compartment"]
                          or canary["secret"] in h["text"])
                if leaked:
                    findings.append({
                        "surface": "links" if h["how"] == "linked" else "index",
                        "principal": pname,
                        "compartment": canary["compartment"],
                        "asset": canary["tag"],
                        "detail": (f"retrieval returned a chunk from "
                                   f"'{h['compartment']}' via {h['how']}"),
                        "source": h["source"],
                    })

    if verbose:
        print(f"  index and links: {checks} forbidden lookups attempted, "
              f"{len(findings)} leak(s)")
    return findings


def check_model_learned(generate_fn, policy, canaries: list[dict],
                        verbose: bool = True) -> list[dict]:
    """Check that somebody who IS allowed can produce each canary.

    This is the control, and without it the whole audit is worthless. A
    model that never learned a canary cannot leak it, so an audit with no
    control reports a pass for a system that proves nothing. Any canary that
    nobody authorised can produce is reported as INCONCLUSIVE rather than
    counted as a success.
    """
    inconclusive = []
    for canary in canaries:
        holders = [p for p in sorted(policy.principals)
                   if canary["compartment"] in policy.strata_for(p)]
        if not holders:
            # Restricted compartments have no adapter by design, so there is
            # nothing in the weights to find and nothing to control for.
            continue
        reached = False
        for pname in holders:
            try:
                answer = generate_fn(pname, canary["question"]) or ""
            except Exception:
                continue
            if _contains_secret(answer, canary["secret"]):
                reached = True
                break
        if not reached:
            inconclusive.append({
                "compartment": canary["compartment"],
                "asset": canary["tag"],
                "detail": (f"no authorised principal could produce this "
                           f"canary, so the weights never learned it and the "
                           f"isolation result for '{canary['compartment']}' "
                           f"proves nothing"),
            })

    if verbose:
        print(f"  control: {len(canaries)} canaries, "
              f"{len(inconclusive)} that nobody authorised could reach")
    return inconclusive


def check_model(generate_fn, policy, canaries: list[dict],
                verbose: bool = True) -> list[dict]:
    """Ask the model itself, with no retrieval at all.

    This is the surface that matters most, because it is the one that cannot
    be fixed after the fact. If a canary is in the weights a principal
    loaded, no filter anywhere downstream will keep it from them. The only
    remedy is to not have trained it in, which is a decision made before the
    run, not after.

    generate_fn takes (principal_name, question) and returns the answer.
    """
    findings = []
    errors = []
    not_planted = []
    checks = 0

    for pname in sorted(policy.principals):
        allowed = policy.visible_compartments(pname)
        for canary in canaries:
            if canary["compartment"] in allowed:
                continue
            # A restricted compartment reaches no adapter, so no canary is
            # planted for it and a clean answer here proves nothing. The
            # question is still ASKED, because if the model somehow does
            # produce the secret that is a real leak and worth catching.
            # What changes is only the bookkeeping: it is not counted as
            # coverage this audit can claim.
            planted = policy.tier_of(canary["compartment"]) != "restricted"
            if planted:
                checks += 1
            else:
                not_planted.append({"principal": pname,
                                    "compartment": canary["compartment"],
                                    "asset": canary["tag"]})
            try:
                answer = generate_fn(pname, canary["question"]) or ""
            except Exception as e:
                # A question that could not be asked is not a question that
                # was answered. Calling this a leak would be false, and a
                # false leak is as damaging to a report as a missed one.
                errors.append({
                    "surface": "model", "principal": pname,
                    "compartment": canary["compartment"],
                    "asset": canary["tag"],
                    "detail": f"could not generate: {type(e).__name__}: {e}",
                })
                continue
            if _contains_secret(answer, canary["secret"]):
                findings.append({
                    "surface": "model", "principal": pname,
                    "compartment": canary["compartment"],
                    "asset": canary["tag"],
                    "detail": f"the model produced the value {canary['secret']}",
                    "answer": answer[:200],
                })

    if verbose:
        print(f"  model weights: {checks} forbidden questions asked, "
              f"{len(findings)} leak(s)"
              + (f", {len(errors)} that could not be asked" if errors else ""))
        if not_planted:
            comps = sorted({f["compartment"] for f in not_planted})
            print(f"    not tested on the weights: {', '.join(comps)} "
                  f"(restricted, so never in any adapter)")
    return findings, errors, not_planted


def _contains_secret(answer: str, secret: str) -> bool:
    """Whether the answer really contains the canary value.

    Matched on a digit boundary so that 42.150 does not count as a hit for
    2.15, which would fail an honest system for no reason. A test that cries
    wolf gets switched off, and then it protects nothing.
    """
    return re.search(rf"(?<![\d.]){re.escape(secret)}(?![\d])", answer) is not None


def run_audit(index=None, generate_fn=None, policy=None, canaries=None,
              expand: int = 4, verbose: bool = True) -> dict:
    """Attack every surface that is available and report.

    Surfaces that are not provided are reported as not tested rather than as
    passed, because a report that counts an untested surface as clean is
    worse than no report.
    """
    if policy is None or not canaries:
        raise ValueError("run_audit needs a policy and a list of canaries.")

    findings = []
    errors = []
    tested, untested = [], []

    if verbose:
        print(f"Access audit over {len(policy.principals)} principals and "
              f"{len(canaries)} canaries")

    if index is not None:
        findings += check_index(index, policy, canaries, expand, verbose)
        tested += ["index", "links"]
    else:
        untested += ["index", "links"]

    inconclusive = []
    if generate_fn is not None:
        # The control runs first. Knowing whether the canaries are in the
        # weights at all decides what a clean result is worth.
        inconclusive = check_model_learned(generate_fn, policy, canaries, verbose)
        model_leaks, errors, not_planted = check_model(generate_fn, policy,
                                                       canaries, verbose)
        findings += model_leaks
        if not_planted:
            untested.append("model weights for "
                            + ", ".join(sorted({f["compartment"]
                                                for f in not_planted})))
        tested.append("model")
    else:
        untested.append("model")

    report = {
        "principals": len(policy.principals),
        "canaries": len(canaries),
        "surfaces_tested": tested,
        "surfaces_not_tested": untested,
        "leaks": findings,
        "errors": errors,
        "inconclusive": inconclusive,
        # A clean result over canaries nobody ever learned is not a pass, so
        # the control counts against passing exactly as a leak does.
        # An audit that could not run is not an audit that passed.
        "passed": not findings and not inconclusive and not errors,
    }

    if verbose:
        print()
        if findings:
            print(f"FAILED. {len(findings)} leak(s):")
            for f in findings:
                print(f"  [{f['surface']}] {f['principal']} reached "
                      f"'{f['compartment']}' ({f['asset']}) - {f['detail']}")
        elif errors:
            print(f"COULD NOT RUN. {len(errors)} question(s) failed before "
                  f"they could be answered:")
            for f in errors[:5]:
                print(f"  [{f['surface']}] {f['principal']} -> "
                      f"{f['compartment']} - {f['detail']}")
            print("  Fix that first. Nothing here says anything about "
                  "isolation either way.")
        elif inconclusive:
            print(f"INCONCLUSIVE. No leaks, but {len(inconclusive)} canary "
                  f"never reached the weights:")
            for f in inconclusive:
                print(f"  [{f['compartment']}] {f['asset']} - {f['detail']}")
            print("  A model that never learned a secret cannot leak it. "
                  "Plant the canary in the")
            print("  training data with `stratum access plant` and rebuild "
                  "before trusting this.")
        else:
            print(f"PASSED on {', '.join(tested)}.")
        if untested:
            print(f"NOT TESTED: {', '.join(untested)}. "
                  f"An untested surface is not a clean one.")
    return report


def save_report(report: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
