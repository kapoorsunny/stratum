"""
Take a plan file and get everything ready, in one command.

Six steps have to happen in order before a company can train anything, and
each already has a verb of its own. Running them by hand is fine once. It is
not fine as the thing a company repeats every time a department is added, a
person changes team, or a document is replaced, because the steps have to
agree with each other and a skipped one shows up much later as a wrong
answer given to the wrong person.

So this runs them in order, stops at the first failure, and prints the
standalone command for every step as it goes. Printing them is deliberate.
The wrapper should be a convenience and not a place the pipeline hides, and
somebody debugging step four needs to be able to run step four.

The last step is the access sweep, which means a run that finishes green has
not merely built an index. It has asked every principal about every
compartment they are not cleared for and got nothing back.

What is deliberately not here is training. Everything above runs on a laptop
in minutes and is safe to repeat. Training strata needs a GPU and hours, and
folding it in would turn a command people run often into one they avoid.
`stratum stack` does that half.
"""
from __future__ import annotations

import time
from pathlib import Path


class PipelineError(Exception):
    """A step failed, with the standalone command to reproduce it."""


class Step:
    """One stage, its output, and the command that does it alone."""

    def __init__(self, name: str, produces: Path, command: str, run):
        self.name = name
        self.produces = produces
        self.command = command
        self.run = run


def _newer(target: Path, *inputs: Path) -> bool:
    """Whether target exists and is at least as new as everything it needs.

    Cheap and slightly conservative. A false negative costs a rebuild, a
    false positive would serve stale material to people whose permissions
    changed, so where it is unsure it rebuilds.
    """
    if not target.exists():
        return False
    t = target.stat().st_mtime
    for i in inputs:
        if not i.exists():
            return False
        if i.is_dir():
            newest = max((p.stat().st_mtime for p in i.rglob("*")
                          if p.is_file()), default=i.stat().st_mtime)
        else:
            newest = i.stat().st_mtime
        if newest > t:
            return False
    return True


def run(plan_path: str, work_dir: str, fetch: bool = True, pause: float = 0.0,
        images: str | None = None, vision_model: str | None = None,
        embedder: str = "hash", embed_model: str | None = None,
        samples: int = 3, k: int = 8, expand: int = 3, force: bool = False,
        verbose: bool = True) -> dict:
    """Plan, corpus, chunks, index, families, access sweep. In that order."""
    from .context import ContextError, build_index
    from .corpus import ingest
    from .corpus_plan import PlanError, build as plan_build, parse
    from .families import FamilyError, centroids_from_chunks, declared
    from .simulate import SimulationError
    from .simulate import run as simulate_run

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    plan_file = Path(plan_path)
    corpus = work / "corpus"
    chunks_dir = work / "chunks"
    chunks = chunks_dir / "chunks.jsonl"
    index = work / "index"
    policy = work / "policy.json"
    families = work / "families.json"
    family_plan = work / "family-plan.json"
    simulation = work / "simulation.json"

    results, started = {}, time.time()

    def say(n, total, title, command, skipped=False):
        if not verbose:
            return
        print()
        print(f"[{n}/{total}] {title}" + ("   already current, skipping"
                                          if skipped else ""))
        print(f"        {command}")
        print()

    total = 6

    # 1. The plan. Everything downstream is derived from this one file, so a
    # problem in it is found before anything is downloaded or written.
    say(1, total, "Read the plan", f"stratum corpus plan check --plan {plan_path}")
    try:
        parsed = parse(plan_path)
    except PlanError as e:
        raise PipelineError(
            f"{e}\n\nRe-run just this step with\n"
            f"  stratum corpus plan check --plan {plan_path}") from e
    results["compartments"] = len(parsed["compartments"])
    results["families_declared"] = len(parsed["families"])
    results["principals"] = len(parsed["principals"])
    if verbose:
        print(f"  {results['compartments']} compartments, "
              f"{results['families_declared']} families, "
              f"{results['principals']} principals")

    # 2. The corpus, the access policy and the family spec, from that one
    # source, so they cannot disagree with each other.
    cmd = f"stratum corpus plan build --plan {plan_path} --out {corpus}"
    fresh = not force and _newer(policy, plan_file) and corpus.exists()
    say(2, total, "Lay out the corpus", cmd, skipped=fresh)
    if not fresh:
        try:
            built = plan_build(plan_path, str(corpus), fetch=fetch,
                               pause=pause, verbose=verbose)
        except PlanError as e:
            raise PipelineError(f"{e}\n\nRe-run just this step with\n  {cmd}") from e
        results["documents"] = sum(r["documents"]
                                   for r in built["report"].values())
        # plan build writes these beside the corpus folder, which is where
        # the rest of this expects them.
        for name in ("policy.json", "families.json"):
            src = corpus.parent / name
            if src.resolve() != (work / name).resolve():
                (work / name).write_text(src.read_text(encoding="utf-8"),
                                         encoding="utf-8")

    # 3. Extraction and chunking. --compartments takes each document's
    # compartment from the folder it sat in, which is what makes every row
    # filterable later.
    cmd = (f"stratum corpus ingest --in {corpus} --out {chunks_dir} "
           f"--compartments")
    if images:
        cmd += f" --images {images}"
    fresh = not force and _newer(chunks, corpus)
    say(3, total, "Extract and chunk", cmd, skipped=fresh)
    if not fresh:
        vision = None
        if images and images != "skip":
            from .vision import get_vision_teacher
            vision = get_vision_teacher(images, model=vision_model)
        try:
            ingest(str(corpus), str(chunks_dir), vision_teacher=vision,
                   compartments=True, verbose=verbose)
        except (ValueError, FileNotFoundError) as e:
            raise PipelineError(f"{e}\n\nRe-run just this step with\n  {cmd}") from e

    # 4. The index. Access lives here rather than in the weights, which is
    # what lets a permission change take effect immediately.
    cmd = (f"stratum context build --chunks {chunks} --out {index} "
           f"--embedder {embedder}")
    fresh = not force and _newer(index / "meta.json", chunks)
    say(4, total, "Build the index", cmd, skipped=fresh)
    if not fresh:
        try:
            build_index(str(chunks), str(index), embedder=embedder,
                        embed_model=embed_model, verbose=verbose)
        except ContextError as e:
            raise PipelineError(f"{e}\n\nRe-run just this step with\n  {cmd}") from e

    # 5. The grouping. Measured from the corpus rather than from a router,
    # because at this point nothing has been trained yet, which is exactly
    # when the number of adapters has to be decided.
    cmd = (f"stratum family plan --chunks {chunks} --declare {families} "
           f"--policy {policy} --out {family_plan}")
    say(5, total, "Check the family grouping", cmd)
    try:
        import json

        from .access import Policy
        from .families import adapters_per_person, audience_check, save_plan

        centroids = centroids_from_chunks(str(chunks), verbose=verbose)
        spec = json.loads(families.read_text(encoding="utf-8"))
        fam = declared(spec, centroids, verbose=verbose)
        pol = Policy.load(str(policy))
        aud = audience_check(fam, pol)
        per = adapters_per_person(fam, pol)
        fam["audience"] = {"safe": aud["safe"],
                           "mixed": [m["family"] for m in aud["mixed"]]}
        fam["per_principal"] = per
        save_plan(fam, str(family_plan))
    except (FamilyError, FileNotFoundError, ValueError) as e:
        raise PipelineError(f"{e}\n\nRe-run just this step with\n  {cmd}") from e

    worst = max((d["adapters"] for d in per.values()), default=0)
    results["max_adapters"] = worst
    results["mixed_families"] = [m["family"] for m in aud["mixed"]]
    if verbose:
        print(f"  most adapters any one principal loads: {worst}")
        if worst > 3:
            print(f"  that is past the limit of about three that merging "
                  f"survives, so either")
            print(f"  use fewer and broader families, or fuse that "
                  f"combination")
        if aud["mixed"]:
            print(f"  {len(aud['mixed'])} family whose members do not share a "
                  f"readership: "
                  f"{', '.join(m['family'] for m in aud['mixed'])}")
            print(f"  those are only safe once the adapter carries language "
                  f"rather than facts")

    # 6. The proof. Everything above can be built correctly and still leak,
    # so the run does not end until the filter has been attacked.
    cmd = (f"stratum access simulate --index {index} --policy {policy} "
           f"--chunks {chunks}")
    say(6, total, "Prove the access filter holds", cmd)
    try:
        report = simulate_run(str(index), str(policy), str(chunks), k=k,
                              expand=expand, samples=samples, verbose=verbose)
    except (SimulationError, FileNotFoundError, ValueError) as e:
        raise PipelineError(f"{e}\n\nRe-run just this step with\n  {cmd}") from e
    from .simulate import save as save_sim
    save_sim(report, str(simulation))
    results["access"] = report["passed"]
    results["queries"] = report["queries"]

    if verbose:
        print()
        print(f"Ready in {time.time() - started:.0f}s")
        print(f"  corpus     {corpus}")
        print(f"  chunks     {chunks}")
        print(f"  index      {index}")
        print(f"  policy     {policy}")
        print(f"  families   {family_plan}")
        print(f"  access     {simulation}")
        print()
        if report["passed"]:
            print("What that proves, and what it does not")
            print()
            print("  Proven. Retrieval and link expansion return nothing a "
                  "person may not read.")
            print("  Try it yourself with")
            print("    stratum context query --index " + str(index) +
                  " --chunks " + str(chunks))
            print("        --policy " + str(policy) +
                  " --principal <name> \"your question\"")
            print()
            print("  Not proven, because no model has been trained yet. Once "
                  "there are strata,")
            print("  `stratum access audit` attacks the weights, which is a "
                  "different surface")
            print("  that fails differently.")
            print()
            print("  Not proven either way. Whether the model declines when "
                  "the material does")
            print("  not hold the answer. A filter that correctly returns "
                  "nothing still leaves")
            print("  a model free to invent, and an invented answer is not a "
                  "leak but it is")
            print("  still wrong. `stratum ground` trains that refusal in on "
                  "purpose.")
            print()
            print("To train one adapter per family from here, write a recipe "
                  "and run")
            print("  stratum stack recipe.yaml")
        else:
            print("Access did NOT pass. Do not serve this. The report above "
                  "names every")
            print("result that reached somebody who may not read it.")

    results["ok"] = report["passed"]
    return results
