"""
STRATUM command-line interface.

    stratum doctor                                   check hardware, recommend size
    stratum plan recipe.yaml                         will this build fit here?
    stratum corpus ingest --in docs/ --out corpus/   documents and images -> chunks
    stratum corpus pairs --chunks ... --out ...      chunks -> training pairs
    stratum train --skill S.jsonl --out strata/x     train one stratum
    stratum merge strata/a strata/b --out model      fuse strata into a model
    stratum eval model --test T.jsonl                score a model (or one stratum)
    stratum chat model                               talk to a model
    stratum distill ...                              student imitates a teacher
    stratum teacher-gen ...                          teacher writes training pairs
    stratum teachers                                 which teacher can this machine run?
    stratum route train strata/*                     learn which skill fits a request
    stratum serve strata/* --router r.json            OpenAI-compatible endpoint
    stratum stack recipe.yaml                        run a whole build from a recipe

Run `stratum <command> -h` for per-command options.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def use_utf8_output() -> None:
    """
    Windows still defaults its console to an eight bit code page, so the
    first time a model writes a Greek letter or a degree sign the print
    raises and takes the whole run with it. That is not a rare corner. A
    model trained on engineering documents writes delta and ohm and micro
    constantly, and losing a completed evaluation to the last line of its
    own output is a bad way to find out.

    Nothing is dropped either way. Characters the terminal genuinely cannot
    draw are replaced rather than raising, so output is always readable and
    never fatal.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # Older Python, or a stream that is not a real file. Printing
            # still works, it just keeps whatever encoding it had.
            pass


def cmd_access_init(args):
    from .access import describe, example_policy
    policy = example_policy()
    policy.save(args.out)
    print(f"Wrote a starting policy to {args.out}\n")
    print(describe(policy))
    print("\nEdit it to match your own compartments, then check it with "
          "`stratum access check`.")


def cmd_access_show(args):
    from .access import AccessError, Policy, describe
    try:
        print(describe(Policy.load(args.policy)))
    except AccessError as e:
        sys.exit(str(e))


def cmd_access_check(args):
    """Check a policy against a real corpus, and say what does not line up."""
    from .access import AccessError, Policy

    try:
        policy = Policy.load(args.policy)
    except AccessError as e:
        sys.exit(str(e))

    problems = []

    if args.chunks:
        from .data import load_jsonl
        rows = load_jsonl(args.chunks, required_keys=("id", "text"))
        seen = {}
        unlabelled = 0
        for r in rows:
            c = r.get("compartment")
            if c is None:
                unlabelled += 1
            else:
                seen[c] = seen.get(c, 0) + 1

        print(f"Corpus: {len(rows)} chunks")
        for c, n in sorted(seen.items()):
            known = c in policy.compartments
            tier = policy.tier_of(c) if known else "NOT IN POLICY"
            print(f"  {c:<16} {n:>6} chunks   {tier}")
            if not known:
                problems.append(
                    f"Chunks are labelled '{c}' but the policy has no such "
                    f"compartment, so nothing decides who may see them.")
        if unlabelled:
            problems.append(
                f"{unlabelled} chunks carry no compartment and would be "
                f"treated as public. Re-run `stratum corpus ingest "
                f"--compartments` over a folder sorted by access.")
        empty = sorted(set(policy.compartments) - set(seen))
        if empty:
            print(f"  compartments with no chunks: {', '.join(empty)}")
        # An empty restricted compartment is merely unused. An empty one in a
        # weights tier is a deployment failure waiting to happen, because the
        # policy tells principals to load an adapter that was never trained.
        for c in empty:
            if policy.tier_of(c) in ("company", "department"):
                who = policy.principals_for(c)
                problems.append(
                    f"Compartment '{c}' is tier '{policy.tier_of(c)}' but has "
                    f"no chunks, so no stratum can be trained for it. "
                    f"{len(who)} principal(s) would be told to load an "
                    f"adapter that does not exist. Either add its documents, "
                    f"or drop it from the policy.")

    # Only a tier MORE exposed than the readership supports is a problem.
    # Choosing a tighter one than allowed is being careful, and nagging
    # somebody for that trains them to ignore this output.
    for name in policy.compartments:
        want = policy.tier_of(name)
        ceiling = policy.suggest_tier(name)
        if policy.tier_risk(want) > policy.tier_risk(ceiling):
            problems.append(
                f"Compartment '{name}' is tier '{want}', which puts its "
                f"content into weights, but only "
                f"{len(policy.principals_for(name))} of "
                f"{len(policy.principals)} principals can see it. "
                f"The most it should be is '{ceiling}'.")

    worst = max((len(policy.strata_for(p)) for p in policy.principals), default=0)
    if worst > 3:
        problems.append(
            f"One principal would load {worst} adapters. More than about "
            f"three degrades a model badly, so route between them instead of "
            f"merging, or move some compartments to tier 'restricted'.")

    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for item in problems:
            print(f"  - {item}")
        sys.exit(1)
    print("Policy is consistent with the corpus.")


def cmd_access_plant(args):
    """
    Run between generating pairs and training. Relying on the teacher to
    quote a planted document is not dependable, and a canary that never
    reaches the weights turns the audit into a test that passes for the
    wrong reason.
    """
    from pathlib import Path as _P

    from .access import AccessError, Policy
    from .leak import add_canary_to_training, make_canary

    try:
        policy = Policy.load(args.policy)
    except AccessError as e:
        sys.exit(str(e))

    total = 0
    for name, comp in policy.compartments.items():
        if comp.tier not in ("company", "department"):
            # Restricted compartments never reach the weights, so there is
            # nothing there to test for and nothing to plant.
            continue
        data = _P(args.data_dir) / f"{name}.jsonl"
        if not data.exists():
            print(f"  {name}: no training data at {data}, skipped")
            continue
        total += add_canary_to_training(str(data), make_canary(name, args.seed),
                                        repeats=args.repeats)
    print(f"\nPlanted {total} canary pairs. Train, then run "
          f"`stratum access audit`.")


def cmd_access_simulate(args):
    """Ask every principal about every compartment and check what comes back."""
    from .simulate import SimulationError, run, save

    try:
        report = run(args.index, args.policy, args.chunks, k=args.k,
                     expand=args.expand, samples=args.samples, seed=args.seed)
    except (SimulationError, FileNotFoundError, ValueError) as e:
        sys.exit(str(e))

    if args.out:
        save(report, args.out)
        print()
        print(f"Report -> {args.out}")
    if not report["passed"]:
        sys.exit(1)


def cmd_access_audit(args):
    """Plant canaries, then try to reach them from every principal."""
    from .access import AccessError, Policy
    from .context import ContextIndex
    from .leak import make_canary, run_audit, save_report

    try:
        policy = Policy.load(args.policy)
    except AccessError as e:
        sys.exit(str(e))

    canaries = [make_canary(c, args.seed) for c in policy.compartments]

    index = None
    if args.index:
        index = ContextIndex(args.index, chunks_path=args.chunks)

    generate_fn = None
    if args.strata_dir:
        generate_fn = _principal_generator(policy, args.strata_dir, args.base)

    report = run_audit(index=index, generate_fn=generate_fn, policy=policy,
                       canaries=canaries, expand=args.expand)
    if args.out:
        save_report(report, args.out)
        print(f"\nReport -> {args.out}")
    if not report["passed"]:
        sys.exit(1)


def _principal_generator(policy, strata_dir, base):
    """Answer as a given principal, loading only their permitted adapters.

    This is the access control claim made concrete. An adapter a principal
    may not have is never attached, so there is nothing to filter at query
    time and nothing that has to be trusted.

    EVERY permitted adapter is asked, and the answers are concatenated.
    Asking only one of them, which an earlier version did, tests a fraction
    of the attack surface and reports a pass for the rest. Somebody trying
    to get at a compartment they should not have would try each adapter they
    can load, so the audit has to as well.
    """
    from pathlib import Path as _P

    from .serve import SkillPool

    available = {d.name: str(d) for d in _P(strata_dir).iterdir() if d.is_dir()}
    pools = {}

    def generate(principal, question):
        allowed = [n for n in policy.strata_for(principal) if n in available]
        if not allowed:
            return ""
        key = tuple(sorted(allowed))
        if key not in pools:
            pools[key] = SkillPool([available[n] for n in key],
                                   base_model=base, verbose=False)
        pool = pools[key]
        answers = []
        for skill in key:
            try:
                answers.append(pool.generate(question, skill=skill,
                                             max_new_tokens=64)["text"])
            except Exception:
                # One adapter failing must not hide what the others say.
                continue
        return "\n".join(answers)

    return generate


def cmd_context_build(args):
    from .context import ContextError, build_index
    try:
        build_index(args.chunks, args.out, embedder=args.embedder,
                    embed_model=args.embed_model, dim=args.dim,
                    link_top=args.link_top,
                    unlabelled_compartment=args.unlabelled_compartment)
    except (ContextError, FileNotFoundError) as e:
        sys.exit(str(e))


def cmd_context_query(args):
    from .access import AccessError, Policy
    from .context import ContextError, ContextIndex

    try:
        index = ContextIndex(args.index, chunks_path=args.chunks, verbose=True)
        allowed = None
        if args.policy and args.principal:
            policy = Policy.load(args.policy)
            allowed = policy.index_compartments(args.principal)
            print(f"As '{args.principal}', who may see: "
                  f"{', '.join(sorted(allowed))}\n")
        hits = index.search(args.query, allowed=allowed, k=args.k,
                            expand=args.expand)
    except (AccessError, ContextError) as e:
        sys.exit(str(e))

    if not hits:
        print("Nothing found that this principal is allowed to see.")
        return
    for h in hits:
        print(f"[{h['how']:<6}] {h['compartment']:<14} {h['source']}")
        print(f"          {h['text'][:160].strip()}...")
        print()


def cmd_context_eval(args):
    from .access import Policy
    from .context import ContextError, ContextIndex, evaluate_recall
    try:
        index = ContextIndex(args.index, chunks_path=args.chunks)
        allowed = None
        if args.policy and args.principal:
            allowed = Policy.load(args.policy).index_compartments(args.principal)
        evaluate_recall(index, args.test, allowed=allowed, expand=args.expand)
    except (ContextError, FileNotFoundError) as e:
        sys.exit(str(e))


def cmd_pack(args):
    from .deploy import DeployError, pack
    try:
        pack(args.out, args.strata, index=args.index, policy=args.policy,
             router=args.router, chunks=args.chunks, base=args.base)
    except DeployError as e:
        sys.exit(str(e))


def cmd_unpack(args):
    """Verify a bundle, then say exactly how to serve it."""
    from .deploy import DeployError, serve_command, verify
    try:
        verify(args.bundle)
    except DeployError as e:
        sys.exit(str(e))

    cmd = serve_command(args.bundle, host=args.host, port=args.port)
    print("\nServe it with:\n")
    print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    if args.run:
        import subprocess
        subprocess.run(cmd)


def _load_centroids(args):
    """One centroid per compartment, from the corpus or from the router.

    The corpus works before anything is trained, which is when the grouping
    has to be decided. The router is the better source once it exists,
    because it is built from the questions people ask rather than from the
    documents alone. Both produce the same kind of vector, so nothing that
    consumes them needs to know which one it got.
    """
    from .families import FamilyError, centroids_from_chunks

    if getattr(args, "chunks", None):
        try:
            return centroids_from_chunks(args.chunks)
        except (FamilyError, FileNotFoundError, ValueError) as e:
            sys.exit(str(e))
    if not getattr(args, "router", None):
        sys.exit("Give either --chunks, the ingested corpus, or --router, a "
                 "router already trained over the compartments.")
    from .router import SkillRouter
    try:
        return SkillRouter.load(args.router).centroids
    except Exception as e:
        sys.exit(f"Could not read the router at {args.router}: {e}")


def cmd_family_init(args):
    """Write a starting family file, one family per compartment, to edit."""
    import json as _json

    from .families import example_spec

    spec = example_spec(_load_centroids(args))
    _P = Path(args.out)
    _P.parent.mkdir(parents=True, exist_ok=True)
    _P.write_text(_json.dumps(spec, indent=2) + chr(10), encoding="utf-8")
    print(f"Wrote {args.out} with {len(spec['families'])} compartments, each "
          f"in a family of its own.")
    print()
    print("Group them by editing the file. A family is a set of compartments "
          "that share")
    print("enough language to be served by one adapter, for example")
    print()
    print('  "technical":  ["engineering", "maintenance", "quality"],')
    print('  "commercial": ["sales", "legal", "procurement"]')
    print()
    print("Then check it with `stratum family plan --declare " + args.out + "`.")


def cmd_family_plan(args):
    """Group compartments into families, declared or measured."""
    import json as _json

    from .families import (FamilyError, adapters_per_person, cluster, cohesion,
                           compare, declared, save_plan)

    centroids = _load_centroids(args)

    try:
        if args.declare:
            spec = _json.loads(Path(args.declare).read_text(encoding="utf-8"))
            plan = declared(spec, centroids,
                            allow_unlisted=args.allow_unlisted)
            print()
            compare(plan, centroids)
        else:
            plan = cluster(centroids, n_families=args.families,
                           max_per_family=args.max_per_family)
    except FamilyError as e:
        sys.exit(str(e))
    except FileNotFoundError:
        sys.exit(f"No family file at {args.declare}. Write one with "
                 f"`stratum family init`.")

    print()
    print("How tight each family is")
    print("  a family whose members sit no closer to each other than to")
    print("  outsiders is not a family, so the margin is what to look at")
    marg = cohesion(centroids, plan["families"])
    weak = []
    for label, m in sorted(marg.items()):
        flag = ""
        if m["members"] > 1 and m["margin"] < args.min_margin:
            flag = "   <- weak, consider splitting"
            weak.append(label)
        print(f"  {label:<20} within {m['within']:.3f}  outside "
              f"{m['outside']:.3f}  margin {m['margin']:+.3f}{flag}")

    if args.policy:
        from .access import AccessError, Policy
        from .families import audience_check
        try:
            policy = Policy.load(args.policy)
        except AccessError as e:
            sys.exit(str(e))

        # Vocabulary similarity says nothing about who may read something,
        # so a family can group compartments with different audiences and
        # produce one adapter that reaches people entitled to only part of it.
        aud = audience_check(plan, policy)
        plan["audience"] = {"safe": aud["safe"],
                            "mixed": [m["family"] for m in aud["mixed"]]}
        if aud["mixed"]:
            print()
            print("Families whose members do not share a readership")
            for m in aud["mixed"]:
                print(f"  {m['family']:<20} {', '.join(m['members'])}")
                for who, cannot in m["exposed"].items():
                    print(f"      {who} would load material from "
                          f"{', '.join(cannot)} that they cannot otherwise read")
            print()
            print("  These are only safe if the adapter carries language rather")
            print("  than facts, which is what `stratum ground` is for and what")
            print("  `stratum access audit` tests. Do not ship one until it has.")
        print()
        print("Adapters each principal would load")
        per = adapters_per_person(plan, policy)
        worst = 0
        for name, d in per.items():
            worst = max(worst, d["adapters"])
            fams = ", ".join(d["families"]) or "none"
            print(f"  {name:<18} sees {d['compartments']:>2} compartment(s), "
                  f"loads {d['adapters']} adapter(s)   [{fams}]")
        print()
        if worst > 3:
            print(f"  {worst} adapters for one person is past the safe merge "
                  f"limit of about three.")
            print("  Either allow more families, or fuse that combination "
                  "with `stratum fuse`.")
        else:
            print(f"  Most any one person loads is {worst}, which is inside "
                  f"the safe merge limit.")

    plan["cohesion"] = marg
    save_plan(plan, args.out)
    print()
    print(f"Plan -> {args.out}")
    if weak:
        sys.exit(1)


def cmd_ground(args):
    """Rewrite training pairs so each prompt carries its source material."""
    from .grounded import GroundedError, ground_pairs
    try:
        ground_pairs(args.pairs, args.chunks, args.out,
                     distractors=args.distractors,
                     abstain_share=args.abstain_share,
                     same_compartment=not args.allow_cross_compartment,
                     seed=args.seed)
    except (GroundedError, FileNotFoundError) as e:
        sys.exit(str(e))


def cmd_setup(args):
    from .setup_env import run_setup
    result = run_setup(dry_run=args.dry_run)
    if result["failed"]:
        sys.exit(1)


def cmd_doctor(args):
    import torch
    print("STRATUM doctor\n" + "-" * 42)
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name}\nVRAM: {vram:.1f} GB")
        if vram >= 40:
            rec = "Qwen3-8B comfortably, or 4B in full precision."
        elif vram >= 24:
            rec = "Qwen3-4B comfortably, 8B with 4-bit."
        elif vram >= 12:
            rec = "Qwen3-4B with 4-bit, 1.7B in bf16."
        elif vram >= 8:
            rec = "Qwen3-1.7B, or 4B with 4-bit + short sequences."
        else:
            rec = "Qwen3-0.6B in bf16, 1.7B with 4-bit."
        print(f"\nRecommended base: {rec}")
        try:
            import bitsandbytes  # noqa
            print("4-bit (QLoRA): available.")
        except Exception:
            print("4-bit (QLoRA): NOT installed. `pip install bitsandbytes` to fit bigger models.")
    else:
        import shutil
        from .hf_utils import detect_hidden_nvidia_gpu, pick_device
        hidden = detect_hidden_nvidia_gpu()
        if hidden:
            print(f"PyTorch cannot see a GPU - but this machine has one: {hidden}.")
            print("The installed PyTorch is the CPU-only build. Fix it with:")
            print("  pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128")
            print("(--force-reinstall matters: a plain install or --upgrade keeps")
            print(" the CPU build, because pip considers it already satisfied.)")
            print("Then re-run `stratum doctor` - training will be many times faster.")
        elif pick_device() == "mps":
            print("Apple GPU (MPS) detected - training and inference use it "
                  "automatically.")
            print("\nRecommended base: Qwen3-1.7B. 4-bit (bitsandbytes) is "
                  "NVIDIA-only, so pick a base that fits in memory as bf16.")
        elif shutil.which("rocm-smi"):
            print("An AMD GPU with ROCm tools was found, but this PyTorch "
                  "build cannot use it.")
            print("On Linux, install the ROCm build of PyTorch (see "
                  "pytorch.org for the current command). On Windows, PyTorch "
                  "has no AMD support - training runs on CPU.")
        else:
            print("No usable GPU found. CPU training works but is slow.")
            print("\nRecommended base: Qwen3-0.6B, small datasets, patience.")

    print()
    from .hf_utils import check_hf_ready, check_torch_stack
    # Versions before Hub access: a stack that cannot load a model at all
    # makes every other line of this report moot.
    stack = check_torch_stack(verbose=True)

    print()
    check_hf_ready(verbose=True)
    if not stack["ok"]:
        print("\nFix the library versions above before training - nothing "
              "that loads a model will run until then.")
        return
    print("\nTo check a specific build against this machine: "
          "`stratum plan recipe.yaml`")


def cmd_train(args):
    from .train import train_tile
    train_tile(
        skill_path=args.skill, out_dir=args.out, base_model=args.base,
        rank=args.rank, lr=args.lr, adamw_lr=args.adamw_lr, epochs=args.epochs,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        max_len=args.max_len, optimizer=args.optimizer,
        system=args.system, load_4bit=not args.no_4bit, seed=args.seed,
    )


def cmd_merge(args):
    from .merge import merge_strata
    try:
        merge_strata(args.strata, args.out, method=args.method,
                     weights=args.weights, density=args.density,
                     drop=args.drop, seed=args.seed,
                     normalize=args.normalize)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))


def cmd_eval(args):
    from .evaluate import run_eval

    index = allowed = None
    if getattr(args, "context", None):
        from .access import Policy
        from .context import ContextIndex
        index = ContextIndex(args.context, chunks_path=args.context_chunks)
        if args.policy and args.principal:
            allowed = Policy.load(args.policy).index_compartments(args.principal)

    report = run_eval(args.model, args.test, scorer=args.scorer,
                      system=args.system, json_out=args.json_out,
                      baseline=args.baseline, context=index,
                      context_style=args.context_style,
                      context_k=args.context_k, allowed=allowed)
    if args.min_score is not None and report["mean"] < args.min_score:
        sys.exit(f"FAIL: score {report['mean']:.1%} is below --min-score "
                 f"{args.min_score:.1%}.")


def cmd_chat(args):
    import torch
    from .data import format_messages, strip_think
    from .hf_utils import encode_for_generation, load_for_inference, pick_device

    # Several strata plus a router: chat against the pool, switching skill per
    # turn, instead of against one merged model.
    if len(args.model) > 1 or args.router:
        from .serve import SkillPool
        pool = SkillPool(args.model, router_path=args.router)
        history = []
        print(f"\nChat across {len(pool.names)} skills. Ctrl-C to quit.\n")
        try:
            while True:
                q = input("you: ").strip()
                if not q:
                    continue
                r = pool.generate(q, system=args.system, history=history)
                history += [{"role": "user", "content": q},
                            {"role": "assistant", "content": r["text"]}]
                print(f"[{r['skill']}, confidence {r['confidence']:.2f}] "
                      f"{r['text']}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
        return

    model, tokenizer = load_for_inference(args.model[0])
    device = pick_device()
    model.to(device).eval()

    # The conversation so far. Each turn is fed back in, so the model sees
    # its own history - without this, "chat" would be amnesiac one-shots.
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print("Chat with your STRATUM model. Ctrl-C to quit.\n")
    try:
        while True:
            q = input("you: ").strip()
            if not q:
                continue
            messages.append({"role": "user", "content": q})
            text = format_messages(tokenizer, messages, add_generation_prompt=True)
            ids = encode_for_generation(tokenizer, text, device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=400, do_sample=False,
                                     temperature=None, top_p=None,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            answer = strip_think(tokenizer.decode(
                out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
            messages.append({"role": "assistant", "content": answer})
            print("stratum:", answer, "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


def cmd_distill(args):
    from .distill import distill_tile
    distill_tile(
        skill_path=args.skill, out_dir=args.out,
        student_model=args.student, teacher_model=args.teacher,
        rank=args.rank, lr=args.lr, epochs=args.epochs,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        max_len=args.max_len, temperature=args.temperature, alpha=args.alpha,
        system=args.system, teacher_4bit=args.teacher_4bit, seed=args.seed,
    )


def cmd_teachers(args):
    from .advisor import CATALOGUE, QUANTS, advise, hardware, print_advice
    if args.quant not in QUANTS:
        sys.exit(f"Unknown quantization '{args.quant}'. "
                 f"Choose from: {', '.join(QUANTS)}.")
    print("Measuring this machine...\n")
    hw = hardware(measure=not args.no_measure)
    rows = advise(hw, min_tok_s=args.min_tok_s, quant=args.quant)
    print_advice(hw, rows, args.quant, args.min_tok_s, top=args.top)


def cmd_route(args):
    from .router import (SkillRouter, evaluate_router, strata_skill_files,
                         train_router)

    if args.route_cmd == "train":
        try:
            files = (dict(s.split("=", 1) for s in args.skills) if args.skills
                     else strata_skill_files(args.strata))
        except (ValueError, FileNotFoundError) as e:
            sys.exit(str(e))
        if len(files) < 2:
            sys.exit("Routing needs at least two skills - with one there is "
                     "nothing to choose between.")
        router = train_router(files, args.out)
        print()
        evaluate_router(router, files)
        print("\nAccuracy on training data is a floor, not a score - hold out")
        print("real requests to know how it behaves on new ones.")
    elif args.route_cmd == "test":
        router = SkillRouter.load(args.router)
        skill, confidence, scores = router.route(args.text)
        print(f"skill      : {skill}")
        print(f"confidence : {confidence:.3f}"
              + ("   (low - the skills overlap for this request)"
                 if confidence < 0.15 else ""))
        for name, score in sorted(scores.items(), key=lambda kv: -kv[1]):
            print(f"   {name:<24} {score:.4f}")


def cmd_serve(args):
    from .serve import SkillPool, serve

    index = policy = None
    try:
        if args.context:
            from .context import ContextIndex
            index = ContextIndex(args.context, chunks_path=args.context_chunks,
                                 verbose=True)
        if args.policy:
            from .access import Policy
            policy = Policy.load(args.policy)
        pool = SkillPool(args.strata, router_path=args.router,
                         base_model=args.base)
    except Exception as e:
        sys.exit(str(e))

    serve(pool, host=args.host, port=args.port, system=args.system,
          index=index, policy=policy, context_style=args.context_style,
          context_k=args.context_k)


def cmd_corpus_fetch(args):
    from .corpus import fetch_urls

    urls = list(args.urls)
    if args.urls_file:
        urls += [l for l in
                 Path(args.urls_file).read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    if not urls:
        sys.exit("No URLs given. Pass them as arguments or with --urls-file.")
    fetch_urls(urls, args.out, pause=args.pause)


def cmd_build(args):
    """Plan, corpus, chunks, index, families, access sweep, in that order."""
    from .pipeline import PipelineError, run

    try:
        result = run(args.plan, args.work, fetch=not args.no_fetch,
                     pause=args.pause, images=args.images,
                     vision_model=args.vision_model, embedder=args.embedder,
                     embed_model=args.embed_model, samples=args.samples,
                     k=args.k, expand=args.expand, force=args.force)
    except PipelineError as e:
        sys.exit(str(e))
    if not result["ok"]:
        sys.exit(1)


def cmd_corpus_plan_init(args):
    from .corpus_plan import PlanError, write_example
    try:
        write_example(args.out)
    except PlanError as e:
        sys.exit(str(e))


def cmd_corpus_plan_check(args):
    """Read the plan and say what it would build, without building it."""
    from .corpus_plan import PlanError, check_policy, parse
    try:
        plan = parse(args.plan)
    except PlanError as e:
        sys.exit(str(e))

    comps = plan["compartments"]
    print(f"{len(comps)} compartment(s) in {len(plan['families'])} "
          f"famil{'y' if len(plan['families']) == 1 else 'ies'}")
    print()
    print(f"  {'compartment':<18} {'family':<14} {'tier':<11} material")
    for name, c in sorted(comps.items()):
        onhand = sum(1 for f in c["folders"]
                     for p in f.rglob("*") if p.is_file())
        bits = []
        if onhand:
            bits.append(f"{onhand} file(s) on disk")
        if c["urls"]:
            bits.append(f"{len(c['urls'])} to download")
        print(f"  {name:<18} {c['family']:<14} {c['tier']:<11} "
              f"{', '.join(bits)}")

    print()
    print("Families")
    for label, members in plan["families"].items():
        print(f"  {label:<20} {', '.join(members)}")

    partial = []
    if plan["principals"]:
        print()
        print("Adapters each principal would load, if every family is trained")
        for name, p in sorted(plan["principals"].items()):
            fams = sorted({comps[c]["family"] for c in p["reads"]
                           if comps[c]["tier"] != "restricted"})
            # A family adapter is only usable by someone who may read every
            # compartment in it. Anything else would hand them language
            # learned from material they are not cleared for.
            usable = [f for f in fams
                      if all(m in p["reads"] for m in plan["families"][f])]
            print(f"  {name:<18} reads {len(p['reads']):>2}, loads "
                  f"{len(usable)} adapter(s)   [{', '.join(usable) or 'none'}]")
            for f in sorted(set(fams) - set(usable)):
                mine = sorted(c for c in plan["families"][f] if c in p["reads"])
                theirs = sorted(c for c in plan["families"][f]
                                if c not in p["reads"])
                partial.append((name, f, mine, theirs))
                print(f"  {'':<18} no {f} adapter, it also carries "
                      f"{', '.join(theirs)}")
                print(f"  {'':<18} so {', '.join(mine)} reaches this "
                      f"principal through the index alone")
        worst = 0
        for name, p in sorted(plan["principals"].items()):
            fams = {comps[c]["family"] for c in p["reads"]
                    if comps[c]["tier"] != "restricted"}
            worst = max(worst, sum(
                1 for f in fams
                if all(m in p["reads"] for m in plan["families"][f])))
        print()
        if worst > 3:
            print(f"  Most any one person loads is {worst}, past the limit of "
                  f"about three that merging survives.")
            print("  Use fewer, broader families, or fuse the common "
                  "combination with `stratum fuse`.")
        else:
            print(f"  Most any one person loads is {worst}, inside the safe "
                  f"merge limit.")

        if partial:
            print()
            print(f"{len(partial)} case(s) where a principal reads part of a "
                  f"family but not all of it")
            print("  Nothing is lost. Those compartments are still searched "
                  "and still answered from,")
            print("  the answer just comes from retrieval rather than from a "
                  "trained adapter, which")
            print("  reads a little less fluently. Handing over the adapter "
                  "instead would give them")
            print("  language learned from material they are not cleared for, "
                  "and no filter can")
            print("  take that back out of a weight.")
            print("  To close a case, either grant the whole family, or move "
                  "that compartment into")
            print("  a family whose members share its audience.")

    complaints = check_policy(plan)
    print()
    if complaints:
        print("The access rules reject this plan.")
        for c in complaints:
            print(f"  {c}")
        sys.exit(1)
    print("The access policy this would produce is valid.")
    print(f"Build it with `stratum corpus plan build --plan {args.plan} "
          f"--out corpus`")


def cmd_corpus_plan_build(args):
    from .corpus_plan import PlanError, build
    try:
        build(args.plan, args.out, fetch=not args.no_fetch, pause=args.pause)
    except PlanError as e:
        sys.exit(str(e))


def cmd_corpus_ingest(args):
    from .corpus import ingest

    vision_teacher = None
    if args.images and args.images != "skip":
        from .vision import get_vision_teacher
        vision_teacher = get_vision_teacher(args.images, model=args.vision_model)
    try:
        ingest(args.in_dir, args.out, vision_teacher=vision_teacher,
               redact_pii=args.redact, chunk_size=args.chunk_size,
               overlap=args.overlap, compartments=args.compartments)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))


def cmd_corpus_pairs(args):
    from .corpus import check_pairs_args, default_concurrency, generate_pairs
    from .teachers import get_teacher

    # Loading a teacher can mean a long download, so every argument is
    # checked before that happens rather than after.
    try:
        check_pairs_args(args.chunks, args.out, args.test_out,
                         args.test_fraction, args.per_chunk)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))

    teacher_fn = get_teacher(args.teacher, model=args.model,
                             url=getattr(args, 'teacher_url', None))
    try:
        concurrency = (args.concurrency if args.concurrency
                       else default_concurrency(args.teacher))

        by_kind = None
        if getattr(args, "image_teacher", None):
            by_kind = {"image": get_teacher(args.image_teacher,
                                            model=args.image_teacher_model)}
            print(f"Image chunks go to the {args.image_teacher} teacher, "
                  f"everything else to {args.teacher}")

        generate_pairs(args.chunks, args.instruction, teacher_fn,
                       out_train=args.out, out_test=args.test_out,
                       per_chunk=args.per_chunk, test_fraction=args.test_fraction,
                       max_chunks=args.max_chunks, concurrency=concurrency,
                       compartment=args.compartment, kind=args.kind,
                       teacher_by_kind=by_kind, seed=args.seed)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))


def cmd_teacher_gen(args):
    from .distill import generate_dataset_from_teacher
    from .teachers import get_teacher

    seeds = [l.strip() for l in
             Path(args.seeds).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(seeds)} seed inputs from {args.seeds}")
    teacher_fn = get_teacher(args.teacher, model=args.model,
                             url=getattr(args, 'teacher_url', None))
    generate_dataset_from_teacher(seeds, args.instruction, teacher_fn, args.out)


def cmd_plan(args):
    """Check a recipe against this machine before spending hours on it."""
    from .plan import (plan_recipe, print_plan, probe_hardware,
                       write_remote_bundle)
    from .recipe import load_recipe

    try:
        recipe = load_recipe(args.recipe)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))

    hw = probe_hardware()
    plan = plan_recipe(recipe, hw)
    print_plan(plan, hw, args.recipe)
    if args.emit_remote:
        print()
        write_remote_bundle(args.emit_remote, args.recipe, recipe)


def _stratum_up_to_date(out_dir, skill, base, rank, epochs):
    """True when a finished stratum already matches this recipe entry.

    The card records what the stratum was trained on - base, rank, the data
    file's hash, and how many epochs completed. If all of it matches, there
    is nothing to redo: a failed eval gate or an edited later stratum should
    not cost retraining the ones that were already right.
    """
    import json as json_mod

    card_path = Path(out_dir) / "stratum_card.json"
    if not card_path.exists():
        return False
    try:
        card = json_mod.loads(card_path.read_text(encoding="utf-8"))
        from .data import file_sha256
        return (card.get("base_model") == base
                and card.get("rank") == rank
                and card.get("epochs_completed", 0) >= epochs
                and card.get("skill_sha256") == file_sha256(skill))
    except Exception:
        return False


def cmd_stack(args):
    """Run a whole build from a YAML recipe: train listed strata, then merge them."""
    from .recipe import load_recipe, stratum_setting
    from .merge import merge_strata

    try:
        recipe = load_recipe(args.recipe)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))

    # Preflight: refuse a build the hardware clearly cannot run, before it
    # trains for an hour and dies. Only a GPU memory wall is a hard stop -
    # CPU-only training is merely slow, so there it warns and continues.
    # The estimate is rough, so --force overrides even a clear no-fit.
    from .plan import NO_FIT, plan_recipe, probe_hardware, rental_advice
    hw = probe_hardware()
    plan = plan_recipe(recipe, hw)
    if plan["verdict"] == NO_FIT:
        if hw["cuda"] and not args.force:
            sys.exit(
                f"This build likely will not fit in this GPU's memory "
                f"(run `stratum plan {args.recipe}` for the details and fixes).\n"
                f"For the training burst, {rental_advice(plan['base_params_b'])}\n"
                f"Use --force to try anyway."
            )
        if not hw["cuda"]:
            print(f"Warning: no GPU here, so this build will be very slow. "
                  f"For a training burst instead, "
                  f"{rental_advice(plan['base_params_b'])}\n")

    base = recipe["base_model"]
    strata_dirs = []
    for st in recipe["strata"]:
        out = st["out"]
        if not args.retrain and "distill" not in st and _stratum_up_to_date(
                out, st["skill"], base, st.get("rank", 16), st.get("epochs", 3)):
            print(f"\n=== Stratum {st['name']} is already trained on this exact "
                  f"data and settings - skipping (--retrain to force) ===")
            strata_dirs.append(out)
            continue
        common = dict(
            rank=st.get("rank", 16),
            epochs=st.get("epochs", 3),
            lr=stratum_setting(recipe, st, "lr", None),
            batch_size=stratum_setting(recipe, st, "batch_size", 4),
            max_len=stratum_setting(recipe, st, "max_len", 1024),
            system=stratum_setting(recipe, st, "system", None),
            seed=stratum_setting(recipe, st, "seed", 42),
        )
        if "distill" in st:
            from .distill import distill_tile
            dcfg = st["distill"]
            print(f"\n=== Distilling stratum: {st['name']} (teacher {dcfg['teacher']}) ===")
            distill_tile(
                skill_path=st["skill"], out_dir=out,
                student_model=base, teacher_model=dcfg["teacher"],
                temperature=dcfg.get("temperature", 2.0),
                alpha=dcfg.get("alpha", 0.5),
                teacher_4bit=dcfg.get("teacher_4bit", False),
                grad_accum=stratum_setting(recipe, st, "grad_accum", 4),
                **{**common, "batch_size": dcfg.get("batch_size",
                                                    common["batch_size"])},
            )
        else:
            from .train import train_tile
            print(f"\n=== Training stratum: {st['name']} ===")
            train_tile(
                skill_path=st["skill"], out_dir=out, base_model=base,
                optimizer=stratum_setting(recipe, st, "optimizer", "muon"),
                adamw_lr=stratum_setting(recipe, st, "adamw_lr", 1e-3),
                grad_accum=stratum_setting(recipe, st, "grad_accum", 4),
                load_4bit=stratum_setting(recipe, st, "load_4bit", True),
                **common,
            )
        strata_dirs.append(out)

    m = recipe.get("merge", {})
    print("\n=== Merging strata ===")
    try:
        merge_strata(strata_dirs, recipe["output_model"],
                     method=m.get("method", "linear"), weights=m.get("weights"),
                     density=m.get("density", 0.2), drop=m.get("drop", 0.9),
                     seed=m.get("seed", 42), normalize=m.get("normalize", False))
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))

    # Eval gates: the recipe tests what it built. A failed gate fails the
    # build, which is what lets CI or a rented box run this unattended.
    failures = []
    if recipe.get("evals"):
        from .evaluate import run_eval
        print("\n=== Evaluating ===")
        for ev in recipe["evals"]:
            report = run_eval(recipe["output_model"], ev["test"],
                              scorer=ev.get("scorer", "contains"),
                              system=ev.get("system", recipe.get("system")))
            bar = ev.get("min_score")
            if bar is not None and report["mean"] < bar:
                failures.append(f"{ev['test']}: {report['mean']:.1%} "
                                f"is below min_score {bar:.1%}")
    if failures:
        sys.exit("Build FAILED its eval gates:\n  " + "\n  ".join(failures))
    print(f"\nBuild complete -> {recipe['output_model']}")


def main():
    use_utf8_output()
    p = argparse.ArgumentParser(prog="stratum",
                                description="Build specialized SLMs from mergeable skill strata.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check hardware, recommend model size")
    d.set_defaults(func=cmd_doctor)

    su = sub.add_parser("setup", help="install what this machine is missing")
    su.add_argument("--dry-run", action="store_true",
                    help="print the commands and install nothing")
    su.set_defaults(func=cmd_setup)

    ac = sub.add_parser("access", help="who may see what, and where it may live")
    acsub = ac.add_subparsers(dest="access_cmd", required=True)

    aci = acsub.add_parser("init", help="write a starting policy file")
    aci.add_argument("--out", default="policy.json")
    aci.set_defaults(func=cmd_access_init)

    acs = acsub.add_parser("show", help="print a policy in readable form")
    acs.add_argument("policy")
    acs.set_defaults(func=cmd_access_show)

    acc = acsub.add_parser("check", help="check a policy against a corpus")
    acc.add_argument("policy")
    acc.add_argument("--chunks", help="chunks.jsonl to check the labels of")
    acc.set_defaults(func=cmd_access_check)

    acp = acsub.add_parser("plant",
                           help="put each compartment's canary into its "
                                "training data, before training")
    acp.add_argument("policy")
    acp.add_argument("--data-dir", required=True,
                     help="folder holding one <compartment>.jsonl per compartment")
    acp.add_argument("--repeats", type=int, default=6,
                     help="how many times to state it. A fact seen once in "
                          "hundreds of examples does not survive training")
    acp.add_argument("--seed", type=int, default=0)
    acp.set_defaults(func=cmd_access_plant)

    acs = acsub.add_parser(
        "simulate",
        help="ask every principal about every compartment and check nothing "
             "forbidden comes back")
    acs.add_argument("--index", required=True)
    acs.add_argument("--policy", required=True)
    acs.add_argument("--chunks", required=True,
                     help="the corpus the index was built from. The queries "
                          "are drawn out of it, so each one is asked in the "
                          "vocabulary of the material it is testing for")
    acs.add_argument("--k", type=int, default=8)
    acs.add_argument("--expand", type=int, default=3,
                     help="links followed out of each hit. A hop is a second "
                          "chance to arrive somewhere forbidden, so this is "
                          "part of the test rather than a setting to turn off")
    acs.add_argument("--samples", type=int, default=3,
                     help="queries drawn per compartment")
    acs.add_argument("--seed", type=int, default=0)
    acs.add_argument("--out", default=None, help="write the report as JSON")
    acs.set_defaults(func=cmd_access_simulate)

    aca = acsub.add_parser("audit", help="plant canaries and try to leak them")
    aca.add_argument("policy")
    aca.add_argument("--index", help="context index to attack")
    aca.add_argument("--chunks", help="chunks.jsonl the index was built from")
    aca.add_argument("--strata-dir",
                     help="folder of trained strata, one per compartment, to "
                          "attack the weights themselves")
    aca.add_argument("--base", default="Qwen/Qwen3-1.7B")
    aca.add_argument("--expand", type=int, default=4,
                     help="how far to follow links when attacking")
    aca.add_argument("--seed", type=int, default=0)
    aca.add_argument("--out", help="write the report as JSON here")
    aca.set_defaults(func=cmd_access_audit)

    cx = sub.add_parser("context",
                        help="the index the model looks facts up in")
    cxsub = cx.add_subparsers(dest="context_cmd", required=True)

    cxb = cxsub.add_parser("build", help="build an index from chunks")
    cxb.add_argument("--chunks", required=True)
    cxb.add_argument("--out", required=True)
    cxb.add_argument("--embedder", choices=("hash", "hf"), default="hash",
                     help="hash needs no download and works offline. hf uses "
                          "a Hugging Face model and is usually better")
    cxb.add_argument("--embed-model", default=None)
    cxb.add_argument("--dim", type=int, default=512)
    cxb.add_argument("--unlabelled-compartment",
                     help="which compartment chunks with no label belong to. "
                          "Without this an unlabelled corpus is refused, "
                          "because treating it as public is the wrong way to "
                          "fail")
    cxb.add_argument("--link-top", type=int, default=12,
                     help="how many links to keep per chunk")
    cxb.set_defaults(func=cmd_context_build)

    cxq = cxsub.add_parser("query", help="search as a given principal")
    cxq.add_argument("query")
    cxq.add_argument("--index", required=True)
    cxq.add_argument("--chunks")
    cxq.add_argument("--policy")
    cxq.add_argument("--principal")
    cxq.add_argument("--k", type=int, default=5)
    cxq.add_argument("--expand", type=int, default=0,
                     help="also follow this many links out of the hits")
    cxq.set_defaults(func=cmd_context_query)

    cxe = cxsub.add_parser("eval",
                           help="recall of the chunk that produced each question")
    cxe.add_argument("--index", required=True)
    cxe.add_argument("--test", required=True)
    cxe.add_argument("--chunks")
    cxe.add_argument("--policy")
    cxe.add_argument("--principal")
    cxe.add_argument("--expand", type=int, default=0)
    cxe.set_defaults(func=cmd_context_eval)

    pk = sub.add_parser("pack",
                        help="bundle a finished build for a serving machine")
    pk.add_argument("strata", nargs="+", help="stratum folders to include")
    pk.add_argument("--out", required=True, help="empty folder for the bundle")
    pk.add_argument("--index", help="context index folder")
    pk.add_argument("--policy", help="access policy file")
    pk.add_argument("--router", help="router file")
    pk.add_argument("--chunks",
                    help="include the source text too. Leave it out when the "
                         "serving machine is not allowed to hold the corpus")
    pk.add_argument("--base", help="base model the strata were trained on. "
                                   "Read from the stratum card when omitted")
    pk.set_defaults(func=cmd_pack)

    up = sub.add_parser("unpack",
                        help="verify a bundle and print how to serve it")
    up.add_argument("bundle")
    up.add_argument("--host", default="0.0.0.0",
                    help="0.0.0.0 to accept connections from the network. "
                         "Put an identity aware proxy in front of it")
    up.add_argument("--port", type=int, default=8927)
    up.add_argument("--run", action="store_true",
                    help="serve it now rather than printing the command")
    up.set_defaults(func=cmd_unpack)

    fm = sub.add_parser("family",
                        help="group compartments by how similarly they are written")
    fmsub = fm.add_subparsers(dest="family_cmd", required=True)
    fmi = fmsub.add_parser("init",
                           help="write a starting family file to edit by hand")
    fmi.add_argument("--router",
                     help="a router already trained over the "
                          "compartments. Use --chunks instead if "
                          "nothing is trained yet")
    fmi.add_argument("--chunks",
                     help="the ingested corpus, chunks.jsonl. Use this "
                          "before anything is trained, which is when the "
                          "grouping has to be decided")
    fmi.add_argument("--out", default="families-spec.json")
    fmi.set_defaults(func=cmd_family_init)

    fmp = fmsub.add_parser("plan", help="build the grouping and check it")
    fmp.add_argument("--router",
                     help="a router already trained over the "
                          "compartments. Better than --chunks once "
                          "it exists, because it is built from the "
                          "questions people ask")
    fmp.add_argument("--chunks",
                     help="the ingested corpus, chunks.jsonl. Use this "
                          "before anything is trained, which is when the "
                          "grouping has to be decided")
    fmp.add_argument("--declare",
                     help="a family file you wrote. With this the grouping "
                          "comes from you and the measurement becomes a "
                          "second opinion printed beside it")
    fmp.add_argument("--allow-unlisted", action="store_true",
                     help="let a compartment missing from the file have a "
                          "family of its own rather than failing")
    fmp.add_argument("--out", required=True)
    fmp.add_argument("--policy",
                     help="also report how many adapters each principal loads")
    fmp.add_argument("--families", type=int, default=None,
                     help="how many to make. Left out, it cuts where the "
                          "cost of joining two groups jumps most")
    fmp.add_argument("--max-per-family", type=int, default=8,
                     help="a family larger than this is a bucket, not a family")
    fmp.add_argument("--min-margin", type=float, default=0.02,
                     help="a family whose members are no closer to each other "
                          "than to outsiders is reported as weak")
    fmp.set_defaults(func=cmd_family_plan)

    gr = sub.add_parser("ground",
                        help="put the source material into each training prompt")
    gr.add_argument("--pairs", required=True, help="training JSONL to rewrite")
    gr.add_argument("--chunks", required=True, help="the corpus it came from")
    gr.add_argument("--out", required=True)
    gr.add_argument("--distractors", type=int, default=3,
                    help="passages beside the right one that do not hold the "
                         "answer. Without these the model learns that the "
                         "first passage is always correct")
    gr.add_argument("--abstain-share", type=float, default=0.15,
                    help="fraction where the source is removed, so declining "
                         "becomes a trained behaviour rather than a hope")
    gr.add_argument("--allow-cross-compartment", action="store_true",
                    help="let distractors come from any compartment. Off by "
                         "default, because a distractor from a compartment "
                         "the reader cannot see puts forbidden text into "
                         "training data others will load")
    gr.add_argument("--seed", type=int, default=42)
    gr.set_defaults(func=cmd_ground)

    t = sub.add_parser("train", help="train one stratum")
    t.add_argument("--skill", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--base", default="Qwen/Qwen3-1.7B")
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--lr", type=float, default=None,
                   help="Muon learning rate. Default scales with base size "
                        "(5e-3 under 2B, 1e-2 under 8B, 2e-2 above) because "
                        "Muon's step is independent of weight magnitude")
    t.add_argument("--adamw-lr", type=float, default=1e-3,
                   help="learning rate for the AdamW side (non-matrix params, or everything with --optimizer adamw)")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--grad-accum", type=int, default=4)
    t.add_argument("--max-len", type=int, default=1024)
    t.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    t.add_argument("--system", default=None, help="optional system prompt")
    t.add_argument("--no-4bit", action="store_true")
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)

    m = sub.add_parser("merge", help="fuse strata into one model")
    m.add_argument("strata", nargs="+")
    m.add_argument("--out", required=True)
    m.add_argument("--method", choices=["linear", "ties", "dare"], default="linear")
    m.add_argument("--weights", nargs="+", type=float)
    m.add_argument("--density", type=float, default=0.2, help="TIES: fraction kept")
    m.add_argument("--drop", type=float, default=0.9, help="DARE: fraction dropped")
    m.add_argument("--seed", type=int, default=42,
                   help="makes DARE's random dropping reproducible")
    m.add_argument("--normalize", action="store_true",
                   help="scale weights to sum to 1 (weighted average instead "
                        "of weighted sum) - the safe shape for 3+ strata")
    m.set_defaults(func=cmd_merge)

    e = sub.add_parser("eval", help="score a model (or a single stratum) on a test set")
    e.add_argument("model")
    e.add_argument("--test", required=True)
    e.add_argument("--scorer", default="contains",
                   choices=["contains", "exact", "json_field", "overlap"],
                   help="contains/exact for short answers, json_field for "
                        "extraction, overlap for free text that paraphrases")
    e.add_argument("--system", default=None)
    e.add_argument("--json-out", default=None,
                   help="write the full report as JSON here (for CI)")
    e.add_argument("--min-score", type=float, default=None,
                   help="exit non-zero if the mean score is below this (for CI gating)")
    e.add_argument("--context", help="a context index to retrieve from")
    e.add_argument("--context-chunks", help="chunks.jsonl the index was built from")
    e.add_argument("--context-style", default="none",
                   choices=["none", "oracle", "chunks", "sentences"],
                   help="what goes in the prompt beside the question. none is "
                        "closed book. oracle uses the chunk the question was "
                        "written from, which is the ceiling and the one worth "
                        "measuring first")
    e.add_argument("--context-k", type=int, default=3,
                   help="how many chunks to retrieve for the chunks and "
                        "sentences arms")
    e.add_argument("--policy", help="access policy, to retrieve as a principal")
    e.add_argument("--principal",
                   help="answer as this principal, seeing only what they may")
    e.add_argument("--baseline", default=None,
                   help="also score this model (usually the base) for comparison")
    e.set_defaults(func=cmd_eval)

    c = sub.add_parser("chat", help="talk to a model, or to a pool of skills")
    c.add_argument("model", nargs="+",
                   help="a merged model, one stratum, or several strata to "
                        "route between")
    c.add_argument("--router", default=None,
                   help="router JSON from `stratum route train` - picks the "
                        "skill for each turn")
    c.add_argument("--system", default=None)
    c.set_defaults(func=cmd_chat)

    th = sub.add_parser("teachers", help="which teacher model can this machine run, and how fast")
    th.add_argument("--quant", default="Q4_K_M",
                    help="quantization to assume (Q8_0/Q6_K/Q5_K_M/Q4_K_M/Q3_K_M/Q2_K)")
    th.add_argument("--min-tok-s", type=float, default=3.0,
                    help="tokens/sec below which a model is not worth using")
    th.add_argument("--top", type=int, default=12, help="rows to show")
    th.add_argument("--no-measure", action="store_true",
                    help="skip the bandwidth measurement and assume a default")
    th.set_defaults(func=cmd_teachers)

    rt = sub.add_parser("route", help="route requests to the right skill instead of merging")
    rtsub = rt.add_subparsers(dest="route_cmd", required=True)
    rtt = rtsub.add_parser("train", help="learn a router from the strata's own training data")
    rtt.add_argument("strata", nargs="*", help="stratum folders")
    rtt.add_argument("--skills", nargs="*", default=None,
                     help="name=file.jsonl pairs, if the strata cannot be read")
    rtt.add_argument("--out", default="router.json")
    rtt.set_defaults(func=cmd_route)
    rte = rtsub.add_parser("test", help="see which skill a request routes to")
    rte.add_argument("text")
    rte.add_argument("--router", default="router.json")
    rte.set_defaults(func=cmd_route)

    sv = sub.add_parser("serve", help="serve strata over an OpenAI-compatible API")
    sv.add_argument("strata", nargs="+", help="stratum folders sharing one base")
    sv.add_argument("--router", default=None, help="router JSON for per-request routing")
    sv.add_argument("--base", default=None, help="override the base model id")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8927)
    sv.add_argument("--system", default=None, help="default system prompt")
    sv.add_argument("--context", help="a context index to retrieve facts from")
    sv.add_argument("--context-chunks", help="chunks.jsonl the index was built from")
    sv.add_argument("--context-style", default="chunks",
                    choices=["chunks", "sentences"])
    sv.add_argument("--context-k", type=int, default=3)
    sv.add_argument("--policy",
                    help="access policy. With one, every request must carry an "
                         "X-Stratum-Principal header, and each principal sees "
                         "only their own compartments")
    sv.set_defaults(func=cmd_serve)

    di = sub.add_parser("distill", help="train a student stratum by imitating a teacher (logit distillation)")
    di.add_argument("--skill", required=True)
    di.add_argument("--out", required=True)
    di.add_argument("--student", default="Qwen/Qwen3-1.7B", help="small model that learns")
    di.add_argument("--teacher", default="Qwen/Qwen3-4B", help="big model to imitate (must share tokenizer)")
    di.add_argument("--rank", type=int, default=16)
    di.add_argument("--lr", type=float, default=2e-2, help="Muon learning rate")
    di.add_argument("--epochs", type=int, default=3)
    di.add_argument("--batch-size", type=int, default=2)
    di.add_argument("--grad-accum", type=int, default=4)
    di.add_argument("--max-len", type=int, default=1024)
    di.add_argument("--temperature", type=float, default=2.0, help="soften distributions")
    di.add_argument("--alpha", type=float, default=0.5, help="soft vs hard loss weight")
    di.add_argument("--system", default=None)
    di.add_argument("--teacher-4bit", action="store_true",
                    help="load the frozen teacher in 4-bit to fit both models (NVIDIA GPU)")
    di.add_argument("--seed", type=int, default=42)
    di.set_defaults(func=cmd_distill)

    bd = sub.add_parser(
        "build",
        help="one plan file to a filtered, searchable, access proven index")
    bd.add_argument("--plan", required=True,
                    help="the corpus plan. `stratum corpus plan init` writes "
                         "one to edit")
    bd.add_argument("--work", required=True,
                    help="folder for everything this produces")
    bd.add_argument("--no-fetch", action="store_true",
                    help="gather local folders only, download nothing")
    bd.add_argument("--pause", type=float, default=0.0,
                    help="seconds between downloads")
    bd.add_argument("--images", default=None,
                    help="a vision teacher to read images with. Left out, "
                         "images are skipped and said to be skipped")
    bd.add_argument("--vision-model", default=None)
    bd.add_argument("--embedder", choices=["hash", "hf"], default="hash",
                    help="hash needs no download and works offline")
    bd.add_argument("--embed-model", default=None)
    bd.add_argument("--samples", type=int, default=3,
                    help="queries per compartment in the access sweep")
    bd.add_argument("--k", type=int, default=8)
    bd.add_argument("--expand", type=int, default=3)
    bd.add_argument("--force", action="store_true",
                    help="redo every step even where the output is current")
    bd.set_defaults(func=cmd_build)

    co = sub.add_parser("corpus", help="turn a folder of documents and images into training data")
    cosub = co.add_subparsers(dest="corpus_cmd", required=True)

    cf = cosub.add_parser("fetch", help="download web pages and files into a corpus folder")
    cf.add_argument("urls", nargs="*", help="URLs to download")
    cf.add_argument("--urls-file", default=None,
                    help="text file with one URL per line (# comments allowed)")
    cf.add_argument("--out", required=True, help="folder to download into (re-run to resume)")
    cf.add_argument("--pause", type=float, default=0.0,
                    help="seconds to wait between downloads. Public sites "
                         "answer a fast run with 429 and most of it fails, "
                         "so try 0.5 if that happens")
    cf.set_defaults(func=cmd_corpus_fetch)

    cpl = cosub.add_parser(
        "plan",
        help="one file saying which compartments exist, what feeds each one, "
             "and which cluster it joins")
    cplsub = cpl.add_subparsers(dest="plan_cmd", required=True)

    cpi = cplsub.add_parser("init", help="write an example plan to edit")
    cpi.add_argument("--out", default="corpus-plan.yaml")
    cpi.set_defaults(func=cmd_corpus_plan_init)

    cpc = cplsub.add_parser("check",
                            help="say what the plan would build, and whether "
                                 "the access rules allow it")
    cpc.add_argument("--plan", required=True)
    cpc.set_defaults(func=cmd_corpus_plan_check)

    cpb = cplsub.add_parser("build",
                            help="lay out the corpus, the access policy and "
                                 "the family spec from the plan")
    cpb.add_argument("--plan", required=True)
    cpb.add_argument("--out", required=True,
                     help="folder to build the corpus under. One sub folder "
                          "per compartment, which is what ingest expects")
    cpb.add_argument("--no-fetch", action="store_true",
                     help="only gather the local folders, download nothing")
    cpb.add_argument("--pause", type=float, default=0.0,
                     help="seconds between downloads. The plan's own fetch "
                          "setting wins over this if it has one")
    cpb.set_defaults(func=cmd_corpus_plan_build)

    ci = cosub.add_parser("ingest", help="extract, deduplicate, and chunk a corpus folder")
    ci.add_argument("--in", dest="in_dir", required=True, help="folder of documents and images")
    ci.add_argument("--out", required=True, help="output folder for chunks.jsonl, manifest, and cache")
    ci.add_argument("--images", default="skip",
                    choices=["skip", "hf", "anthropic", "openai", "gemini", "echo"],
                    help="vision teacher for image files. hf runs locally - "
                         "anthropic/openai/gemini send every image to that API")
    ci.add_argument("--vision-model", default=None,
                    help="vision model id for the chosen backend")
    ci.add_argument("--redact", action="store_true",
                    help="baseline scrub of emails, phone and card numbers - "
                         "not a substitute for your own compliance pipeline")
    ci.add_argument("--chunk-size", type=int, default=2400,
                    help="target chunk length in characters. 2000-3000 suits "
                         "most prose - smaller for dense tables, larger when "
                         "answers span several paragraphs")
    ci.add_argument("--overlap", type=int, default=240,
                    help="characters shared between neighboring chunks")
    ci.add_argument("--compartments", action="store_true",
                    help="label every chunk with its top-level folder, which "
                         "is how access control gets its unit. Doc 17 explains "
                         "what to sort the folders by")
    ci.set_defaults(func=cmd_corpus_ingest)

    cp = cosub.add_parser("pairs", help="have a teacher write grounded Q/A pairs per chunk")
    cp.add_argument("--chunks", required=True, help="chunks.jsonl from corpus ingest")
    cp.add_argument("--instruction", required=True,
                    help="what kind of pairs to write, one line - e.g. "
                         "'Write questions a field engineer would ask.'")
    cp.add_argument("--out", required=True, help="training JSONL (re-run to resume)")
    cp.add_argument("--test-out", default=None,
                    help="held-out test JSONL - required when --test-fraction > 0")
    cp.add_argument("--teacher-url", default=None,
                    help="base URL for --teacher llama-cpp "
                         "(default http://127.0.0.1:8080/v1)")
    cp.add_argument("--teacher", default="hf",
                    choices=["hf", "llama-cpp", "claude-cli", "openai",
                             "anthropic", "gemini", "echo"],
                    help="teacher backend. claude-cli uses your Claude Code "
                         "subscription, no API key. Everything except hf/echo "
                         "sends chunks to that provider - use hf for data "
                         "that must not leave")
    cp.add_argument("--model", default=None, help="teacher model id for the chosen backend")
    cp.add_argument("--compartment",
                    help="only use chunks from this access compartment, which "
                         "is how one stratum per compartment gets trained")
    cp.add_argument("--kind", choices=["document", "image"],
                    help="only use chunks of this sort. Text lifted out of a "
                         "diagram reads nothing like prose, so it is usually "
                         "worth its own skill")
    cp.add_argument("--image-teacher",
                    choices=("hf", "llama-cpp", "claude-cli", "openai",
                             "anthropic", "gemini", "echo"),
                    help="a different teacher for image-derived chunks. Leave "
                         "unset to use --teacher for everything")
    cp.add_argument("--image-teacher-model", default=None)
    cp.add_argument("--concurrency", type=int, default=None,
                    help="teacher calls to run at once. Defaults to 1 for a "
                         "local model and several for an API, since an API "
                         "spends its time waiting. Lower it if you get rate "
                         "limited")
    cp.add_argument("--per-chunk", type=int, default=3,
                    help="pairs to request per chunk. 2-4 is sensible - more "
                         "than that and the teacher starts repeating itself")
    cp.add_argument("--max-chunks", type=int, default=None,
                    help="cap teacher cost by sampling this many chunks, spread "
                         "evenly across the corpus. Aim for 500-1500 training "
                         "pairs per skill (chunks x per-chunk) - doc 7 sizes it")
    cp.add_argument("--test-fraction", type=float, default=0.1,
                    help="fraction of CHUNKS whose pairs go to the test set")
    cp.add_argument("--seed", type=int, default=42, help="makes the train/test split stable")
    cp.set_defaults(func=cmd_corpus_pairs)

    tg = sub.add_parser("teacher-gen", help="data distillation: a teacher writes training pairs from seed inputs")
    tg.add_argument("--seeds", required=True,
                    help="text file, one seed input per line. 500-1500 seeds "
                         "makes a solid skill, under 50 only proves the "
                         "pipeline (doc 7)")
    tg.add_argument("--instruction", required=True, help="what the skill should do, one line")
    tg.add_argument("--out", required=True, help="output training JSONL (re-run to resume)")
    tg.add_argument("--teacher-url", default=None,
                    help="base URL for --teacher llama-cpp "
                         "(default http://127.0.0.1:8080/v1)")
    tg.add_argument("--teacher", default="hf",
                    choices=["hf", "llama-cpp", "claude-cli", "openai",
                             "anthropic", "gemini", "echo"],
                    help="teacher backend. claude-cli uses your Claude Code "
                         "subscription, no API key. Everything except hf/echo "
                         "sends seeds to that provider - use hf for data "
                         "that must not leave")
    tg.add_argument("--model", default=None, help="teacher model name/id for the chosen backend")
    tg.set_defaults(func=cmd_teacher_gen)

    pl = sub.add_parser("plan", help="check a recipe against this machine, or plan a remote build")
    pl.add_argument("recipe")
    pl.add_argument("--emit-remote", default=None, metavar="DIR",
                    help="write a build-and-test script for a rented GPU box here")
    pl.set_defaults(func=cmd_plan)

    s = sub.add_parser("stack", help="run a full build from a YAML recipe")
    s.add_argument("recipe")
    s.add_argument("--force", action="store_true",
                   help="run even when the preflight says it will not fit")
    s.add_argument("--retrain", action="store_true",
                   help="retrain strata even when their output already matches "
                        "the recipe (same data, base, and settings)")
    s.set_defaults(func=cmd_stack)

    args = p.parse_args()

    # Every command below loads a model, so a mismatched torch/transformers
    # pair is fatal for all of them. Check once, here, rather than letting
    # each one crash inside a library import with an unreadable traceback.
    if args.cmd in {"train", "merge", "eval", "chat", "distill",
                    "teacher-gen", "stack", "serve"}:
        from .hf_utils import require_torch_stack
        require_torch_stack()

    args.func(args)


if __name__ == "__main__":
    main()
