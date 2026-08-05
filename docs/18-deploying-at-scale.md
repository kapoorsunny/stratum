# 18 - Deploying at scale

*One laptop is where you start. A company is where this ends up, and that means several machines, some with graphics cards and some without, people on Macs, and a model that has to be there in the morning. This chapter covers the shapes that work and the ones that do not.*

---

## The three machines

Almost every deployment sorts into three roles, and they want completely different hardware. Trying to make one machine do all three is the most common way this gets expensive.

| Role | What it does | What it needs | How long it runs |
|---|---|---|---|
| **Build** | reads documents, runs the teacher, trains strata | a GPU, for an hour or two | in bursts, then idle |
| **Index** | embeds chunks, builds the term index and links | processor and memory, no GPU needed | minutes |
| **Serve** | answers requests | modest and constant, near the users | permanently |

The build machine is expensive and idle most of the time. The serving machines are cheap and busy all of the time. Keeping them separate is what makes the cost sensible, and it is why `stratum pack` exists.

---

## Moving a build to where it will be served

```bash
# on the build machine
stratum pack strata/* --out bundle/ \
  --index index/ --policy policy.json --router router.json
```

```
Packed 3 strata into bundle
  base model : Qwen/Qwen3-1.7B
  files      : 28
  size       : 251.5 MB
```

Copy that folder however your organisation moves files. Then on each serving node:

```bash
stratum unpack bundle/ --run --host 0.0.0.0
```

`unpack` checks every file against a sha256 recorded at pack time, and refuses to serve if any of them changed. This matters more than it looks. A truncated adapter does not throw an error. It loads, it attaches, it generates, and it is quietly worse than it should be, and you find out weeks later from a user complaint rather than a stack trace.

### What is deliberately not in the bundle

**The base model.** It is gigabytes, it is byte-identical on every node, and each node can pull it from your own mirror or cache. Shipping it with every bundle would multiply your transfer by a hundred for no benefit. The manifest records *which* base is required, so a serving node with the wrong one fails immediately and clearly rather than attaching adapters to the wrong shapes.

**The source text**, unless you ask for it with `--chunks`. Whether the serving tier is allowed to hold the corpus is a policy decision, not a technical one, so it is a flag rather than a default. Without it the index still ranks and scores, but it cannot return the passage text, so the serving node needs to reach the same `chunks.jsonl` some other way.

---

## Which machine builds what

Splitting the build across machines is easy because the pieces are independent by construction.

**One compartment per machine.** Compartments do not share training data, so they train in parallel with no coordination at all:

```bash
# machine A
stratum corpus pairs --chunks chunks.jsonl --compartment engineering ...
stratum train --skill data/engineering.jsonl --out strata/engineering

# machine B, at the same time
stratum corpus pairs --chunks chunks.jsonl --compartment safety ...
stratum train --skill data/safety.jsonl --out strata/safety
```

Then pack them together on whichever machine collects the results. There is nothing to merge and no ordering to respect.

**The teacher pass is the expensive part, and it parallelises best.** Generating training pairs is thousands of independent calls. Against an API teacher, `--concurrency 8` turns an overnight job into a lunch break, because nearly all of that time was spent waiting for replies rather than computing anything. Against a local model, leave it at 1, because the model is already using the whole card and a second request wins nothing.

**Splitting the teacher pass across machines** is best done by compartment, as above, because that is the split the tool already understands.

Do not try to slice one compartment across machines with `--max-chunks` and different `--seed` values. `--max-chunks` samples with a fixed stride over the chunk file, so it is a pure function of that file and every machine would select **the same chunks**. And `--seed` decides only which side of the train and test split a chunk falls on, so different seeds across machines put the same chunk in one machine's training file and another machine's test file. Concatenating those contaminates the test set, and the resulting score is meaningless in the flattering direction.

If one compartment really is too big for one machine, split its **documents** into separate folders and ingest them as separate compartments, or run the pass once with `--concurrency` raised, which is where the real speedup is anyway.

---

## People on Macs

Macs are first class here, and the advice for them is genuinely different rather than a footnote.

**On Apple silicon** the graphics processor is on the same chip and the way software reaches it is called MPS. The important part is that **the ordinary PyTorch download already supports it**. There is no CUDA-equivalent to hunt for, and if you went looking for one you were looking for something that does not exist.

Apple silicon also uses **unified memory**, so the graphics side and the processor share one pool. There is no separate VRAM figure to check. The number that matters is simply how much memory the machine has.

**4-bit compression does not work on a Mac at all.** Not a version problem, not a missing package. The library that does it is written for NVIDIA hardware. So pick a base model that fits as it is. `Qwen3-1.7B` is the usual answer, `Qwen3-4B` if the machine has plenty of memory. `stratum setup` says this plainly rather than installing something that would import and then fail at the point of use.

**On an Intel Mac** there is no graphics acceleration at all and everything runs on the processor. It works and it is slow. `Qwen3-0.6B` with a small dataset is the realistic size.

**The sensible pattern for a Mac-heavy team** is that Macs do everything except training. Ingest, indexing, packing, serving and evaluation are all fine on a Mac. Training goes to a shared Linux box with a graphics card, or to a rented hour, and `stratum plan --emit-remote` writes the script that runs the identical build there and brings the result back.

---

## Serving on a network

```bash
stratum unpack bundle/ --run --host 0.0.0.0 --port 8927
```

`--host 0.0.0.0` accepts connections from the network rather than only from the machine itself.

**Put something in front of it.** With `--policy`, each request has to carry an `X-Stratum-Principal` header naming who is asking, and **that header is not authentication**. Anyone can set it. It is where a deployment puts the identity its own gateway has already established, after that gateway has actually checked a token or a certificate or a session. The server prints this warning on startup rather than letting anyone assume otherwise.

A workable shape:

```
users -> identity aware proxy -> stratum serve (several nodes) -> shared bundle
             establishes who         sets X-Stratum-Principal
             they are                from the verified identity
```

### How many nodes

Serving is memory-bound rather than compute-bound, and the memory is dominated by one thing: the base model, held once per process.

| | |
|---|---|
| Base model, 1.7B at bf16 | about 3.4 GB |
| Each additional stratum attached to it | a few MB |
| Context index resident set, 1,100 chunks | under 100 MB |

So twenty skills on one base is about 3.5 GB, not twenty models. One process serves them all, and you scale by adding processes for concurrency rather than because you ran out of room.

---

## When the corpus gets large

The retrieval side is a flat scan over a memory-mapped array, which is linear and unglamorous and beats an approximate index for far longer than people expect.

| Chunks | Corpus size | Scan time | Against generation |
|---|---|---|---|
| 1,100 | a few MB of text | 1.4 ms | invisible |
| 100,000 | a few hundred MB | 15 ms | invisible |
| 1,000,000 | a few GB | 150 ms | still under generation |

You need roughly **ten million chunks** before the index leaves memory and its layout becomes a question at all. That is around twenty gigabytes of text, and at that point the right answer is a purpose-built library rather than anything in this project.

The reason a flat scan wins is measured rather than assumed, in [engine/README.md](../engine/README.md). On real storage what costs is **the number of reads you have to do one after another**, not their size. Every read carries about 205 microseconds of fixed cost regardless, so a 256 KB read gets 64 times the data of a 4 KB read for 2.4 times the time. A flat scan is one read. A graph walk is many, each waiting on the last.

That also explains why sharding a graph index goes badly. In a 100M-vector HNSW index across five nodes, [over 80% of search steps become cross-node](https://arxiv.org/pdf/2512.17264), and every one is a dependent network round trip. Whereas concurrent reads are nearly free, measured here at 67 times the throughput at 64 threads. **Shard so that one request stays on one node, and scale with replicas.**

---

## Keeping it current

| What changed | What has to happen | How long |
|---|---|---|
| A restricted document | re-embed the changed chunks | seconds |
| A department document | retrain that one compartment, repack, redeploy | minutes |
| A company document | retrain the shared adapter | scheduled, rarely |
| The access policy | edit the file, re-run `access check` and `access audit` | seconds |

Because compartments are independent, redeploying one does not disturb the others. And because volatile material was never allowed into the weights in the first place ([doc 17](17-access-control-and-context.md)), the things that change most often are the things that cost nothing to change.

**Run the audit on every build.** It is fast, it needs no GPU, and it fails loudly:

```bash
stratum access plant policy.json --data-dir data/    # before training
stratum access audit policy.json --index index/ --strata-dir strata/ --out audit.json
```

Exit code is non-zero on a leak, on a canary nobody authorised could reach, and on a question that could not be asked at all. All three mean the same thing for a gate: do not ship this build. The output says which of the three it was.

---

## What will bite you

- **The header is not identity.** Said three times in this project because it is the one that gets skipped.
- **PII is your pipeline's job.** `--redact` is a regex second net, and in the training path it is the *last* net, because anything it misses goes into weights you cannot filter.
- **No audit log of what was retrieved for whom.** Responses carry their sources, but nothing here writes them down. If you need that record, capture it at the proxy.
- **The base model has to match.** Adapters only fit the base they were trained on. The manifest catches it, but only if you use `unpack` rather than copying files around by hand.
- **Retrieval might not help.** Strata are trained closed-book, so context in the prompt is out of distribution for them. Measure it with `stratum eval --context-style oracle` before building a retrieval tier you may not want.

---

## What you now know

- **Three roles, three machines.** Build wants a GPU in bursts, serving wants to be small and permanent, and keeping them apart is what makes the cost work.
- **A bundle is the unit that moves**, and it is verified by hash on arrival, because a half-copied adapter fails silently.
- **The base model stays out of the bundle** and is pinned by name instead.
- **Compartments build in parallel** with no coordination, because they share no data.
- **Macs do everything but train**, and need genuinely different advice rather than a footnote.
- **Shard so a request stays on one node**, because dependent reads across machines are what kills distributed graph indexes.
- **The audit is a build gate**, not a report. And it has to run on a build that was planted, or it reports INCONCLUSIVE rather than a pass.

Next: [access control in full ->](17-access-control-and-context.md), or back to [the index ->](README.md)
