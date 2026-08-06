<div align="center">

<img src="docs/logo.png" width="200" alt="STRATUM logo"/>

# STRATUM

### Turn your company's documents into a language model that only tells each person what they are allowed to know

**S**pecialized **T**raining via **R**eusable **A**dapter **T**iles and **U**nified **M**erging

*Train small skill layers one at a time on a laptop, fuse or route between them, and serve the result behind your own address. No data-center GPUs. No ML background. Every concept explained from zero.*

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![platforms](https://img.shields.io/badge/runs%20on-Mac%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)](docs/16-what-changed.md)

</div>

---

## Point it at a folder. Get a model.

```bash
stratum corpus ingest --in company-docs/ --out chunks/ --compartments
stratum corpus pairs  --chunks chunks/chunks.jsonl --teacher claude-cli --out data/skill.jsonl --test-out data/skill-test.jsonl --instruction "Write questions a field engineer would ask."
stratum train --skill data/skill.jsonl --out strata/engineering
stratum serve strata/* --context index/ --policy policy.json
```

PDFs, Word files, spreadsheets, slide decks, web pages and **photographs of equipment** go in one end. A tested, servable model comes out the other, on a laptop, in an afternoon.

---

## What makes it different

<table>
<tr><td width="33%" valign="top">

### Skills are separate things

Each skill is a **stratum** of a few megabytes, trained on its own. Fuse them into one model, or keep them apart and route per request. Adding a skill never means retraining what you already had.

</td><td width="33%" valign="top">

### Access control that actually holds

**You cannot filter a weight.** So who may see what is decided *before* training. Compartments that only some people can read never enter shared parameters, and a canary audit tries to break it on every build.

</td><td width="33%" valign="top">

### It tells you when it will not work

Merging three skills produced empty output, and that is in the docs. Routing failed on overlapping skills, and that is in the docs with the numbers. Measurements beat opinions here, including when they are unflattering.

</td></tr>
</table>

---

## The whole thing, in one picture

```mermaid
flowchart LR
    D[PDFs, Word, slides<br/>spreadsheets, web pages]
    I[photographs<br/>and diagrams]
    D --> C[corpus ingest<br/>chunk + label by compartment]
    I -- vision model reads them --> C
    C --> P[corpus pairs<br/>a teacher writes the training data]
    C --> X[context build<br/>vectors + terms + links]
    P --> T[train<br/>one stratum per skill]
    T --> M{{merge<br/>or route}}
    M --> S[serve<br/>OpenAI-compatible, per person]
    X --> S
    A[access policy] --> S
    A --> T

    classDef src fill:#2d1a52,stroke:#1b1035,color:#fff
    classDef mid fill:#7F77DD,stroke:#1b1035,color:#fff
    classDef fuse fill:#1D9E75,stroke:#1b1035,color:#fff
    classDef out fill:#EF9F27,stroke:#1b1035,color:#1b1035
    class D,I,A src
    class C,P,X,T mid
    class M fuse
    class S out
```

---

## Real numbers, measured on a laptop

An RTX 4070 with 8.6 GB of VRAM, on a 1,115-chunk corpus of energy engineering documents plus real equipment photographs.

| | |
|---|---|
| Corpus ingested, including 5 images read by a local vision model | **14 seconds** cached, 2 minutes cold |
| Context index over 1,115 chunks, 24,886 terms, 13,356 links | **4.5 MB**, builds in seconds |
| Retrieval latency against 4 to 6 seconds of generation | **10 to 15 ms**, about 0.3% of a request |
| One stratum, 300 training pairs, 3 epochs | **2 minutes** |
| Two teachers compared head to head on identical chunks | Qwen3-8B won 2 of 3 skills against Claude |

Two findings worth having, both of which went against what was expected:

- **Giving a fine-tuned model the exactly correct chunk sometimes makes it worse.** Measured across three skills: safety improved by 5.5 points, extract was flat, explain got slightly worse. `stratum eval --context-style oracle` measures this on your own corpus in an afternoon.
- **Read *order* on an NVMe drive is worth 2%. Read *size* is worth 2.2x.** Every read costs about 205 microseconds no matter how small, so what to minimise is the number of reads that happen one after another. The tool that measured it ships in [`engine/`](engine/README.md).

---

## Company data, without the model telling everyone everything

This is the part most tools do not have, and the reason is that it cannot be bolted on afterwards.

A row in a database has an owner and a query can be filtered. A number inside a matrix of weights has neither. Once a salary figure is in the parameters, no check downstream keeps it from a contractor, because the model does not look it up, it just knows it.

So STRATUM decides at build time, by asking two questions about every set of documents: **how many people can see it**, and **how often does it change**.

| Tier | When | Where the knowledge goes |
|---|---|---|
| **company** | everyone sees it, changes rarely | into weights, one shared adapter |
| **department** | a department shares it, changes rarely | into weights, that department's adapter |
| **restricted** | narrow, sensitive, or volatile | **index only, never the weights** |

```bash
stratum access init  --out policy.json                 # who sees what
stratum access check policy.json --chunks chunks/...   # does it match the corpus
stratum access plant policy.json --data-dir data/      # canaries, BEFORE training
# ... train one stratum per compartment ...
stratum context build --chunks chunks/... --out index/
stratum access audit policy.json --index index/ --strata-dir strata/
```

A **canary** is a fact that exists in exactly one compartment and could not be guessed, invented or reasoned out. `plant` puts one into each compartment's training data before training, and `audit` then asks every person for every canary they should not be able to reach.

The order matters. Auditing a build that was never planted reports **INCONCLUSIVE** rather than a pass, because a model that never learned a secret cannot leak it, and reporting that as clean would be the most dangerous kind of green tick.

Retrieval, link expansion and the model weights are attacked separately, because they fail differently. A surface that was not tested is reported as **not tested**, never as passed.

Right to erasure falls out of the same rule. Deletion requests are nearly always about restricted material, and restricted material was never in the weights, so it is a row deletion rather than a retrain.

[Chapter 17](docs/17-access-control-and-context.md) explains all of it from zero.

---

## A real run, sixteen departments, one command

Everything below actually ran. The corpus is public Wikipedia material so you can reproduce it exactly, and the plan file is [`examples/enterprise/plan.yaml`](examples/enterprise/plan.yaml).

You describe your company once, in one file. Which departments exist, what material feeds each one, which cluster it belongs to, and who reads what.

```yaml
compartments:
  engineering:
    tier: department        # a group reads it, so it gets an adapter
    family: technical       # which cluster carries its language
    folders: [//fileserver/engineering]

  payroll:
    tier: restricted        # index only, never enters any weights
    volatile: true
    folders: [//fileserver/hr/pay]

principals:
  reliability_lead:
    reads: [public, engineering, maintenance, quality, instrumentation, project_atlas]
  contractor:
    reads: [public]
```

Then one command turns that into a working, filtered, access proven system.

```bash
stratum corpus plan init --out plan.yaml     # a plan to edit
stratum build --plan plan.yaml --work work/  # everything else
```

### What that command does

```mermaid
flowchart TD
    Y[plan.yaml<br/>departments, clusters, people]

    Y --> S1["1 read the plan<br/>refuse a broken one before<br/>anything is downloaded"]
    S1 --> S2["2 lay out the corpus<br/>share drives, Word, PDF, slides<br/>spreadsheets, images, web"]
    S2 --> POL[policy.json]
    S2 --> FAM[families.json]
    S2 --> S3["3 extract and chunk<br/>every row labelled with the<br/>compartment it came from"]
    S3 --> S4["4 build the index<br/>vectors, terms and links"]
    S4 --> S5["5 check the grouping<br/>how many adapters does<br/>any one person load"]
    POL --> S5
    FAM --> S5
    S5 --> S6["6 attack the filter<br/>every person, every<br/>compartment they cannot read"]
    S6 --> OUT[servable, and proven]

    classDef src fill:#2d1a52,stroke:#1b1035,color:#fff
    classDef mid fill:#7F77DD,stroke:#1b1035,color:#fff
    classDef gate fill:#1D9E75,stroke:#1b1035,color:#fff
    classDef out fill:#EF9F27,stroke:#1b1035,color:#1b1035
    class Y,POL,FAM src
    class S1,S2,S3,S4 mid
    class S5,S6 gate
    class OUT out
```

Each step prints the standalone command that does it alone, so the wrapper is a convenience rather than a place the pipeline hides.

### What came out

| | |
|---|---|
| Compartments, from 16 folders of documents | **16 departments** in 7 clusters |
| People simulated, up to 5 departments plus a project each | **12 principals** |
| Files in, one duplicate dropped, zero extraction errors | **122 in, 121 kept** |
| Chunks out, every one labelled with its compartment | **2,502** |
| Index over the lot, 46,732 terms and 30,024 links | **10.1 MB** |
| Most adapters any single person has to load | **3**, inside the merge limit |
| Access sweep, every person against every forbidden department | **576 queries, 0 leaks** |

### The bit that matters

The same question, asked by three different people, on the same index.

> *"what are the terms for ending a supplier agreement early"*

| Who they are | What came back |
|---|---|
| **counsel**, reads legal and procurement | contract management and vendor material, the actual answer |
| **reliability_lead**, reads four technical departments | maintenance material only, never sees that procurement exists |
| **contractor**, reads public only | public governance material, nothing else |

Nobody was told they were being filtered. There is no refusal message to work around, because the forbidden rows were removed before ranking rather than after. A hidden chunk cannot even push a permitted one out of the results, which would leak through what is missing.

### Two things the run found on its own

**Two clusters do not share a readership.** `commercial` holds legal, procurement and sales, and the auditor reads legal but not the other two. The tool refuses to hand that person the cluster adapter and says why. They still get answers from legal, through retrieval instead. Nothing is lost except a little fluency, and the alternative would put language learned from material they are not cleared for inside a model they load.

**The count of adapters per person is the number that decides whether this scales.** Merging more than about three adapters destroys the model. Sixteen departments where a person belongs to six of them still comes to three, because adapters follow how departments *write* and access stays in the index. Forty departments behave the same way.

### Proving the proof works

An access sweep that cannot fail proves nothing, so the test suite breaks the filter three ways on purpose and checks the sweep catches each one. A search that ignores the permitted set. A filter applied to ranking but dropped on link expansion, which is the subtle one. And an index that returns nothing at all, which leaks nothing and is reported as **INCONCLUSIVE** rather than as a pass.

```bash
stratum access simulate --index work/index --policy work/policy.json \
                        --chunks work/chunks/chunks.jsonl
```

Full walkthrough with every command and its output in [Chapter 19](docs/19-a-real-enterprise-run.md).

---

## How a question actually gets answered

Take an energy company. Priya is a rotating equipment engineer. She asks:

> *"What is the certified bearing clearance on pump P-4471, and what are we paying Vendor A for maintenance?"*

That question deliberately spans two things she has different rights to. Here is what happens.

### Part one, what already exists before she asks

This all happened at build time, weeks ago.

**The company sorted its documents by who may read them.**

```
corpus/
  public/        everyone
  engineering/   the engineering department
  safety/        the safety department
  finance/       the finance team
  hr/            HR only
```

They did not invent this. It is the share structure they already had. `corpus ingest --compartments` walked it and stamped every chunk with its folder, so each of the 1,115 chunks now carries a label saying which compartment it belongs to.

**Someone wrote down who is who.** A policy file. Priya is `engineer`, and she may see `public` and `engineering`. It also records a **tier** per compartment, which decides *where that knowledge is allowed to live*:

| Compartment | Tier | Consequence |
|---|---|---|
| public | company | trained into an adapter everyone loads |
| engineering | department | trained into an adapter engineers load |
| safety | department | trained into an adapter safety staff load |
| finance | **restricted** | **never trained into anything. Index only** |
| hr | **restricted** | **never trained into anything. Index only** |

The finance and HR decision is the important one, and it is irreversible in the right direction. Their content never enters any set of weights, so there is no adapter anywhere that knows what Vendor A costs.

**Three things got built.**

*Adapters.* One per weights-tier compartment. `public`, `engineering`, `safety`. Each is a few megabytes trained on that compartment's documents alone. Finance and HR got none.

*An index.* Every chunk from all five compartments, with a vector, its terms, and links to related chunks. The index holds everything, including finance and HR, because the index can be filtered at query time. Weights cannot.

*A router.* Learned from the training questions, it knows which compartment a question is about.

### Part two, the request

**Step 1. Something else decides who she is.**

Priya's request hits the company's identity proxy first. It checks her token, confirms she is who she says, looks her up, and sets a header:

```
X-Stratum-Principal: engineer
```

**STRATUM does not authenticate anyone.** It trusts that header completely, which is only safe because nothing can reach it except through the proxy. This is printed on the server's startup banner, because it is the thing people skip.

**Step 2. The policy is consulted twice, for two different questions.**

```
policy.index_compartments("engineer")  ->  {public, engineering}
policy.strata_for("engineer")          ->  {public, engineering}
```

These look the same here but they are not the same question. The first is *what may she read*. The second is *which adapters may she invoke*. For a finance manager they differ, because she may read finance documents but there is no finance adapter to invoke.

Getting only the first one right was a real hole in this system, found by an adversarial review of this repo. Retrieval was filtered and adapter selection was not, so naming a department adapter simply handed it over.

**Step 3. Retrieval, filtered before anything is scored.**

The index turns Priya's question into a vector and scores it against all 1,115 chunks. Then:

```
mask = [chunk.compartment in {public, engineering} for every chunk]
dense_scores[not mask]   = -infinity
lexical_scores[not mask] = -infinity
```

The forbidden rows are knocked out **before** ranking, not after. This ordering is the whole thing. Rank first and filter after, and a finance chunk that scores highly takes a slot and then vanishes, so Priya gets four results where there should have been five, and the gap itself tells her something exists that she cannot see.

Two rankings are produced, a semantic one and an exact-term one, combined by reciprocal rank fusion, which needs no tuning and does not care that the two scores are on different scales. The term index is what finds `P-4471` exactly. The vector is what understands "bearing clearance" when the document says "running fit tolerance".

**Step 4. Links, checked on every hop.**

The top hits can pull in chunks they are linked to, so a question about a pump reaches the maintenance procedure that governs it without naming it.

**Every neighbour is permission checked individually.** Not the starting chunk, every hop. An unfiltered edge walks straight around step 3, silently, and it is the hole people leave open because they filter the search, feel finished, and then let expansion go wherever the graph goes.

**Step 5. The prompt is assembled.**

The surviving chunks go into the prompt, each carrying its source, with an instruction to say so if the material does not answer the question.

What Priya's prompt contains: the engineering document with the pump specification.

What it does not contain: anything from `finance/vendor-contract-summary.txt`. That chunk exists in the index, scored well on the second half of her question, and was eliminated at step 3 before it was ever ranked.

**Step 6. Choosing which adapter answers.**

The router reads her question and says `engineering`. That answer is then filtered against what she may invoke:

```
router says              ->  engineering
permitted                ->  {public, engineering}
engineering in permitted ->  use it
```

If the router had said `safety`, she would get `public` instead, not `safety`. If she had explicitly asked for `model: "safety"` in her request, she gets a **403**, not a quiet substitution, because silently answering from a different adapter would hide the denial.

The base model is loaded once with all three adapters attached. Selecting one is a pointer change rather than a load, so this costs nothing measurable.

**Step 7. Generation, and what comes back.**

The `engineering` adapter answers. The response carries a block saying exactly what happened:

```json
"stratum": {
  "principal": "engineer",
  "skill": "engineering",
  "confidence": 0.41,
  "sources": [
    {"source": "engineering/pumps.txt", "compartment": "engineering"},
    {"source": "public/equipment-tag-convention.txt", "compartment": "public"}
  ]
}
```

Every source is traceable to a document, with the compartment it came from. An answer nobody can trace is an answer nobody can check.

Priya gets her bearing clearance. She does not get the contract value, and cannot, by three independent mechanisms.

### Why it holds, in three layers

```
Priya asks about the Vendor A contract
        |
        +-- Layer 1  the index knows the answer but never ranks it
        |            filtered before scoring, so no displacement and no timing gap
        |
        +-- Layer 2  the links to it are checked on every hop
        |            so she cannot arrive sideways from a chunk she can see
        |
        +-- Layer 3  no adapter she can load ever learned it
                     finance is restricted tier, so it entered no weights at all
```

Layers one and two are query-time and could in principle be bugged. **Layer three cannot.** The knowledge is not in any parameter she has access to, so there is nothing to filter and nothing to trust.

That is the reason for the tier system. A retrieval filter is a check you have to get right every time. A weight that was never trained is a fact that does not exist for her.

### The same question, three people

| Who | Gets the pump spec | Gets the contract value | Why |
|---|---|---|---|
| **contractor**, public only | no | no | Neither compartment is theirs |
| **engineer**, Priya | **yes** | no | Engineering is hers, finance is not |
| **commercial-lead**, public and engineering and finance | **yes** | **yes, from retrieval only** | May read finance, but loads no finance adapter, because there is not one |

That last row is the interesting one. The commercial lead gets the contract value **as retrieved text in the prompt**, never as something the model knew. So when that contract is renegotiated next month, one row changes in the index and the answer is immediately correct. Nothing is retrained, and nothing stale was baked into a weight.

### The honest caveats

- **A small model will invent a figure when the context lacks one.** The prompt tells it to say it does not know, and a 1.7B model ignores that. Measured here, three models produced three confident and completely different numbers when the right chunk was missing. So retrieval quality is not a nicety, it is the line between an answer and a fabrication.
- **Retrieval quality depends heavily on the embedder.** Asked in the documents' own words, the zero-dependency default finds things fine. Asked in a person's own words it missed the contract entirely, where a real embedding model put it first. That is 77.8% against 100% recall on the same held-out questions.
- **Nothing logs what was retrieved for whom.** The response carries its sources but nothing writes them down. Capture that at your proxy if you need the record.

---

## Quick start

```bash
git clone https://github.com/sarkar4777/stratum.git && cd stratum
pip install -e .

stratum setup     # installs what THIS machine is missing
stratum doctor    # and tells you what it found
```

`stratum setup` knows a Mac from an NVIDIA box. On Apple silicon it knows the ordinary PyTorch wheel already uses the GPU and there is no CUDA-style download to hunt for, and that 4-bit compression will never work there at any version. On an NVIDIA machine it knows that `pip install --upgrade torch` silently leaves the CPU build in place, which is the single most common way this project fails for people.

```bash
# understand it first, no GPU, about 20 seconds
python scripts/demo_concepts.py

# train two skills, fuse them, measure, use
stratum train --skill examples/extract.jsonl  --out strata/extract  --base Qwen/Qwen3-1.7B
stratum train --skill examples/classify.jsonl --out strata/classify --base Qwen/Qwen3-1.7B
stratum merge strata/extract strata/classify --out models/my-slm
stratum eval  models/my-slm --test examples/test-extract.jsonl --scorer json_field
stratum chat  models/my-slm

# or keep them apart and route between them
stratum route train strata/extract strata/classify --out router.json
stratum serve strata/* --router router.json --port 8927
```

Or run the whole build from one recipe, with its own pass or fail gates, so a finished build is a tested build:

```bash
stratum plan  examples/recipe.yaml   # will it fit here? if not, what will, and where to rent it
stratum stack examples/recipe.yaml   # train everything, fuse, run the gates
```

For a whole company rather than a couple of skills, describe it in one plan file and let one command do the rest. No GPU needed for any of this:

```bash
stratum corpus plan init  --out plan.yaml                # a plan to edit
stratum corpus plan check --plan plan.yaml               # before downloading anything
stratum build --plan plan.yaml --work work/              # corpus, index, grouping, proof
```

The last step ends by attacking its own access filter, so a run that finishes green has asked every person about every department they cannot read and got nothing back. [Chapter 19](docs/19-a-real-enterprise-run.md) is that run written down, sixteen departments, every command and its output.

---

## Build here, serve there

The machine that builds a model and the machines that serve it are rarely the same. Building wants a GPU for an hour. Serving wants to be near the people asking, on several machines, for months.

```bash
# on the GPU box
stratum pack strata/* --out bundle/ --index index/ --policy policy.json --router router.json

# on each serving node
stratum unpack bundle/ --run --host 0.0.0.0
```

The bundle carries a manifest with a sha256 for every file, and `unpack` verifies all of them before anything is served. A half-copied adapter does not announce itself, it just answers slightly worse, and that is exactly the failure this catches.

The base model is deliberately **not** in the bundle. It is gigabytes, it is identical everywhere, and every node can pull it from one place. The manifest records which one is required so a mismatch is caught rather than discovered.

---

## Different models for different data

A model good at reading a P&ID diagram is not the model you want writing training questions about a maintenance contract. So pick per data type:

```bash
stratum corpus ingest --in docs/ --out chunks/ --images hf          # local vision model
stratum corpus pairs  --chunks chunks/chunks.jsonl \
  --teacher claude-cli --image-teacher hf --image-teacher-model Qwen/Qwen2.5-VL-3B-Instruct
```

Train a separate skill per data type with `--kind image` or `--kind document`, and the router picks between them at serve time. A question mentioning a diagram, a nameplate or an axis leans toward the skill that learned from pictures.

`stratum teachers` measures your machine and tells you which models it can actually run, and how fast, before you download 60 GB to find out.

---

## Documentation, written as a short book

Read in order, or jump to what you need. Zero prior knowledge assumed anywhere.

| # | Doc | You will understand |
|---|-----|-------------------|
| 0 | [What is a language model?](docs/00-what-is-a-language-model.md) | Tokens, parameters, training, from zero |
| 1 | [The memory problem](docs/01-the-memory-problem.md) | Why laptops struggle, with real numbers |
| 2 | [The STRATUM idea](docs/02-the-stratum-idea.md) | Why build in independent layers |
| 3 | [LoRA and adapters](docs/03-lora-and-adapters.md) | Training 1% of a model, and rank |
| 4 | [Muon, explained fully](docs/04-muon-explained.md) | Why Muon beats AdamW here |
| 5 | [Merging strata](docs/05-merging.md) | The fuse math, three methods, honest limits |
| 6 | [Training internals](docs/06-training.md) | The loss mask and every default |
| 7 | [Distillation](docs/07-distillation.md) | Teaching a small model from a big one |
| 8 | [Evaluation](docs/08-evaluation.md) | Proving it works rather than guessing |
| 9 | [Full walkthrough](docs/09-full-walkthrough.md) | Empty folder to working model |
| 10 | [Scaling and production](docs/10-scaling-and-production.md) | Bigger models, serving, industry patterns |
| 11 | [Glossary](docs/11-glossary.md) | Every term, full form, plain definition |
| 12 | [For experienced developers](docs/12-for-experienced-developers.md) | Every concept mapped to patterns you know |
| 13 | [Troubleshooting](docs/13-troubleshooting.md) | The problems people actually hit, with fixes |
| 14 | [From a corpus to a model](docs/14-from-corpus-to-model.md) | Thousands of documents and images in, tested model out |
| 15 | [Routing and serving](docs/15-routing-and-serving.md) | Picking a skill per request instead of merging |
| 16 | [What changed, and why](docs/16-what-changed.md) | Every command explained from zero, and what `engine/` is |
| 17 | [Access control and context](docs/17-access-control-and-context.md) | Company data in a model, safely |
| 18 | [Deploying at scale](docs/18-deploying-at-scale.md) | Laptop to a network of GPU machines |
| 19 | [A real enterprise run](docs/19-a-real-enterprise-run.md) | Sixteen departments end to end, every command and its output |

**Coming back to a version you have not used?** Start at [16](docs/16-what-changed.md). It stands alone and assumes nothing.

A complete worked example lives in [`examples/energy/`](examples/energy/): a public web corpus turned into a tested energy-sector model, with every command, the measured gains over the base model, and the two failures the build hit on the way.

---

## Who it is for

- You have a laptop with a consumer GPU, 8 GB of VRAM is plenty, or a patient CPU.
- You want a model good at **your** tasks, extraction, classification, domain questions, not a general chatbot.
- Your documents cannot leave your environment.
- Different people in your organisation are allowed to see different things.
- You want to understand why it works, not copy commands.

**Coming from Java, C++, C# or backend Python?** [Doc 12](docs/12-for-experienced-developers.md) maps every concept here to patterns you already know. Adapters as plugins, merging as patch composition, distillation as caching a senior engineer's judgement.

## What STRATUM is not

- **Not a general-knowledge model.** It is good at the specific things you train it on. That is the point.
- **Not instant.** One stratum is minutes to a couple of hours on a laptop. Start it before lunch.
- **Not magic merging.** Strata from different bases, or deeply conflicting skills, will not fuse cleanly, and [doc 5](docs/05-merging.md) is honest about where the wall is.
- **Not a substitute for good data.** The biggest quality lever by far is the quality of your examples.
- **Not an identity provider.** `stratum serve` reads who is asking from a header. Put something in front of it that actually establishes that.

## Verify it yourself

```bash
python scripts/demo_concepts.py    # the core ideas, real numbers, no GPU
python -m pytest tests/ -v         # unit tests plus a full pipeline run on a tiny model
cd engine && make test             # the C tools, 31 checks including 116 corrupt files
stratum doctor                     # your GPU and library versions
```

Every code path here is covered: the optimizer math, delta extraction, all three merge methods, the loss mask, the scorers, the distillation loss, the access filtering, and the leak audit. The pipeline test builds a tiny model from scratch, trains two strata, merges them with every method, checks the merged weights are exactly base plus deltas, and evaluates the result, on CPU, in seconds. The same suite runs in CI on every push.

The access tests are written as attacks rather than checks, because a bug there is not a bad answer, it is a disclosure.

## Acknowledgements

STRATUM assembles published research and open tooling into one teachable pipeline. Credit for the underlying methods belongs to their authors.

- **LoRA** - Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- **QLoRA** - Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
- **Muon** - Keller Jordan (2024), building on Jeremy Bernstein's work on orthogonalized updates
- **TIES merging** - Yadav et al., "TIES-Merging: Resolving Interference When Merging Models" (2023)
- **DARE** - Yu et al., "Language Models are Super Mario" (2023)
- **Distillation** - Hinton, Vinyals and Dean, "Distilling the Knowledge in a Neural Network" (2015)
- **Compartmented adapters for access control** - [AC-LoRA](https://arxiv.org/abs/2505.11557) (NeurIPS 2025) reached this architecture first, and names the combinatorial problem that the tiering here exists to avoid
- **Parametric RAG** - [PRAG](https://github.com/oneal2000/PRAG) (SIGIR 2025), encoding documents into adapters
- **Retriever supervision from generated questions** - [Promptagator](https://arxiv.org/abs/2209.11755) (ICLR 2023), and doc2query, InPars and GPL before it
- **Graph building without a model per document** - LazyGraphRAG (Microsoft) and [LinearRAG](https://arxiv.org/pdf/2510.10114) (ICLR 2026)
- **GGUF and the block quantization formats** - Georgi Gerganov and the [ggml and llama.cpp](https://github.com/ggml-org/llama.cpp) contributors
- **The idea behind the C engine** - [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) by Fareed Khan. [engine/README.md](engine/README.md) explains the debt in full, including the measurement that killed the original plan
- **Tooling** - Hugging Face Transformers and PEFT, bitsandbytes, PyTorch, and the Qwen team's open models

Thanks also to the people who ran this on hardware the author does not have and reported what broke. The adapter dispatch failure on Apple silicon and the Muon learning rate problem on small models were both found that way, on an Intel Mac and a Colab T4, and both are fixed. Bug reports from a machine nobody on the project owns are worth more than they look.

## License

MIT. Use it commercially, modify it, ship it. Contributions welcome, see [CONTRIBUTING.md](CONTRIBUTING.md).

<div align="center">

*Small layers. One model.*

</div>
