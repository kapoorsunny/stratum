# 17 - Access control, and the context index

*Putting a company's documents into a model is easy. Making sure the model only tells each person what they are allowed to know is the hard part, and it has to be decided before training rather than after. This chapter covers both, and the test that proves it worked.*

---

## The problem in one sentence

**You cannot filter a weight.**

Everything else follows from that. If a salary figure gets trained into a model's parameters, there is no check you can add later that keeps it away from a contractor. The model does not look the number up and it does not retrieve it. It simply knows it, the same way it knows what a pump is, and the only way to make it not know is to train again without it.

This is different from every access control problem you have solved before. A database row has an owner and a query can be filtered. A file has permissions and a read can be denied. A number inside a matrix of weights has neither.

So the decision about who may see what has to be made at build time. That is what this chapter is about.

---

## Two ideas

### A compartment

A **compartment** is a set of documents that share an access policy. It is usually something your company already has: a SharePoint site, a network share, a folder, a Confluence space.

STRATUM takes it from the folder your documents are in:

```
corpus/
  public/        <- everyone
  engineering/   <- the engineering department
  safety/        <- the safety department
  finance/       <- the finance team
  hr/            <- HR only
```

```bash
stratum corpus ingest --in corpus/ --out chunks/ --compartments
```

Every chunk comes out labelled with its top folder. Reusing the sort your company has already done is better than inventing a second one, because the one that already exists is the one somebody keeps up to date.

### A tier

A **tier** says where that compartment's knowledge is allowed to live. There are three, and which one a compartment gets is forced by two questions rather than chosen by preference.

| Tier | When | Where the knowledge goes |
|---|---|---|
| **company** | everyone can see it, and it changes rarely | into the weights, in one shared adapter |
| **department** | a whole department shares it, and it changes rarely | into the weights, in that department's adapter |
| **restricted** | narrow, or sensitive, or changes often | **never into the weights.** Index only |

The two questions are **how many people can see it** and **how often does it change**. Nothing else.

---

## Why not just compartment everything

Because it does not survive contact with a real organisation, and it is worth understanding exactly how it fails.

Suppose every compartment gets its own adapter, and a person loads the adapters for the compartments they can see. That sounds right. Now count. If somebody holds permissions to a particular combination of compartments, and adapters had to be trained for that combination, then with n compartments there are 2ⁿ possible combinations. At fifty compartments that is more adapters than there are atoms worth caring about.

This is not a hypothetical. It is the stated open problem in [AC-LoRA](https://arxiv.org/abs/2505.11557) (NeurIPS 2025), which is the closest published work to this design and which names the exponential blow-up as the thing it does not solve.

There is a second, more practical limit. Adapters do not compose indefinitely. Merging more than about **three** degrades a model badly, and this project measured that directly: three strata merged at full strength produced a model that generated **nothing at all**, empty output on every prompt, with healthy training losses right to the end. The published literature lands in the same place.

**The way out is that real permissions are not arbitrary combinations.** They are a lattice. Nobody holds a random twelve of fifty compartments. They hold *public, plus their department, plus their team, plus a couple of projects.* So only the broad, stable, hierarchical tiers go into weights, where the count per person is two or three. Everything narrow stays in the index, where filtering a row is easy and deleting one is instant.

It also puts each mechanism on the job it is actually good at:

- **Training is good at form.** Vocabulary, phrasing, how your organisation answers a question of this kind.
- **Retrieval is good at facts.** Specific numbers, current values, the thing that changed last Tuesday.

Which means the volatile facts never go into the weights, so there is nothing stale baked in and nothing to unlearn when a document is withdrawn.

---

## Writing the policy

```bash
stratum access init --out policy.json
```

That writes a starting policy and prints it:

```
Compartments
  public         company     weights + index  5 principal(s)
  engineering    department  weights + index  3 principal(s)
  safety         department  weights + index  3 principal(s)
  finance        restricted  index only       1 principal(s)
  hr             restricted  index only       1 principal(s)
                 volatile, so it stays out of the weights

Principals
  contractor     sees 1 compartment(s), loads 1 adapter(s)
                 adapters: public
                 blocked : engineering, finance, hr, safety
  ...
Most adapters any one principal loads: 3
```

It is a plain JSON file on purpose. An access policy nobody can read is an access policy nobody can audit, and the first question in any security review is going to be *show me exactly who can see this.*

### What the policy refuses to let you write

These are errors, not warnings, because each one is a disclosure waiting to happen:

- **A `company` tier compartment that somebody cannot see.** Its content goes into an adapter everybody loads, so the tier is a lie.
- **A `department` tier compartment only one person can see.** Training a shared adapter on one person's data puts it in a model other people load.
- **A `volatile` compartment in the weights.** Volatile means it changes, and changing means retraining, so it must not be in there.
- **A grant to a compartment that does not exist.** Almost always a typo in the name of a real one, which silently grants nothing while looking like it granted something.

### Checking it against reality

```bash
stratum access check policy.json --chunks chunks/chunks.jsonl
```

This compares what the policy says against what the corpus actually contains, and reports chunks labelled with compartments the policy has never heard of, compartments with no chunks, chunks with no label at all, and any compartment whose tier is not what its visibility supports.

---

## The context index

The other half. Facts the model should look up rather than memorise.

```bash
stratum context build --chunks chunks/chunks.jsonl --out index/
```

```
Indexing 1097 chunks
  compartments: engineering, finance, hr, public, safety
  embedding with the built-in hashing embedder, dim 512
  term index: 24830 terms, 204407 postings
  links: 13164 edges over 1097 chunks (12.0 per chunk)
  index -> index/  (4.4 MB)
```

Three parts, and each does a different job.

**Vectors** find text that means the same thing as your question even when it uses none of the same words. The default embedder needs no download, no GPU and no internet. It hashes words and word pairs into a fixed number of positions, which sounds crude and works surprisingly well on technical material because the vocabulary is doing the work.

It has a real limit, and it is worth knowing before you rely on it. Measured on the same held-out questions, over 1,115 chunks:

| Compartment | built-in hashing | `--embedder hf` |
|---|---|---|
| public | 77.8% recall@5 | **100%** |
| engineering | 86.4% | **95.5%** |
| safety | 88.9% | **100%** |

The gap opens on **paraphrase**. Asked "What is the maintenance contract with Vendor A worth?", the hashing embedder missed the contract entirely and returned three unrelated chunks, while the model put it first. Ask the same thing with the document's own words, "Vendor A maintenance framework annual value payment terms net 45", and the hashing embedder finds it at rank one.

So use the built-in one when your users search with the vocabulary of the documents, which in engineering is more often than you would expect. Use `--embedder hf` when they ask questions in their own words. It costs one download of about 90 MB and runs on the processor.

**A term index** finds the chunk that contains the exact part number. This matters more than it sounds. Somebody asking about `V-201` wants the document that says `V-201`, and no amount of semantic similarity beats an exact match on a code. The two are combined by reciprocal rank fusion, which needs no tuning and does not care that the two scores are on completely different scales.

**Links** connect chunks that mention the same distinctive things. Two chunks get an edge when they share terms that are rare across the corpus. Rare is what carries the signal, because every chunk mentions "the" and only a few mention "blowout preventer". What this buys is reach: a question about a valve can arrive at the procedure that governs it without ever naming it, because the two share an equipment tag.

Building the graph costs **no language model calls at all**. That is not novel and should not be. Microsoft's LazyGraphRAG and [LinearRAG](https://arxiv.org/pdf/2510.10114) both do structure extraction without paying a model per document, which is what makes graph building affordable at enterprise scale in the first place.

### Filtered before scored, and why the order matters

```bash
stratum context query "What is the certified clearance of asset ENG-4064?" \
  --index index/ --policy policy.json --principal contractor
```

A principal who cannot see a compartment never has its chunks **ranked**, not even to be ranked and then discarded.

That ordering is the whole thing. Filtering after ranking leaks in two ways. It leaks through *displacement*: a hidden chunk that scores highly takes a slot, so the caller gets four results where they should have had five, and the gap tells them something exists. And it leaks through timing. Both survive a demo and fail an audit.

**Links are filtered on every hop too.** An unfiltered edge is a hole straight through everything above, and it is the hole people leave open, because they filter the search, feel finished, and then let expansion walk wherever the graph goes.

---

## Proving it, with canaries

Every design in this area claims isolation. What an enterprise actually asks for is a test that attacks the claim and can be re-run on every rebuild.

```bash
stratum access audit policy.json \
  --index index/ --chunks chunks/chunks.jsonl \
  --strata-dir strata/ --out audit.json
```

A **canary** is a fact that exists in exactly one compartment and could not be guessed, invented, or reasoned out from public material:

```
Asset ENG-4064 carries a certified clearance of 42.418 millimetres
as recorded in the engineering register.
```

Then for every principal, and every compartment they cannot see, the audit asks a question whose only answer is that canary. Anyone who produces it is a leak.

Canaries are the right instrument precisely **because they cannot be arrived at honestly**. If a contractor's model says `42.418`, there is no innocent explanation, no partial credit, and no argument about whether the answer was close enough. That is what makes it a test rather than an impression.

Three surfaces get attacked, because they fail differently:

| Surface | The failure |
|---|---|
| `index` | retrieval returns a forbidden chunk |
| `links` | following an edge reaches one |
| `model` | the loaded adapters produce the canary **with no retrieval at all** |

The third is the one that matters most, because it is the one that cannot be fixed afterwards. If a canary is in the weights somebody loaded, no filter anywhere downstream will keep it from them.

A surface that was not tested is reported as **not tested**, never as passed. A report that counts what it did not check as clean is worse than no report.

---

## Serving it

```bash
stratum serve strata/* --router router.json \
  --context index/ --context-chunks chunks/chunks.jsonl \
  --policy policy.json
```

Each request names who is asking:

```bash
curl http://127.0.0.1:8927/v1/chat/completions \
  -H "X-Stratum-Principal: engineer" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"..."}]}'
```

Every response reports which sources went into it and which compartment each came from, so an answer can always be traced to documents rather than believed.

**That header is not authentication and must never be treated as any.** Anyone can set it. It is the place a real deployment puts the identity its own gateway has already established, and the server says so on startup rather than letting anyone assume otherwise.

---

## Updating quickly

This is where the tiering pays off a second time.

| What changed | What has to happen | How long |
|---|---|---|
| A restricted document | re-embed the changed chunks | seconds |
| A department document | retrain that one compartment's adapter | minutes |
| A company document | retrain the shared adapter | scheduled, rarely |

And for **right to erasure**, which is the one that ends pilots: a deletion request is almost always about restricted-tier material, somebody's record, a retracted contract, a customer file. Delete the row, it is gone, nothing to retrain. The company and department tiers hold broadly-visible stable material, which is exactly the data nobody asks you to erase.

That is not a coincidence. It is the same rule, applied twice. Data that is broadly visible and stable is both safe to train and unlikely to be withdrawn.

---

## From a laptop to a company

The artifact does not change.

On a laptop you have one compartment, everything you can see, so it is one adapter and one flat index. That is what STRATUM did before this chapter existed.

At company scale, the compartment is the shard key. Same file format, same code path, larger `n`.

A flat scan over the vectors is linear and beats an approximate index until roughly a **million chunks**, because at that size the scan takes about 150 ms and generating the answer takes seconds. Retrieval is a fraction of a percent of a request, and optimising it further is unobservable.

The reason a flat scan wins is measured rather than assumed, and it is in [engine/README.md](../engine/README.md): on real storage, **what costs is the number of reads you have to do one after another**, not how big they are. Every read carries about 205 microseconds of fixed cost regardless of size, so a 256 KB read gets you 64 times the data of a 4 KB read for 2.4 times the time. A flat scan is one read. A graph walk is many, each waiting on the last.

That also explains why sharded graph indexes fall over: in a 100M-vector HNSW index across five nodes, [over 80% of search steps become cross-node traversals](https://arxiv.org/pdf/2512.17264), and every one of them is a dependent round trip.

---

## What this does not solve

- **The header is not identity.** Put a real gateway in front of it.
- **PII is still your pipeline's job.** `--redact` is a regex second net, and in the training path it is the last net, because anything it misses goes into the weights.
- **Retrieval may not help a fine-tuned model at all.** Strata are trained closed-book, so context in the prompt is out of distribution for them. [ChipLingo](https://arxiv.org/abs/2604.27415) measures retrieval as worth +7.3 to a base model and **−5.5** to the same model after domain adaptation. Measure it on your own corpus before assuming, with `stratum eval --context-style oracle`, which hands the model the exact right chunk. If it cannot use that, no retriever will save it.
- **There is no audit log of what was retrieved for whom.** The response carries its sources, but nothing here writes them down.
- **A small model will invent an answer when the context does not contain one.** The prompt says to say so if the material does not answer the question, and a 1.7B model ignores that instruction. Measured here: with the right chunk missing from the context, three separate models produced three confident and completely different figures. Retrieval quality is therefore not a nicety. It is the difference between an answer and a fabrication, and it is why the recall numbers above matter more than they look.

---

## What you now know

- **You cannot filter a weight**, so who-sees-what is decided before training, not after.
- A **compartment** is a set of documents sharing a policy, taken from folders you already have.
- A **tier** decides where its knowledge may live, and is forced by **how many people see it** and **how often it changes**.
- Compartmenting everything into adapters **explodes combinatorially** and hits a hard merge limit of about three. Tiering is what avoids both.
- Retrieval is **filtered before scoring**, and **links are filtered on every hop**, because filtering afterwards leaks through displacement.
- **Canaries** turn an isolation claim into a test that fails loudly, on three separate surfaces.
- The **update story falls out of the tiering**: the volatile data was never in the weights, so withdrawing it costs nothing.

Next: [what changed and why ->](16-what-changed.md), or back to [the index ->](README.md)
