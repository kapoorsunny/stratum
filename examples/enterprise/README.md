# Enterprise example: five departments, one model, nobody sees too much

A worked build of an access-controlled model over a mixed corpus, with every command and every measured number. It is the reference for [doc 17](../../docs/17-access-control-and-context.md) and [doc 18](../../docs/18-deploying-at-scale.md).

Everything here ran on one laptop: an RTX 4070 with 8.6 GB of VRAM.

---

## The corpus

62 documents and 5 photographs, sorted into folders by who is allowed to read them.

| Compartment | Contents | Chunks |
|---|---|---|
| `public` | site glossary, tag conventions, published standards | 132 |
| `engineering` | equipment and process material, plus 5 real equipment images | 848 |
| `safety` | hazard studies, integrity, inspection, operating practice | 127 |
| `finance` | contracts, capital plan, insurance position | 4 |
| `hr` | pay bands, allowances, procedure | 4 |
| | **total** | **1,115** |

The public, engineering and safety material is real energy engineering content from Wikipedia and a US EIA report. **The finance and hr documents are synthetic and say so in every file.** They exist to be things that must not leak, not to be true.

The images are real equipment photographs and diagrams from Wikimedia Commons.

---

## The policy

Seven principals across five compartments.

| Principal | Sees | Loads adapters |
|---|---|---|
| `contractor` | public | 1 |
| `engineer` | public, engineering | 2 |
| `safety-officer` | public, safety | 2 |
| `plant-manager` | public, engineering, safety | 3 |
| `commercial-lead` | public, engineering, finance | 2 |
| `hr-lead` | public, hr | 1 |
| `executive` | everything | 3 |

Note the asymmetry: `commercial-lead` **sees** three compartments but loads only **two** adapters, because finance is restricted tier and never enters weights. `hr-lead` sees two and loads one, for the same reason.

The most any principal loads is three, which is the measured safe merge limit.

---

## Every command

```bash
E=examples/enterprise

# 1. documents and images in, labelled chunks out
stratum corpus ingest --in $E/corpus --out $E/chunks --compartments --images hf --redact

# 2. does the policy match what is actually there
stratum access check $E/policy.json --chunks $E/chunks/chunks.jsonl

# 3. a teacher writes training data, one compartment at a time
for C in public engineering safety finance hr; do
  stratum corpus pairs --chunks $E/chunks/chunks.jsonl --compartment $C \
    --instruction "..." --out $E/data/$C.jsonl --test-out $E/data/$C-test.jsonl \
    --teacher claude-cli --concurrency 6 --per-chunk 3 --test-fraction 0.1
done

# 4. plant a canary in each compartment that reaches the weights
stratum access plant $E/policy.json --data-dir $E/data

# 5. one stratum per weights-tier compartment
for C in public engineering safety; do
  stratum train --skill $E/data/$C.jsonl --out $E/strata/$C \
    --base Qwen/Qwen3-1.7B --epochs 4 --rank 16
done

# 6. the index, and a router over the compartments
stratum context build --chunks $E/chunks/chunks.jsonl --out $E/index --embedder hf
stratum route train $E/strata/* --out $E/router.json

# 7. try to break it, and fail the build if it breaks
stratum access audit $E/policy.json --index $E/index \
  --chunks $E/chunks/chunks.jsonl --strata-dir $E/strata --expand 8

# 8. bundle it for the serving machines
stratum pack $E/strata/* --out $E/bundle --index $E/index \
  --policy $E/policy.json --router $E/router.json --chunks $E/chunks/chunks.jsonl

# 9. on each serving node
stratum unpack $E/bundle --run --host 0.0.0.0
```

---

## What it measured

### The audit passes, and the pass means something

```
Access audit over 7 principals and 5 canaries
  index and links: 17 forbidden lookups attempted, 0 leak(s)
  control        : 5 canaries, 0 that nobody authorised could reach
  model weights  : 17 forbidden questions asked, 0 leak(s)

PASSED on index, links, model.
```

The **control** line is the one that matters. It confirms every canary really was learned by somebody authorised, so the zero leaks below it is a genuine result rather than a model that never knew anything. An earlier run reported `INCONCLUSIVE` for exactly this reason, which is what the control is for.

### Routing

99.8% over 420 held-out requests, because compartments are topically distinct in a way that task-type skills are not. Compare [doc 15](../../docs/15-routing-and-serving.md), where three task skills over one corpus routed at near-chance confidence.

| Compartment | Accuracy |
|---|---|
| public | 100.0% (120/120) |
| engineering | 99.4% (176/177) |
| safety | 100.0% (123/123) |

### Retrieval, and why the embedder choice matters

Recall of the exact chunk each held-out question was written from.

| Compartment | built-in hashing | `--embedder hf` |
|---|---|---|
| public | 77.8% @5 | **100%** |
| engineering | 86.4% @5 | **95.5%** |
| safety | 88.9% @5 | **100%** |

The gap is entirely about paraphrase. "What is the maintenance contract with Vendor A worth?" found nothing useful with the hashing embedder and found the contract at rank one with the model. Ask it in the document's own words and both find it immediately.

### Serving, per person

The same question, asked by three people, through the OpenAI-compatible endpoint:

| Principal | Sources returned |
|---|---|
| `contractor` | 3 chunks, all `public` |
| `engineer` | 3 chunks, all `engineering` |
| `executive` | `safety` and `engineering` |

A request with no `X-Stratum-Principal` header is refused. An unknown principal is refused by name, with the list of valid ones.

### The image pipeline

Five images read by a local `Qwen2.5-VL-3B-Instruct`, on an 8.6 GB card.

| | |
|---|---|
| Before downscaling | over 15 minutes for one image |
| After capping the long edge at 1024 px | 4 images in 2 minutes |

Quality splits by image type, and it is worth knowing which is which. Diagrams read well: the crude oil distillation figure came back as `Crude Oil FURNACE Gasoline (Petrol) 150°C Kerosene 200°C Diesel Oil 300°C Fuel Oil 370°C`, and the gas turbine efficiency chart came back as a table of values. Photographs read poorly: a turbine hall photo produced one character, and a rotor photo produced only the manufacturer name off the casing. Use a larger vision model, or an API one, if photographs carry information you need.

---

## Three things this build got wrong first

Kept because the failures are more useful than the successes.

**The audit passed for the wrong reason.** The first version planted canary documents in the corpus and trusted the teacher to write about them. It did not. The canaries never reached the weights, so nothing could leak them, and the audit reported a clean pass. The fix was `stratum access plant`, which writes the canary pairs directly, plus a control that reports `INCONCLUSIVE` when no authorised principal can produce a canary.

**The audit only attacked one adapter.** The generator asked the first skill alphabetically rather than every skill the principal could load. For `plant-manager`, holding three adapters, two thirds of the attack surface went untested. Somebody trying to reach a compartment they should not have would try all of them, so now the audit does too.

**The link graph was empty and said nothing.** Edges were selected by keeping terms above the median IDF, which sounds reasonable and is wrong, because most terms in any corpus appear exactly once, so the median sits on singletons and every term that could connect two chunks gets excluded. The index built successfully with zero edges. Now selection is by document frequency directly, and the builder warns loudly if it ever produces no links again.

---

## Reproducing it

The corpus in this folder is the sorted document set and the policy. Fetching the source material, the teacher pass and the training runs are all in the commands above. The teacher pass is the only part that costs anything.

Expect about 40 minutes for the teacher pass at `--concurrency 6`, and about 2 minutes per stratum on a laptop GPU.
