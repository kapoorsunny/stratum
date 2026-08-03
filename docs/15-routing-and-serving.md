# 15 - Routing and serving

*Merging is not the only way to use several strata. You can keep them apart and pick one per request, which costs nothing in interference and lets you add a skill without rebuilding anything. Then serve the result so Claude Code, an IDE, or any OpenAI-compatible client can use the model you built.*

---

## The problem with merging everything

Doc 5 is honest about where fusing runs out: strata interfere, and past three or four skills each signal gets fainter. This project hit the wall itself. Three strata merged at full weight produced a model that generated nothing at all - empty output on every prompt, with healthy training losses right up to the end.

Averaging the weights fixed that particular failure, but the underlying tension does not go away. Every skill you fuse in dilutes the others a little, and there is no setting at which twenty skills coexist cleanly in one set of weights.

## The other answer

Keep the strata separate. Load the frozen base once, attach every adapter to it, and choose which one answers each request.

```bash
stratum route train strata/extract strata/classify strata/policy --out router.json
stratum serve strata/* --router router.json
```

What that buys:

- **No interference at all.** One adapter is active at a time, running at exactly the strength it was trained to.
- **Adding a skill is training one adapter.** No re-merge, no re-evaluation of the skills you already had.
- **Removing one is deleting a folder.**
- **Memory is one base plus a few megabytes per skill.** A 1.7B base with twenty skills is about 3.5 GB, not twenty models.

Switching skills costs nothing measurable. Every adapter is attached to the same loaded base, so selecting one is a pointer change rather than a load.

## Routing at the request level, not the token level

Large mixture-of-experts models route per token, choosing a handful of experts for each one. That is the finest possible granularity and the most expensive to serve, because the working set changes constantly.

STRATUM routes per **request**, and that difference matters more than it sounds. A request's subject does not change halfway through the answer, so one routing decision covers hundreds of generated tokens and its cost disappears into the noise. The locality is structural rather than statistical.

There is a cautionary tale in the token-level approach. A well-known engine that streams a 2.78-trillion-parameter model from disk publishes its own cache measurements, and they show the expert cache stuck at a 36% hit rate from 8 GB all the way to 64 GB - an eightfold range of memory that buys nothing. Its author names the reason: modern MoE training deliberately *balances* expert usage for training efficiency, and flat usage is exactly what defeats a cache. Token-level routing fights the model's own training objective. Request-level routing does not.

## Building a router

```bash
stratum route train strata/extract strata/classify --out router.json
```

The training data is free. Every stratum's card records which JSONL trained it, so the prompts are the examples and the skill name is the label - no annotation, no extra collection. The router itself is TF-IDF over words, word pairs and character n-grams with one centroid per skill. No model to load, microseconds per decision, and the result is a JSON file you can read and hand-correct.

Character n-grams matter more than they look for technical domains. They catch equipment codes and units - `V-201`, `84 bar` - that whole-word matching misses when the exact token never appeared in training.

Check where a request would go:

```bash
stratum route test "Extract the total from this invoice: 'Amount due 412 EUR'" --router router.json
```

```
skill      : extract
confidence : 0.816
   extract                  0.9012
   classify                 0.0852
```

## Confidence, and when routing is the wrong tool

**Confidence is the margin between the best skill and the runner-up**, not the raw score. A request that suits two skills equally reports a low number rather than presenting a coin flip as a decision. Under about 0.15, treat it as unclear.

That number is also the honest test of whether your skills should be routed at all. Building this feature produced both outcomes:

| Skills | Training accuracy | Unseen requests | Verdict |
|---|---|---|---|
| extract vs classify | 100% | 4 of 4 correct, confidence 0.38-0.82 | route them |
| three skills from one corpus | 98.9% | misrouted, confidence 0.02-0.13 | merge them |
| the same three from a corpus 3.6x larger | 92.5% | 3 of 3 correct, confidence 0.003-0.035 | merge them |

The third row was run to check whether more data would rescue the second, and it does not. A corpus of 1,097 chunks instead of 300 moved training accuracy around a little and left confidence where it was, near zero.

It is worth looking at what that row actually did, because it is the router behaving well rather than badly. Asked three clearly different questions, it picked the right skill every time, and reported that it could not tell. Both of those are true at once. The winner really was slightly ahead, and slightly ahead of two others out of three is not a decision worth acting on. A router that reported 0.9 there would be lying.

The second row is the interesting one. Those three skills were generated from the same document set with different instructions, and the teacher wrote similarly-worded questions for all of them. They are not separable from the request text, and no router can fix that - the information simply is not in the input.

**So the rule is:** skills a reader could tell apart from the request alone should be routed. Skills that only differ in what you want done with a similar-looking request should be merged. High training accuracy with low confidence on real requests is the signature of the second case, and `stratum route train` prints a warning naming the skills that overlap.

## Serving it

```bash
stratum serve strata/* --router router.json --port 8927
```

```
Serving 3 skills on http://127.0.0.1:8927
  skills : energy-qa, energy-extract, energy-explain
  base   : Qwen/Qwen3-1.7B

Point any OpenAI-compatible client at it:
  base URL  http://127.0.0.1:8927/v1
  api key   any non-empty string
  model     'auto' to route, or a skill name to force one
```

The endpoint speaks the OpenAI chat dialect, which is what makes the model usable from tools that have never heard of STRATUM - Claude Code, IDE assistants, `curl`, any client with a base-URL setting. Every response also carries a `stratum` block reporting which skill answered and how confident the router was, so routing is never invisible.

```bash
curl http://127.0.0.1:8927/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"How does a combined cycle plant recover waste heat?"}]}'
```

Naming a skill instead of `auto` bypasses the router entirely - a caller who knows what it wants is never second-guessed.

## Chatting with what you built

```bash
stratum chat models/my-slm                          # a merged model
stratum chat strata/* --router router.json          # a pool of skills
```

The second form keeps conversation history and prints which skill handled each turn, so you can watch routing behave on real questions rather than trusting the accuracy number.

## Merge or route

| | Merge | Route |
|---|---|---|
| Skills that interfere | dilutes both | unaffected |
| Adding a skill | rebuild and re-evaluate | train one adapter |
| Artifact | one standalone model | base plus adapters |
| Serving | any stack that loads a model | `stratum serve`, or PEFT |
| Deploy to llama.cpp or a quantized runtime | straightforward | needs the adapters merged first |
| Best for | two or three complementary skills | many skills, or skills that fight |

They are not exclusive. Grouping is often right: merge the two or three skills that genuinely belong together, keep the rest as separate strata, and route between the groups. Doc 10's pattern A and pattern B are the same idea from the production side.

## What you now know

- **Merging is one option, routing is the other**, and the choice is about whether your skills interfere.
- Routing keeps skills at **full strength**, makes adding one cheap, and costs one base plus megabytes.
- STRATUM routes per **request**, where locality is structural - unlike token-level MoE caching, which fights the model's own load balancing.
- **Confidence tells you whether routing suits your skills at all.** Low confidence on real requests means merge them instead.
- `stratum serve` exposes the result on the **OpenAI dialect**, so existing tools can use a model you trained yourself.

Next: [the glossary ->](11-glossary.md), or back to [merging ->](05-merging.md)
