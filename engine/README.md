# The STRATUM engine

A small C engine for reading GGUF checkpoints and running sparse
mixture-of-experts models whose weights do not fit in memory.

This is a separate, optional part of STRATUM. Nothing in the Python
toolchain needs it, and you can build strata, merge, route, serve and
evaluate without ever compiling a line of C.

## Why it exists

STRATUM's `teacher-gen` and `corpus pairs` commands send thousands of
independent prompts to a teacher model and collect the answers. Nobody
watches a token appear. It is a throughput job, and the better the teacher
the better the resulting skill.

That makes very large sparse models interesting as teachers, and they are
exactly the models that will not fit on a laptop. This engine is about
running one anyway, for that workload specifically.

## Credit where it is due

**The inspiration is [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)
by Fareed Khan**, Apache-2.0 licensed. It runs a 2.78-trillion-parameter
model on one CPU in 8.24 GB by keeping the always-on part of the network
resident and streaming the sleeping experts from disk. It is careful, honest
work and worth reading in full.

Two things in that project shaped this one directly.

The first is a measurement it publishes against itself. Its expert cache
holds a 36.24% hit rate from 8 GB of cache all the way to 64 GB, and its
documentation names the reason plainly: mixture-of-experts training
deliberately balances how often each expert is used, and flat usage is
exactly what defeats a cache. Being told, by the author, precisely why the
obvious approach stops working is what pointed at trying a different one.

The second is a bug it documents rather than hides. Its batch prefetch marked
a cache slot empty before reading into it, so a fast path in the eviction
scan handed the same slot to the next expert in the same batch, several
parallel reads landed in one buffer, and the model multiplied garbage. The
comment explaining this sits in
[src/cache/k3_cache.c](https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/src/cache/k3_cache.c).
Anyone writing a scheduler here should read it first, because the same trap
is waiting.

No code was taken from it. The two engines differ in format and in I/O. That
project reads safetensors with `O_DIRECT` `pread` into a fixed slot arena.
This one reads GGUF through a memory mapping with residency hints.

**The GGUF format** is from [ggml and llama.cpp](https://github.com/ggml-org/llama.cpp)
by Georgi Gerganov and contributors, MIT licensed. The block layouts in
`gguf.c` follow that specification. Reading somebody else's format is the
point of a format, but the work of designing it was theirs.

**The block quantization schemes** (Q4_K, Q6_K, Q8_0 and the rest) are also
ggml's design.

**The scheduling idea below is not ours either.** Grouping a batch of tokens
by expert so that one weight load serves many tokens is published work, and
it was already published before this was written:

- **MoE-Gen** ([arXiv 2503.09716](https://arxiv.org/abs/2503.09716)) states
  it directly. It defers loading an expert until enough tokens are waiting
  on it, and enlarges the batch specifically to amortize the cost of the
  load.
- **Klotski** (ASPLOS 2025,
  [arXiv 2502.06888](https://arxiv.org/abs/2502.06888)) builds an
  expert-aware multi-batch pipeline across GPU, host memory and disk, using
  compute on one layer to hide the next layer's load.
- **LayerScope and PreScope**
  ([arXiv 2509.23638](https://arxiv.org/abs/2509.23638)) do predictive
  cross-layer scheduling for multi-batch inference on ordinary servers.
- **llama.cpp** has grouped matmuls by expert across a batch since 2023, in
  `ggml_mul_mat_id`. Expert-major compute order was never new.

The narrow thing that is not already covered is the operating point. Those
systems assume the checkpoint fits in host memory, and their eval hardware
has hundreds of gigabytes of it. None of them run at twelve times
oversubscription with a drive as the primary tier rather than an overflow.
Their cost models are also calibrated on Mixtral shaped models, with eight
experts and two firing, and applied to a model with 896 experts and sixteen
firing they predict a speedup about fifty times larger than the real one.
Getting that arithmetic right for the many-experts case is the only original
work here, and it is arithmetic, not architecture.

## What we do differently

The reference engine's unit of work is one token. For each token it finds
the handful of experts that token needs, reads them, and multiplies. An LRU
cache tries to keep useful experts around between tokens, and the balanced
usage defeats it.

This engine's unit of work is one layer, for every token in flight at once.

A forward pass is a loop nest. Layers have to run in order because layer L+1
needs layer L's output, but the tokens inside a layer do not depend on each
other at all. So the loops can be swapped. Run every token through layer L,
then move the whole batch to layer L+1.

Three things follow, and none of them follow from a better cache.

**The resident set becomes one layer instead of the model.** You need
layer L's experts, plus whatever the prefetcher is pulling in for L+1.

**Each expert is read exactly once per pass.** With enough tokens in flight,
the set of experts that somebody chose covers essentially all of them. There
is nothing to keep and nothing to evict, so the cache hit rate stops being a
number that matters.

**The reads can be coalesced.** The read order is completely known in
advance, because layer order is fixed and the experts within a layer can be
sorted by where they sit in the file. Neighbours in that sorted order can
then be fetched as one large read rather than several small ones.

That third point started life as a different and stronger claim, that
putting reads in file order would itself make them much faster. Measuring it
showed otherwise, and the measurement is below.

## What the drive actually does

`stratum-bandwidth` reads the same blocks from the same file three times, in
scattered order, in file order, and straight through. Same bytes, same
sizes, only the order changes. Reads bypass the operating system cache, so
this is the drive and not memory.

On the machine this was developed on, a laptop NVMe drive:

| Block size | Scattered | In file order | Straight through | Gain from ordering |
|---|---|---|---|---|
| 1 MB | 1.34 GB/s | 1.35 GB/s | 1.42 GB/s | 1.01x |
| 4 MB | 2.07 GB/s | 2.18 GB/s | 2.28 GB/s | 1.05x |
| 16 MB | 2.67 GB/s | 2.72 GB/s | 3.12 GB/s | 1.02x |
| 64 MB | 2.89 GB/s | 2.89 GB/s | 3.39 GB/s | 1.00x |

**Read order is worth nothing here.** Between zero and five percent, at every
size tried. On this drive, at the block sizes an expert actually occupies,
where the blocks sit relative to one another simply does not matter.

**Read size is worth a great deal.** Going from 1 MB reads to 64 MB reads
more than doubles the speed, from 1.34 to 2.89 GB/s.

That inverts the third argument above. Sorting experts into file order is
not useful because sorting is fast. It is useful because sorting puts
neighbours next to each other, and neighbours can be **merged into one big
read**. The win comes from the merging, not from the ordering.

Two caveats, because a measurement is only as good as its conditions. These
reads are issued one at a time, and NVMe drives reach their rated speed only
with several requests outstanding, which is why nothing here exceeds 3.4
GB/s on a drive rated well above that. And this is one drive. Run the tool
on yours before believing any of it.

## The numbers for the model itself

Everything below is for Kimi K3, which is the largest thing anyone has tried
this on. They come from the reference engine's own published measurements
and from the model's shape.

| | |
|---|---|
| Layers | 93, of which 92 are mixture-of-experts |
| Experts per layer | 896, of which 16 fire per token |
| Read per token, experts | 25.83 GB |
| Read per token, always-on weights | 108.81 GB, re-read every token |
| Read per token, total | 134.64 GB |
| State held per sequence | about 232 MB, which caps a batch near 320 on a 128 GB machine |
| Speedup, prefill | about 18x |
| Speedup, generation | about 6x |

Two of those deserve a second look.

The always-on weights are five times heavier than the experts and are read
again for every single token, so they dominate everything at small batch
sizes. They stop dominating above a batch of about four, after which expert
traffic takes over and the gap widens.

Those drive figures are from the reference engine's own hardware and show a
much wider spread between scattered and sequential reads than the drive
measured above. Which is the point of shipping the measuring tool rather
than a number. Run it on the machine you intend to use.

## The thing that decides whether any of this is worth doing

Everything above is about reading weights off the drive faster. It turns out
that is not the slow part, and finding that out changes what should be built
first.

Some words first, because the rest of this section leans on them.

A **floating point operation**, or **flop**, is one arithmetic step on one
decimal number. A multiply is one. An add is one. Running a language model
is almost entirely multiplying long lists of numbers together and adding up
the results, so counting flops is a reasonable way to count the work.

**Flops per second** is therefore a speed, in the same way miles per hour is
a speed. **GFLOP/s** is a billion of them per second. You will also see
**TFLOP/s**, which is a thousand billion.

Every processor has a **peak** figure, which is what it manages when the
arithmetic is arranged exactly the way the hardware likes. Nothing hits its
peak in practice. What matters is the fraction of it you get.

Now the measurement.

The test machine here is an ordinary high end laptop, an Intel Core i9 with
24 cores and 128 GB of memory. Arranged well, a processor like that does
somewhere near **9,700 GFLOP/s**.

The reference engine's arithmetic runs at about **21 GFLOP/s**. That is
**two tenths of one percent** of what the chip can do.

The reason is visible in its code, and it is a deliberate choice rather than
an oversight. The engine multiplies weights against **one token at a time**.
A processor is fast when it can reuse a number it has already fetched, and
with one token there is nothing to reuse, so the chip spends its time
waiting on memory rather than calculating. The engine also unpacks each
compressed weight into a small scratch array before using it, and it
accumulates in the slowest and most precise arithmetic available, on purpose,
so that its results can be checked bit for bit against a reference. Both
choices trade speed for confidence, which is a fair trade when correctness is
the thing you are trying to establish.

Put the two costs side by side, for a batch of 320 tokens:

| | Time |
|---|---|
| Reading the weights off the drive | about 4 minutes |
| Doing the arithmetic on them | about 50 minutes |

The drive is not what you are waiting for. It is outnumbered roughly twelve
to one.

**So the plan in this file is in the wrong order.** Scheduling reads more
cleverly saves four minutes out of fifty-four. Until the arithmetic gets
faster, a better read schedule is decoration.

The good news is that the fix and the schedule want the same thing. Once a
batch of tokens is gathered for one expert, that expert's weights can be
unpacked once and multiplied against all 320 tokens instead of one, which is
exactly the arrangement processors are fast at. The arithmetic and the reads
improve for the same reason.

But they are separate pieces of work, and only one of them is on the
critical path. **Anyone starting from this design should write the batched
multiply first and the read scheduler second.**

## Limitations

A sweep costs the same whether one token rides it or a million.

One person typing one message into a chat box gets no benefit from any of
this, and pays a full sweep for every token they wait on. This is a
throughput design, and it only pays when there is enough work to fill the
sweep. Batch generation fills it. Interactive chat does not.

The honest comparison is not against the reference engine, it is against not
needing any of this. A dense 70B model at 4 bits, on two second hand
graphics cards, writes a thousand training answers in about twenty minutes.
The best case here is a few days instead of several months. That is the
difference between impossible and merely slow, which is a real difference,
but it is worth having only if a very large sparse teacher is worth days
when a smaller one is worth minutes. That is a question about what you are
building, not about engineering, and it deserves an answer before anybody
writes the scheduler.

If you want a big model to answer you quickly and interactively, use a model
that fits in your memory, and use [llama.cpp](https://github.com/ggml-org/llama.cpp)
or [Ollama](https://ollama.com) to run it. STRATUM's `stratum teachers`
command will tell you which ones those are on your machine.

## Building

```bash
cd engine
make
make test
```

Any C99 compiler works. There are no dependencies.

## Layout

| File | What it is |
|---|---|
| `src/map.c` | read only file mapping, and the residency hints, for POSIX and Windows |
| `src/gguf.c` | GGUF parser that treats the file as hostile and never reads past the end |

## License

MIT, the same as the rest of STRATUM.
