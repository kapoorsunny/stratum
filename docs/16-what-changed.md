# 16 - What changed, and why

*Written for someone who used the previous version to build a model and came back to find new commands. Nothing here assumes you read the other chapters first, and every term gets explained where it appears.*

---

## The short version

You could already turn documents into a model. Five things are new.

| | What it is |
|---|---|
| `stratum setup` | Installs whatever your machine is missing, and knows the difference between a Mac, an NVIDIA machine and everything else |
| `stratum route` | Keeps your skills separate and picks one per question, instead of blending them into one model |
| `stratum serve` | Puts your model behind a web address so other software can use it |
| `stratum teachers` | Tells you which large models your machine can actually run, and how fast |
| `--concurrency` | Asks the teacher several questions at once instead of one at a time |

Two bugs that people reported are fixed, and there is a new folder called `engine/` containing some C code that is optional and that most people should ignore.

The rest of this chapter explains each one properly.

---

## Some words, first

If you have been using STRATUM without being sure what the words mean, this section is for you. Skip it if you already know.

**A model** is a very large pile of numbers that, given some text, guesses what text comes next. That is genuinely all it does. Everything else is built on top of that one trick.

**Training** means adjusting those numbers so the guesses get better at something you care about. You show it examples of question and answer, it guesses, you measure how wrong it was, and you nudge the numbers slightly in the direction that would have been less wrong. Then you do that a few hundred thousand times.

**A base model** is somebody else's finished model that you start from. Training one from nothing costs millions. Adjusting an existing one costs an afternoon.

**An adapter**, which this project calls a **stratum**, is a small set of extra numbers that sit alongside a base model and change its behaviour. The base model is left completely untouched. This matters because a base model is several gigabytes and an adapter is a few megabytes, so you can have twenty adapters and still only one base.

**A skill** is one thing you have taught the model, held in one adapter. Answering questions about pump maintenance is a skill. Pulling numbers out of a report is a different skill.

**Merging** is combining several adapters into one, so a single model can do several things.

**A teacher** is a bigger, smarter model that writes your training examples for you. You give it your documents, it writes the questions and answers, and you train your small model on what it wrote. This is how you get training data without a team of people writing it by hand.

**A GPU** is a graphics card. It happens to be very good at the kind of arithmetic models need, so training on one is often twenty to fifty times faster than training on the main processor.

**VRAM** is the memory on the graphics card. It is separate from your computer's normal memory and it is usually much smaller, which is why a model that fits on your hard disk may still not fit on your card.

---

## `stratum setup` - because the hardest part was never the code

The single most common way this project failed for people had nothing to do with STRATUM. It was that PyTorch, the library underneath everything, comes in several different builds, and installing the wrong one leaves you with a machine that quietly ignores your graphics card.

This is worth understanding because it is invisible when it happens. Nothing errors. Training just runs thirty times slower than it should, and unless you know what to expect, it looks like that is simply how long training takes.

It happened to this project's own author on this project's own machine, for several hours, on a laptop with a perfectly good graphics card sitting idle.

The really unpleasant part is that the obvious fix does not work:

```bash
pip install --upgrade torch          # does nothing useful
```

Because the wrong build is still called `torch` and still satisfies the requirement, so pip looks, sees it is already installed, and stops. You need this instead:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

Which nobody would guess.

So now there is a command:

```bash
stratum setup
```

It looks at your machine, works out what is wrong, and fixes it. If you want to see what it intends to do without letting it do anything:

```bash
stratum setup --dry-run
```

### What it does on a Mac

Macs need saying separately, because the advice for them is different in a way that catches people out.

**On an Apple silicon Mac** (M1, M2, M3, M4 and later), the graphics processor is built into the same chip as the main one, and Apple calls the way software reaches it **MPS**. The important thing is that **the normal PyTorch download already supports it**. There is no special version to hunt for. If you went looking for a Mac equivalent of that CUDA download line above, you were looking for something that does not exist, and `stratum setup` will now tell you so instead of leaving you searching.

Apple silicon also uses **unified memory**, meaning the graphics side and the normal side share one pool. There is no separate VRAM number to check. The number that matters is just how much memory the machine has.

**4-bit compression does not work on a Mac at all.** This one is worth being blunt about, because it is not a version problem or a missing package, and no amount of installing will fix it.

Compression here means storing each number in the model using fewer digits, so the whole thing takes less room. Full precision uses about two bytes per number. Four-bit uses about half a byte, so the model becomes roughly a quarter of the size and fits where it otherwise would not. The library that does this, `bitsandbytes`, is written for NVIDIA hardware specifically.

So on a Mac, you pick a model that fits as it is. `Qwen3-1.7B` is the usual answer. `Qwen3-4B` works if the machine has plenty of memory. `stratum setup` says this plainly rather than installing a package that would import successfully and then fail the moment you used it.

**On an older Intel Mac**, there is no graphics acceleration available at all and everything runs on the main processor. It works. It is slow. `Qwen3-0.6B` with a small dataset is the realistic size, and the command will say so rather than let you discover it after two hours.

---

## `stratum route` - the alternative to merging

### The problem

Say you have three skills and you want one model that does all three. The obvious answer, and the one this project has always offered, is to merge the adapters together.

Merging works, and it has a limit. Every skill you fuse in dilutes the others slightly. Two or three skills coexist fine. Past that, each one gets a little fainter.

While building the example in [chapter 14](14-from-corpus-to-model.md), this project ran into the extreme version of that. Three skills merged at full strength produced a model that generated **nothing at all**. Empty output on every question. The training had gone fine, the loss numbers looked healthy right to the end, and the result was a model that had been pulled so far in three directions at once that it landed nowhere.

### The other answer

Do not merge them. Keep every skill in its own adapter, load the base model once, attach all the adapters to it, and choose which one answers each question.

```bash
stratum route train strata/explain strata/extract strata/safety --out router.json
stratum serve strata/* --router router.json
```

What that buys you:

- **No dilution at all.** One skill answers at a time, at exactly the strength it was trained to.
- **Adding a skill means training one adapter.** No rebuilding, no re-testing what you already had.
- **Removing one means deleting a folder.**
- **The memory cost is one base model plus a few megabytes per skill.** Twenty skills on a 1.7B base is about 3.5 gigabytes, not twenty separate models.

Switching between skills costs nothing you could measure, because every adapter is already attached to the same loaded base. Choosing one is changing a pointer, not loading anything.

### Where the router comes from

You do not have to label anything. Every stratum records which file it was trained on, so the questions in that file are the examples and the skill's own name is the answer. The router is built out of training data you already had.

The router itself is not a model. It counts words, pairs of words, and short runs of characters, and keeps an average of those counts per skill. A new question gets compared against each average and the closest one wins. It loads instantly, decides in microseconds, and the result is a readable JSON file you can open and correct by hand if it gets something wrong.

Counting short runs of characters matters more than it sounds for technical work. It is what catches `V-201` and `84 bar` as recognisable shapes even when that exact code never appeared in training.

### Confidence, and knowing when routing is wrong for you

```bash
stratum route test "Extract the total from this invoice: 'Amount due 412 EUR'" --router router.json
```

```
skill      : extract
confidence : 0.816
   extract                  0.9012
   classify                 0.0852
```

**Confidence is the gap between the winner and the runner up**, not the winner's raw score. That choice is deliberate. A question that suits two skills equally reports a low number instead of dressing up a coin flip as a decision. Below about 0.15, treat it as undecided.

That number also tells you whether your skills should be routed at all, and building this feature produced both answers:

| The skills | Accuracy on training data | On new questions | What that means |
|---|---|---|---|
| extract against classify | 100% | 4 out of 4 right, confidence 0.38 to 0.82 | route them |
| three skills from one document set | 98.9% | wrong answers, confidence 0.02 to 0.13 | merge them instead |

The second row is the interesting one and it is worth sitting with. Those three skills came from the same documents with different instructions, and the teacher wrote similarly worded questions for all three. Nothing in the question itself says which skill should handle it. No router can fix that, because the information is not there to find.

**The rule:** skills a person could tell apart just by reading the question should be routed. Skills that differ only in what you want *done* with a similar looking question should be merged. High accuracy on training data combined with low confidence on real questions is exactly what the second case looks like, and `stratum route train` now prints a warning naming the skills that overlap.

---

## `stratum serve` - using what you built

A trained model sitting in a folder is not much use until something can talk to it.

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

The phrase **OpenAI-compatible** needs unpacking. When OpenAI published their web interface, everyone else copied its shape, and it became the common language that tools speak to models. Almost every coding assistant, chat client and IDE plugin has a box where you can put a different address.

So by speaking that language, a model you trained this morning can be used by software that has never heard of STRATUM. Point your editor at it. Point a script at it. Point `curl` at it.

```bash
curl http://127.0.0.1:8927/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"How does a combined cycle plant recover waste heat?"}]}'
```

Every answer carries an extra block saying which skill handled it and how confident the router was, so the routing is never hidden from you. Naming a skill instead of `auto` skips the router entirely, because a caller that already knows what it wants should not be second guessed.

To just talk to it yourself:

```bash
stratum chat models/my-slm                        # a merged model
stratum chat strata/* --router router.json        # a pool of skills
```

The second form keeps the conversation going and prints which skill took each turn, which is a better way to judge routing than staring at an accuracy number.

---

## `stratum teachers` - what can this machine actually run

Your model is only as good as the teacher that wrote its training data. So the useful question is: what is the biggest, best teacher this particular machine can run?

That question is annoying to answer by hand, because it depends on how much memory you have, how much of it is on the graphics card, how fast your memory is, and how heavily the model is compressed.

```bash
stratum teachers
```

It measures your machine, then works through a catalogue of open models and tells you which fit and roughly how fast each would be. On the laptop this was developed on, it found that a 120-billion-parameter model would run at about 14 words per second, and correctly refused to recommend a much larger one that would have managed less than one.

### Why some enormous models are fast and others are not

This is the single most useful idea in that command, and it is not obvious.

Some large models are **sparse**, which means that although the model contains a great many numbers, only a small fraction of them are used for any given word. The rest sit idle. Such a model is described with two figures: the **total** size and the **active** size.

Speed depends almost entirely on the **active** figure, because generating a word means reading the numbers you actually use, and reading is the slow part.

Which is why a model with 80 billion numbers in total but only 3 billion active can be **faster** than a model with 8 billion numbers that all get used. The bigger one is doing less work.

---

## `--concurrency` - stop asking one question at a time

When a teacher writes your training data, STRATUM asks it about one chunk of your documents, waits for the answer, then asks about the next.

If the teacher is a service on the internet, nearly all of that time is spent waiting for a reply. Sending one question at a time is like posting one letter and standing at the mailbox until it is answered before writing the next.

```bash
stratum corpus pairs --chunks corpus/chunks.jsonl --teacher claude-cli \
  --concurrency 8 --instruction "..." --out data/explain.jsonl
```

The sensible default depends on the teacher, so you do not have to think about it. A local model runs one at a time, because it is already using the whole graphics card and a second request would win nothing and might run it out of memory. A service runs several, because it is only waiting.

If a provider starts rejecting your requests for going too fast, lower the number. The error message will say so.

---

## Local teachers now compress themselves to fit

A related change with a similar theme.

Running a teacher on your own machine keeps your documents on your machine, which matters a great deal if those documents are confidential. But a good teacher is large, and large does not fit.

An 8-billion-parameter model needs about 17 gigabytes at full precision. A common laptop graphics card has 8. So it does not fit, and it falls back to the main processor, where it runs roughly twenty times slower.

But a teacher is only ever **read**. It is never trained, never adjusted. That makes compressing it nearly free in terms of quality, and it takes that same model down to about 5.5 gigabytes, which does fit.

So STRATUM now checks before loading, and compresses when it has to:

```
Loading Hugging Face teacher: Qwen/Qwen3-8B
 loading in 4-bit so it fits the GPU
```

It does not compress anything that already fits, and it does not try on a Mac, where the necessary library does not work.

---

## Two bugs, both found by other people

Both of these came from people running this on hardware nobody on the project owns, which is the only way some bugs are ever going to surface.

### Training could throw away its best work

Training runs in **epochs**, meaning complete passes over your data. More passes usually means better, until it does not, at which point the model starts getting worse.

The old code saved whatever it had at the end of the last epoch. So if your third epoch went badly, the good model from epoch two was overwritten by the bad one from epoch three, and you had no way of knowing.

Someone reported exactly this, with numbers showing the loss going 1.63, then 1.88, then 4.90. That last figure is a model that has come apart. And that is what got saved.

Now STRATUM watches every epoch, keeps the weights from the best one, says so plainly when it sees things going backwards, and records in the model's own record card which epoch it kept.

The underlying cause was interesting enough to fix separately. The **Muon** optimizer, which is what adjusts the numbers during training, deliberately makes each adjustment the same size regardless of how large the numbers being adjusted are. That is normally a strength. But it means that on a small model, where the numbers are smaller, each adjustment is proportionally larger, and a setting that is fine on a 7-billion-parameter model is too aggressive on a 600-million one. The default now scales with the size of the model you chose.

### Adapters failed to attach on some machines

The second report was a crash when attaching an adapter, on an Apple silicon Mac, that only happened after a multi-gigabyte model had finished downloading.

The underlying issue was a version mismatch between two libraries. The genuinely annoying part was the timing: the download had to complete before the failure appeared, so every attempt cost the download again.

STRATUM now tests this on a tiny model built in memory, in about a second, before anything is downloaded. If it is going to fail, it fails immediately and tells you which versions to change.

The same idea got applied more widely. Any argument that can be checked without loading a model is now checked first. During the testing for this release, a run spent thirty-six minutes downloading a teacher and then stopped on a bad argument that could have been caught in the first second. That is fixed.

---

## The `engine/` folder, and why you can ignore it

There is now some C code in the repository. Nothing in STRATUM uses it. You never need to compile it. This section explains what it is for and why it is not finished, because a folder of unexplained C code in a Python project is worse than no folder at all.

### Where it came from

A project called [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) by Fareed Khan does something that sounds impossible. It runs a model with 2.78 **trillion** numbers in it on one ordinary processor, using 8 gigabytes of memory, from a file that is 1.56 terabytes on disk.

The way it works is the sparsity idea from earlier, pushed hard. The model is divided into 896 **experts** per layer, of which only 16 are used for any given word. So the vast majority of the file is asleep at any moment. Keep the always-needed part in memory, leave the rest on disk, and fetch the handful you need.

It works, and it is slow. About ten to thirty seconds per word.

### The idea

That project publishes a measurement about itself which is more interesting than it first appears. It keeps recently used experts in memory hoping to reuse them, and that only works 36% of the time. Giving it eight times more memory to work with does not improve that number at all. It stays at 36%.

Its author explains why, and the explanation is the good bit: models like this are **deliberately trained to spread work evenly across their experts**, because that is what makes them efficient to train in the first place. But even spreading is precisely what makes keeping things around useless. Nothing is used often enough to be worth keeping.

That suggested trying the opposite. Instead of processing one word and fetching what it needs, gather a large number of words, and for each expert, handle everyone who needs it while it is in memory. Group the work by expert rather than by word.

The appeal is that it changes the *shape* of the reading, not just the amount. Reading scattered pieces of a huge file is slow. Reading it front to back is several times faster on the same drive. If you know in advance which experts you need, you can sort them into file order and sweep through once.

### Why it stopped there

Before writing it, the design was handed to several independent reviewers with instructions to attack it. They came back with fourteen objections and two of them mattered.

**The idea is not new.** A paper called MoE-Gen describes the same technique in almost the same words, published in 2025. Klotski, from a 2025 conference, does a more sophisticated version. And llama.cpp, the most widely used tool of its kind, has grouped work by expert across a batch since 2023.

There is a narrow thing left that is genuinely not covered, which is that all of those assume the model fits in the computer's memory, and none of them deal with a file twelve times larger than the machine. But that is a much smaller claim than the one that had been made.

**The disk was never the bottleneck.** This is the finding that stopped the work, and it is a good lesson in measuring before building.

The reference engine's arithmetic runs at roughly two tenths of one percent of what the processor is capable of. Put the two costs side by side for a batch of 320 words: reading the weights takes about four minutes, and doing the arithmetic on them takes about fifty. The disk is outnumbered twelve to one.

So the whole plan was aimed at the wrong thing. Making the reading cleverer saves four minutes out of fifty-four. The arithmetic is what needs fixing first, and that is a separate job.

### What is actually in the folder

Two things that are finished, tested and useful on their own:

- **A GGUF reader.** GGUF is the standard file format that quantized models are distributed in. This one refuses to read past the end of a truncated or corrupt file, which the tests check by feeding it 116 different broken versions of a valid file.
- **A measurement tool.** It reads the same blocks from the same file in three different orders, scattered, sorted and sequential, and reports the speed of each. That settles by measurement whether reordering reads is worth anything on your particular drive, rather than assuming it from arithmetic.

Everything else is documented in [engine/README.md](../engine/README.md), including the full list of what came from where, and an honest note that if you want a large model to answer you quickly and interactively, none of this helps and you should use [llama.cpp](https://github.com/ggml-org/llama.cpp) with a model that fits in your memory.

---

## What you now know

- **`stratum setup` fixes the machine**, and knows that a Mac needs different advice from an NVIDIA box, including that 4-bit compression will never work there.
- **Routing is the alternative to merging.** Keep skills separate, pick one per question, and nothing gets diluted. Use it when a person could tell your skills apart from the question alone.
- **Confidence is a gap, not a score**, and a low one is telling you these skills should be merged instead.
- **`stratum serve` speaks the common language**, so anything with an address box can use what you built.
- **Speed depends on active size, not total size**, which is why some enormous models are quick.
- **Ask a service several questions at once**, and a local model one at a time.
- **A teacher only ever gets read**, so compressing it is nearly free and is often what makes it fit at all.
- **Measure before you build.** A plan to make reading faster died on the discovery that reading was never the slow part.

Next: [routing and serving in more depth ->](15-routing-and-serving.md), or back to [the index ->](README.md)
