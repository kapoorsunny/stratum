# 13 - Troubleshooting

*The problems people actually hit, in the order they usually hit them. Every error STRATUM raises on purpose is listed here with the fix.*

---

## Out of memory, on a machine that should have plenty

Check this before touching any setting, because the usual cause is not the settings.

```bash
stratum free --dry-run
```

An earlier STRATUM run can still be alive and holding the card. A training run stopped with Ctrl-C, a server whose terminal was closed, a notebook that was never restarted. The process disappears from view and keeps its memory, and the next thing you start fails for memory on a machine that looks idle.

On the laptop this was written on, six forgotten servers were holding the whole 8 GB card while a training run waited for memory that had already been paid for.

```
GPU 0  NVIDIA GeForce RTX 4070 Laptop GPU
  7,933 of 8,188 MB in use (97%)
System memory  38,928 of 137,031 MB in use (28%)

5 STRATUM process(es) still running
      pid  command      GPU MB    RAM MB  age
    66964  train        on GPU     6,313  5m10s
    63780  train             -        12  53m21s
```

Release them with `stratum free`. It stops children first, asks politely before insisting, and reports how much actually came back rather than assuming.

Two things it will not do. It only ever touches processes whose command line runs STRATUM, so on a shared machine somebody else's job is safe. And it never touches the process it is running in or anything that process descends from, since freeing memory by killing yourself achieves nothing.

`stratum doctor` says the same thing unprompted, and also prints how much of the card is free right now rather than only how large it is.

Three platforms, three different pictures, and the command handles all of them.

- **NVIDIA on Linux** reports how much each process holds.
- **NVIDIA on Windows** names the processes on the card but answers `[N/A]` for their sizes, because the display driver runs in WDDM mode. The listing says `on GPU` rather than inventing a number, and releasing works exactly the same.
- **Apple silicon** has no separate card. The GPU shares system memory, so the system memory figure is the one that matters and ending the process is what gives it back.

If the card is full and `stratum free` finds nothing of ours, something else on the machine is holding it. The command says so rather than reporting a cheerful success, and `nvidia-smi` will name it.

## The model states things that are not in the documents

Turn on the support check, which is on by default whenever `--context` is given.

```bash
stratum serve strata/* --context index/ --context-chunks chunks/chunks.jsonl \
                       --policy policy.json          # the check is already on
```

Every answer is compared against the passages it was given, and anything they do not carry becomes a refusal before it leaves the server. The reason comes back in the response so a front end can explain it:

```json
"support": {"supported": false, "reason": "number not in the material",
            "bad_numbers": ["1.5", "2"]}
```

Measured on three trained strata over 49 cases, this took unsupported statements delivered from 24 to 0 for a closed book model and from 9 to 0 for a grounded one, and on the grounded model it cost no correct answers at all.

If it refuses too much, `--min-overlap 0.2` is more permissive, and `--no-require-support` turns it off entirely. Before doing either, look at what it held back. On the run this was written from, two of the refusals that looked like over-caution were the check correctly suppressing a fabricated pressure ratio and a patent dated ninety-four years wrong.

## Training data that teaches the model to invent

Check the last line `stratum ground` prints.

```
  checked, every answerable row's material really does carry its answer
```

If it instead warns that rows still claim an answer the material does not carry, do not train on that file. A row that says a question is answerable while showing material without the answer is a worked example of inventing, and the model will learn it.

This used to happen silently. Chunks were cut at their first 1200 characters, so any answer sitting past the cut disappeared while the row still claimed to be answerable. The window now follows the answer, and rows that still cannot be answered become decline rows instead.

## Out of memory during training

If the card really is free and training still runs out, `stratum plan recipe.yaml` estimates what each stratum needs on this machine and suggests the fix, or tells you honestly that the build belongs on rented hardware (doc 10). The levers, in the order to pull them:

1. `--batch-size 1 --grad-accum 16` - same effective batch (doc 6), a fraction of the memory.
2. Make sure 4-bit is on. It's the default on NVIDIA GPUs, but only if `bitsandbytes` is installed - `stratum doctor` tells you.
3. `--max-len 512` if your pairs are short. Memory grows with sequence length.
4. A smaller base. Run `stratum doctor` and use its recommendation. A pipeline proven on 0.6B scales to 4B later by changing one argument.

## "NameError: name 'torch' is not defined"

The libraries pip installed do not work together. transformers refuses to use a PyTorch older than its minimum - transformers 5 needs PyTorch 2.4 or newer - and rather than failing at install time it quietly disables its own PyTorch half. The first import that reaches modelling code then dies with a `NameError` that names neither package and neither version.

`stratum doctor` diagnoses it: the "Library versions" section ends with `these versions work together: yes` or names the mismatch and prints the fix. Every command that loads a model checks the same thing first, so you get five readable lines instead of a twenty-frame traceback.

The fix depends on your machine:

```bash
pip install -U torch            # where a newer PyTorch exists
pip install "transformers<5"    # where it does not - keep this torch, older libraries
```

**On Intel Macs, only the second one works.** PyTorch published no macOS x86_64 build after 2.2.2, so "upgrade torch" is a dead end there - the install is capped to `transformers<5` automatically on that platform for exactly this reason. Doctor knows the difference and only offers the advice that can succeed.

## "The following model_kwargs are not used by the model"

The tokenizer returned something the model has no argument for - `token_type_ids` is the usual culprit, produced by BERT-family tokenizers and rejected by causal models. STRATUM filters generation inputs down to what a causal model accepts (`encode_for_generation` in `stratum/hf_utils.py`), so this should not reach you. If it does with a custom loading path, route your tokenizer output through that helper.

## My GPU isn't being used

The most common cause is not a missing GPU - it's the CPU-only build of PyTorch sitting in front of a perfectly good one. `stratum doctor` now checks for this: if the NVIDIA driver reports a card that PyTorch cannot see, it prints the fix. The command matters:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

A plain `pip install torch` or `--upgrade` quietly keeps the CPU build, because pip considers the installed version already satisfied. On Apple silicon the GPU (MPS) is used automatically with the normal PyTorch install. AMD cards need the ROCm build on Linux and are not supported by PyTorch on Windows.

## `bitsandbytes` won't install or won't load

4-bit (QLoRA) needs `bitsandbytes`, which needs an NVIDIA GPU. On Windows, recent versions install normally with `pip install bitsandbytes` - if an old pinned version fails, upgrade pip and retry. On Mac or CPU-only machines it isn't available: train in bf16 (`--no-4bit` makes this explicit) with a smaller base.

## Model downloads fail

Run `stratum doctor` first - it checks the exact things that go wrong:

- **Misspelled id.** Hub ids look like `Qwen/Qwen3-1.7B`, namespace and all.
- **Gated model.** Some releases need a (free) license acceptance: `huggingface-cli login`.
- **Locked-down network.** Set `HF_HUB_OFFLINE=1` and point `--base` at a local folder, after downloading on a connected machine with `huggingface-cli download <model>` and copying the folder over.

## "A training row has no response tokens within --max-len"

One of your prompts alone is longer than `--max-len`, so the response would be cut off entirely and the row would teach nothing. STRATUM stops instead of training on it silently. Raise `--max-len`, or shorten or drop that row - the error prints the start of the offending prompt.

## The model writes `<think>` blocks or reasons before answering

It's a thinking model (doc 6). STRATUM disables thinking in every template it renders and strips think blocks from output it scores or displays, so you'd normally never see one. If you're serving the merged model with your own stack (vLLM and the like), render prompts with `enable_thinking=False` there too - the serving stack doesn't know what STRATUM knows.

## The merged model outputs nothing at all

Empty answers on every prompt, while each stratum works fine on its own. The merge summed several full-strength deltas and overshot the base. Check the weight-shift line the merge prints - well above 25% confirms it. Re-merge with `--normalize` (or lower `--weights`), which turns the sum into an average. Doc 5 has the worked example.

## A skill scores worse after merging than alone

Some merge cost is normal (a couple of points - doc 8 shows a healthy example). A collapse means conflict:

1. Try `--method ties`, then `--method dare` (doc 5 explains when each helps).
2. Turn that skill's weight up with `--weights`.
3. If two skills genuinely fight, keep them as separate swappable strata (doc 10, pattern B).
4. If the stratum was trained with 4-bit and the drop is small but real, retrain it with `--no-4bit` - see the QLoRA seam note in doc 5.

## "Strata have different base models"

Merging refuses because the strata weren't trained from the same base, and deltas against one base are nonsense against another (doc 5). Retrain the odd one out on the shared base - the error names each stratum's base so you can see which it is.

## Merging refuses a DoRA stratum or one with modules_to_save

Both change the model in ways that aren't plain additive deltas, so merging them with this math would be silently wrong - STRATUM says so instead. Retrain the stratum as a standard LoRA adapter, or keep it as a separate runtime adapter.

## `teacher-gen` died halfway through a big run

Nothing is lost. Pairs are written as they're generated, so re-run the exact same command - seeds already answered are skipped and only the missing ones go to the teacher. Failed seeds retry with growing pauses automatically.

## CPU-only training is very slow

It works, but set expectations: a few hundred pairs on a 0.6B base is an hours-not-minutes job on CPU. Use it to prove your data and pipeline, then do the real training burst on any machine with a GPU - the commands are identical.

## Training loss is flat

Either the learning rate is too low for the optimizer you chose (Muon's default is `2e-2`, AdamW wants around `1e-4` to `1e-3`), or the rank is below what the skill needs and you've hit the ceiling from doc 3. Raise `--rank` before blaming the method.

## Corpus ingest says a format needs an extra library

The document parsers are optional dependencies so the core install stays light. `pip install 'stratum-slm[corpus]'` brings PDF, Word, PowerPoint, Excel, and image support in one go.

## "no extractable text - this is probably a scanned PDF"

The PDF is pictures of pages, with no text layer to read. Two routes: run OCR over it before ingesting, or export its pages as images and ingest those with a vision teacher (`--images hf`). Doc 14 covers both.

## Something else

Open an issue with your `stratum doctor` output and the full error. Hardware reports (GPU, VRAM, which base worked) are welcome even when nothing is broken - they build the community table in CONTRIBUTING.md.
