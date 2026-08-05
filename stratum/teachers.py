"""
Teacher backends for data distillation (`stratum teacher-gen`).

A "teacher" is any callable str -> str that answers a prompt well. STRATUM
stays vendor-neutral so you can use whatever you have access to:

  hf         a local Hugging Face model (from the Hub or a local path) -
             nothing leaves your machine
  claude-cli Claude through the installed Claude Code CLI - uses your Claude
             subscription, no API key needed
  openai     the OpenAI API (needs OPENAI_API_KEY and `pip install openai`)
  anthropic  the Anthropic API (needs ANTHROPIC_API_KEY and `pip install anthropic`)
  gemini     the Google Gemini API (needs GEMINI_API_KEY and `pip install google-genai`)
  echo       a no-op teacher for testing the pipeline without any model

Everything except hf and echo sends your data to that provider - for corpora
that must stay in your environment, hf is the one to use.

Every backend fails LOUDLY and CLEARLY if something is missing - a normal
enterprise developer (Java/.NET/Python) should never be left guessing.

The API backends carry a default model id, but provider lineups change - pass
--model with your provider's current recommendation rather than relying on the
default staying fresh forever.
"""
from __future__ import annotations

import os


TEACHER_BACKENDS = ("hf", "llama-cpp", "claude-cli", "openai", "anthropic",
                    "gemini", "echo")


def get_teacher(backend: str, model: str | None = None,
                url: str | None = None):
    """Return a callable str->str for the requested teacher backend."""
    backend = backend.lower()
    if backend == "hf":
        return _hf_teacher(model or "Qwen/Qwen3-4B")  # 4-bit when it must
    if backend == "llama-cpp":
        return _local_server_teacher(url or "http://127.0.0.1:8080/v1", model)
    if backend == "claude-cli":
        return _claude_cli_teacher(model)
    if backend == "openai":
        return _openai_teacher(model or "gpt-4o-mini")
    if backend == "anthropic":
        return _anthropic_teacher(model or "claude-opus-5")
    if backend == "gemini":
        return _gemini_teacher(model or "gemini-2.5-flash")
    if backend == "echo":
        return _echo_teacher()
    raise ValueError(f"Unknown teacher backend '{backend}'. "
                     f"Choose from: {', '.join(TEACHER_BACKENDS)}.")


def _echo_teacher():
    """A teacher that needs no model, so the pipeline can be run end to end.

    It has to answer in the shape the caller expects. `corpus pairs` throws
    away any answer that is not a JSON array of prompt and response objects,
    while `teacher-gen` takes the answer verbatim. Returning prose to both
    meant every chunk failed, three retries deep, and the run finished with
    an empty file and a successful exit code.
    """
    import json
    import re

    filler = "(echo teacher - replace with a real backend)"

    def teacher(prompt: str) -> str:
        m = re.search(r"(\d+)\s+question", prompt)
        if m:
            n = max(1, int(m.group(1)))
            return json.dumps([{"prompt": f"(echo question {i + 1})",
                                "response": filler} for i in range(n)])
        return filler

    return teacher


def _local_server_teacher(base_url: str, model_name: str | None):
    """Any local server that speaks the OpenAI chat API.

    This is how a large model becomes your teacher without a byte leaving the
    machine. It works with llama.cpp's llama-server, Ollama, LM Studio, vLLM,
    and with STRATUM's own `stratum serve` - anything on that dialect. Run
    `stratum teachers` first to see which model your hardware can actually
    hold, and at what speed.

    Only stdlib is used, so no dependency is added for the privilege of
    talking to a local port.
    """
    import json
    import urllib.error
    import urllib.request

    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    endpoint = f"{base_url}/chat/completions"

    # Fail now, with a usable message, rather than on the first of 5,000 calls.
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=10) as r:
            served = json.loads(r.read().decode("utf-8")).get("data", [])
        names = [m.get("id") for m in served if m.get("id")]
        print(f"Local teacher at {base_url}"
              + (f" serving: {', '.join(names[:4])}" if names else ""))
        if model_name is None and names:
            model_name = names[0]
    except Exception as e:
        raise EnvironmentError(
            f"No local model server answering at {base_url} ({e}).\n"
            f" - Start one, e.g. llama-server -m model.gguf -c 8192 --port 8080\n"
            f" - Or point elsewhere with --teacher-url\n"
            f" - Run `stratum teachers` to see which model this machine can run."
        ) from e

    def teacher(prompt: str) -> str:
        body = json.dumps({
            "model": model_name or "local",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 2048,
        }).encode("utf-8")
        req = urllib.request.Request(
            endpoint, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer local"})
        with urllib.request.urlopen(req, timeout=900) as r:
            payload = json.loads(r.read().decode("utf-8"))
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"Server returned no choices: {payload}")
        from .data import strip_think
        return strip_think(choices[0]["message"]["content"] or "")

    return teacher


def _claude_cli_teacher(model_name: str | None):
    """Claude through the Claude Code CLI, on your existing subscription.

    No API key involved - each call runs `claude -p` as a subprocess, so
    whatever plan the CLI is signed into pays for the tokens. Leave the model
    unset to use the CLI's default, or pass --model to pick one.
    """
    import shutil
    import subprocess

    exe = shutil.which("claude")
    if exe is None:
        raise EnvironmentError(
            "The claude-cli teacher needs the Claude Code CLI on PATH.\n"
            " - Install it from https://claude.com/claude-code and sign in, or\n"
            " - use --teacher anthropic with an API key, or --teacher hf for "
            "a local model."
        )

    def teacher(prompt: str) -> str:
        cmd = [exe, "-p", "--output-format", "text"]
        if model_name:
            cmd += ["--model", model_name]
        result = subprocess.run(cmd, input=prompt, capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                f"claude CLI returned {result.returncode}: "
                f"{(result.stderr or result.stdout).strip()[:300]}")
        return result.stdout

    return teacher


def _gemini_teacher(model_name: str):
    """Google Gemini API as teacher. Needs GEMINI_API_KEY (or GOOGLE_API_KEY)
    and `pip install google-genai`."""
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Export it first:\n"
            " export GEMINI_API_KEY=...\n"
            "Or use `--teacher hf` for a local model instead."
        )
    try:
        from google import genai
    except ImportError as e:
        raise ImportError("pip install google-genai") from e

    client = genai.Client()

    def teacher(prompt: str) -> str:
        resp = client.models.generate_content(model=model_name, contents=prompt)
        return resp.text or ""

    return teacher


def should_quantize_teacher(size_b: float | None, vram_gb: float) -> bool:
    """Decide whether a teacher of this size needs 4-bit to fit the card.

    Full precision costs about 2 bytes per parameter and the run needs a
    little room on top for activations, so the test is generous rather than
    exact. With no GPU or no known size the answer is no, because guessing
    wrong here wastes a long download.
    """
    if not vram_gb or not size_b:
        return False
    return size_b * 2.2 > vram_gb * 0.85


def _hf_teacher(model_name: str, load_4bit: str | bool = "auto"):
    """A local Hugging Face model as teacher.

    Downloads from the Hub on first use, or loads from a local folder. Runs
    on the GPU when there is one.

    A teacher is only ever read, never trained, so quantizing it costs very
    little quality and roughly quarters its memory. With load_4bit="auto" the
    model is quantized when it would not otherwise fit in VRAM, which is what
    lets an 8B teacher run on an 8 GB card instead of falling back to the CPU
    and taking twenty times longer.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "The 'hf' teacher needs transformers + torch. "
            "Install with: pip install transformers torch"
        ) from e

    print(f"Loading Hugging Face teacher: {model_name}")
    print("(first run downloads the model - this can take a while)")

    kwargs = dict(dtype=torch.bfloat16)
    if load_4bit == "auto":
        from .plan import model_params_b
        vram = (torch.cuda.get_device_properties(0).total_memory / 1e9
                if torch.cuda.is_available() else 0)
        load_4bit = should_quantize_teacher(model_params_b(model_name), vram)
    if load_4bit and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            print(" loading in 4-bit so it fits the GPU")
        except Exception as e:
            print(f" 4-bit unavailable ({e}) - loading in bf16")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Could not load teacher '{model_name}'. Common causes:\n"
            f" - No internet, or the model id is misspelled.\n"
            f" - The model is gated: run `huggingface-cli login` first.\n"
            f" - Out of memory: pick a smaller teacher.\n"
            f"Original error: {e}"
        ) from e

    from .hf_utils import pick_device
    device = pick_device()
    if "quantization_config" not in kwargs:
        model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def teacher(prompt: str) -> str:
        from .data import strip_think
        messages = [{"role": "user", "content": prompt}]
        try:
            # enable_thinking=False keeps thinking models (Qwen3) from writing
            # reasoning blocks into your training data. Templates that don't
            # know the flag ignore it.
            text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True,
                                                 enable_thinking=False)
        except Exception:
            text = prompt
        from .hf_utils import encode_for_generation
        ids = encode_for_generation(tokenizer, text, device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=512, do_sample=False,
                                 temperature=None, top_p=None,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        raw = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        # Belt and braces - some checkpoints think even when asked not to.
        return strip_think(raw)

    return teacher


def _openai_teacher(model_name: str):
    """OpenAI API as teacher. Needs OPENAI_API_KEY and `pip install openai`."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export it first:\n"
            " export OPENAI_API_KEY=sk-...\n"
            "Or use `--teacher hf` for a local model instead."
        )
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("pip install openai") from e

    client = OpenAI()

    def teacher(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content

    return teacher


def _anthropic_teacher(model_name: str):
    """Anthropic API as teacher. Needs ANTHROPIC_API_KEY and `pip install anthropic`."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Export it first:\n"
            " export ANTHROPIC_API_KEY=sk-ant-...\n"
            "Or use `--teacher hf` for a local model instead."
        )
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("pip install anthropic") from e

    client = anthropic.Anthropic()

    def teacher(prompt: str) -> str:
        resp = client.messages.create(
            model=model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    return teacher
