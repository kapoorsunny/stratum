"""
Serving: one base model, many skills, routed per request.

Two pieces:

  SkillPool   the frozen base loaded once with every stratum attached, and a
              router that picks which one answers each request
  serve()     an OpenAI-compatible HTTP endpoint, so Claude Code, an IDE, or
              any client that speaks that API can talk to a model you built

The memory arithmetic is the point. Merging N strata means one model per
combination you might want. Routing them means one base plus a few megabytes
per skill - a 1.7B base with twenty skills is about 3.5 GB, and switching
between them is a pointer change, not a load.

Dependencies stay at what STRATUM already needs: the server is stdlib
http.server, not a web framework.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class SkillPool:
    """A base model with every stratum attached, routed per request."""

    def __init__(self, strata_dirs: list[str], router_path: str | None = None,
                 base_model: str | None = None, verbose: bool = True):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .hf_utils import pick_device
        from .merge import read_stratum_card

        if not strata_dirs:
            raise ValueError("No strata given - nothing to serve.")

        cards = [read_stratum_card(d) for d in strata_dirs]
        bases = {c["base_model"] for c in cards}
        if base_model is None:
            if len(bases) != 1:
                raise ValueError(
                    f"Strata come from different bases: {bases}. Routing needs "
                    f"one shared base so a single loaded model can carry every "
                    f"adapter. Retrain the odd ones out, or serve them separately."
                )
            base_model = bases.pop()

        self.names = [c.get("stratum_name", Path(d).name)
                      for c, d in zip(cards, strata_dirs)]
        self.cards = dict(zip(self.names, cards))
        self.device = pick_device()

        if verbose:
            print(f"Loading base {base_model} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(strata_dirs[0])
        model = AutoModelForCausalLM.from_pretrained(base_model,
                                                     dtype=torch.bfloat16)

        # Every adapter attaches to the SAME loaded base. Switching skills is
        # then set_adapter(), which costs nothing - no reload, no copy.
        for i, (name, d) in enumerate(zip(self.names, strata_dirs)):
            if i == 0:
                model = PeftModel.from_pretrained(model, d, adapter_name=name)
            else:
                model.load_adapter(d, adapter_name=name)
            if verbose:
                print(f" + {name}")

        self.model = model.to(self.device).eval()
        self.base_model = base_model
        self.router = None
        if router_path and Path(router_path).exists():
            from .router import SkillRouter
            self.router = SkillRouter.load(router_path)
            if verbose:
                print(f"Router loaded: {', '.join(self.router.skills)}")
        elif verbose:
            print("No router - requests use the first stratum unless one is named.")

    def pick(self, prompt: str, skill: str | None = None):
        """Decide which skill answers. An explicit skill always wins over the
        router, so a caller who knows what they want is never second-guessed."""
        if skill:
            if skill not in self.names:
                raise ValueError(f"Unknown skill '{skill}'. "
                                 f"Available: {', '.join(self.names)}")
            return skill, 1.0, {}
        if self.router is None:
            return self.names[0], 0.0, {}
        name, confidence, scores = self.router.route(prompt)
        if name not in self.names:
            return self.names[0], 0.0, scores
        return name, confidence, scores

    def generate(self, prompt: str, skill: str | None = None,
                 system: str | None = None, max_new_tokens: int = 400,
                 history: list[dict] | None = None) -> dict:
        """Answer one request with the skill that fits it."""
        import torch

        from .data import format_messages, strip_think
        from .hf_utils import encode_for_generation

        chosen, confidence, scores = self.pick(prompt, skill)
        self.model.set_adapter(chosen)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(history or [])
        messages.append({"role": "user", "content": prompt})

        text = format_messages(self.tokenizer, messages, add_generation_prompt=True)
        ids = encode_for_generation(self.tokenizer, text, self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        new = out[0][ids["input_ids"].shape[1]:]
        answer = strip_think(self.tokenizer.decode(new, skip_special_tokens=True))
        return {
            "text": answer,
            "skill": chosen,
            "confidence": round(confidence, 4),
            "scores": {k: round(v, 4) for k, v in sorted(
                scores.items(), key=lambda kv: -kv[1])[:4]},
            "tokens": int(new.shape[0]),
            "seconds": round(time.perf_counter() - t0, 2),
        }


def serve(pool: SkillPool, host: str = "127.0.0.1", port: int = 8927,
          system: str | None = None) -> None:
    """Expose the pool over an OpenAI-compatible HTTP API.

    Speaking that dialect is what makes the model usable from tools you
    already have - Claude Code, an IDE assistant, curl - without any of them
    needing to know STRATUM exists. Routing is reported back in the response
    so you can always see which skill answered.
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # the request log below is more useful than the default

        def _send(self, code: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.end_headers()

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                return self._send(200, {"object": "list", "data": [
                    {"id": n, "object": "model", "owned_by": "stratum",
                     "stratum_card": pool.cards[n]} for n in pool.names]})
            if self.path.rstrip("/") in ("", "/health"):
                return self._send(200, {"status": "ok", "base": pool.base_model,
                                        "skills": pool.names,
                                        "router": pool.router is not None})
            self._send(404, {"error": {"message": f"No route {self.path}"}})

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/chat/completions",
                                             "/chat/completions"):
                return self._send(404, {"error": {"message": f"No route {self.path}"}})
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                messages = req.get("messages") or []
                user = next((m["content"] for m in reversed(messages)
                             if m.get("role") == "user"), None)
                if not user:
                    raise ValueError("No user message in the request.")
                sys_msg = next((m["content"] for m in messages
                                if m.get("role") == "system"), system)
                history = [m for m in messages[:-1] if m.get("role") in
                           ("user", "assistant")]
                # An explicit model name selects that skill, so a client can
                # bypass routing by asking for it directly.
                asked = req.get("model")
                skill = asked if asked in pool.names else None

                result = pool.generate(
                    user, skill=skill, system=sys_msg,
                    max_new_tokens=int(req.get("max_tokens") or 400),
                    history=history or None)
                print(f"  {result['skill']:<20} conf {result['confidence']:.3f}  "
                      f"{result['tokens']} tok in {result['seconds']}s")
                self._send(200, {
                    "id": f"stratum-{int(time.time()*1000)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": result["skill"],
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant",
                                             "content": result["text"]}}],
                    "usage": {"completion_tokens": result["tokens"],
                              "prompt_tokens": 0,
                              "total_tokens": result["tokens"]},
                    "stratum": {"skill": result["skill"],
                                "confidence": result["confidence"],
                                "scores": result["scores"],
                                "seconds": result["seconds"]},
                })
            except Exception as e:
                self._send(400, {"error": {"message": str(e),
                                           "type": type(e).__name__}})

    server = HTTPServer((host, port), Handler)
    print(f"\nServing {len(pool.names)} skills on http://{host}:{port}")
    print(f"  skills : {', '.join(pool.names)}")
    print(f"  base   : {pool.base_model}")
    print(f"\nPoint any OpenAI-compatible client at it:")
    print(f"  base URL  http://{host}:{port}/v1")
    print(f"  api key   any non-empty string")
    print(f"  model     'auto' to route, or a skill name to force one")
    print("\nCtrl-C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        server.server_close()
