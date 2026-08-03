"""
Teacher advisor: which model can this machine actually run, and how fast.

Picking a teacher is the highest-leverage decision in the whole pipeline -
the reference build in examples/energy scored 5.3% with a small teacher and
56.4% with a strong one, changing nothing else. But "which model can I run"
usually gets answered by trial and error, a 40 GB download, and an out of
memory error an hour later.

This answers it arithmetically instead.

The number that decides speed is not the one people quote. Generating a token
means moving every ACTIVE parameter through memory, and doing only about two
arithmetic operations per parameter, so decoding is bound by bandwidth rather
than compute:

    tokens/sec  ~=  bandwidth / (active_params * bits_per_param / 8)

For a dense model every parameter is active. For a mixture-of-experts model
only a slice is - which is why an 80B model with 3B active runs many times
faster than a 30B dense one, despite being larger on disk. Total parameters
buy quality and cost disk. ACTIVE parameters cost time. The catalogue below
reports both, and the ratio between them.

Everything here is an estimate, deliberately conservative, and good enough to
sort "this will run well" from "this will crawl" - which is the only decision
being made.
"""
from __future__ import annotations

import shutil
from pathlib import Path

# Effective bits per parameter for common GGUF quantizations, including the
# per-block scales that make the real file bigger than the nominal bit width.
QUANTS = {
    "Q8_0": 8.5,
    "Q6_K": 6.6,
    "Q5_K_M": 5.7,
    "Q4_K_M": 4.8,
    "Q3_K_M": 3.9,
    "Q2_K": 3.35,
}

# A curated starting point, not a complete index. Sizes are in billions of
# parameters: `active` equals `total` for dense models. Any Hugging Face id
# works with the tools - this list exists to make the tradeoffs visible.
CATALOGUE = [
    # name,                    hf id,                          total, active, license
    ("Qwen3-0.6B",            "Qwen/Qwen3-0.6B",                0.6,   0.6,  "Apache-2.0"),
    ("Qwen3-1.7B",            "Qwen/Qwen3-1.7B",                1.7,   1.7,  "Apache-2.0"),
    ("Qwen3-4B",              "Qwen/Qwen3-4B",                  4.0,   4.0,  "Apache-2.0"),
    ("Qwen3-8B",              "Qwen/Qwen3-8B",                  8.0,   8.0,  "Apache-2.0"),
    ("Qwen3-14B",             "Qwen/Qwen3-14B",                14.0,  14.0,  "Apache-2.0"),
    ("Qwen3-32B",             "Qwen/Qwen3-32B",                32.0,  32.0,  "Apache-2.0"),
    ("Qwen3-30B-A3B",         "Qwen/Qwen3-30B-A3B",            30.5,   3.3,  "Apache-2.0"),
    ("Qwen3-Next-80B-A3B",    "Qwen/Qwen3-Next-80B-A3B-Instruct", 80.0, 3.0, "Apache-2.0"),
    ("Qwen3-235B-A22B",       "Qwen/Qwen3-235B-A22B",         235.0,  22.0,  "Apache-2.0"),
    ("gpt-oss-20b",           "openai/gpt-oss-20b",            21.0,   3.6,  "Apache-2.0"),
    ("gpt-oss-120b",          "openai/gpt-oss-120b",          117.0,   5.1,  "Apache-2.0"),
    ("Mixtral-8x7B",          "mistralai/Mixtral-8x7B-Instruct-v0.1", 46.7, 12.9, "Apache-2.0"),
    ("Mixtral-8x22B",         "mistralai/Mixtral-8x22B-Instruct-v0.1", 141.0, 39.0, "Apache-2.0"),
    ("Llama-3.1-8B",          "meta-llama/Llama-3.1-8B-Instruct", 8.0,  8.0,  "Llama 3.1 (gated)"),
    ("Llama-3.3-70B",         "meta-llama/Llama-3.3-70B-Instruct", 70.0, 70.0, "Llama 3.3 (gated)"),
    ("DeepSeek-V3",           "deepseek-ai/DeepSeek-V3",      671.0,  37.0,  "DeepSeek (open)"),
]


def measure_bandwidth(quick: bool = True) -> dict:
    """Measure this machine's memory bandwidth rather than assuming it.

    Bandwidth is what sets the speed ceiling, and it varies by a factor of
    five across laptops. Two seconds of measurement beats a guess.
    """
    import time

    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    n = 25_000_000 if quick else 100_000_000
    threads = min(8, (shutil.os.cpu_count() or 4))
    arrays = [np.ones(n, dtype=np.float32) for _ in range(threads)]
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(lambda a: float(a.sum()), arrays))  # warm
        t0 = time.perf_counter()
        list(ex.map(lambda a: float(a.sum()), arrays))
        elapsed = time.perf_counter() - t0
    total = sum(a.nbytes for a in arrays)
    return {"ram_gbs": total / elapsed / 1e9, "threads": threads}


def hardware(measure: bool = True) -> dict:
    """What this machine has to work with."""
    import torch

    hw = {"ram_gb": 0.0, "vram_gb": 0.0, "free_disk_gb": 0.0,
          "gpu": None, "ram_gbs": 25.0}
    try:
        import psutil  # optional
        hw["ram_gb"] = psutil.virtual_memory().total / 1e9
    except Exception:
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            hw["ram_gb"] = stat.ullTotalPhys / 1e9
        except Exception:
            try:
                hw["ram_gb"] = (shutil.os.sysconf("SC_PAGE_SIZE")
                                * shutil.os.sysconf("SC_PHYS_PAGES") / 1e9)
            except Exception:
                hw["ram_gb"] = 16.0  # a safe floor when nothing reports

    if torch.cuda.is_available():
        hw["gpu"] = torch.cuda.get_device_name(0)
        hw["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    hw["free_disk_gb"] = shutil.disk_usage(Path.home()).free / 1e9
    if measure:
        try:
            hw["ram_gbs"] = measure_bandwidth()["ram_gbs"]
        except Exception:
            pass
    return hw


def estimate(total_b: float, active_b: float, bits: float, hw: dict) -> dict:
    """Size a model against this machine and estimate its decode speed."""
    disk_gb = total_b * 1e9 * bits / 8 / 1e9
    active_gb = active_b * 1e9 * bits / 8 / 1e9

    # Where the weights end up dictates the bandwidth they move at. VRAM is
    # roughly 4x system RAM on a laptop dGPU, and disk is roughly 15x slower.
    if hw["vram_gb"] and disk_gb < hw["vram_gb"] * 0.85:
        where, bandwidth = "VRAM", hw["ram_gbs"] * 4.0
    elif disk_gb < hw["ram_gb"] * 0.75:
        where, bandwidth = "RAM", hw["ram_gbs"]
    else:
        # Partly resident, partly streamed. Weight the two by how much fits.
        cached = max(0.0, hw["ram_gb"] * 0.7) / disk_gb
        cached = min(1.0, cached)
        disk_gbs = 2.5
        bandwidth = cached * hw["ram_gbs"] + (1 - cached) * disk_gbs
        where = f"{cached:.0%} cached"

    tok_s = bandwidth / active_gb if active_gb else 0.0
    return {
        "disk_gb": disk_gb,
        "active_gb": active_gb,
        "where": where,
        "tok_s": tok_s,
        "fits_disk": disk_gb < hw["free_disk_gb"] * 0.9,
    }


def advise(hw: dict, min_tok_s: float = 3.0, quant: str = "Q4_K_M",
           catalogue=None) -> list[dict]:
    """Rank catalogue models by what this machine can run usefully."""
    bits = QUANTS[quant]
    rows = []
    for name, hf_id, total_b, active_b, lic in (catalogue or CATALOGUE):
        est = estimate(total_b, active_b, bits, hw)
        rows.append({
            "name": name, "hf_id": hf_id, "total_b": total_b,
            "active_b": active_b, "sparsity": total_b / active_b,
            "license": lic, "usable": est["fits_disk"] and est["tok_s"] >= min_tok_s,
            **est,
        })
    # Usable first, then the largest model that stays usable - bigger teachers
    # write better training data, so quality wins where speed is adequate.
    rows.sort(key=lambda r: (r["usable"], r["total_b"]), reverse=True)
    return rows


def print_advice(hw: dict, rows: list[dict], quant: str, min_tok_s: float,
                 top: int = 12) -> None:
    print("STRATUM teacher advisor\n" + "-" * 74)
    gpu = f"{hw['gpu']}, {hw['vram_gb']:.1f} GB VRAM" if hw["gpu"] else "no GPU"
    print(f"This machine : {hw['ram_gb']:.0f} GB RAM at "
          f"{hw['ram_gbs']:.0f} GB/s, {gpu}")
    print(f"Free disk    : {hw['free_disk_gb']:.0f} GB")
    print(f"Assuming     : {quant} quantization, useful means "
          f"{min_tok_s:.0f}+ tokens/sec\n")

    print(f"{'model':<22}{'total':>7}{'active':>8}{'sparse':>8}"
          f"{'disk':>8}{'runs in':>12}{'tok/s':>8}")
    print("-" * 74)
    shown = 0
    for r in rows:
        if shown >= top:
            break
        mark = "  " if r["usable"] else "x "
        speed = f"{r['tok_s']:.0f}" if r["tok_s"] >= 1 else f"{r['tok_s']:.2f}"
        if not r["fits_disk"]:
            speed = "no disk"
        print(f"{mark}{r['name']:<20}{r['total_b']:>6.0f}B{r['active_b']:>7.1f}B"
              f"{r['sparsity']:>7.1f}x{r['disk_gb']:>7.0f}G"
              f"{r['where']:>12}{speed:>8}")
        shown += 1

    best = next((r for r in rows if r["usable"]), None)
    print("-" * 74)
    if best is None:
        print("\nNothing in the catalogue clears the bar on this machine.")
        print("Lower it with --min-tok-s, try a smaller quantization "
              "(--quant Q3_K_M),")
        print("or free disk space if the 'no disk' marks are what stopped it.")
        return

    print(f"\nBest teacher that runs well here: {best['name']} "
          f"({best['total_b']:.0f}B total, {best['active_b']:.1f}B active, "
          f"{best['license']})")
    print(f"  about {best['tok_s']:.0f} tokens/sec, {best['disk_gb']:.0f} GB on disk\n")
    print("Run it locally, then point STRATUM at it:")
    print(f"  1. Download a {quant} GGUF of {best['name']} "
          f"(from {best['hf_id']} or a GGUF mirror)")
    print(f"  2. llama-server -m <file>.gguf -c 8192 --port 8080")
    print(f"  3. stratum corpus pairs --chunks corpus/chunks.jsonl \\")
    print(f"       --instruction \"...\" --teacher llama-cpp \\")
    print(f"       --out data/skill.jsonl --test-out data/skill-test.jsonl")
    print("\nNothing leaves the machine on that path, which is the point for "
          "regulated data.")

    sparse = [r for r in rows if r["usable"] and r["sparsity"] > 4]
    if sparse:
        s = sparse[0]
        print(f"\nWorth knowing: {s['name']} activates only "
              f"{s['active_b']:.1f}B of its {s['total_b']:.0f}B per token "
              f"({s['sparsity']:.0f}x sparse),")
        print("which is why it runs far faster than a dense model of similar "
              "size. When you\ncompare teachers, compare the active column, "
              "not the total.")
    print("\nEstimates are rough and deliberately conservative - they sort "
          "'runs well' from\n'will crawl'. Measure the one you pick.")
