"""
Work out what this machine is missing and install it.

`stratum doctor` tells you what is wrong. This fixes it. The difference
matters because the most common failure in this project is not a bug in the
code, it is a machine with the wrong PyTorch build on it, and the person
hitting that has no reason to know which of the eight PyTorch download URLs
is theirs.

Everything here is deliberately conservative. It never removes a package, it
never touches anything outside the running Python environment, and with
--dry-run it prints the exact commands and does nothing at all.

The three platforms behave quite differently and the differences are not
cosmetic:

  Apple silicon   the GPU is called MPS and the normal PyTorch wheel already
                  supports it. There is nothing to install for GPU support,
                  which surprises people who expect a CUDA style download.
                  4-bit quantization does not exist here at all.
  NVIDIA          needs a CUDA build of PyTorch, and pip will not give you
                  one by upgrading, because the CPU build satisfies the
                  requirement and pip stops there.
  everything else CPU works, slowly, and saying so is more useful than
                  pretending otherwise.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def machine_facts() -> dict:
    """Everything the decisions below depend on, gathered in one place."""
    facts = {
        "os": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "apple_silicon": False,
        "torch": None,
        "torch_cuda_build": False,
        "cuda_visible": False,
        "mps_visible": False,
        "nvidia_present": False,
        "bitsandbytes": False,
        "torchvision": False,
        "peft": None,
        "transformers": None,
        "compiler": None,
    }

    facts["apple_silicon"] = (facts["os"] == "Darwin" and
                              facts["arch"] in ("arm64", "aarch64"))

    try:
        import torch
        facts["torch"] = torch.__version__
        facts["torch_cuda_build"] = bool(getattr(torch.version, "cuda", None))
        facts["cuda_visible"] = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None)
        facts["mps_visible"] = bool(mps and mps.is_available())
    except Exception:
        pass

    for name in ("transformers", "peft"):
        try:
            mod = __import__(name)
            facts[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            pass

    try:
        import bitsandbytes  # noqa: F401
        facts["bitsandbytes"] = True
    except Exception:
        pass

    try:
        import torchvision  # noqa: F401
        facts["torchvision"] = True
    except Exception:
        pass

    from .hf_utils import detect_hidden_nvidia_gpu
    facts["nvidia_present"] = bool(detect_hidden_nvidia_gpu())

    for cc in ("cc", "gcc", "clang"):
        if shutil.which(cc):
            facts["compiler"] = cc
            break

    return facts


def cuda_index_url() -> str:
    """Which PyTorch wheel index a CUDA machine should use.

    Pinned to a recent CUDA that current NVIDIA drivers all support. A driver
    older than the wheel is the one case this gets wrong, and the error
    message from that is clear enough to act on.
    """
    return "https://download.pytorch.org/whl/cu128"


def plan_actions(facts: dict) -> list[dict]:
    """Decide what needs doing. Pure, so the tests can check the decisions
    without installing anything."""
    todo = []

    if facts["torch"] is None:
        if facts["apple_silicon"]:
            todo.append({
                "why": "PyTorch is not installed. On Apple silicon the normal "
                       "wheel already uses the GPU, so no special index is needed.",
                "cmd": [sys.executable, "-m", "pip", "install", "torch"],
                "key": "torch",
            })
        elif facts["nvidia_present"]:
            todo.append({
                "why": "PyTorch is not installed and this machine has an "
                       "NVIDIA GPU, so it needs the CUDA build.",
                "cmd": [sys.executable, "-m", "pip", "install", "torch",
                        "--index-url", cuda_index_url()],
                "key": "torch",
            })
        else:
            todo.append({
                "why": "PyTorch is not installed. No GPU was found, so this "
                       "is the CPU build and training will be slow.",
                "cmd": [sys.executable, "-m", "pip", "install", "torch"],
                "key": "torch",
            })

    elif facts["nvidia_present"] and not facts["torch_cuda_build"]:
        todo.append({
            "why": "This machine has an NVIDIA GPU but the installed PyTorch "
                   "is the CPU-only build, so training would run on the "
                   "processor and take many times longer.\n"
                   "  --force-reinstall is required. A plain install or an "
                   "upgrade leaves the CPU build in place, because pip sees "
                   "torch is present and considers the requirement met.",
            "cmd": [sys.executable, "-m", "pip", "install", "--force-reinstall",
                    "torch", "--index-url", cuda_index_url()],
            "key": "torch-cuda",
        })

    for name in ("transformers", "peft"):
        if facts[name] is None:
            todo.append({
                "why": f"{name} is needed to load and adapt models.",
                "cmd": [sys.executable, "-m", "pip", "install", name],
                "key": name,
            })

    # bitsandbytes is the 4-bit library and it is CUDA only. Installing it on
    # a Mac gets you a package that imports and then fails at the point of
    # use, which is worse than not having it, so it is skipped there.
    if not facts["bitsandbytes"]:
        if facts["nvidia_present"] or facts["cuda_visible"]:
            todo.append({
                "why": "bitsandbytes gives 4-bit loading, which is what lets a "
                       "bigger base model or teacher fit on this card.",
                "cmd": [sys.executable, "-m", "pip", "install", "bitsandbytes"],
                "key": "bitsandbytes",
            })

    # Reading images needs torchvision, and without it the error names
    # pytorch.org rather than the package, which sends people the wrong way.
    if not facts["torchvision"] and facts["torch"] is not None:
        cmd = [sys.executable, "-m", "pip", "install", "torchvision"]
        if facts["torch_cuda_build"]:
            cmd += ["--index-url", cuda_index_url()]
        todo.append({
            "why": "torchvision is missing, which is what image models need to "
                   "read a picture. Ingesting a corpus that contains images "
                   "fails without it.",
            "cmd": cmd,
            "key": "torchvision",
        })

    return todo


def platform_notes(facts: dict) -> list[str]:
    """Things worth saying that no install can fix."""
    notes = []

    if facts["apple_silicon"]:
        notes.append(
            "Apple silicon. The GPU here is called MPS and PyTorch uses it "
            "automatically, so there is no CUDA style download to do and "
            "nothing missing if you were looking for one.")
        notes.append(
            "4-bit quantization is NVIDIA only, so it is not available on "
            "this machine at any version. Pick a base model that fits in "
            "memory as it is. Qwen3-1.7B is the usual answer, and Qwen3-4B "
            "works on a machine with plenty of unified memory.")
        notes.append(
            "Unified memory means the GPU shares the machine's RAM, so the "
            "number that matters is total memory rather than a separate VRAM "
            "figure.")
    elif facts["os"] == "Darwin":
        notes.append(
            "Intel Mac. There is no GPU acceleration available here, so "
            "everything runs on the processor. It works, and it is slow. "
            "Qwen3-0.6B with a small dataset is the realistic size.")

    if facts["os"] == "Windows" and facts["arch"].lower() in ("arm64", "aarch64"):
        notes.append(
            "Windows on ARM. PyTorch wheels for this platform are limited, "
            "so check what installed actually runs before planning a build.")

    if not facts["compiler"]:
        notes.append(
            "No C compiler was found. That only matters if you want to build "
            "the optional engine in engine/, and nothing in the Python "
            "toolchain needs it.")

    return notes


def run_setup(dry_run: bool = False, verbose: bool = True) -> dict:
    """Look at the machine, then fix what can be fixed."""
    facts = machine_facts()

    if verbose:
        print("STRATUM setup\n" + "-" * 42)
        gpu = ("NVIDIA, visible to PyTorch" if facts["cuda_visible"] else
               "NVIDIA, NOT visible to PyTorch" if facts["nvidia_present"] else
               "Apple MPS" if facts["mps_visible"] else "none found")
        print(f"System   : {facts['os']} {facts['arch']}")
        print(f"Python   : {facts['python']}")
        print(f"PyTorch  : {facts['torch'] or 'not installed'}")
        print(f"GPU      : {gpu}")
        print()

    actions = plan_actions(facts)
    notes = platform_notes(facts)

    if verbose:
        for n in notes:
            print(f"note: {n}\n")

    if not actions:
        if verbose:
            print("Nothing to install. Run `stratum doctor` for the full "
                  "report, or `stratum teachers` to see which models this "
                  "machine can run.")
        return {"facts": facts, "actions": [], "installed": [], "failed": []}

    installed, failed = [], []
    for a in actions:
        if verbose:
            print(f"{a['why']}")
            print(f"  {' '.join(a['cmd'])}")
        if dry_run:
            continue
        try:
            subprocess.run(a["cmd"], check=True)
            installed.append(a["key"])
            if verbose:
                print(f"  done\n")
        except subprocess.CalledProcessError as e:
            failed.append(a["key"])
            if verbose:
                print(f"  FAILED with exit code {e.returncode}. Run the "
                      f"command above by hand to see the full output.\n")

    if verbose:
        if dry_run:
            print("\nThis was a dry run and nothing was installed. Drop "
                  "--dry-run to do it.")
        elif failed:
            print(f"\n{len(failed)} step(s) failed. Fix those, then re-run "
                  f"`stratum setup`.")
        else:
            print("\nDone. Run `stratum doctor` to confirm, then "
                  "`stratum teachers` to see what this machine can run.")

    return {"facts": facts, "actions": actions,
            "installed": installed, "failed": failed}
