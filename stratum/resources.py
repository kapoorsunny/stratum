"""
Find and release memory that an earlier STRATUM run is still holding.

A training run that was interrupted, a server closed by shutting its terminal,
a notebook that was never restarted. Each of those can leave a process alive
holding several gigabytes of GPU memory, and the next thing you start fails
with an out of memory error that names a number far smaller than the card you
bought. People conclude the card is too small and go and rent one.

Six leftover servers were found on the machine this was written on, holding
the whole card, while a training run sat waiting for memory that was already
paid for.

So this looks for STRATUM's own leftovers, says exactly what it found, and
lets you release them.

Two rules it does not break.

It only ever touches processes whose command line invokes STRATUM. Not python
in general, not anything sharing the GPU. A machine can be a shared build box
and somebody else's job is not this tool's business.

It never touches the process it is running in, or anything that process is
descended from. Freeing memory by killing yourself midway through is not
freeing memory.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

# How long to wait for a process to end politely before insisting. Torch
# processes flush CUDA context on the way out, which is worth allowing since
# a hard kill can leave the driver holding memory a little longer.
GRACE_SECONDS = 5.0

# Words that mean a command line belongs to this project. Checked against the
# whole command line, which on every platform contains either the module form
# or the console script.
MARKERS = ("-m stratum", "stratum serve", "stratum train", "stratum stack",
           "stratum distill", "stratum build", "stratum chat", "stratum eval",
           "\\stratum.exe", "/stratum")


class ResourceError(Exception):
    """Memory cannot be inspected or released on this machine."""


def _nvidia_smi(*args: str) -> str | None:
    """Run nvidia-smi, or return None where there is no NVIDIA driver.

    Absent is a normal answer rather than an error. Apple silicon and CPU
    machines have no such tool and are not broken for lacking it.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, *args], capture_output=True, text=True,
                             timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _rocm_smi() -> list[dict]:
    """AMD cards on Linux, through rocm-smi.

    Its output format has changed more than once, so this reads the JSON form
    and takes what it recognises rather than parsing columns by position.
    """
    exe = shutil.which("rocm-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--showmeminfo", "vram", "--json"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0 or not out.stdout.strip():
        return []
    try:
        import json
        data = json.loads(out.stdout)
    except Exception:
        return []
    devices = []
    for i, (card, fields) in enumerate(sorted(data.items())):
        used = total = None
        for key, value in fields.items():
            k = key.lower()
            try:
                v = int(value)
            except (TypeError, ValueError):
                continue
            if "total" in k and "vram" in k:
                total = v
            elif "used" in k and "vram" in k:
                used = v
        if used is not None and total:
            devices.append({"kind": "amd", "index": i, "name": card,
                            "used_mb": used // (1024 * 1024),
                            "total_mb": total // (1024 * 1024)})
    return devices


def system_memory() -> dict | None:
    """Used and total system memory, on any operating system.

    This matters everywhere, and on Apple silicon it is the whole story. There
    the GPU has no memory of its own, it shares this pool, so a model held by
    a forgotten process is taking exactly the memory the next run needs and
    there is no separate card to look at.
    """
    try:
        import psutil
        m = psutil.virtual_memory()
    except Exception:
        return None
    return {"used_mb": round((m.total - m.available) / 1e6),
            "total_mb": round(m.total / 1e6)}


def apple_gpu() -> bool:
    """Whether this is an Apple GPU sharing system memory."""
    if sys.platform != "darwin":
        return False
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        # Every Apple silicon Mac has one, so on darwin the honest default
        # when torch cannot be asked is yes rather than no.
        import platform
        return platform.machine() in ("arm64", "aarch64")


def gpu_memory() -> list[dict]:
    """Used and total memory per discrete GPU, in megabytes.

    Empty on a Mac and on a machine with no accelerator. That is a real
    answer rather than a failure, and callers pair it with system_memory,
    which is where the memory actually is on those machines.
    """
    out = _nvidia_smi("--query-gpu=index,name,memory.used,memory.total",
                      "--format=csv,noheader,nounits")
    if not out:
        return _rocm_smi()
    devices = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            devices.append({"kind": "nvidia", "index": int(parts[0]),
                            "name": parts[1], "used_mb": int(parts[2]),
                            "total_mb": int(parts[3])})
        except ValueError:
            continue
    return devices


def gpu_processes() -> dict:
    """Which process ids the driver says are holding GPU memory.

    Asked of the driver rather than guessed, because a process can hold GPU
    memory without looking busy in any other way, and this is the only place
    that truth lives.

    The value is how many megabytes each one holds, or None where the driver
    knows the process is on the card but will not say how much. That happens
    on ordinary Windows machines, where the display driver runs in WDDM mode
    and answers this question with [N/A].

    Keeping the process with an unknown size, rather than dropping the row
    that failed to parse, is the whole point. Dropping it loses the fact that
    the process is on the GPU at all, and then a command that acts only on
    GPU holders quietly acts on nothing and reports success.
    """
    out = _nvidia_smi("--query-compute-apps=pid,used_memory",
                      "--format=csv,noheader,nounits")
    if not out:
        return {}
    held = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        try:
            held[pid] = int(parts[1])
        except ValueError:
            held[pid] = None
    return held


def attribution_available() -> bool:
    """Whether this driver will say how much GPU memory each process holds.

    False on a normal Windows desktop or laptop. The totals for the card are
    still correct there, so the useful report is the card's total plus which
    processes are on it, without a per process breakdown.
    """
    held = gpu_processes()
    if not held:
        return True
    return any(v is not None for v in held.values())


def _protected_pids() -> set[int]:
    """This process and every process it descends from.

    Killing an ancestor takes this process with it, so the whole chain is off
    limits however well it matches the markers.
    """
    import psutil

    safe = {os.getpid()}
    try:
        p = psutil.Process(os.getpid())
        for parent in p.parents():
            safe.add(parent.pid)
    except Exception:
        pass
    return safe


def _looks_like_stratum(cmdline: str) -> bool:
    low = cmdline.lower().replace("/", "\\")
    return any(m.lower().replace("/", "\\") in low for m in MARKERS)


def find(include_gpu_only: bool = False) -> list[dict]:
    """Every STRATUM process running now, other than this one.

    include_gpu_only narrows it to those the driver says are actually holding
    GPU memory, which is the set worth killing when the complaint is that a
    card is full.
    """
    import psutil

    protected = _protected_pids()
    held = gpu_processes()
    found = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            pid = proc.info["pid"]
            if pid in protected:
                continue
            cmd = " ".join(proc.info.get("cmdline") or [])
            if not cmd or not _looks_like_stratum(cmd):
                continue
            if include_gpu_only and pid not in held:
                continue
            try:
                rss = proc.memory_info().rss
            except Exception:
                rss = 0
            found.append({
                "pid": pid,
                "verb": _verb_of(cmd),
                "cmdline": cmd,
                "rss_mb": round(rss / 1e6),
                # on_gpu is what the driver confirms. gpu_mb is how much, and
                # is None where the driver knows the first but not the second.
                # Two fields rather than one, so a machine that cannot report
                # sizes never reads as a machine holding nothing.
                "on_gpu": pid in held,
                "gpu_mb": held.get(pid),
                "age_s": round(time.time() - (proc.info.get("create_time") or 0)),
            })
        except Exception:
            # A process that ended while being read is not an error, it is
            # the thing being looked for having gone away by itself.
            continue
    # On the card first, then by how much they hold, then by size in RAM.
    # Sorted on three keys rather than one because the middle key is missing
    # on drivers that will not report per process sizes, and the ones on the
    # GPU are what somebody short of GPU memory wants at the top either way.
    return sorted(found, key=lambda d: (not d["on_gpu"], -(d["gpu_mb"] or 0),
                                        -d["rss_mb"]))


def _verb_of(cmdline: str) -> str:
    """The STRATUM subcommand, for a listing somebody has to read quickly."""
    parts = cmdline.replace("\\", "/").split()
    for i, p in enumerate(parts):
        if p == "stratum" or p.endswith("/stratum") or p.endswith("stratum.exe"):
            if i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                return parts[i + 1]
            return "stratum"
    return "stratum"


def release(dry_run: bool = False, gpu_only: bool = False,
            verbose: bool = True) -> dict:
    """Stop STRATUM's leftovers and report what came back.

    Polite first, then insistent. A torch process asked to stop will flush its
    CUDA context on the way out, and a process killed outright can leave the
    driver holding its memory for a moment longer, so the grace period is
    worth having even though it makes the command slower.
    """
    import psutil

    before = gpu_memory()
    ram_before = system_memory()
    targets = find(include_gpu_only=gpu_only)

    if verbose:
        _print_state(before, targets)

    if not targets:
        if verbose:
            print("Nothing of STRATUM's is running, so there is nothing here "
                  "to release.")
            if before and max(d["used_mb"] for d in before) > 512:
                print()
                print("The card is not empty though. Something else on this "
                      "machine is holding it,")
                print("and this command will not touch another program's "
                      "memory. `nvidia-smi` names it.")
        return {"found": [], "stopped": [], "survived": [],
                "before": before, "after": before, "freed_mb": 0}

    if dry_run:
        if verbose:
            print()
            print("Nothing was stopped. Run the same command without "
                  "--dry-run to release them.")
        return {"found": targets, "stopped": [], "survived": [],
                "before": before, "after": before, "freed_mb": 0}

    procs = []
    for t in targets:
        try:
            procs.append(psutil.Process(t["pid"]))
        except psutil.NoSuchProcess:
            continue

    # Children first. A server that spawned workers will otherwise have them
    # reparented and left holding memory after the parent is gone, which is
    # the exact failure this command exists to clear up.
    everything = []
    for p in procs:
        try:
            everything.extend(p.children(recursive=True))
        except Exception:
            pass
        everything.append(p)

    for p in everything:
        try:
            p.terminate()
        except Exception:
            pass
    gone, alive = psutil.wait_procs(everything, timeout=GRACE_SECONDS)

    for p in alive:
        try:
            p.kill()
        except Exception:
            pass
    if alive:
        _, alive = psutil.wait_procs(alive, timeout=GRACE_SECONDS)

    # The driver does not always give the memory back the instant a process
    # dies, so the after reading is taken once it has settled rather than
    # immediately, or the report understates what was recovered.
    time.sleep(1.5)
    after = gpu_memory()
    ram_after = system_memory()
    freed = _freed(before, after)

    stopped = [t for t in targets if t["pid"] not in {p.pid for p in alive}]
    survived = [t for t in targets if t["pid"] in {p.pid for p in alive}]

    if verbose:
        print()
        print(f"Stopped {len(stopped)} process(es)")
        for d in after:
            print(f"  GPU {d['index']}  {d['used_mb']:,} of {d['total_mb']:,} "
                  f"MB in use now")
        if freed > 0:
            print(f"  {freed:,} MB of GPU memory released")
        elif before:
            print("  no GPU memory came back, so what was holding it was not "
                  "on the card")
        # Reported on every platform, and the only figure that means anything
        # on a Mac, where the GPU shares this pool and there is no card to
        # read a number off.
        if ram_after and ram_before:
            back = ram_before["used_mb"] - ram_after["used_mb"]
            print(f"  System memory {ram_after['used_mb']:,} of "
                  f"{ram_after['total_mb']:,} MB in use now")
            if back > 0:
                print(f"  {back:,} MB of system memory released")
        if survived:
            print()
            print(f"{len(survived)} process(es) would not stop. Something is "
                  f"holding them open,")
            print("often a debugger or a stuck driver call. They need "
                  "stopping by hand.")
            for s in survived:
                print(f"  pid {s['pid']}  {s['verb']}")

    return {"found": targets, "stopped": stopped, "survived": survived,
            "before": before, "after": after, "freed_mb": freed}


def _freed(before: list[dict], after: list[dict]) -> int:
    a = {d["index"]: d["used_mb"] for d in after}
    return sum(max(0, d["used_mb"] - a.get(d["index"], d["used_mb"]))
               for d in before)


def _print_state(devices: list[dict], targets: list[dict]) -> None:
    for d in devices:
        share = 100 * d["used_mb"] / d["total_mb"] if d["total_mb"] else 0
        label = "GPU" if d.get("kind") == "nvidia" else "AMD GPU"
        print(f"{label} {d['index']}  {d['name']}")
        print(f"  {d['used_mb']:,} of {d['total_mb']:,} MB in use "
              f"({share:.0f}%)")

    mem = system_memory()
    if not devices:
        if apple_gpu():
            print("Apple GPU, which has no memory of its own and shares "
                  "system memory.")
            print("So the number below is the one that matters, and ending a "
                  "process is what")
            print("gives that memory back.")
        else:
            print("No GPU found, so there is no separate pool to release. "
                  "Anything held is")
            print("system memory, below.")
    if mem:
        share = 100 * mem["used_mb"] / mem["total_mb"] if mem["total_mb"] else 0
        print(f"System memory  {mem['used_mb']:,} of {mem['total_mb']:,} MB "
              f"in use ({share:.0f}%)")
    print()

    if not targets:
        return

    known = attribution_available()
    print(f"{len(targets)} STRATUM process(es) still running")
    print(f"  {'pid':>7}  {'command':<10} {'GPU MB':>8} {'RAM MB':>9}  age")
    for t in targets:
        age = t["age_s"]
        when = (f"{age // 3600}h{(age % 3600) // 60:02d}m" if age >= 3600
                else f"{age // 60}m{age % 60:02d}s")
        if t["gpu_mb"] is None:
            gpu = "on GPU" if t["on_gpu"] else "-"
        else:
            gpu = f"{t['gpu_mb']:,}" if t["on_gpu"] else "-"
        print(f"  {t['pid']:>7}  {t['verb']:<10} {gpu:>8} "
              f"{t['rss_mb']:>9,}  {when}")
    if not known:
        print()
        print("  This driver will not say how much GPU memory each process "
              "holds, which is")
        print("  normal on Windows and changes nothing about releasing them. "
              "The card total")
        print("  above is still correct, and 'on GPU' means the driver "
                "confirms that process")
        print("  is using it.")


def summary_line() -> str | None:
    """One line for `stratum doctor`, or None when there is nothing to say."""
    try:
        devices = gpu_memory()
        targets = find()
    except Exception:
        return None
    if not targets:
        return None
    # Sizes may be unavailable while the fact of being on the GPU is not, so
    # the wording follows what is actually known rather than treating a
    # missing number as a zero.
    gpu = sum(t["gpu_mb"] or 0 for t in targets)
    on_gpu = sum(1 for t in targets if t["on_gpu"])
    ram = sum(t["rss_mb"] for t in targets)
    total = devices[0]["total_mb"] if devices else 0

    if gpu:
        where = f"{gpu:,} MB of GPU memory"
        if total:
            where += f" of {total:,}"
    elif on_gpu:
        where = f"GPU memory, {on_gpu} of them on the card"
    else:
        where = f"{ram:,} MB of system memory"

    return (f"{len(targets)} earlier STRATUM process(es) are still running and "
            f"holding {where}.\n"
            f"  That is why a new run can fail for memory on a machine that "
            f"looks idle.\n"
            f"  Release them with `stratum free`, or `stratum free --dry-run` "
            f"to look first.")
