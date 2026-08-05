"""
Turn a finished build into one artifact, and put it somewhere else.

The machine that builds a model and the machine that serves it are almost
never the same. Building wants a GPU for an hour. Serving wants to be near
the people asking, on several machines, for months. So the useful unit is a
bundle that can be handed from one to the other and checked on arrival.

What goes in a bundle:

    strata/       one adapter per skill or compartment
    index/        the context index, if there is one
    policy.json   who may see what, if there is one
    router.json   which skill answers which request, if there is one
    MANIFEST.json everything above, with a sha256 for every file

The manifest is the point. A serving node that loads a bundle without
checking it is trusting a file copy, and a half-copied adapter does not
announce itself, it just answers slightly worse. `stratum unpack` verifies
every hash before anything is served, and refuses on a mismatch.

What deliberately does NOT go in:

  - the base model. It is gigabytes, it is identical everywhere, and every
    serving node can pull it from the same place. The manifest records
    which one is required so a mismatch is caught rather than discovered.
  - the corpus. The index refers to chunks.jsonl, and whether that travels
    depends on whether the serving node is allowed to hold the source text.
    That is a policy decision, so it is a flag rather than a default.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

MANIFEST_VERSION = 1


class DeployError(Exception):
    """A bundle is malformed, incomplete, or does not match its manifest."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _walk(root: Path):
    """Every file under root, relative, in a stable order.

    Sorted so that two bundles built from the same inputs produce the same
    manifest, which is what makes them comparable.
    """
    for p in sorted(root.rglob("*")):
        if p.is_file():
            yield p


def pack(out_dir: str, strata: list[str], index: str | None = None,
         policy: str | None = None, router: str | None = None,
         chunks: str | None = None, base: str | None = None,
         verbose: bool = True) -> dict:
    """Collect a build into one directory with a checked manifest."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise DeployError(
            f"{out_dir} already has something in it. Packing into a used "
            f"folder would mix two builds, so pick an empty path or delete "
            f"that one first.")
    out.mkdir(parents=True, exist_ok=True)

    manifest = {"version": MANIFEST_VERSION, "base_model": base,
                "strata": [], "files": {}}

    strata_out = out / "strata"
    strata_out.mkdir()
    for s in strata:
        src = Path(s)
        if not src.is_dir():
            raise DeployError(f"{s} is not a stratum folder.")
        shutil.copytree(src, strata_out / src.name)
        manifest["strata"].append(src.name)
        # The base model has to match on the serving side or the adapter
        # attaches to the wrong shapes, so it is read from the card here and
        # checked on arrival.
        card = src / "stratum_card.json"
        if card.exists():
            try:
                got = json.loads(card.read_text(encoding="utf-8")).get("base_model")
                if got and base and got != base:
                    raise DeployError(
                        f"Stratum '{src.name}' was trained on {got} but this "
                        f"bundle says {base}. Adapters only fit the base they "
                        f"were trained on.")
                manifest["base_model"] = manifest["base_model"] or got
            except json.JSONDecodeError:
                pass

    for name, path in (("index", index), ("policy.json", policy),
                       ("router.json", router)):
        if not path:
            continue
        src = Path(path)
        if not src.exists():
            raise DeployError(f"{path} does not exist.")
        if src.is_dir():
            shutil.copytree(src, out / name)
        else:
            shutil.copy2(src, out / name)

    if chunks:
        src = Path(chunks)
        if not src.exists():
            raise DeployError(f"{chunks} does not exist.")
        shutil.copy2(src, out / "chunks.jsonl")
        manifest["chunks_included"] = True
    else:
        manifest["chunks_included"] = False

    # Hash last, over everything that actually landed.
    for p in _walk(out):
        manifest["files"][str(p.relative_to(out)).replace("\\", "/")] = _sha256(p)

    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                       encoding="utf-8")

    total = sum(p.stat().st_size for p in _walk(out))
    if verbose:
        print(f"Packed {len(manifest['strata'])} strata into {out}")
        print(f"  base model : {manifest['base_model'] or 'not recorded'}")
        print(f"  files      : {len(manifest['files'])}")
        print(f"  size       : {total / 1e6:.1f} MB")
        if not manifest["chunks_included"] and index:
            print("  note: the index is here but the source text is not. Pass "
                  "--chunks to include it,")
            print("        or make sure the serving node can reach the same "
                  "chunks.jsonl.")
        print(f"\nCopy {out} to the serving machine, then run "
              f"`stratum unpack {out.name}`.")
    return manifest


def verify(bundle_dir: str, verbose: bool = True) -> dict:
    """Check a bundle against its own manifest.

    Runs before anything is served, because a truncated adapter does not
    fail loudly. It attaches, it generates, and it is quietly worse.
    """
    root = Path(bundle_dir)
    mpath = root / "MANIFEST.json"
    if not mpath.exists():
        raise DeployError(
            f"No MANIFEST.json in {bundle_dir}, so there is nothing to check "
            f"this against. Was it made by `stratum pack`?")

    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("version") != MANIFEST_VERSION:
        raise DeployError(
            f"This bundle was packed by a different version of STRATUM. "
            f"Re-pack it on the build machine.")

    expected = manifest["files"]
    problems = []

    for rel, want in expected.items():
        if rel == "MANIFEST.json":
            continue
        p = root / rel
        if not p.exists():
            problems.append(f"missing: {rel}")
            continue
        if _sha256(p) != want:
            problems.append(f"changed since packing: {rel}")

    on_disk = {str(p.relative_to(root)).replace("\\", "/") for p in _walk(root)}
    for rel in sorted(on_disk - set(expected)):
        if rel != "MANIFEST.json":
            problems.append(f"not in the manifest: {rel}")

    if verbose:
        print(f"Bundle {bundle_dir}")
        print(f"  strata     : {', '.join(manifest['strata'])}")
        print(f"  base model : {manifest.get('base_model') or 'not recorded'}")
        print(f"  files      : {len(expected)}")

    if problems:
        if verbose:
            print(f"\nFAILED. {len(problems)} problem(s):")
            for p in problems[:20]:
                print(f"  - {p}")
        raise DeployError(
            f"{len(problems)} file(s) do not match the manifest. Copy the "
            f"bundle again rather than serving from it.")

    if verbose:
        print("  every file matches its hash")
    return manifest


def serve_command(bundle_dir: str, host: str = "0.0.0.0",
                  port: int = 8927) -> list[str]:
    """The command that serves a verified bundle.

    Printed rather than run, so whoever is deploying can see exactly what is
    about to happen and put it in whatever supervisor they already use.
    """
    root = Path(bundle_dir)
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))

    cmd = ["stratum", "serve"]
    cmd += [str(root / "strata" / s) for s in manifest["strata"]]
    if (root / "router.json").exists():
        cmd += ["--router", str(root / "router.json")]
    if (root / "index").exists():
        cmd += ["--context", str(root / "index")]
        chunks = root / "chunks.jsonl"
        if chunks.exists():
            cmd += ["--context-chunks", str(chunks)]
    if (root / "policy.json").exists():
        cmd += ["--policy", str(root / "policy.json")]
    if manifest.get("base_model"):
        cmd += ["--base", manifest["base_model"]]
    cmd += ["--host", host, "--port", str(port)]
    return cmd
