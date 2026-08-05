"""
The context index. Facts the model should look up rather than memorise.

A stratum learns the shape of your domain, how a question in it gets
answered and in what words. What it should not be asked to hold is every
number in every document, because a number changes on Tuesday and a model
does not.

So this is the other half. It holds the chunks, a vector for each, a term
index, and the links between them, and it answers one question: given this
request from this person, which pieces of text should go in the prompt.

Three things make it different from a bolted-on vector store.

It is FILTERED BEFORE IT IS SCORED. A person who cannot see a compartment
never has its chunks ranked, not even to be discarded afterwards. Filtering
after ranking leaks through timing and through top-k displacement, and it is
the kind of thing that survives a demo and fails an audit.

It has LINKS, and the links are filtered too. A hit can pull in the chunks
it is connected to, which is what lets an answer reach a definition that the
question never named. An unfiltered edge is a hole straight through the
filtering above, so expansion checks permission on every hop.

It is ORDINARY FILES. Vectors are a .npy read through a memory map, the term
index and the links are arrays in an .npz, and the text stays in the
chunks.jsonl that produced it. Nothing here is a format anyone has to learn,
and a person can open every part of it.

On scale, plainly: a flat scan over the vectors is linear and beats an
approximate index until roughly a million chunks, because at that size the
scan is about 150 ms and generating the answer takes seconds. This project
measured the storage behind it (`engine/README.md`) and the finding was that
what costs is the number of reads you have to do ONE AFTER ANOTHER, not
their size. A flat scan is one read. That is the whole reason it wins.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

INDEX_VERSION = 3
# Word characters rather than a literal a-z range. The range version
# produced zero tokens for Arabic, Hebrew, CJK, Cyrillic and Greek, and cut
# accented Latin into fragments, which turned every search into "return the
# first k chunks" with no error anywhere. Technical shapes still survive
# intact, which is what the range was there to protect.
_TOKEN = re.compile(r"[^\W_]+(?:[-_.][^\W_]+)*")


class ContextError(Exception):
    """Something is wrong with the index or how it is being used."""


def _tokens(text: str) -> list[str]:
    """Words, plus the technical shapes plain word splitting throws away.

    Codes like V-201 and 480V are the whole vocabulary of an engineering
    corpus, and a tokenizer that splits them into 'v' and '201' loses the
    only part that identified the equipment.
    """
    return _TOKEN.findall(text.lower())


def _hash_embed(text: str, dim: int) -> "list[float]":
    """A vector with no model behind it.

    Every token, and every adjacent pair of tokens, is hashed to a position
    and added there. It is the hashing trick, it needs no download and no
    GPU, and on a technical corpus it does most of what a small sentence
    model does because the vocabulary is doing the work rather than the
    semantics.

    It is the default so that the index builds on any machine, offline, in
    seconds. Pass --embedder hf when quality matters more than that.
    """
    import numpy as np

    v = np.zeros(dim, dtype="float32")
    toks = _tokens(text)
    for t in toks:
        h = hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        # The sign comes from the hash too, so unrelated tokens landing on
        # the same position tend to cancel instead of piling up.
        sign = 1.0 if h[4] & 1 else -1.0
        v[idx] += sign
    for a, b in zip(toks, toks[1:]):
        h = hashlib.blake2b(f"{a}_{b}".encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if h[4] & 1 else -1.0
        v[idx] += sign * 0.5
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _hf_embed_batch(texts, model_name, device=None, batch_size=16):
    """Mean pooled hidden states from an ordinary Hugging Face model.

    transformers is already required by this project, so this adds no new
    dependency. Any encoder works, and a small sentence model works best.
    """
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    from .hf_utils import pick_device
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        # Padding positions must not be averaged in, or a short chunk in a
        # batch of long ones gets its vector pulled toward nothing.
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        out.append(pooled.float().cpu().numpy())
    return np.vstack(out).astype("float32")


def build_index(chunks_path: str, out_dir: str, embedder: str = "hash",
                embed_model: str | None = None, dim: int = 512,
                link_top: int = 12, unlabelled_compartment: str | None = None,
                verbose: bool = True) -> dict:
    """Turn chunks.jsonl into a searchable, access aware index.

    Everything written here is derived. Delete the folder and rebuild and
    you get the same thing back, so it never needs backing up separately
    from the corpus.
    """
    import numpy as np

    from .data import load_jsonl

    rows = load_jsonl(chunks_path, required_keys=("id", "text", "source"))
    if not rows:
        raise ContextError(f"{chunks_path} has no chunks in it.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    texts = [r["text"] for r in rows]

    # An unlabelled chunk used to become public. That is the wrong way to
    # fail, because the corpora most likely to be ingested without the flag
    # are exactly the ones with something in them worth compartmenting, and
    # nothing anywhere said it had happened.
    unlabelled = sum(1 for r in rows if not r.get("compartment"))
    if unlabelled and not unlabelled_compartment:
        raise ContextError(
            f"{unlabelled} of {len(rows)} chunks carry no compartment, and "
            f"treating them as public would make them readable by every "
            f"principal.\n"
            f" - Re-run `stratum corpus ingest --compartments` over a folder "
            f"sorted by who may read it, or\n"
            f" - pass --unlabelled-compartment NAME to say which compartment "
            f"they belong to.")
    comps = [r.get("compartment") or unlabelled_compartment for r in rows]

    if verbose:
        print(f"Indexing {len(rows)} chunks from {chunks_path}")
        print(f"  compartments: {', '.join(sorted(set(comps)))}")

    if embedder == "hash":
        if verbose:
            print(f"  embedding with the built-in hashing embedder, dim {dim}")
        vecs = np.vstack([_hash_embed(t, dim) for t in texts])
    elif embedder == "hf":
        # The RESOLVED name is what gets recorded below. Storing the argument
        # instead wrote a null into the index whenever the default was used,
        # and every query then crashed asking the Hub for a model called
        # None. A default that is not written down is not a default.
        embed_model = embed_model or "sentence-transformers/all-MiniLM-L6-v2"
        if verbose:
            print(f"  embedding with {embed_model}")
        vecs = _hf_embed_batch(texts, embed_model)
        dim = int(vecs.shape[1])
    else:
        raise ContextError(
            f"Unknown embedder '{embedder}'. Use 'hash' for the built-in one "
            f"that needs no download, or 'hf' for a Hugging Face model.")

    np.save(out / "vectors.npy", vecs)

    postings = _build_postings(texts, verbose=verbose)
    np.savez(out / "postings.npz", **postings)

    links = _build_links(postings, len(rows), link_top, verbose=verbose)
    np.savez(out / "links.npz", **links)

    meta = {
        "version": INDEX_VERSION,
        "chunks": chunks_path,
        # The corpus fingerprint. Loading an index against a corpus it was
        # not built from is the most common silent failure in a home built
        # retrieval stack, and it produces answers that look fine.
        "corpus_sha256": _file_sha(chunks_path),
        "n": len(rows),
        "dim": int(dim),
        "embedder": embedder,
        "embed_model": embed_model if embedder == "hf" else None,
        "ids": [r["id"] for r in rows],
        "compartments": comps,
        "sources": [r.get("source", "") for r in rows],
        # What each chunk came from. Text lifted out of a diagram reads
        # nothing like prose, and a question about a diagram usually wants
        # the diagram, so it is worth being able to prefer one or the other.
        "kinds": [r.get("kind", "document") for r in rows],
    }
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    size_mb = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    if verbose:
        print(f"  index -> {out}  ({size_mb:.1f} MB)")
    return {"n": len(rows), "dim": dim, "size_mb": size_mb,
            "compartments": sorted(set(comps))}


def _file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _build_postings(texts: list[str], min_df: int = 1, verbose: bool = True) -> dict:
    """A term index, stored as plain arrays.

    This is a normal inverted index with TF-IDF weights, kept in compressed
    sparse row form so a query touches only the terms it mentions. It exists
    alongside the vectors because the two fail differently. A vector finds
    text that means the same thing, and a term index finds the document that
    contains the exact part number, which is what somebody usually wants.
    """
    import numpy as np

    df: dict[str, int] = {}
    doc_terms = []
    for t in texts:
        counts: dict[str, int] = {}
        for tok in _tokens(t):
            counts[tok] = counts.get(tok, 0) + 1
        doc_terms.append(counts)
        for tok in counts:
            df[tok] = df.get(tok, 0) + 1

    vocab = {t: i for i, t in enumerate(sorted(k for k, v in df.items() if v >= min_df))}
    n = len(texts)
    idf = np.zeros(len(vocab), dtype="float32")
    for t, i in vocab.items():
        idf[i] = math.log((n + 1) / (df[t] + 1)) + 1.0

    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for counts in doc_terms:
        row_i, row_d = [], []
        for tok, c in counts.items():
            j = vocab.get(tok)
            if j is None:
                continue
            # Sublinear term frequency. A word appearing forty times in a
            # chunk does not make that chunk forty times more about it.
            row_i.append(j)
            row_d.append((1.0 + math.log(c)) * float(idf[j]))
        norm = math.sqrt(sum(x * x for x in row_d)) or 1.0
        indices.extend(row_i)
        data.extend(x / norm for x in row_d)
        indptr.append(len(indices))

    if verbose:
        print(f"  term index: {len(vocab)} terms, {len(indices)} postings")

    return {
        "indptr": np.array(indptr, dtype="int64"),
        "indices": np.array(indices, dtype="int32"),
        "data": np.array(data, dtype="float32"),
        "idf": idf,
        "vocab_terms": np.array(sorted(vocab), dtype=object),
    }


def _build_links(postings: dict, n: int, top: int, verbose: bool = True) -> dict:
    """Edges between chunks that talk about the same distinctive things.

    This is the linkage layer, and it is built with no language model calls
    at all. Two chunks are connected when they share terms that are rare
    across the corpus. Rare is what carries the signal, because every chunk
    mentions 'the' and only a few mention 'blowout preventer'.

    What it buys is reach. A question about a valve can arrive at the
    procedure that governs it without ever naming it, because the two are
    connected through the equipment tag they share.

    Building the graph this way is not new, and it should not be. Microsoft's
    LazyGraphRAG and LinearRAG both extract structure without paying a model
    per document, which is what makes graph building affordable at all.
    """
    import numpy as np

    indptr = postings["indptr"]
    indices = postings["indices"]
    idf = postings["idf"]

    # Which terms are allowed to create an edge, chosen by how many chunks
    # they appear in rather than by a score threshold.
    #
    # Selecting on IDF looks equivalent and is not. Word frequencies follow
    # a very steep curve, so most terms in any corpus appear exactly once,
    # the median IDF sits on those singletons, and a rule like "above the
    # median" throws away every term that could possibly connect two chunks.
    # That produced an index with zero edges and no error.
    #
    # A term in one chunk connects nothing. A term in a large fraction of
    # the corpus connects everything to everything, which is the same as
    # connecting nothing and costs far more to compute.
    inverted: dict[int, list[int]] = {}
    for doc in range(n):
        for k in range(indptr[doc], indptr[doc + 1]):
            inverted.setdefault(int(indices[k]), []).append(doc)

    max_df = max(4, min(64, int(n * 0.05)))

    scores: dict[int, dict[int, float]] = {}
    for term, docs in inverted.items():
        if len(docs) < 2 or len(docs) > max_df:
            continue
        w = float(idf[term])
        for i, a in enumerate(docs):
            for b in docs[i + 1:]:
                scores.setdefault(a, {})
                scores.setdefault(b, {})
                scores[a][b] = scores[a].get(b, 0.0) + w
                scores[b][a] = scores[b].get(a, 0.0) + w

    l_indptr = [0]
    l_indices: list[int] = []
    l_data: list[float] = []
    for doc in range(n):
        nbrs = sorted(scores.get(doc, {}).items(), key=lambda kv: -kv[1])[:top]
        for j, w in nbrs:
            l_indices.append(j)
            l_data.append(w)
        l_indptr.append(len(l_indices))

    if verbose:
        print(f"  links: {len(l_indices)} edges over {n} chunks "
              f"({len(l_indices) / max(n, 1):.1f} per chunk), from terms "
              f"appearing in 2 to {max_df} chunks")
    if n > 20 and not l_indices:
        # An index with no edges still answers searches, so this would
        # otherwise go unnoticed until link expansion silently did nothing.
        print("  WARNING: no links were built, so link expansion will do "
              "nothing. Every term is either\n  unique to one chunk or spread "
              "across too many for an edge to mean anything.")

    return {
        "indptr": np.array(l_indptr, dtype="int64"),
        "indices": np.array(l_indices, dtype="int32"),
        "data": np.array(l_data, dtype="float32"),
    }


class ContextIndex:
    """A built index, ready to answer questions from a given principal."""

    def __init__(self, index_dir: str, chunks_path: str | None = None,
                 verbose: bool = False):
        import numpy as np

        d = Path(index_dir)
        meta_path = d / "meta.json"
        if not meta_path.exists():
            raise ContextError(
                f"No index at {index_dir}. Build one with "
                f"`stratum context build --chunks chunks.jsonl --out {index_dir}`.")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if self.meta.get("version") != INDEX_VERSION:
            raise ContextError(
                f"The index at {index_dir} was built by another version of "
                f"STRATUM. Rebuild it with `stratum context build`.")

        # Read only and memory mapped. The pages a query touches are paged
        # in and the rest never leaves the disk, so opening a large index
        # costs nothing.
        self.vectors = np.load(d / "vectors.npy", mmap_mode="r")
        self.postings = dict(np.load(d / "postings.npz", allow_pickle=True))
        self.links = dict(np.load(d / "links.npz"))

        self.ids = self.meta["ids"]
        self.compartments = self.meta["compartments"]
        self.sources = self.meta["sources"]
        self.kinds = self.meta.get("kinds") or ["document"] * len(self.ids)
        self.dim = self.meta["dim"]
        self.vocab = {t: i for i, t in enumerate(self.postings["vocab_terms"])}

        self.chunks_path = chunks_path or self.meta["chunks"]
        self._texts: list[str] | None = None

        # Checked here rather than left for a caller to remember. Every
        # construction site used to skip it, so an index built from an older
        # version of the corpus would score rows normally and hand back
        # citations with no text in them.
        self.check_corpus()

        if verbose:
            print(f"Index: {self.meta['n']} chunks, "
                  f"{len(set(self.compartments))} compartments, "
                  f"embedder {self.meta['embedder']}")

    def check_corpus(self, chunks_path: str | None = None) -> None:
        """Refuse to serve an index built from a different corpus.

        Silently answering from stale text is worse than failing, because
        nothing about the answer looks wrong.
        """
        path = chunks_path or self.chunks_path
        if not Path(path).exists():
            raise ContextError(f"The chunk file {path} is missing.")
        if _file_sha(path) != self.meta["corpus_sha256"]:
            raise ContextError(
                f"{path} has changed since this index was built, so the index "
                f"points at text that is no longer there.\n"
                f" - Rebuild with `stratum context build`.")

    @property
    def texts(self) -> list[str]:
        """Chunk text, loaded on first use.

        Kept out of the index deliberately. The text already exists in
        chunks.jsonl and copying it would mean two things to keep in step.
        """
        if self._texts is None:
            from .data import load_jsonl
            rows = load_jsonl(self.chunks_path, required_keys=("id", "text"))
            by_id = {r["id"]: r["text"] for r in rows}
            missing = [i for i in self.ids if i not in by_id]
            if missing:
                # Substituting an empty string here produced a citation with
                # a real source, a real score and no content, which reads as
                # an answer with nothing behind it.
                raise ContextError(
                    f"{self.chunks_path} no longer holds {len(missing)} of "
                    f"the chunks this index was built from, starting with "
                    f"{missing[0]}.\n"
                    f" - Rebuild with `stratum context build`.")
            self._texts = [by_id[i] for i in self.ids]
        return self._texts

    def allowed_mask(self, allowed: set[str] | None):
        """Which rows this principal may have scored at all.

        Returned as a boolean array so the filter happens before ranking
        rather than after. None means no filtering, which is the single
        machine case.
        """
        import numpy as np

        if allowed is None:
            return np.ones(len(self.ids), dtype=bool)
        return np.array([c in allowed for c in self.compartments], dtype=bool)

    def _embed_query(self, text: str):
        import numpy as np

        if self.meta["embedder"] == "hash":
            return _hash_embed(text, self.dim)
        vec = _hf_embed_batch([text], self.meta["embed_model"])[0]
        return np.asarray(vec, dtype="float32")

    def _lexical_scores(self, query: str, mask):
        """Score every permitted chunk against the query's terms."""
        import numpy as np

        scores = np.zeros(len(self.ids), dtype="float32")
        qcounts: dict[int, float] = {}
        for tok in _tokens(query):
            j = self.vocab.get(tok)
            if j is not None:
                qcounts[j] = qcounts.get(j, 0.0) + 1.0
        if not qcounts:
            return scores

        idf = self.postings["idf"]
        qvec = {j: (1.0 + math.log(c)) * float(idf[j]) for j, c in qcounts.items()}
        norm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0

        indptr = self.postings["indptr"]
        indices = self.postings["indices"]
        data = self.postings["data"]
        for doc in np.nonzero(mask)[0]:
            s = 0.0
            for k in range(indptr[doc], indptr[doc + 1]):
                w = qvec.get(int(indices[k]))
                if w is not None:
                    s += w * float(data[k])
            scores[doc] = s / norm
        return scores

    def search(self, query: str, allowed: set[str] | None = None, k: int = 5,
               expand: int = 0, dense_weight: float = 1.0,
               lexical_weight: float = 1.0, kind: str | None = None) -> list[dict]:
        """Find the chunks this principal should see for this request.

        Dense and lexical rankings are combined by reciprocal rank fusion,
        which needs no tuning and does not care that the two scores are on
        different scales. Adding them directly would let whichever happens
        to be larger win everything.
        """
        import numpy as np

        mask = self.allowed_mask(allowed)
        if kind:
            mask = mask & np.array([k == kind for k in self.kinds], dtype=bool)
        if not mask.any():
            return []

        q = self._embed_query(query)
        dense = np.asarray(self.vectors @ q, dtype="float32")
        # Forbidden rows are removed before ranking, not after. Ranking them
        # and dropping them later would let a hidden chunk push a permitted
        # one out of the results, which leaks through what is missing.
        dense[~mask] = -np.inf

        lexical = self._lexical_scores(query, mask)
        lexical[~mask] = -np.inf

        pool = min(len(self.ids), max(k * 10, 50))
        fused: dict[int, float] = {}
        for arr, weight in ((dense, dense_weight), (lexical, lexical_weight)):
            if weight <= 0:
                continue
            order = np.argsort(-arr)[:pool]
            for rank, doc in enumerate(order):
                # A score of zero is not weak evidence, it is no evidence.
                # Crediting it anyway gave rows with no term overlap at all
                # the same rank-based weight as genuine matches, and since
                # they all tie at zero their order came from the sort rather
                # than from anything about the text.
                if not np.isfinite(arr[doc]) or arr[doc] <= 0:
                    continue
                fused[int(doc)] = fused.get(int(doc), 0.0) + weight / (60.0 + rank)

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        hits = [self._hit(doc, score, "direct") for doc, score in ranked]

        if expand > 0 and hits:
            hits.extend(self._expand([h["row"] for h in hits], mask, expand,
                                     {h["row"] for h in hits}))
        return hits

    def _expand(self, seed_rows, mask, limit, seen):
        """Follow links out of the hits, one hop, permission checked.

        The check is the point. An edge that crosses into a compartment the
        caller cannot see would walk straight around the filtering above,
        and it would do it silently, so every neighbour is tested rather
        than trusted because its neighbour was allowed.
        """
        indptr = self.links["indptr"]
        indices = self.links["indices"]
        data = self.links["data"]

        found: dict[int, float] = {}
        for row in seed_rows:
            for k in range(indptr[row], indptr[row + 1]):
                nbr = int(indices[k])
                if nbr in seen or not mask[nbr]:
                    continue
                found[nbr] = max(found.get(nbr, 0.0), float(data[k]))

        best = sorted(found.items(), key=lambda kv: -kv[1])[:limit]
        return [self._hit(doc, score, "linked") for doc, score in best]

    def _hit(self, row: int, score: float, how: str) -> dict:
        return {
            "row": int(row),
            "id": self.ids[row],
            "compartment": self.compartments[row],
            "source": self.sources[row],
            "kind": self.kinds[row],
            "score": float(score),
            "how": how,
            "text": self.texts[row],
        }


def evaluate_recall(index: ContextIndex, test_path: str, k_values=(1, 5, 20),
                    allowed: set[str] | None = None, expand: int = 0,
                    verbose: bool = True) -> dict:
    """How often the chunk that produced a question is actually found.

    The teacher recorded which chunk it was looking at when it wrote each
    question, so the labels were free and they are exact. This is scored on
    the same held-out split the generator is scored on, which means the
    retriever and the model can be compared on the same footing.
    """
    from .data import load_jsonl

    rows = load_jsonl(test_path, required_keys=("prompt",))
    rows = [r for r in rows if r.get("source_chunk")]
    if not rows:
        raise ContextError(
            f"{test_path} has no source_chunk fields, so there is nothing to "
            f"score recall against. Test files written by "
            f"`stratum corpus pairs` carry them.")

    kmax = max(k_values)
    hits = {k: 0 for k in k_values}
    for r in rows:
        found = index.search(r["prompt"], allowed=allowed, k=kmax, expand=expand)
        ids = [h["id"] for h in found]
        for k in k_values:
            if r["source_chunk"] in ids[:k]:
                hits[k] += 1

    report = {"n": len(rows),
              "recall": {k: hits[k] / len(rows) for k in k_values}}
    if verbose:
        print(f"Recall over {len(rows)} held-out questions")
        for k in k_values:
            print(f"  recall@{k:<3} {report['recall'][k]:.1%}")
    return report


def build_prompt(question: str, hits: list[dict], budget_chars: int = 2000,
                 style: str = "chunks") -> str:
    """Assemble the context the model actually sees.

    Style matters more than it looks and this project does not yet know
    which one wins here, so both are available and doc 17 says how to find
    out on your own corpus.

      chunks     the retrieved text as it stands
      sentences  only the sentences that share wording with the question,
                 which fits far more distinct sources in the same budget

    Every piece carries its source, so an answer can be traced back to a
    document rather than believed.
    """
    if not hits:
        return question

    parts, used = [], 0
    for h in hits:
        text = h["text"]
        if style == "sentences":
            text = _relevant_sentences(question, text)
        head = f"[{h['source']}]"
        piece = f"{head}\n{text}".strip()
        if used + len(piece) > budget_chars:
            piece = piece[:max(0, budget_chars - used)]
        if not piece:
            break
        parts.append(piece)
        used += len(piece)
        if used >= budget_chars:
            break

    return ("Use the reference material below to answer. If it does not "
            "contain the answer, say so.\n\n"
            + "\n\n".join(parts)
            + f"\n\nQuestion: {question}")


# Words that say a question is about something that was a picture. A
# diagram, a nameplate, a chart. They are the only signal available, because
# the question arrives as text either way.
_VISUAL_WORDS = frozenset("""
diagram diagrams drawing drawings schematic schematics figure figures chart
charts graph graphs plot plots photo photograph photographs image images
picture pictures nameplate label labels labelled labeled legend axis axes
sketch layout arrangement shown depicts depicted illustration illustrated
screenshot scan scanned plate plan drawing-number
""".split())


def looks_visual(question: str) -> bool:
    """Whether a question is asking about something that was a picture.

    Used to prefer image-derived chunks, and to prefer a skill trained on
    them. It is a hint and never a filter, because somebody asking about a
    pump does not say diagram and still wants the diagram.
    """
    return bool(_VISUAL_WORDS & set(_tokens(question)))


_SENT = re.compile(r"(?<=[.!?])\s+")


def _relevant_sentences(question: str, text: str, keep: int = 4) -> str:
    """The sentences in a chunk that share distinctive words with the
    question, in the order they appeared."""
    qt = set(_tokens(question))
    sents = _SENT.split(text)
    scored = []
    for i, s in enumerate(sents):
        st = set(_tokens(s))
        if not st:
            continue
        scored.append((len(qt & st) / math.sqrt(len(st)), i, s))
    scored.sort(key=lambda x: -x[0])
    chosen = sorted(scored[:keep], key=lambda x: x[1])
    return " ".join(s for _, _, s in chosen) or text[:400]
