"""The access sweep, including proof that it can fail.

A sweep that reports PASSED on a broken index is worse than no sweep, so
the tests below break the filter on purpose and check that the sweep says
so. Without those, a green result means nothing.
"""
import json

import pytest

from stratum.simulate import SimulationError, run

# Two compartments whose words do not overlap, so a hit from the wrong one
# can only be a filter failure and never a coincidence of vocabulary.
TECH = ["turbine blade erosion inspection interval",
        "compressor surge margin during startup",
        "bearing vibration spectrum shows misalignment",
        "heat exchanger fouling factor calculation"]
LEGAL = ["indemnity clause survives termination of agreement",
         "arbitration seat and governing law provisions",
         "breach of warranty remedies and liquidated damages",
         "assignment of intellectual property on completion"]


def build(tmp_path, extra_principal_reads=None):
    """A real corpus, a real index and a real policy, all on disk."""
    from stratum.context import build_index

    chunks = tmp_path / "chunks.jsonl"
    rows = []
    for i, t in enumerate(TECH):
        rows.append({"id": f"tech-{i}", "text": t, "source": f"tech{i}.txt",
                     "compartment": "engineering", "kind": "document"})
    for i, t in enumerate(LEGAL):
        rows.append({"id": f"legal-{i}", "text": t, "source": f"legal{i}.txt",
                     "compartment": "legal", "kind": "document"})
    chunks.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                      encoding="utf-8")

    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "compartments": [
            {"name": "engineering", "tier": "department"},
            {"name": "legal", "tier": "department"},
        ],
        "principals": [
            {"name": "engineer_a", "compartments": ["engineering"]},
            {"name": "engineer_b", "compartments": ["engineering"]},
            {"name": "counsel_a", "compartments": ["legal"]},
            {"name": "counsel_b", "compartments": ["legal"]},
        ],
    }), encoding="utf-8")

    index = tmp_path / "index"
    build_index(str(chunks), str(index), verbose=False)
    return str(index), str(policy), str(chunks)


def test_a_correctly_filtered_index_passes(tmp_path):
    index, policy, chunks = build(tmp_path)
    report = run(index, policy, chunks, samples=2, verbose=False)
    assert report["passed"]
    assert report["leaks"] == []
    assert report["inconclusive"] == []


def test_the_positive_control_actually_retrieves(tmp_path):
    """A pass has to mean the index answered, not that it stayed silent."""
    index, policy, chunks = build(tmp_path)
    report = run(index, policy, chunks, samples=2, verbose=False)
    for name, d in report["per_principal"].items():
        assert d["hits_from_permitted"] > 0, \
            f"{name} retrieved nothing even from their own compartment"


def test_every_principal_is_asked_about_every_forbidden_compartment(tmp_path):
    """Coverage is the whole point. A sweep that skips a pair proves nothing
    about that pair."""
    index, policy, chunks = build(tmp_path)
    report = run(index, policy, chunks, samples=2, verbose=False)
    # Four principals, two compartments, two samples each.
    assert report["queries"] == 4 * 2 * 2
    for name, d in report["per_principal"].items():
        assert d["asked_about_denied"] == 2, name
        assert d["asked_about_permitted"] == 2, name


def test_a_filter_that_is_not_applied_is_caught(tmp_path, monkeypatch):
    """The negative control.

    This is the bug the sweep exists to find, a search that ignores the
    permitted set. If the sweep still reports a pass here then it is not
    testing anything and every green run above is meaningless.
    """
    from stratum.context import ContextIndex

    index, policy, chunks = build(tmp_path)
    real = ContextIndex.search

    def unfiltered(self, query, allowed=None, **kw):
        return real(self, query, allowed=None, **kw)

    monkeypatch.setattr(ContextIndex, "search", unfiltered)
    report = run(index, policy, chunks, samples=2, verbose=False)

    assert not report["passed"]
    assert report["leaks"], "an unfiltered search must be reported as a leak"
    leaked = {(lk["principal"], lk["leaked_from"]) for lk in report["leaks"]}
    assert ("engineer_a", "legal") in leaked
    assert ("counsel_a", "engineering") in leaked


def test_a_filter_dropped_only_on_link_expansion_is_caught(tmp_path, monkeypatch):
    """The subtler bug.

    Ranking is filtered, the hop out of a hit is not. A sweep that ran with
    expansion turned off would pass this and the index would still leak, so
    expansion is part of the test rather than an option.
    """
    from stratum.context import ContextIndex

    index, policy, chunks = build(tmp_path)
    real = ContextIndex.search

    def leaky_expand(self, query, allowed=None, expand=0, **kw):
        hits = real(self, query, allowed=allowed, expand=expand, **kw)
        if expand:
            # What an unchecked hop looks like, a neighbour appended without
            # asking whether this principal may read it.
            for i, comp in enumerate(self.compartments):
                if allowed and comp not in allowed:
                    hits.append({"id": self.ids[i], "compartment": comp,
                                 "source": self.sources[i], "text": "",
                                 "score": 0.1, "how": "expand"})
                    break
        return hits

    monkeypatch.setattr(ContextIndex, "search", leaky_expand)
    report = run(index, policy, chunks, samples=2, expand=3, verbose=False)

    assert not report["passed"]
    assert any(lk["via"] == "expand" for lk in report["leaks"]), \
        "a leak arriving through expansion has to be reported as such"


def test_an_index_that_returns_nothing_is_inconclusive_not_a_pass(tmp_path,
                                                                  monkeypatch):
    """Zero results leak nothing and prove nothing."""
    from stratum.context import ContextIndex

    index, policy, chunks = build(tmp_path)
    monkeypatch.setattr(ContextIndex, "search",
                        lambda self, query, **kw: [])
    report = run(index, policy, chunks, samples=2, verbose=False)

    assert not report["passed"]
    assert report["leaks"] == []
    assert sorted(report["inconclusive"]) == ["counsel_a", "counsel_b",
                                              "engineer_a", "engineer_b"]


def test_the_same_seed_asks_the_same_questions(tmp_path):
    """Two runs that disagree cannot be used as evidence of anything."""
    index, policy, chunks = build(tmp_path)
    a = run(index, policy, chunks, samples=2, seed=7, verbose=False)
    b = run(index, policy, chunks, samples=2, seed=7, verbose=False)
    assert a["per_principal"] == b["per_principal"]
    assert a["queries"] == b["queries"]


def test_a_corpus_with_no_compartments_never_reaches_the_sweep(tmp_path):
    """Two guards, and the outer one fires first.

    An unlabelled corpus is refused at index build, because treating an
    unlabelled chunk as public is the wrong way to fail. The sweep carries
    the same check anyway, since it is the thing that would otherwise report
    a confident pass over a corpus with no compartments in it at all.
    """
    from stratum.context import ContextError, build_index
    from stratum.simulate import _queries_for

    bare = tmp_path / "bare.jsonl"
    bare.write_text(json.dumps({"id": "a", "text": "hello",
                                "source": "a.txt"}) + "\n", encoding="utf-8")
    with pytest.raises(ContextError):
        build_index(str(bare), str(tmp_path / "index"), verbose=False)

    assert _queries_for([{"id": "a", "text": "hello"}], 3, 0) == {}


def test_a_corpus_the_index_was_not_built_from_is_refused(tmp_path):
    """Sweeping the wrong corpus would report on rows the index never held."""
    index, policy, chunks = build(tmp_path)
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({"id": "tech-0", "text": "different text",
                                 "source": "tech0.txt",
                                 "compartment": "engineering"}) + "\n",
                     encoding="utf-8")
    with pytest.raises(SimulationError, match="has changed since this index"):
        run(index, str(policy), str(other), verbose=False)


def test_restricted_material_is_searchable_by_the_people_cleared_for_it(tmp_path):
    """Restricted means out of the weights, not out of the index. A sweep
    that treated it as unreadable would report a pass for the wrong reason."""
    from stratum.context import build_index

    chunks = tmp_path / "chunks.jsonl"
    rows = [{"id": "p-0", "text": "the code of conduct applies to everyone",
             "source": "p.txt", "compartment": "public", "kind": "document"},
            {"id": "r-0", "text": "salary band four midpoint and review cycle",
             "source": "r.txt", "compartment": "payroll", "kind": "document"}]
    chunks.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                      encoding="utf-8")
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "compartments": [{"name": "public", "tier": "company"},
                         {"name": "payroll", "tier": "restricted",
                          "volatile": True}],
        "principals": [{"name": "hr_lead", "compartments": ["public", "payroll"]},
                       {"name": "everyone_else", "compartments": ["public"]}],
    }), encoding="utf-8")
    index = tmp_path / "index"
    build_index(str(chunks), str(index), verbose=False)

    report = run(str(index), str(policy), str(chunks), samples=1,
                 verbose=False)
    assert report["passed"]
    assert "payroll" in report["per_principal"]["hr_lead"]["visible"]
    assert "payroll" not in report["per_principal"]["everyone_else"]["visible"]
    assert report["per_principal"]["hr_lead"]["hits_from_permitted"] > 0
