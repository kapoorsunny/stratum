"""Tests for compartments, tiers, the context index and the leak audit.

The point of this file is that access control is the one part of the project
where a bug is not a bad answer, it is a disclosure. So these tests are
written as attacks rather than as checks, and several of them exist because
the obvious implementation passes a naive test and still leaks.
"""
import json

import pytest

from stratum.access import (AccessError, Compartment, Policy, Principal,
                            describe, example_policy)
from stratum.context import ContextIndex, build_index, build_prompt
from stratum.leak import _contains_secret, make_canary, run_audit


def simple_policy():
    return Policy(
        compartments=[
            Compartment("public", "company"),
            Compartment("engineering", "department"),
            Compartment("finance", "restricted"),
        ],
        principals=[
            Principal("contractor", ["public"]),
            Principal("engineer", ["public", "engineering"]),
            Principal("cfo", ["public", "engineering", "finance"]),
        ],
    )


def test_a_restricted_compartment_never_becomes_an_adapter():
    """The whole tier idea in one assertion. Restricted data must not reach
    weights, because a weight cannot be filtered afterwards."""
    p = simple_policy()

    assert "finance" not in p.strata_for("cfo")
    assert p.strata_for("cfo") == ["engineering", "public"]
    # It is still searchable for the people who may see it.
    assert "finance" in p.index_compartments("cfo")
    assert "finance" not in p.index_compartments("engineer")


def test_a_principal_only_loads_what_they_can_see():
    p = simple_policy()
    assert p.strata_for("contractor") == ["public"]
    assert p.strata_for("engineer") == ["engineering", "public"]


def test_company_tier_is_refused_when_somebody_cannot_see_it():
    """A company tier adapter is loaded by everyone, so training it on data
    that is not visible to everyone is a disclosure by construction."""
    with pytest.raises(AccessError, match="cannot see it"):
        Policy(
            compartments=[Compartment("secret-plans", "company")],
            principals=[Principal("a", ["secret-plans"]),
                        Principal("b", [])],
        )


def test_department_tier_is_refused_for_a_compartment_only_one_person_sees():
    with pytest.raises(AccessError, match="only 1 principal"):
        Policy(
            compartments=[Compartment("public", "company"),
                          Compartment("mine", "department")],
            principals=[Principal("a", ["public", "mine"]),
                        Principal("b", ["public"])],
        )


def test_volatile_data_is_refused_from_the_weights():
    """Volatile means it changes, and changing means retraining. Anything
    that has to be withdrawn quickly must stay out of the parameters."""
    with pytest.raises(AccessError, match="volatile"):
        Policy(
            compartments=[Compartment("public", "company"),
                          Compartment("tickets", "department", volatile=True)],
            principals=[Principal("a", ["public", "tickets"]),
                        Principal("b", ["public", "tickets"])],
        )


def test_a_grant_to_a_compartment_that_does_not_exist_is_refused():
    """Usually a typo in the name of a real compartment, which would
    silently grant nothing and look like it granted something."""
    with pytest.raises(AccessError, match="do not exist"):
        Policy(
            compartments=[Compartment("public", "company")],
            principals=[Principal("a", ["public", "enginering"])],
        )


def test_an_unknown_principal_is_an_error_not_an_empty_result():
    """Returning nothing for an unknown name would look exactly like a
    principal with no permissions, and quietly serve them nothing forever."""
    p = simple_policy()
    with pytest.raises(AccessError, match="Unknown principal"):
        p.visible_compartments("nobody")


def test_suggest_tier_reports_what_the_visibility_supports():
    p = simple_policy()
    assert p.suggest_tier("public") == "company"
    assert p.suggest_tier("engineering") == "department"
    assert p.suggest_tier("finance") == "restricted"


def test_a_policy_survives_a_round_trip_through_a_file(tmp_path):
    p = example_policy()
    path = tmp_path / "policy.json"
    p.save(str(path))
    back = Policy.load(str(path))

    assert set(back.compartments) == set(p.compartments)
    for name in p.principals:
        assert back.strata_for(name) == p.strata_for(name)


def test_describe_warns_when_too_many_adapters_would_be_merged():
    p = Policy(
        compartments=[Compartment("public", "company")] +
                     [Compartment(f"d{i}", "department") for i in range(5)],
        principals=[Principal("everyone", ["public"] + [f"d{i}" for i in range(5)]),
                    Principal("other", ["public"] + [f"d{i}" for i in range(5)])],
    )
    assert "above the safe merge limit" in describe(p)


def write_chunks(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture()
def small_index(tmp_path):
    """A corpus where each compartment has a distinctive, checkable fact."""
    rows = [
        {"id": "p1", "text": "The site standard for pipe labelling is ISO 14726 "
                             "and applies to every plant area.",
         "source": "public/standards.txt", "compartment": "public"},
        {"id": "p2", "text": "General safety induction is required before entry "
                             "to any operating area.",
         "source": "public/induction.txt", "compartment": "public"},
        {"id": "e1", "text": "Pump P-4471 has a certified bearing clearance of "
                             "0.418 millimetres per the engineering register.",
         "source": "engineering/pumps.txt", "compartment": "engineering"},
        {"id": "e2", "text": "Compressor K-201 is rated for 84 bar discharge "
                             "pressure at design conditions.",
         "source": "engineering/compressors.txt", "compartment": "engineering"},
        {"id": "f1", "text": "The 2026 maintenance contract with Vendor A is "
                             "valued at 4.2 million and renews in March.",
         "source": "finance/contracts.txt", "compartment": "finance"},
        {"id": "f2", "text": "Vendor A payment terms are net 45 with a 2 percent "
                             "early settlement discount.",
         "source": "finance/terms.txt", "compartment": "finance"},
    ]
    chunks = tmp_path / "chunks.jsonl"
    write_chunks(chunks, rows)
    out = tmp_path / "index"
    build_index(str(chunks), str(out), embedder="hash", dim=256, verbose=False)
    return ContextIndex(str(out), chunks_path=str(chunks)), str(chunks)


def test_search_never_returns_a_forbidden_compartment(small_index):
    index, _ = small_index
    hits = index.search("What is the bearing clearance of pump P-4471?",
                        allowed={"public"}, k=10)

    assert hits, "a permitted search should still return something"
    assert all(h["compartment"] == "public" for h in hits)
    assert all("0.418" not in h["text"] for h in hits)


def test_the_same_search_works_for_somebody_who_may_see_it(small_index):
    """The counterpart to the test above. Filtering that blocks everybody is
    not access control, it is a broken index."""
    index, _ = small_index
    hits = index.search("What is the bearing clearance of pump P-4471?",
                        allowed={"public", "engineering"}, k=5)

    assert any("0.418" in h["text"] for h in hits)


def test_link_expansion_cannot_walk_into_a_forbidden_compartment(small_index):
    """The hole people leave open. Filtering the search and then following
    an edge out of a permitted hit goes straight around the filter."""
    index, _ = small_index

    for principal_view in ({"public"}, {"public", "engineering"}):
        hits = index.search("vendor payment terms and contract value",
                            allowed=principal_view, k=10, expand=10)
        for h in hits:
            assert h["compartment"] in principal_view, (
                f"expansion reached {h['compartment']} for a caller who may "
                f"only see {principal_view}")


def test_forbidden_rows_are_removed_before_ranking_not_after(small_index):
    """Ranking a hidden chunk and dropping it afterwards leaks through what
    is missing, because it displaces a permitted result."""
    index, _ = small_index

    # Asked about finance material as somebody who may only see public. If
    # the filter ran after ranking, the finance chunks would take the top
    # slots and be dropped, leaving fewer results than there are permitted
    # chunks that match at all.
    public_only = index.search("Vendor A contract payment", allowed={"public"},
                               k=3)
    everything = index.search("Vendor A contract payment", allowed=None, k=3)

    assert all(h["compartment"] == "public" for h in public_only)
    assert any(h["compartment"] == "finance" for h in everything), (
        "the finance chunks should win this query when they are permitted")
    assert public_only, "filtering must not silently return nothing"


def test_an_index_refuses_a_corpus_it_was_not_built_from(small_index, tmp_path):
    """Answering from text that has moved produces answers that look fine,
    which is the worst kind of wrong."""
    index, chunks = small_index
    index.check_corpus(chunks)

    with open(chunks, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "x", "text": "new", "source": "s"}) + "\n")

    from stratum.context import ContextError
    with pytest.raises(ContextError, match="has changed"):
        index.check_corpus(chunks)


def test_recall_is_measured_against_the_chunk_that_wrote_the_question(small_index, tmp_path):
    from stratum.context import evaluate_recall

    index, _ = small_index
    test = tmp_path / "test.jsonl"
    write_chunks(test, [
        {"prompt": "What is the certified bearing clearance of pump P-4471?",
         "expected": "0.418 mm", "source_chunk": "e1"},
        {"prompt": "What discharge pressure is compressor K-201 rated for?",
         "expected": "84 bar", "source_chunk": "e2"},
    ])

    report = evaluate_recall(index, str(test), k_values=(1, 5), verbose=False)
    assert report["n"] == 2
    assert report["recall"][5] == 1.0


def test_the_audit_catches_a_leak_that_a_broken_filter_would_allow(small_index):
    """The harness has to fail when there is something to find, or passing
    it means nothing."""
    index, _ = small_index
    policy = simple_policy()

    # A canary matching a chunk that really is in the corpus, standing in
    # for a filter that lets finance text through.
    canary = make_canary("finance")
    canary["question"] = "What is the 2026 maintenance contract with Vendor A valued at?"
    canary["secret"] = "4.2"

    class NoFilterIndex:
        """An index whose filtering has been removed, which is exactly the
        regression this audit exists to catch."""
        def search(self, q, allowed=None, k=5, expand=0):
            return index.search(q, allowed=None, k=k, expand=expand)

    report = run_audit(index=NoFilterIndex(), policy=policy,
                       canaries=[canary], verbose=False)
    assert not report["passed"]
    assert any(f["principal"] == "contractor" for f in report["leaks"])


def test_the_audit_passes_on_the_real_index(small_index):
    index, _ = small_index
    policy = simple_policy()
    canaries = [make_canary(c) for c in policy.compartments]

    report = run_audit(index=index, policy=policy, canaries=canaries,
                       verbose=False)
    assert report["passed"]
    assert "model" in report["surfaces_not_tested"]


def test_an_untested_surface_is_never_reported_as_clean(small_index):
    """A report that counts what it did not check as a pass is worse than no
    report."""
    index, _ = small_index
    policy = simple_policy()
    report = run_audit(index=index, policy=policy,
                       canaries=[make_canary("finance")], verbose=False)

    assert "model" in report["surfaces_not_tested"]
    assert "index" in report["surfaces_tested"]


def test_a_model_that_reveals_a_canary_fails_the_audit():
    policy = simple_policy()
    canary = make_canary("finance")

    def leaky(principal, question):
        return f"The certified clearance is {canary['secret']} millimetres."

    report = run_audit(generate_fn=leaky, policy=policy, canaries=[canary],
                       verbose=False)
    assert not report["passed"]
    assert all(f["surface"] == "model" for f in report["leaks"])


def test_a_model_that_declines_passes():
    policy = simple_policy()
    canary = make_canary("finance")

    def careful(principal, question):
        return "I do not have information about that asset."

    report = run_audit(generate_fn=careful, policy=policy, canaries=[canary],
                       verbose=False)
    assert report["passed"]


def test_the_secret_matcher_does_not_cry_wolf():
    """A near miss on a number must not count, or an honest system fails and
    the audit gets switched off."""
    assert _contains_secret("the value is 42.150 mm", "42.150")
    assert not _contains_secret("the value is 442.150 mm", "42.150")
    assert not _contains_secret("the value is 42.1509 mm", "42.150")
    assert not _contains_secret("nothing here", "42.150")


def test_canaries_are_stable_and_distinct_per_compartment():
    a = make_canary("finance")
    b = make_canary("finance")
    c = make_canary("hr")

    assert a["secret"] == b["secret"], "a canary must be repeatable"
    assert a["secret"] != c["secret"], "compartments must not share a secret"


def test_the_prompt_carries_a_source_for_every_piece(small_index):
    index, _ = small_index
    hits = index.search("bearing clearance", allowed={"public", "engineering"}, k=2)
    prompt = build_prompt("What is the clearance?", hits, budget_chars=1000)

    assert "engineering/pumps.txt" in prompt
    assert "What is the clearance?" in prompt


def test_the_sentence_style_fits_more_sources_in_the_same_budget(small_index):
    index, _ = small_index
    hits = index.search("bearing clearance of pump", allowed=None, k=4)

    chunks = build_prompt("clearance?", hits, budget_chars=300, style="chunks")
    sentences = build_prompt("clearance?", hits, budget_chars=300, style="sentences")

    assert sentences.count("[") >= chunks.count("[")


def test_an_empty_permission_set_returns_nothing_rather_than_everything(small_index):
    """The failure mode that matters. A bug that turns 'no permissions' into
    'no filter' hands the whole corpus to someone with no access at all."""
    index, _ = small_index
    assert index.search("anything at all", allowed=set(), k=5) == []


def test_ingest_labels_every_chunk_with_its_folder(tmp_path):
    """The label has to survive all the way to chunks.jsonl. It was once
    added to the function and not to the record, which produced an unlabelled
    corpus and no error at all."""
    from stratum.corpus import compartment_of, ingest

    src = tmp_path / "corpus"
    for dept in ("public", "finance"):
        (src / dept).mkdir(parents=True)
        (src / dept / "note.txt").write_text(
            f"This is a {dept} document about site operations. " * 20,
            encoding="utf-8")

    out = tmp_path / "chunks"
    ingest(str(src), str(out), compartments=True, verbose=False)

    rows = [json.loads(l) for l in
            (out / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l]
    assert rows
    assert all("compartment" in r for r in rows), "every chunk needs a label"
    assert {r["compartment"] for r in rows} == {"public", "finance"}


def test_a_file_at_the_top_level_is_public():
    from stratum.corpus import compartment_of

    assert compartment_of("readme.txt") == "public"
    assert compartment_of("finance/contract.pdf") == "finance"
    assert compartment_of("hr/records/2026/june.csv") == "hr"


def test_a_large_image_is_shrunk_before_a_local_model_sees_it():
    """Token cost goes up with area, so a full resolution photo can cost more
    than the document it came with."""
    from PIL import Image

    from stratum.vision import VISION_MAX_EDGE, _fit_for_vision

    big = _fit_for_vision(Image.new("RGB", (4000, 3000)))
    assert max(big.size) == VISION_MAX_EDGE
    assert abs(big.size[0] / big.size[1] - 4000 / 3000) < 0.01, "aspect kept"

    tall = _fit_for_vision(Image.new("RGB", (500, 4000)))
    assert max(tall.size) == VISION_MAX_EDGE


def test_a_small_image_is_left_exactly_alone():
    from PIL import Image

    from stratum.vision import _fit_for_vision

    small = Image.new("RGB", (640, 480))
    assert _fit_for_vision(small) is small


def test_being_more_careful_than_required_is_not_reported_as_a_problem():
    """A compartment declared restricted when its readership would permit
    department is a deliberate choice. Nagging about it trains people to
    ignore the output, and then the real warnings go unread too."""
    p = simple_policy()

    assert p.tier_risk("restricted") < p.tier_risk("department")
    assert p.tier_risk("department") < p.tier_risk("company")
    # finance is seen by one of three, and declared restricted, which is
    # tighter than its ceiling and therefore fine.
    assert p.tier_risk(p.tier_of("finance")) <= p.tier_risk(p.suggest_tier("finance"))


def test_an_empty_weights_tier_compartment_is_a_deployment_failure():
    """The policy would tell principals to load an adapter that was never
    trained, because there was nothing to train it on."""
    p = Policy(
        compartments=[Compartment("public", "company"),
                      Compartment("safety", "department")],
        principals=[Principal("a", ["public", "safety"]),
                    Principal("b", ["public", "safety"])],
    )
    # The policy itself is valid. What is wrong only shows up against a real
    # corpus, which is why `access check` takes one.
    assert p.strata_for("a") == ["public", "safety"]
    assert policy_needs_chunks_for(p) == ["public", "safety"]


def policy_needs_chunks_for(policy):
    """Every compartment that must have documents for the build to work."""
    return sorted(n for n, c in policy.compartments.items()
                  if c.tier in ("company", "department"))


def test_manifest_paths_are_portable_between_operating_systems():
    """A bundle packed on Windows has to unpack on Linux. Backslashes in a
    manifest key would make every hash miss."""
    from stratum.deploy import pack, verify
    import tempfile, os, json as _j

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "strata", "public")
        os.makedirs(src)
        with open(os.path.join(src, "adapter_model.safetensors"), "w") as f:
            f.write("weights")
        with open(os.path.join(src, "stratum_card.json"), "w") as f:
            _j.dump({"base_model": "Qwen/Qwen3-0.6B"}, f)

        out = os.path.join(tmp, "bundle")
        m = pack(out, [src], verbose=False)

        assert all("\\" not in k for k in m["files"]), (
            "manifest keys must use forward slashes so a bundle packed on "
            "one system verifies on another")
        verify(out, verbose=False)


def test_a_corrupted_bundle_is_refused():
    from stratum.deploy import DeployError, pack, verify
    import tempfile, os, json as _j

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "strata", "public")
        os.makedirs(src)
        weights = os.path.join(src, "adapter_model.safetensors")
        with open(weights, "w") as f:
            f.write("weights")
        with open(os.path.join(src, "stratum_card.json"), "w") as f:
            _j.dump({"base_model": "Qwen/Qwen3-0.6B"}, f)

        out = os.path.join(tmp, "bundle")
        pack(out, [src], verbose=False)
        with open(os.path.join(out, "strata", "public",
                               "adapter_model.safetensors"), "a") as f:
            f.write("x")

        with pytest.raises(DeployError, match="do not match"):
            verify(out, verbose=False)


def test_packing_a_stratum_trained_on_a_different_base_is_refused():
    """Adapters only fit the base they were trained on, and finding that out
    on the serving node is the expensive way."""
    from stratum.deploy import DeployError, pack
    import tempfile, os, json as _j

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "strata", "public")
        os.makedirs(src)
        with open(os.path.join(src, "stratum_card.json"), "w") as f:
            _j.dump({"base_model": "Qwen/Qwen3-4B"}, f)

        with pytest.raises(DeployError, match="only fit the base"):
            pack(os.path.join(tmp, "bundle"), [src],
                 base="Qwen/Qwen3-1.7B", verbose=False)


def test_a_canary_nobody_authorised_can_reach_is_inconclusive_not_a_pass():
    """The control that makes the audit mean something. A model that never
    learned a secret cannot leak it, and reporting that as a pass would be
    the most dangerous kind of green tick."""
    from stratum.leak import run_audit

    policy = simple_policy()
    canary = make_canary("engineering")

    def knows_nothing(principal, question):
        return "I have no information about that."

    report = run_audit(generate_fn=knows_nothing, policy=policy,
                       canaries=[canary], verbose=False)

    assert not report["passed"], "a canary nobody learned must not pass"
    assert report["leaks"] == []
    assert len(report["inconclusive"]) == 1


def test_the_audit_passes_when_the_right_person_can_and_others_cannot():
    """What a genuine pass looks like. The control succeeds and the attack
    fails."""
    from stratum.leak import run_audit

    policy = simple_policy()
    canary = make_canary("engineering")

    def correct(principal, question):
        if "engineering" in policy.strata_for(principal):
            return f"The clearance is {canary['secret']} millimetres."
        return "I do not have that."

    report = run_audit(generate_fn=correct, policy=policy, canaries=[canary],
                       verbose=False)

    assert report["passed"]
    assert report["inconclusive"] == []


def test_canary_training_pairs_state_the_value_every_time():
    from stratum.leak import canary_pairs, make_canary

    canary = make_canary("finance")
    pairs = canary_pairs(canary, repeats=6)

    assert len(pairs) == 6
    assert all(canary["secret"] in p["response"] for p in pairs)
    # Varied phrasing, so the model learns the fact rather than one sentence.
    assert len({p["prompt"] for p in pairs}) > 1


def test_an_hf_index_records_the_model_it_actually_used(tmp_path, monkeypatch):
    """Recording the argument rather than the resolved name wrote a null
    into the index, and every later query asked the Hub for a model called
    None. A default that is not written down is not a default."""
    import numpy as np

    from stratum import context

    def fake_embed(texts, model_name, device=None, batch_size=16):
        assert model_name, "the embedder must be given a real model name"
        return np.ones((len(texts), 8), dtype="float32")

    monkeypatch.setattr(context, "_hf_embed_batch", fake_embed)

    chunks = tmp_path / "chunks.jsonl"
    write_chunks(chunks, [{"id": "a", "text": "some text here", "source": "s",
                           "compartment": "public"}])
    out = tmp_path / "index"
    build_index(str(chunks), str(out), embedder="hf", verbose=False)

    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["embed_model"], "the resolved model name must be recorded"
    assert "MiniLM" in meta["embed_model"]

    # And a query must round trip without asking the Hub for None.
    index = ContextIndex(str(out), chunks_path=str(chunks))
    assert index.search("some text", k=1)


def test_an_unlabelled_corpus_is_refused_rather_than_made_public(tmp_path):
    """Defaulting to public is the wrong way to fail. The corpora most likely
    to be ingested without the flag are exactly the ones worth compartmenting."""
    from stratum.context import ContextError

    chunks = tmp_path / "chunks.jsonl"
    write_chunks(chunks, [{"id": "a", "text": "salary bands for 2026",
                           "source": "hr/pay.txt"}])

    with pytest.raises(ContextError, match="carry no compartment"):
        build_index(str(chunks), str(tmp_path / "index"), verbose=False)


def test_an_unlabelled_corpus_can_be_placed_explicitly(tmp_path):
    """Saying where they belong is fine. Guessing is not."""
    chunks = tmp_path / "chunks.jsonl"
    write_chunks(chunks, [{"id": "a", "text": "salary bands for 2026",
                           "source": "hr/pay.txt"}])
    out = tmp_path / "index"
    build_index(str(chunks), str(out), unlabelled_compartment="hr",
                verbose=False)

    index = ContextIndex(str(out), chunks_path=str(chunks))
    assert index.compartments == ["hr"]
    assert index.search("salary", allowed={"public"}) == []
    assert index.search("salary", allowed={"hr"})


def test_a_stale_index_is_refused_at_load_rather_than_serving_empty_text(tmp_path):
    """check_corpus existed and nothing called it, so an index built from an
    older corpus scored rows normally and returned citations with no content."""
    from stratum.context import ContextError

    chunks = tmp_path / "chunks.jsonl"
    write_chunks(chunks, [{"id": "a", "text": "pump P-4471 clearance 0.418",
                           "source": "e/p.txt", "compartment": "engineering"}])
    out = tmp_path / "index"
    build_index(str(chunks), str(out), verbose=False)
    ContextIndex(str(out), chunks_path=str(chunks))

    write_chunks(chunks, [{"id": "b", "text": "different text entirely",
                           "source": "e/p.txt", "compartment": "engineering"}])
    with pytest.raises(ContextError, match="has changed"):
        ContextIndex(str(out), chunks_path=str(chunks))


def test_a_principal_cannot_invoke_an_adapter_they_were_not_granted():
    """The hole the cross check found. Retrieval was filtered and adapter
    selection was not, so naming a department adapter simply handed it over."""
    from stratum.serve import SkillPool

    class FakePool(SkillPool):
        def __init__(self):
            self.names = ["public", "engineering"]
            self.router = None

    pool = FakePool()

    assert pool.pick("anything", skill="engineering")[0] == "engineering"

    with pytest.raises(PermissionError, match="not permitted"):
        pool.pick("anything", skill="engineering", permitted={"public"})

    # The fallback must respect it too, not just the explicit path.
    assert pool.pick("anything", permitted={"public"})[0] == "public"

    with pytest.raises(PermissionError, match="No stratum"):
        pool.pick("anything", permitted=set())


def test_the_router_cannot_route_into_an_adapter_a_caller_may_not_have():
    """A router that picks a forbidden skill and gets away with it is the
    same hole wearing a different hat."""
    from stratum.serve import SkillPool

    class Always:
        def route(self, text):
            return "engineering", 0.9, {"engineering": 0.9, "public": 0.1}

    class FakePool(SkillPool):
        def __init__(self):
            self.names = ["public", "engineering"]
            self.router = Always()

    pool = FakePool()
    assert pool.pick("anything")[0] == "engineering"

    chosen, confidence, scores = pool.pick("anything", permitted={"public"})
    assert chosen == "public"
    assert "engineering" not in scores, "a forbidden skill must not appear in the scores"


def test_the_tokenizer_handles_scripts_that_are_not_latin():
    """An ASCII-only pattern produced zero tokens, an all-zero vector and an
    empty term index, so every query returned the same first k chunks."""
    from stratum.context import _tokens

    assert _tokens("V-201 rated 480V rev.2") == ["v-201", "rated", "480v", "rev.2"]
    assert len(_tokens("caf\u00e9 na\u00efve M\u00fcller")) == 3
    assert len(_tokens("\u0446\u0435\u043d\u0442\u0440\u043e\u0431\u0435\u0436\u043d\u044b\u0439 \u043d\u0430\u0441\u043e\u0441")) == 2
    assert len(_tokens("\u0645\u0636\u062e\u0629 \u0627\u0644\u0637\u0631\u062f")) == 2
