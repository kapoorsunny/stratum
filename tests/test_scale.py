"""Tests for the pieces that let this work at forty departments.

Two of these are access tests wearing a different hat. Grouping compartments
into families and grounding training data both decide what ends up inside
shared weights, so a mistake there is a disclosure rather than a bad answer,
and they are written as attacks accordingly.
"""
import json

import pytest

from stratum.access import Compartment, Policy, Principal
from stratum.families import (FamilyError, adapters_per_person, audience_check,
                              cluster, cohesion, compare, declared,
                              example_spec, similarity)
from stratum.grounded import (GROUNDED_TEMPLATE, REFUSAL, GroundedError,
                              build_prompt, ground_pairs, is_refusal)


def centroid(**weights):
    """A normalised centroid, the shape the router produces."""
    norm = sum(v * v for v in weights.values()) ** 0.5
    return {k: v / norm for k, v in weights.items()}


@pytest.fixture()
def three_families():
    """Compartments that clearly fall into technical, money and people."""
    return {
        "engineering":  centroid(pump=0.9, valve=0.8, pressure=0.7, torque=0.5),
        "maintenance":  centroid(pump=0.8, valve=0.6, bearing=0.8, torque=0.6),
        "accounting":   centroid(ledger=0.9, accrual=0.8, invoice=0.7, audit=0.5),
        "treasury":     centroid(ledger=0.7, hedge=0.9, invoice=0.6, audit=0.5),
        "hr":           centroid(employee=0.9, leave=0.8, appraisal=0.7),
        "training":     centroid(employee=0.8, course=0.9, appraisal=0.6),
    }


# ------------------------------------------------------------ clustering
def test_departments_that_write_alike_land_together(three_families):
    plan = cluster(three_families, n_families=3, verbose=False)

    assert plan["n_families"] == 3
    of = plan["of"]
    assert of["engineering"] == of["maintenance"]
    assert of["accounting"] == of["treasury"]
    assert of["hr"] == of["training"]
    # And the three groups are genuinely apart from one another.
    assert of["engineering"] != of["accounting"] != of["hr"]


def test_a_family_is_never_bigger_than_asked(three_families):
    plan = cluster(three_families, n_families=1, max_per_family=2, verbose=False)
    assert all(len(m) <= 2 for m in plan["families"].values())


def test_cohesion_reports_a_positive_margin_for_a_real_family(three_families):
    plan = cluster(three_families, n_families=3, verbose=False)
    report = cohesion(three_families, plan["families"])

    for label, m in report.items():
        if m["members"] > 1:
            assert m["margin"] > 0, (
                f"family '{label}' members sit no closer to each other than "
                f"to outsiders, so it is not a family")


def test_similarity_is_symmetric_and_bounded(three_families):
    a, b = three_families["engineering"], three_families["accounting"]
    assert abs(similarity(a, b) - similarity(b, a)) < 1e-9
    assert 0.0 <= similarity(a, b) <= 1.0
    assert abs(similarity(a, a) - 1.0) < 1e-9


# ------------------------------------------------------ declared families
def test_a_company_can_declare_its_own_grouping(three_families):
    """The measurement is a proposal. A regulator or an owner may require a
    grouping the vocabulary would never produce, and that has to win."""
    spec = {"families": {"plant": ["engineering", "maintenance", "hr"],
                         "money": ["accounting", "treasury"],
                         "people": ["training"]}}
    plan = declared(spec, three_families, verbose=False)

    assert plan["of"]["hr"] == "plant", "the declared grouping must win"
    assert plan["source"] == "declared"


def test_a_compartment_in_two_families_is_refused(three_families):
    spec = {"families": {"a": ["engineering", "hr"], "b": ["hr"]}}
    with pytest.raises(FamilyError, match="in both"):
        declared(spec, three_families, verbose=False)


def test_a_compartment_the_router_never_saw_is_refused(three_families):
    """Usually a typo in a folder name, which would otherwise silently create
    a family with nothing in it."""
    spec = {"families": {"a": ["enginering"]}}
    with pytest.raises(FamilyError, match="never seen"):
        declared(spec, three_families, verbose=False)


def test_a_compartment_left_out_of_the_file_is_refused_by_default(three_families):
    spec = {"families": {"a": ["engineering", "maintenance"]}}
    with pytest.raises(FamilyError, match="not in"):
        declared(spec, three_families, verbose=False)

    plan = declared(spec, three_families, allow_unlisted=True, verbose=False)
    assert plan["of"]["hr"] == "hr", "an unlisted one gets a family of its own"


def test_compare_flags_a_declared_grouping_the_writing_disagrees_with(three_families):
    """Not a veto. A disagreement is often correct and always worth seeing."""
    spec = {"families": {"odd": ["engineering", "accounting"],
                         "rest": ["maintenance", "treasury", "hr", "training"]}}
    plan = declared(spec, three_families, verbose=False)
    report = compare(plan, three_families, verbose=False)

    assert report["disagree"] > 0
    assert report["total"] == 6


def test_the_starting_file_is_valid_input_to_the_thing_that_reads_it(three_families):
    spec = example_spec(three_families)
    plan = declared(spec, three_families, verbose=False)
    assert plan["n_families"] == len(three_families)


# -------------------------------------------------------------- audience
def audience_policy():
    return Policy(
        compartments=[Compartment("public", "company"),
                      Compartment("engineering", "department"),
                      Compartment("maintenance", "department")],
        principals=[Principal("everyone", ["public"]),
                    Principal("eng", ["public", "engineering", "maintenance"]),
                    Principal("tech", ["public", "engineering", "maintenance"])],
    )


def test_a_family_mixing_readerships_is_reported_not_silently_allowed():
    """The failure clustering can create on its own. Vocabulary similarity
    says nothing about who is allowed to read something."""
    policy = audience_policy()
    plan = {"families": {"tech": ["public", "engineering"]},
            "of": {"public": "tech", "engineering": "tech"}}

    report = audience_check(plan, policy)

    assert report["mixed"], "a family spanning two readerships must be flagged"
    flagged = report["mixed"][0]
    assert "everyone" in flagged["exposed"], (
        "the principal who sees only public would load engineering material")
    assert flagged["exposed"]["everyone"] == ["engineering"]


def test_a_family_whose_members_share_a_readership_is_safe():
    policy = audience_policy()
    plan = {"families": {"tech": ["engineering", "maintenance"]},
            "of": {"engineering": "tech", "maintenance": "tech"}}

    report = audience_check(plan, policy)
    assert report["mixed"] == []
    assert "tech" in report["safe"]


def test_nobody_is_counted_as_loading_an_adapter_they_may_not_have():
    """Counting one they are not entitled to would report a smaller adapter
    number by quietly assuming a leak."""
    policy = audience_policy()
    plan = {"families": {"tech": ["public", "engineering"]},
            "of": {"public": "tech", "engineering": "tech"}}

    per = adapters_per_person(plan, policy)

    # 'everyone' sees only public, so the mixed family is not theirs to load.
    # The count is zero rather than one. There is no separate company adapter
    # sitting outside the grouping to fall back on. Public was put in a family
    # with engineering, and a family is loaded whole or not at all, so this
    # person is served by the base model and retrieval alone. Reporting one
    # here would hide exactly the situation the grouping needs to be changed
    # to fix.
    assert per["everyone"]["families"] == []
    assert per["everyone"]["adapters"] == 0
    # 'eng' may read both members, so the family is legitimately theirs.
    assert per["eng"]["families"] == ["tech"]
    assert per["eng"]["adapters"] == 1


# -------------------------------------------------------------- grounding
def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture()
def corpus(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    write(chunks, [
        {"id": "e1", "text": "Pump P-4471 has a bearing clearance of 0.418 mm.",
         "source": "engineering/pumps.txt", "compartment": "engineering"},
        {"id": "e2", "text": "Compressor K-201 is rated for 84 bar discharge.",
         "source": "engineering/comp.txt", "compartment": "engineering"},
        {"id": "e3", "text": "Cooling tower CT-9 has a design approach of 4 K.",
         "source": "engineering/ct.txt", "compartment": "engineering"},
        {"id": "f1", "text": "The Vendor A contract is worth 4.2 million.",
         "source": "finance/contract.txt", "compartment": "finance"},
        {"id": "f2", "text": "Payment terms with Vendor A are net 45.",
         "source": "finance/terms.txt", "compartment": "finance"},
    ])
    pairs = tmp_path / "pairs.jsonl"
    write(pairs, [
        {"prompt": "What is the bearing clearance of pump P-4471?",
         "response": "0.418 mm", "source_chunk": "e1",
         "source": "engineering/pumps.txt"},
        {"prompt": "What pressure is K-201 rated for?",
         "response": "84 bar", "source_chunk": "e2",
         "source": "engineering/comp.txt"},
    ])
    return str(chunks), str(pairs)


def test_grounding_puts_the_source_into_the_prompt(corpus, tmp_path):
    chunks, pairs = corpus
    out = tmp_path / "grounded.jsonl"
    ground_pairs(pairs, chunks, str(out), distractors=1, abstain_share=0.0,
                 verbose=False)

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert rows
    for r in rows:
        assert "reference material" in r["prompt"].lower()
        assert "Question:" in r["prompt"]
        assert r["grounded"] is True


def test_distractors_never_come_from_another_compartment(corpus, tmp_path):
    """The one that matters. A distractor pulled from a compartment the
    reader cannot see would put forbidden text into training data for a model
    other people load, which is the exact failure the tiers prevent."""
    chunks, pairs = corpus
    out = tmp_path / "grounded.jsonl"
    ground_pairs(pairs, chunks, str(out), distractors=2, abstain_share=0.0,
                 verbose=False)

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        assert "4.2 million" not in r["prompt"], "finance text leaked in"
        assert "net 45" not in r["prompt"], "finance text leaked in"
        assert "finance/" not in r["prompt"], "a finance source was cited"


def test_abstention_rows_remove_the_answer_and_teach_the_refusal(corpus, tmp_path):
    chunks, pairs = corpus
    out = tmp_path / "grounded.jsonl"
    ground_pairs(pairs, chunks, str(out), distractors=2, abstain_share=1.0,
                 verbose=False)

    by_id = {c["id"]: c for c in
             (json.loads(l) for l in
              open(chunks, encoding="utf-8").read().splitlines() if l.strip())}

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert rows
    for r in rows:
        assert r["abstain"] is True
        assert r["response"] == REFUSAL
        # The invariant is that THIS row's own source is gone. Another
        # chunk's numbers may legitimately appear as a distractor, and often
        # should, because a distractor that is obviously irrelevant teaches
        # the model nothing.
        own = by_id[r["source_chunk"]]["text"]
        assert own not in r["prompt"], (
            "an abstention row still contains its own source, so the model "
            "would learn to decline while looking straight at the answer")


def test_grounding_is_identical_on_a_rerun(corpus, tmp_path):
    """A resumed job must build the same dataset it started, and two people
    running the same command must get the same model."""
    chunks, pairs = corpus
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    ground_pairs(pairs, chunks, str(a), distractors=2, seed=7, verbose=False)
    ground_pairs(pairs, chunks, str(b), distractors=2, seed=7, verbose=False)
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")


def test_a_different_seed_gives_different_distractors(corpus, tmp_path):
    chunks, pairs = corpus
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    ground_pairs(pairs, chunks, str(a), distractors=1, abstain_share=0.0,
                 seed=1, verbose=False)
    ground_pairs(pairs, chunks, str(b), distractors=1, abstain_share=0.0,
                 seed=2, verbose=False)
    assert a.read_text(encoding="utf-8") != b.read_text(encoding="utf-8")


def test_pairs_naming_no_real_chunk_are_refused(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    write(chunks, [{"id": "a", "text": "something", "source": "s"}])
    pairs = tmp_path / "pairs.jsonl"
    write(pairs, [{"prompt": "q", "response": "r", "source_chunk": "nope"}])

    with pytest.raises(GroundedError, match="name a chunk that exists"):
        ground_pairs(str(pairs), str(chunks), str(tmp_path / "o.jsonl"),
                     verbose=False)


def test_the_serve_time_prompt_matches_the_training_shape():
    """A model that learned one layout and is given another is back to having
    context out of distribution, which is the whole problem being fixed."""
    p = build_prompt("What is the clearance?",
                     [("engineering/pumps.txt", "Pump P-4471 clearance 0.418 mm")])
    assert p.startswith(GROUNDED_TEMPLATE.split("{")[0])
    assert "[engineering/pumps.txt]" in p
    assert p.rstrip().endswith("What is the clearance?")


def test_a_refusal_is_recognised_even_when_paraphrased():
    assert is_refusal(REFUSAL)
    assert is_refusal("The reference material provided does not contain "
                      "the answer to that question.")
    assert is_refusal("Unfortunately the material does not contain the answer.")
    assert not is_refusal("The clearance is 0.418 millimetres.")
