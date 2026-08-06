"""Attacks on the corpus plan.

The plan decides which folder a compartment is written to and which people
may read it, so a mistake here is an access mistake rather than a cosmetic
one. Every test below is a thing somebody could put in a plan file, either
by accident or on purpose, and what has to happen instead of it working.
"""
import json

import pytest

from stratum.corpus_plan import (PlanError, build, check_policy, family_spec,
                                 known_suffixes, parse, policy_dict, write_example)


def write(tmp_path, body, name="plan.yaml"):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return str(p)


def material(tmp_path, *names):
    """A folder with a readable document in it, so a compartment has content."""
    d = tmp_path / "raw"
    d.mkdir(exist_ok=True)
    for n in names:
        sub = d / n
        sub.mkdir(exist_ok=True)
        (sub / "doc.txt").write_text(f"material for {n}", encoding="utf-8")
    return d


TWO_DEPTS = """
compartments:
  public:
    tier: company
    family: general
    folders: [raw/public]
  engineering:
    tier: department
    family: technical
    folders: [raw/engineering]
principals:
  a:
    reads: [public, engineering]
  b:
    reads: [public, engineering]
"""


def test_a_valid_plan_parses(tmp_path):
    material(tmp_path, "public", "engineering")
    plan = parse(write(tmp_path, TWO_DEPTS))
    assert set(plan["compartments"]) == {"public", "engineering"}
    assert plan["families"] == {"general": ["public"],
                                "technical": ["engineering"]}
    assert check_policy(plan) == []


def test_a_compartment_named_dotdot_cannot_escape_the_output_folder(tmp_path):
    """The dangerous name that carries no dangerous character.

    '..' passes any check written as a list of forbidden characters, and
    would make the build write a compartment into the parent of the folder
    it was told to use.
    """
    material(tmp_path, "public")
    body = """
compartments:
  "..":
    tier: company
    folders: [raw/public]
principals:
  a: [".."]
"""
    with pytest.raises(PlanError, match="not usable as a folder name"):
        parse(write(tmp_path, body))


@pytest.mark.parametrize("name", ["../etc", "a/b", "a\\b", ".hidden", "-flag",
                                  "has space", "a:b", "a*b"])
def test_names_that_are_not_safe_path_components_are_refused(tmp_path, name):
    material(tmp_path, "public")
    body = (f'compartments:\n  "{name}":\n    tier: company\n'
            f'    folders: [raw/public]\n')
    with pytest.raises(PlanError, match="not usable as a folder name"):
        parse(write(tmp_path, body))


def test_a_principal_cannot_be_granted_a_compartment_that_does_not_exist(tmp_path):
    """A typo in a grant must not silently create an empty permission."""
    material(tmp_path, "public")
    body = """
compartments:
  public:
    tier: company
    folders: [raw/public]
principals:
  a: [public, enginering]
"""
    with pytest.raises(PlanError, match="does not define"):
        parse(write(tmp_path, body))


def test_a_compartment_with_no_material_is_refused(tmp_path):
    body = """
compartments:
  ghost:
    tier: company
"""
    with pytest.raises(PlanError, match="no material"):
        parse(write(tmp_path, body))


def test_a_folder_that_does_not_exist_is_refused(tmp_path):
    """Caught before the download, not after it."""
    body = """
compartments:
  public:
    tier: company
    folders: [raw/typo]
"""
    with pytest.raises(PlanError, match="does not exist"):
        parse(write(tmp_path, body))


def test_something_that_is_not_a_web_address_is_refused(tmp_path):
    body = """
compartments:
  public:
    tier: company
    urls: ["raw/public/doc.txt"]
"""
    with pytest.raises(PlanError, match="not a web address"):
        parse(write(tmp_path, body))


def test_company_tier_that_not_everyone_can_read_is_rejected(tmp_path):
    """Company tier means shared weights, so it has to be visible to all."""
    material(tmp_path, "public", "secret")
    body = """
compartments:
  public:
    tier: company
    folders: [raw/public]
  secret:
    tier: company
    folders: [raw/secret]
principals:
  a:
    reads: [public, secret]
  b:
    reads: [public]
"""
    plan = parse(write(tmp_path, body))
    assert check_policy(plan), "a company tier compartment only one person " \
                              "can read must not be accepted"


def test_build_refuses_before_downloading_when_the_policy_is_rejected(tmp_path):
    """The order matters. Finding this out after an eight hour fetch is not
    finding it out usefully."""
    material(tmp_path, "public", "secret")
    body = """
compartments:
  public:
    tier: company
    folders: [raw/public]
  secret:
    tier: company
    urls: ["https://example.invalid/never-fetched"]
principals:
  a:
    reads: [public, secret]
  b:
    reads: [public]
"""
    out = tmp_path / "corpus"
    with pytest.raises(PlanError, match="access policy the rules do not allow"):
        build(write(tmp_path, body), str(out), verbose=False)
    assert not (out / "secret").exists(), "nothing should have been written"


def test_a_compartment_belongs_to_exactly_one_family_by_construction(tmp_path):
    """There is no way to write a plan that puts one compartment in two
    families, because the family is a field on the compartment."""
    material(tmp_path, "public", "engineering")
    plan = parse(write(tmp_path, TWO_DEPTS))
    seen = [c for members in plan["families"].values() for c in members]
    assert len(seen) == len(set(seen))


def test_family_defaults_to_the_compartments_own_name(tmp_path):
    """Left out, a compartment gets a family of its own, which costs an
    adapter but can never mix audiences."""
    material(tmp_path, "public")
    body = """
compartments:
  public:
    tier: company
    folders: [raw/public]
"""
    plan = parse(write(tmp_path, body))
    assert plan["families"] == {"public": ["public"]}


def test_build_copies_and_never_moves(tmp_path):
    """A wrong path in a plan must not be able to destroy the material."""
    raw = material(tmp_path, "public", "engineering")
    out = tmp_path / "corpus"
    build(write(tmp_path, TWO_DEPTS), str(out), fetch=False, verbose=False)
    assert (raw / "public" / "doc.txt").exists()
    assert (out / "public" / "doc.txt").exists()
    assert (out / "engineering" / "doc.txt").exists()


def test_build_writes_a_policy_the_access_rules_can_load(tmp_path):
    from stratum.access import Policy

    material(tmp_path, "public", "engineering")
    out = tmp_path / "corpus"
    result = build(write(tmp_path, TWO_DEPTS), str(out), fetch=False,
                   verbose=False)
    policy = Policy.load(result["policy"])
    assert policy.tier_of("public") == "company"
    assert policy.tier_of("engineering") == "department"
    assert policy.visible_compartments("a") == {"public", "engineering"}


def test_build_writes_a_family_spec_the_family_planner_can_read(tmp_path):
    from stratum.families import declared

    material(tmp_path, "public", "engineering")
    out = tmp_path / "corpus"
    result = build(write(tmp_path, TWO_DEPTS), str(out), fetch=False,
                   verbose=False)
    spec = json.loads((tmp_path / "families.json").read_text(encoding="utf-8"))
    # Fake centroids, because the grouping here is declared and the numbers
    # are only used to check the names line up.
    plan = declared(spec, {"public": [1.0, 0.0], "engineering": [0.0, 1.0]},
                    verbose=False)
    assert plan["of"]["engineering"] == "technical"
    assert result["families"].endswith("families.json")


def test_nested_folders_are_flattened_without_collisions(tmp_path):
    """Two files with the same name in different sub folders must both
    survive, or half the corpus disappears without a word."""
    raw = tmp_path / "raw" / "public"
    (raw / "one").mkdir(parents=True)
    (raw / "two").mkdir(parents=True)
    (raw / "one" / "report.txt").write_text("first", encoding="utf-8")
    (raw / "two" / "report.txt").write_text("second", encoding="utf-8")
    body = """
compartments:
  public:
    tier: company
    folders: [raw/public]
principals:
  a: [public]
  b: [public]
"""
    out = tmp_path / "corpus"
    build(write(tmp_path, body), str(out), fetch=False, verbose=False)
    got = sorted(p.name for p in (out / "public").iterdir())
    assert got == ["one__report.txt", "two__report.txt"]


def test_formats_ingest_cannot_read_are_left_out_not_pretended(tmp_path):
    raw = tmp_path / "raw" / "public"
    raw.mkdir(parents=True)
    (raw / "keep.txt").write_text("readable", encoding="utf-8")
    (raw / "drop.zip").write_bytes(b"PK\x03\x04")
    body = """
compartments:
  public:
    tier: company
    folders: [raw/public]
principals:
  a: [public]
  b: [public]
"""
    out = tmp_path / "corpus"
    r = build(write(tmp_path, body), str(out), fetch=False, verbose=False)
    assert r["report"]["public"]["documents"] == 1
    assert r["report"]["public"]["skipped_type"] == 1


def test_the_file_types_come_from_ingest_and_are_not_a_second_list():
    """A separate list would drift and documents would vanish quietly."""
    from stratum.corpus import DOC_TYPES, IMAGE_TYPES
    assert known_suffixes() == DOC_TYPES | IMAGE_TYPES


def test_a_single_string_is_accepted_where_a_list_is_expected(tmp_path):
    material(tmp_path, "public")
    body = """
compartments:
  public:
    tier: company
    folders: raw/public
"""
    plan = parse(write(tmp_path, body))
    assert len(plan["compartments"]["public"]["folders"]) == 1


def test_url_files_are_read_and_comments_ignored(tmp_path):
    material(tmp_path, "public")
    (tmp_path / "urls.txt").write_text(
        "# a comment\nhttps://example.invalid/a\n\nhttps://example.invalid/b\n",
        encoding="utf-8")
    body = """
compartments:
  public:
    tier: company
    url_files: [urls.txt]
"""
    plan = parse(write(tmp_path, body))
    assert plan["compartments"]["public"]["urls"] == [
        "https://example.invalid/a", "https://example.invalid/b"]


def test_the_example_plan_is_itself_valid(tmp_path):
    """The thing `plan init` hands people has to survive `plan check`, or
    the first command anybody runs after it fails."""
    p = tmp_path / "example.yaml"
    raw = tmp_path / "raw"
    for d in ("public", "engineering", "maintenance", "legal", "payroll"):
        (raw / d).mkdir(parents=True)
        (raw / d / "doc.txt").write_text("x", encoding="utf-8")
    write_example(str(p))
    plan = parse(str(p))
    assert check_policy(plan) == []
    assert plan["compartments"]["payroll"]["tier"] == "restricted"
    assert plan["compartments"]["payroll"]["volatile"] is True


def test_write_example_refuses_to_overwrite(tmp_path):
    p = tmp_path / "plan.yaml"
    p.write_text("mine", encoding="utf-8")
    with pytest.raises(PlanError, match="already exists"):
        write_example(str(p))
    assert p.read_text(encoding="utf-8") == "mine"


def test_the_shipped_enterprise_plan_is_valid():
    """The worked example in the repository has to stay correct."""
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "examples/enterprise/plan.yaml"
    if not p.exists():
        pytest.skip("enterprise example not present")
    plan = parse(str(p))
    assert check_policy(plan) == []
    assert len(plan["compartments"]) >= 16
    # Nobody may end up loading more adapters than merging can survive.
    comps = plan["compartments"]
    for name, pr in plan["principals"].items():
        fams = {comps[c]["family"] for c in pr["reads"]
                if comps[c]["tier"] != "restricted"}
        usable = [f for f in fams
                  if all(m in pr["reads"] for m in plan["families"][f])]
        assert len(usable) <= 3, f"{name} would load {len(usable)} adapters"


def test_policy_and_family_spec_describe_the_same_compartments(tmp_path):
    """The two outputs come from one source, so they cannot disagree."""
    material(tmp_path, "public", "engineering")
    plan = parse(write(tmp_path, TWO_DEPTS))
    in_policy = {c["name"] for c in policy_dict(plan)["compartments"]}
    in_families = {c for m in family_spec(plan)["families"].values() for c in m}
    assert in_policy == in_families
