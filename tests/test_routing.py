"""Tests for skill routing, serving, and the teacher advisor.

Routing is the alternative to merging: keep strata separate and pick one per
request. These cover the router's accuracy, the honesty of its confidence
score, the pool that hot-swaps adapters on one loaded base, and the advisor
that decides which teacher a machine can run.
"""
import json
from pathlib import Path

import pytest

from stratum.advisor import CATALOGUE, QUANTS, advise, estimate
from stratum.router import SkillRouter, evaluate_router, train_router

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture()
def two_skill_router(tmp_path):
    """A router over two genuinely different skills."""
    files = {"extract": str(EXAMPLES / "extract.jsonl"),
             "classify": str(EXAMPLES / "classify.jsonl")}
    router = train_router(files, str(tmp_path / "router.json"), verbose=False)
    return router, files


def test_router_separates_distinct_skills(two_skill_router):
    router, files = two_skill_router
    report = evaluate_router(router, files, verbose=False)
    assert report["accuracy"] > 0.9
    assert set(report["per_skill"]) == {"extract", "classify"}


def test_router_generalizes_to_unseen_requests(two_skill_router):
    """The point of routing is requests nobody trained on."""
    router, _ = two_skill_router
    unseen = [
        ("Extract the total from this invoice: 'Amount due 412 EUR'", "extract"),
        ("Invoice shows a grand total of 1,200 GBP", "extract"),
        ("Classify this support ticket: the app crashes on startup", "classify"),
        ("Classify this ticket: I was billed twice this month", "classify"),
    ]
    for text, expected in unseen:
        assert router.route(text)[0] == expected, text


def test_confidence_is_honest_about_ambiguity(two_skill_router):
    """A request that fits one skill clearly should score higher than one
    that fits both - otherwise the number is decoration."""
    router, _ = two_skill_router
    clear = router.route("Extract the total from this invoice: 'Total: $88'")[1]
    vague = router.route("hello there")[1]
    assert clear > vague


def test_router_round_trips(tmp_path, two_skill_router):
    router, _ = two_skill_router
    path = str(tmp_path / "saved.json")
    router.save(path)
    reloaded = SkillRouter.load(path)
    q = "Extract the total from this invoice: 'Total: $99'"
    assert reloaded.route(q)[0] == router.route(q)[0]
    assert reloaded.skills == router.skills


def test_router_needs_two_skills(tmp_path):
    with pytest.raises(ValueError, match="two skills"):
        train_router({"only": str(EXAMPLES / "extract.jsonl")},
                     str(tmp_path / "r.json"), verbose=False)


def test_missing_router_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="route train"):
        SkillRouter.load(str(tmp_path / "nope.json"))


def test_router_recovers_skill_files_from_cards(two_strata):
    """Routing should be buildable from a folder of strata alone - the cards
    already record which data trained each one."""
    from stratum.router import strata_skill_files
    files = strata_skill_files(two_strata)
    assert len(files) == 2
    assert all(Path(p).exists() for p in files.values())


def test_advisor_ranks_by_what_actually_runs():
    """A big model that crawls must rank below a smaller one that works."""
    hw = {"ram_gb": 32.0, "vram_gb": 8.0, "free_disk_gb": 500.0,
          "gpu": "test", "ram_gbs": 40.0}
    rows = advise(hw, min_tok_s=3.0)
    assert rows[0]["usable"]
    # DeepSeek-V3 is the largest in the catalogue and cannot run usefully on
    # 32 GB - it must not be the recommendation.
    assert rows[0]["name"] != "DeepSeek-V3"
    deepseek = next(r for r in rows if r["name"] == "DeepSeek-V3")
    assert not deepseek["usable"]


def test_advisor_prefers_sparse_models_for_speed():
    """The whole point: active parameters set speed, not total ones."""
    hw = {"ram_gb": 64.0, "vram_gb": 0.0, "free_disk_gb": 500.0,
          "gpu": None, "ram_gbs": 40.0}
    bits = QUANTS["Q4_K_M"]
    sparse = estimate(80.0, 3.0, bits, hw)     # 80B with 3B active
    dense = estimate(32.0, 32.0, bits, hw)     # 32B dense
    assert sparse["tok_s"] > dense["tok_s"] * 3
    assert sparse["disk_gb"] > dense["disk_gb"]  # bigger on disk, still faster


def test_advisor_flags_models_that_do_not_fit_disk():
    hw = {"ram_gb": 32.0, "vram_gb": 0.0, "free_disk_gb": 20.0,
          "gpu": None, "ram_gbs": 40.0}
    rows = advise(hw)
    big = next(r for r in rows if r["name"] == "DeepSeek-V3")
    assert not big["fits_disk"] and not big["usable"]


def test_catalogue_entries_are_well_formed():
    for name, hf_id, total, active, lic in CATALOGUE:
        assert active <= total, name
        assert "/" in hf_id, name
        assert lic


def test_local_server_teacher_fails_clearly_when_nothing_is_running():
    from stratum.teachers import get_teacher
    with pytest.raises(EnvironmentError, match="stratum teachers"):
        get_teacher("llama-cpp", url="http://127.0.0.1:9")


def test_pool_refuses_strata_from_different_bases(two_strata, tmp_path):
    """One loaded base carries every adapter, so mixed bases cannot work."""
    import shutil
    from stratum.serve import SkillPool

    other = tmp_path / "foreign-stratum"
    shutil.copytree(two_strata[0], other)
    card = json.loads((other / "stratum_card.json").read_text(encoding="utf-8"))
    card["base_model"] = "somewhere/else"
    (other / "stratum_card.json").write_text(json.dumps(card), encoding="utf-8")

    with pytest.raises(ValueError, match="different bases"):
        SkillPool([two_strata[0], str(other)], verbose=False)


def test_peft_dispatch_probe_passes_on_a_working_install():
    """Issue #2: peft's optional backends can be at versions it rejects, and
    the failure otherwise lands eleven frames deep AFTER the base model has
    downloaded. The probe runs first and costs milliseconds."""
    from stratum.hf_utils import check_peft_dispatch
    ok, message = check_peft_dispatch()
    assert ok, message


def test_peft_dispatch_advice_names_the_package():
    from stratum.hf_utils import peft_dispatch_advice
    torchao = peft_dispatch_advice(
        "Found an incompatible version of torchao. Found version 0.10.0, "
        "but only versions above 0.16.0 are supported")
    assert "pip uninstall -y torchao" in torchao
    assert "Colab" in torchao
    # A backend nobody has seen yet must still produce a usable instruction.
    other = peft_dispatch_advice(
        "Found an incompatible version of eetq. Found version 1.0.0, "
        "but only versions above 2.0.0 are supported")
    assert "eetq" in other and "pip uninstall" in other


def test_healthy_run_records_its_best_epoch(tiny_base, tmp_path):
    from stratum.train import train_tile

    out = tmp_path / "healthy"
    train_tile(skill_path=str(EXAMPLES / "extract.jsonl"), out_dir=str(out),
               base_model=tiny_base, rank=2, epochs=2, batch_size=2,
               max_len=96, load_4bit=False, seed=3)
    card = json.loads((out / "stratum_card.json").read_text(encoding="utf-8"))
    assert card["diverged"] is False
    assert 1 <= card["best_epoch"] <= 2


def test_divergence_verdict_matches_the_reported_failure():
    """Issue #3, replayed with the exact losses from the bug report.

    Run 1 was 1.6340 -> 1.8839 -> 4.9027: the run completed, reported
    success, saved a stratum, and scored 0.0%. The rule must call epochs 2
    and 3 both non-improving and diverging, so the weights kept are epoch
    1's and the card says what happened.
    """
    from stratum.train import epoch_verdict

    first = 1.6340
    e2 = epoch_verdict(1.8839, first, first)
    e3 = epoch_verdict(4.9027, first, first)
    assert not e2["improved"] and e2["diverging"]
    assert not e3["improved"] and e3["diverging"]

    # The fixed run from the same report, at lr 5e-3: 2.3063 -> 1.3667 -> 0.8416
    healthy = epoch_verdict(1.3667, 2.3063, 2.3063)
    assert healthy["improved"] and not healthy["diverging"]
    assert epoch_verdict(0.8416, 1.3667, 2.3063)["improved"]


def test_muon_lr_default_scales_with_base_size():
    """Muon's step size is set by the learning rate and matrix shape, not by
    how large the weights are - so the same rate is a bigger relative push on
    a small model. Both reported divergences were on small bases."""
    from stratum.train import default_muon_lr

    assert default_muon_lr("Qwen/Qwen3-0.6B") == 5e-3
    assert default_muon_lr("Qwen/Qwen3-1.7B") == 5e-3
    assert default_muon_lr("Qwen/Qwen3-4B") == 1e-2
    assert default_muon_lr("Qwen/Qwen3-8B") == 2e-2
    # Bigger bases keep the rate that has been fine for them.
    assert default_muon_lr("Qwen/Qwen3-32B") == 2e-2
    # A local folder gives no size to read, so take the middle road.
    assert default_muon_lr("/some/local/checkpoint") == 1e-2
    # Monotonic: a bigger base never gets a smaller rate.
    sizes = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]
    rates = [default_muon_lr(s) for s in sizes]
    assert rates == sorted(rates)


def test_explicit_lr_is_never_overridden(tiny_base, tmp_path):
    """A rate the user asked for must survive - the default only fills a gap."""
    from stratum.train import train_tile

    out = tmp_path / "explicit-lr"
    train_tile(skill_path=str(EXAMPLES / "extract.jsonl"), out_dir=str(out),
               base_model=tiny_base, rank=2, epochs=1, batch_size=2,
               max_len=96, lr=7e-3, load_4bit=False, seed=3)
    card = json.loads((out / "stratum_card.json").read_text(encoding="utf-8"))
    assert card["lr"] == 7e-3


def test_a_worse_epoch_that_is_not_divergence_still_keeps_the_best():
    """Loss can wobble upward without the run diverging. That epoch's weights
    are still not the ones to keep, but it must not be reported as a blow-up."""
    from stratum.train import epoch_verdict
    v = epoch_verdict(1.10, 1.00, 2.00)   # worse than best, still below start
    assert not v["improved"] and not v["diverging"]


def test_diverged_run_keeps_the_better_weights(tiny_base, tmp_path, capsys):
    """The end to end guarantee: a run driven to diverge still leaves a
    loadable adapter from its best epoch, and flags itself in the card."""
    from stratum.merge import load_stratum_factors
    from stratum.train import train_tile

    out = tmp_path / "diverged"
    train_tile(skill_path=str(EXAMPLES / "extract.jsonl"), out_dir=str(out),
               base_model=tiny_base, rank=4, epochs=3, batch_size=2,
               grad_accum=1, max_len=96, lr=200.0, load_4bit=False, seed=1)
    card = json.loads((out / "stratum_card.json").read_text(encoding="utf-8"))

    assert card["best_epoch"] <= card["epochs"]
    assert "diverged" in card
    # Whatever happened to the loss, the artifact on disk is intact.
    assert load_stratum_factors(str(out))
    if card["diverged"]:
        assert "--lr" in capsys.readouterr().out


def test_pool_serves_and_switches_skills(two_strata, tmp_path):
    """The end to end claim: one base in memory, several skills, routed."""
    from stratum.serve import SkillPool

    files = {Path(d).name: str(EXAMPLES / f"{Path(d).name}.jsonl")
             for d in two_strata}
    router_path = str(tmp_path / "pool-router.json")
    train_router(files, router_path, verbose=False)

    pool = SkillPool(two_strata, router_path=router_path, verbose=False)
    assert len(pool.names) == 2

    routed = pool.generate("Extract the total from this invoice: 'Total: $88'",
                           max_new_tokens=4)
    assert routed["skill"] == "extract"
    assert routed["tokens"] > 0

    # An explicit skill overrides the router and reports full confidence.
    forced = pool.generate("anything at all", skill="classify", max_new_tokens=4)
    assert forced["skill"] == "classify"
    assert forced["confidence"] == 1.0

    with pytest.raises(ValueError, match="Unknown skill"):
        pool.generate("hello", skill="not-a-skill")


def test_teacher_quantizes_only_when_it_will_not_fit():
    """An 8B teacher on an 8 GB card must drop to 4-bit, a 1.7B must not."""
    from stratum.teachers import should_quantize_teacher

    assert should_quantize_teacher(8.0, 8.6) is True
    assert should_quantize_teacher(4.0, 8.6) is True
    assert should_quantize_teacher(1.7, 8.6) is False
    assert should_quantize_teacher(8.0, 80.0) is False


def test_teacher_never_guesses_without_a_gpu_or_a_size():
    """No card and no known size both mean leave it alone."""
    from stratum.teachers import should_quantize_teacher

    assert should_quantize_teacher(70.0, 0) is False
    assert should_quantize_teacher(None, 8.6) is False
