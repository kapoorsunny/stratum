"""Tests for the setup command.

The decisions live in a pure function so every platform can be checked from
one machine. That matters here more than usual, because the bugs this
command exists to prevent were all reported by people on hardware the author
does not own.
"""
from stratum.setup_env import cuda_index_url, plan_actions, platform_notes


def facts(**over):
    """A machine with nothing installed, overridden per test."""
    base = {
        "os": "Linux", "arch": "x86_64", "python": "3.11.0",
        "apple_silicon": False, "torch": None, "torch_cuda_build": False,
        "cuda_visible": False, "mps_visible": False, "nvidia_present": False,
        "bitsandbytes": False, "peft": None, "transformers": None,
        "compiler": "cc",
    }
    base.update(over)
    return base


def keys(actions):
    return [a["key"] for a in actions]


def joined(actions):
    return " ".join(" ".join(a["cmd"]) for a in actions)


def test_apple_silicon_gets_plain_torch_and_never_bitsandbytes():
    """The wheel already does MPS, and 4-bit does not exist on a Mac."""
    a = plan_actions(facts(os="Darwin", arch="arm64", apple_silicon=True))

    assert "torch" in keys(a)
    assert "--index-url" not in joined(a)
    assert "bitsandbytes" not in keys(a)


def test_apple_silicon_is_told_4bit_will_never_work():
    notes = " ".join(platform_notes(facts(os="Darwin", arch="arm64",
                                          apple_silicon=True)))
    assert "4-bit" in notes
    assert "MPS" in notes


def test_intel_mac_is_told_it_has_no_gpu_option():
    notes = " ".join(platform_notes(facts(os="Darwin", arch="x86_64")))
    assert "Intel Mac" in notes
    assert "processor" in notes


def test_nvidia_machine_with_no_torch_gets_the_cuda_index():
    a = plan_actions(facts(nvidia_present=True))
    assert cuda_index_url() in joined(a)


def test_cpu_build_on_an_nvidia_machine_is_force_reinstalled():
    """The bug that cost hours. A plain install leaves the CPU build alone,
    because pip sees torch is already there."""
    a = plan_actions(facts(torch="2.11.0+cpu", torch_cuda_build=False,
                           nvidia_present=True))

    fix = [x for x in a if x["key"] == "torch-cuda"]
    assert len(fix) == 1
    assert "--force-reinstall" in fix[0]["cmd"]
    assert cuda_index_url() in fix[0]["cmd"]
    assert "pip sees torch is present" in fix[0]["why"]


def test_a_working_cuda_machine_is_left_alone():
    a = plan_actions(facts(torch="2.11.0+cu128", torch_cuda_build=True,
                           cuda_visible=True, nvidia_present=True,
                           bitsandbytes=True, peft="0.20.0",
                           transformers="5.14.1"))
    assert a == []


def test_a_working_mac_is_left_alone():
    a = plan_actions(facts(os="Darwin", arch="arm64", apple_silicon=True,
                           torch="2.11.0", mps_visible=True, peft="0.20.0",
                           transformers="5.14.1"))
    assert a == []


def test_missing_libraries_are_installed_on_any_platform():
    a = plan_actions(facts(torch="2.11.0", transformers=None, peft=None))
    assert "transformers" in keys(a)
    assert "peft" in keys(a)


def test_a_missing_compiler_is_only_a_note_not_an_action():
    """The engine is optional, so a machine without a compiler is fine."""
    f = facts(compiler=None, torch="2.11.0", peft="0.20.0",
              transformers="5.14.1")

    assert plan_actions(f) == []
    assert "compiler" in " ".join(platform_notes(f))


def test_a_local_teacher_runs_one_call_at_a_time():
    """A local model already has the whole GPU, so a second call wins
    nothing and can run it out of memory."""
    from stratum.corpus import default_concurrency

    assert default_concurrency("hf") == 1
    assert default_concurrency("claude-cli") > 1
    assert default_concurrency("openai") > 1


def test_chunks_map_yields_every_item_at_any_concurrency():
    from stratum.corpus import _map_chunks

    items = list(range(50))
    for workers in (1, 4, 16):
        got = sorted(_map_chunks(lambda x: x * 2, items, workers))
        assert got == [x * 2 for x in items]


def test_printing_survives_characters_a_console_cannot_draw(capsys):
    """A model trained on engineering documents writes Greek letters, and a
    Windows console defaults to an encoding that cannot hold them."""
    import io
    import sys

    from stratum.__main__ import use_utf8_output

    old = sys.stdout
    try:
        sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        use_utf8_output()
        print("delta \u0394 ohm \u03a9 micro \u03bc degree \u00b0")
        sys.stdout.flush()
    finally:
        sys.stdout = old
