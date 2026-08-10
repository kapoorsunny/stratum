"""Finding and releasing memory an earlier run is still holding.

Nothing here starts or stops a real process. The dangerous part of this
module is deciding WHICH process to stop, so that is what the tests pin
down, along with the reporting that has to stay honest on a machine whose
driver will not answer the question.

Every test runs the same on Windows, Linux and a Mac, because the platform
differences are faked rather than depended on. A test that only runs where
there is an NVIDIA card would not protect the two thirds of users who do not
have one.
"""
import os

import pytest

from stratum import resources


# ------------------------------------------------------ which process is ours
@pytest.mark.parametrize("cmdline", [
    "python -m stratum serve strata/a --port 8927",
    "/usr/bin/python3 -m stratum train --skill x.jsonl",
    "C:\\proj\\.venv\\Scripts\\python.exe -m stratum build --plan p.yaml",
    "/opt/venv/bin/stratum serve strata/a",
    "C:\\proj\\.venv\\Scripts\\stratum.exe train --skill x.jsonl",
])
def test_our_own_processes_are_recognised(cmdline):
    assert resources._looks_like_stratum(cmdline)


@pytest.mark.parametrize("cmdline", [
    "python -m http.server",
    "python train.py --model llama",
    "/usr/bin/jupyter-lab",
    "python -c import torch",
    "ollama serve",
    # The word appears but not as this tool being run. A shared build box is
    # somebody else's machine too.
    "grep -r stratum /home/other/notes.txt",
    "vim stratum_ideas.md",
])
def test_other_peoples_processes_are_left_alone(cmdline):
    assert not resources._looks_like_stratum(cmdline)


@pytest.mark.parametrize("cmdline,verb", [
    ("python -m stratum serve strata/a", "serve"),
    ("python -m stratum train --skill x", "train"),
    ("/opt/venv/bin/stratum build --plan p.yaml", "build"),
    ("python -m stratum", "stratum"),
    ("python -m stratum --help", "stratum"),
])
def test_the_subcommand_is_read_off_the_command_line(cmdline, verb):
    assert resources._verb_of(cmdline) == verb


# --------------------------------------------------------------- self defence
def test_this_process_is_never_a_target():
    """Freeing memory by killing yourself halfway through is not freeing it."""
    assert os.getpid() in resources._protected_pids()
    assert all(t["pid"] != os.getpid() for t in resources.find())


def test_the_whole_parent_chain_is_protected():
    """Killing an ancestor takes this process with it."""
    import psutil

    protected = resources._protected_pids()
    for parent in psutil.Process(os.getpid()).parents():
        assert parent.pid in protected


# ------------------------------------------- reading a driver that will not say
def fake_smi(text):
    def _smi(*args):
        if "query-compute-apps" in " ".join(args):
            return text
        return None
    return _smi


def test_a_driver_that_reports_sizes_is_read_normally(monkeypatch):
    monkeypatch.setattr(resources, "_nvidia_smi",
                        fake_smi("1234, 5120\n5678, 2048\n"))
    assert resources.gpu_processes() == {1234: 5120, 5678: 2048}
    assert resources.attribution_available()


def test_a_process_the_driver_will_not_size_is_still_a_process(monkeypatch):
    """The bug this exists to prevent.

    On an ordinary Windows machine the driver names the process but answers
    [N/A] for its memory. Dropping the row because the number would not parse
    loses the fact that the process is on the GPU at all, and then a command
    that acts only on GPU holders quietly acts on nothing and says it worked.
    """
    monkeypatch.setattr(resources, "_nvidia_smi", fake_smi("66404, [N/A]\n"))
    held = resources.gpu_processes()
    assert 66404 in held, "the process must survive an unreadable size"
    assert held[66404] is None
    assert not resources.attribution_available()


def test_a_machine_with_no_driver_reports_nothing_rather_than_failing(monkeypatch):
    """A Mac is not a broken Windows box."""
    monkeypatch.setattr(resources, "_nvidia_smi", lambda *a: None)
    monkeypatch.setattr(resources, "_rocm_smi", lambda: [])
    assert resources.gpu_processes() == {}
    assert resources.gpu_memory() == []


def test_rubbish_from_the_driver_does_not_crash_the_command(monkeypatch):
    monkeypatch.setattr(resources, "_nvidia_smi",
                        fake_smi("nonsense\n\n, ,\nabc, def\n"))
    assert resources.gpu_processes() == {}


def test_gpu_only_still_selects_when_sizes_are_unavailable(monkeypatch):
    """The consequence of the fix above, checked end to end.

    Selection is on whether the driver confirms the process is on the card,
    which it does say, and not on a size, which it may not.
    """
    monkeypatch.setattr(resources, "_nvidia_smi", fake_smi("4242, [N/A]\n"))

    class FakeProc:
        info = {"pid": 4242, "name": "python.exe", "create_time": 0,
                "cmdline": ["python", "-m", "stratum", "serve", "strata/a"]}

        def memory_info(self):
            class M:
                rss = 1_000_000_000
            return M()

    monkeypatch.setattr(resources, "_protected_pids", lambda: {os.getpid()})
    import psutil
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: [FakeProc()])

    found = resources.find(include_gpu_only=True)
    assert len(found) == 1
    assert found[0]["pid"] == 4242
    assert found[0]["on_gpu"] is True
    assert found[0]["gpu_mb"] is None


def test_the_listing_sorts_without_a_size_to_sort_on(monkeypatch):
    """Two of these hold an unknown amount and one holds nothing. Sorting on
    the missing number alone would raise rather than print a list."""
    monkeypatch.setattr(resources, "_nvidia_smi", fake_smi("1, [N/A]\n2, [N/A]\n"))

    def proc(pid, rss):
        class P:
            info = {"pid": pid, "name": "python", "create_time": 0,
                    "cmdline": ["python", "-m", "stratum", "serve"]}

            def memory_info(self):
                class M:
                    pass
                m = M()
                m.rss = rss
                return m
        return P()

    monkeypatch.setattr(resources, "_protected_pids", lambda: {os.getpid()})
    import psutil
    # Sizes in bytes, far enough apart to survive rounding to megabytes.
    monkeypatch.setattr(psutil, "process_iter",
                        lambda attrs=None: [proc(3, 10_000_000_000),
                                            proc(1, 2_000_000_000),
                                            proc(2, 9_000_000_000)])

    found = resources.find()
    # The two the driver confirms are on the card come first.
    assert [f["pid"] for f in found][:2] == [2, 1]
    assert found[-1]["pid"] == 3 and found[-1]["on_gpu"] is False


# ----------------------------------------------------------------- the report
def test_dry_run_stops_nothing(monkeypatch, capsys):
    calls = []

    class FakeProc:
        info = {"pid": 999_999, "name": "python", "create_time": 0,
                "cmdline": ["python", "-m", "stratum", "serve"]}

        def memory_info(self):
            class M:
                rss = 1
            return M()

        def terminate(self):
            calls.append("terminate")

        def kill(self):
            calls.append("kill")

    monkeypatch.setattr(resources, "_protected_pids", lambda: {os.getpid()})
    monkeypatch.setattr(resources, "_nvidia_smi", lambda *a: None)
    monkeypatch.setattr(resources, "_rocm_smi", lambda: [])
    import psutil
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: [FakeProc()])

    report = resources.release(dry_run=True, verbose=True)
    assert calls == [], "a dry run must not touch anything"
    assert report["stopped"] == []
    assert len(report["found"]) == 1
    assert "Nothing was stopped" in capsys.readouterr().out


def test_a_clean_machine_says_so_and_does_nothing(monkeypatch, capsys):
    monkeypatch.setattr(resources, "find", lambda include_gpu_only=False: [])
    monkeypatch.setattr(resources, "_nvidia_smi", lambda *a: None)
    monkeypatch.setattr(resources, "_rocm_smi", lambda: [])
    report = resources.release(verbose=True)
    assert report["found"] == []
    assert report["freed_mb"] == 0
    assert "nothing here to release" in capsys.readouterr().out


def test_a_full_card_held_by_another_program_is_not_ours_to_touch(monkeypatch,
                                                                  capsys):
    """Saying so matters. Otherwise the answer to a full card is a command
    that reports success and changes nothing."""
    monkeypatch.setattr(resources, "find", lambda include_gpu_only=False: [])
    monkeypatch.setattr(resources, "gpu_memory", lambda: [
        {"kind": "nvidia", "index": 0, "name": "Test GPU",
         "used_mb": 7000, "total_mb": 8000}])
    resources.release(verbose=True)
    out = capsys.readouterr().out
    assert "not touch another program" in out


def test_freed_counts_only_memory_that_came_back():
    before = [{"index": 0, "used_mb": 7000, "total_mb": 8000}]
    after = [{"index": 0, "used_mb": 1000, "total_mb": 8000}]
    assert resources._freed(before, after) == 6000
    # A card that filled up further did not free anything, and must never
    # report a negative as though it had.
    assert resources._freed(after, before) == 0


def test_the_doctor_line_appears_only_when_there_is_something_to_say(monkeypatch):
    monkeypatch.setattr(resources, "find", lambda include_gpu_only=False: [])
    assert resources.summary_line() is None

    monkeypatch.setattr(resources, "find", lambda include_gpu_only=False: [
        {"pid": 1, "verb": "serve", "cmdline": "", "rss_mb": 4000,
         "on_gpu": True, "gpu_mb": None, "age_s": 60}])
    monkeypatch.setattr(resources, "gpu_memory", lambda: [
        {"kind": "nvidia", "index": 0, "name": "Test", "used_mb": 7000,
         "total_mb": 8000}])
    line = resources.summary_line()
    assert "stratum free" in line
    # No size available, so it must not invent one or say zero.
    assert "0 MB" not in line


def test_system_memory_is_reported_everywhere():
    """The only figure that means anything on a Mac, where the GPU shares it."""
    mem = resources.system_memory()
    assert mem is not None
    assert mem["total_mb"] > 0
    assert 0 <= mem["used_mb"] <= mem["total_mb"]
