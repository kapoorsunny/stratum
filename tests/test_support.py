"""The support check, which is the last thing between a model and a person.

Every case below is either a fabrication observed on a real build, or a
correct answer that must not be refused. Both matter equally. A check that
only catches inventions by refusing everything has moved the problem rather
than fixed it, so roughly half these tests exist to stop it doing that.
"""
import pytest

from stratum.grounded import REFUSAL
from stratum.support import check, explain, gate, passages_from_prompt

PASSAGES = """\
[engineering/pumps.txt]
Pump P-4471 is a single stage centrifugal pump. The certified bearing
clearance is 0.418 mm measured at the coupling end. Inspection is due every
4000 running hours.

[engineering/compressor.txt]
Compressor K-201 is rated for 84 bar discharge pressure. Surge control is by
recycle valve.
"""


# ------------------------------------------------ the fabrications it must stop
def test_a_number_that_is_nowhere_in_the_material_is_refused():
    """The failure this whole module exists for.

    Served to somebody with no access to engineering material, on a build
    where retrieval had correctly returned nothing about pumps.
    """
    answer = ("The limits are 1.5 to 2.0 times the pump's rated discharge "
              "capacity.")
    result = check(answer, PASSAGES, "what are the bearing clearance limits")
    assert not result["supported"]
    assert result["bad_numbers"] == ["1.5", "2"]
    assert "1.5" in explain(result)


def test_invented_percentages_are_refused():
    """Observed verbatim on the grounded model, which still invented here."""
    answer = ("In 2030 the market is expected to grow at 6.5% per year, and "
              "the CAGR is projected to exceed 10% by 2030.")
    result = check(answer, PASSAGES, "what market size and CAGR by 2030")
    assert not result["supported"]
    assert "6.5" in result["bad_numbers"]


def test_an_answer_with_nothing_in_common_with_the_material_is_refused():
    """The qualitative version, where no number is invented but the reason is.

    Also observed. The model restated the question and appended a cause of
    its own.
    """
    answer = ("Carbon dioxide is the fluid most commonly used for miscible "
              "displacement because of its low purchase cost relative to "
              "other injected chemicals.")
    result = check(answer, PASSAGES, "why is carbon dioxide used")
    assert not result["supported"]
    assert result["reason"] == "almost nothing in common with the material"


def test_an_invented_equipment_tag_is_refused():
    answer = "Pump P-9999 has a clearance of 0.418 mm."
    result = check(answer, PASSAGES, "what is the clearance")
    assert not result["supported"]
    assert result["bad_identifiers"] == ["P9999"]


def test_the_same_tag_written_with_or_without_a_dash_is_one_tag():
    """P-4471 and P4471 are the same pump. Calling one invented because the
    source wrote the other would fire on correct answers constantly."""
    assert check("Pump P4471 has a clearance of 0.418 mm.", PASSAGES,
                 "")["supported"]


# ------------------------------------------- the answers it must NOT interfere with
def test_a_correct_answer_quoting_the_material_passes():
    answer = "The certified bearing clearance on P-4471 is 0.418 mm."
    assert check(answer, PASSAGES, "clearance of P-4471")["supported"]


def test_a_correct_answer_that_paraphrases_passes():
    """Refusing a right answer costs the same trust as passing a wrong one."""
    answer = ("Bearings on that pump should be inspected every 4000 hours of "
              "running, and the clearance certified at the coupling is "
              "0.418 mm.")
    assert check(answer, PASSAGES, "inspection interval")["supported"]


def test_a_number_written_differently_still_counts_as_present():
    """A source saying 1,200 and an answer saying 1200 are the same figure."""
    passages = "The unit produced 1,200 tonnes in the period."
    assert check("It produced 1200 tonnes.", passages, "")["supported"]
    assert check("It produced 1200.0 tonnes.", passages, "")["supported"]


def test_a_number_the_asker_supplied_is_not_an_invention():
    """Restating a figure from the question is not making one up."""
    result = check("Pump P-4471 runs at 2950 rpm as you say, and the "
                   "clearance is 0.418 mm.",
                   PASSAGES, "what about pump P-4471 running at 2950 rpm")
    assert result["supported"]


def test_written_out_numbers_are_not_checked():
    """"Three reasons" is discourse, not a measurement. Refusing on it would
    make the check fire constantly on ordinary prose."""
    answer = ("There are three considerations when inspecting the bearing "
              "clearance on pump P-4471 at the coupling end.")
    assert check(answer, PASSAGES, "")["supported"]


def test_a_short_answer_is_not_judged_on_overlap():
    """Too little content for a proportion to mean anything. Its numbers are
    still checked, which is the part that matters."""
    assert check("0.418 mm", PASSAGES, "clearance")["supported"]
    assert not check("0.911 mm", PASSAGES, "clearance")["supported"]


def test_a_refusal_is_never_itself_refused():
    """It claims nothing, so there is nothing in it to support. Checking one
    would refuse the very behaviour this is meant to encourage."""
    result = check(REFUSAL, PASSAGES, "anything at all")
    assert result["supported"]
    assert result["reason"] == "refusal"


# --------------------------------------------------------------------- the gate
def test_the_gate_returns_the_refusal_in_place_of_an_invention():
    answer = "The limits are 1.5 to 2.0 times rated discharge capacity."
    out, why = gate(answer, PASSAGES, "clearance limits")
    assert out == REFUSAL
    assert not why["supported"]


def test_the_gate_passes_a_supported_answer_through_untouched():
    answer = "The certified bearing clearance is 0.418 mm."
    out, why = gate(answer, PASSAGES, "clearance")
    assert out == answer
    assert why["supported"]


def test_the_refusal_the_gate_returns_is_the_one_ground_trains():
    """One behaviour, whether the model declined or the check made it. A
    caller matching on refusals must not need to know which happened."""
    from stratum.grounded import is_refusal

    out, _ = gate("The value is 99.9 mm.", PASSAGES, "")
    assert is_refusal(out)


def test_with_no_material_at_all_a_specific_claim_cannot_be_supported():
    """What happens when retrieval found nothing, which is exactly the
    situation the contractor was in."""
    out, why = gate("The clearance limit is 0.418 mm.", "", "clearance")
    assert out == REFUSAL
    assert why["bad_numbers"] == ["0.418"]


# ------------------------------------------------------------------ the plumbing
def test_passages_are_recovered_from_a_grounded_prompt():
    """So a recorded evaluation is checked by the same code that serves, and
    the two cannot drift apart."""
    from stratum.grounded import build_prompt

    prompt = build_prompt("What is the clearance?",
                          [("engineering/pumps.txt", "The clearance is 0.418 mm.")])
    got = passages_from_prompt(prompt)
    assert "0.418" in got
    assert "What is the clearance?" not in got


def test_a_passage_containing_the_word_question_is_not_truncated():
    """Split from the right, or a passage discussing questions loses
    everything after the word and takes its numbers with it."""
    from stratum.grounded import build_prompt

    prompt = build_prompt(
        "How many?",
        [("doc.txt", "Question: what is the limit? The answer is 42 units.")])
    got = passages_from_prompt(prompt)
    assert "42 units" in got


def test_the_min_overlap_dial_moves_the_line():
    answer = ("Carbon dioxide is used because of its low purchase cost "
              "relative to other injected chemicals in the field.")
    assert not check(answer, PASSAGES, "", min_overlap=0.35)["supported"]
    # Permissive enough and it stops firing, which is what the dial is for.
    assert check(answer, PASSAGES, "", min_overlap=0.0)["supported"]


@pytest.mark.parametrize("answer", ["", "   ", "\n"])
def test_an_empty_answer_does_not_crash_the_check(answer):
    result = check(answer, PASSAGES, "anything")
    assert isinstance(result["supported"], bool)
