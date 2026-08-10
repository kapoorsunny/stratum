"""
Check an answer against the material it was supposed to come from, and refuse
it when it is not there.

Training reduces invention. It does not end it, and no amount of it will,
because a language model's job is to continue text plausibly and a plausible
continuation of a question it cannot answer is an answer. Measured on a real
build, adding grounded training took invention from 28 cases out of 28 down
to 13 out of 28. Better, and nowhere near good enough to put in front of
somebody who will act on the number.

So the last line is not training, it is a check after the fact. The model
writes an answer, and before anybody sees it, the answer is compared against
the passages it was given. Claims that are not in those passages do not go
out.

That turns a tendency into a rule. It is also auditable, which matters more
in an enterprise than the accuracy of any single model, because the question
after an incident is never what the model usually does, it is what happened
that time and how anybody could have known.

Two rules, chosen because they catch different failures.

A number that is not in the source. This is the dangerous one and the easiest
to be sure about. "The limits are 1.5 to 2.0 times rated discharge capacity"
was produced for somebody with no access to engineering material, and neither
number appears anywhere in what they were shown. A wrong figure is what gets
acted on, and a figure is either present in the source or it is not, so this
check is close to exact rather than a judgement.

An answer with almost nothing in common with the source. This catches the
qualitative version, where a model restates the question and appends a reason
of its own invention. "Carbon dioxide is used because of its low cost" says
nothing the passages say.

Both are tunable, and both report which rule fired and on what, so a refusal
can be explained rather than merely delivered.

What this does not do is check that a supported answer is a correct one. An
answer can quote the right number from the right passage and still be a wrong
reading of it. This is a floor, not a guarantee of truth.
"""
from __future__ import annotations

import re

# Digits, decimals, and ranges. Written out numbers are deliberately not
# matched. "There are three reasons" is discourse rather than a measurement,
# and a model that says three where the source says four is not the failure
# this is here to catch.
_NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?![\w])")

# Codes and identifiers, which behave like numbers for this purpose. A tag
# such as P-4471 or ISO-9000 either appears in the source or was invented,
# and inventing one is how an answer ends up describing a pump that does not
# exist.
_IDENT = re.compile(r"\b[A-Za-z]+[-_]?\d+[A-Za-z0-9]*\b")

_WORD = re.compile(r"[^\W_]+", re.UNICODE)

# Words that carry no content, so overlap computed on them would say a model
# and a source agree because both are written in English.
_STOP = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "he",
    "she", "they", "them", "their", "we", "our", "you", "your", "i", "not",
    "no", "do", "does", "did", "have", "has", "had", "can", "could", "will",
    "would", "should", "may", "might", "must", "there", "here", "what", "which",
    "who", "when", "where", "why", "how", "also", "such", "some", "any", "all",
    "more", "most", "other", "into", "about", "over", "under", "between",
    "because", "so", "very", "much", "many", "one", "two", "three", "both",
}

# How much of the answer's content has to appear in the source before it is
# treated as coming from there. Low on purpose. A high floor would refuse
# correct answers that paraphrase, and a refusal of a good answer costs the
# same trust as an invention.
MIN_OVERLAP = 0.35

# Answers shorter than this are not checked for overlap, because a three word
# answer has too little content for the proportion to mean anything. Their
# numbers are still checked.
MIN_WORDS_FOR_OVERLAP = 6


def _numbers(text: str) -> set[str]:
    """Numbers in a piece of text, normalised so formatting does not matter.

    A source writing 1,200 and an answer writing 1200 are the same figure, and
    a check that called that a fabrication would be useless.
    """
    out = set()
    for raw in _NUMBER.findall(text or ""):
        n = raw.replace(",", "")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.add(n or "0")
    return out


def _identifiers(text: str) -> set[str]:
    """Equipment tags and standard numbers, with separators removed.

    P-4471 and P4471 are the same pump, and a check that called one of them
    invented because the other was written with a dash would fire on correct
    answers constantly.
    """
    return {m.upper().replace("-", "").replace("_", "")
            for m in _IDENT.findall(text or "")}


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text or "")
            if w.lower() not in _STOP and len(w) > 2}


def passages_from_prompt(prompt: str) -> str:
    """The reference material out of a grounded prompt.

    Lets the check run on a recorded evaluation, where the passages are part
    of the prompt and there is no index to consult, using exactly the same
    code that runs when serving. Two implementations would eventually
    disagree, and the one that mattered would be the one not being tested.
    """
    from .grounded import GROUNDED_TEMPLATE

    head = GROUNDED_TEMPLATE.split("{passages}")[0].strip()
    body = prompt
    if head and head in body:
        body = body.split(head, 1)[1]
    # Everything before the question is the material. Split from the right,
    # because a passage may legitimately contain the word Question.
    if "\nQuestion:" in body:
        body = body.rsplit("\nQuestion:", 1)[0]
    return body


def check(answer: str, passages: str, question: str = "",
          min_overlap: float = MIN_OVERLAP) -> dict:
    """Is this answer carried by this material?

    question is included as a source of allowed numbers and terms, because an
    answer restating a figure the asker supplied has not invented it.

    Returns what was found rather than only a verdict, so a refusal can say
    which claim was not there.
    """
    from .grounded import is_refusal

    answer = answer or ""
    # A refusal claims nothing, so there is nothing in it to support. Checking
    # one would refuse the very behaviour this exists to encourage.
    if is_refusal(answer):
        return {"supported": True, "reason": "refusal", "bad_numbers": [],
                "bad_identifiers": [], "overlap": 1.0}

    allowed_text = f"{passages}\n{question}"
    bad_numbers = sorted(_numbers(answer) - _numbers(allowed_text))
    bad_ident = sorted(_identifiers(answer) - _identifiers(allowed_text))

    answer_words = _content_words(answer)
    source_words = _content_words(allowed_text)
    if answer_words:
        overlap = len(answer_words & source_words) / len(answer_words)
    else:
        overlap = 0.0
    long_enough = len(answer_words) >= MIN_WORDS_FOR_OVERLAP

    if bad_numbers:
        reason = "number not in the material"
    elif bad_ident:
        reason = "identifier not in the material"
    elif long_enough and overlap < min_overlap:
        reason = "almost nothing in common with the material"
    else:
        reason = ""

    return {"supported": not reason, "reason": reason,
            "bad_numbers": bad_numbers, "bad_identifiers": bad_ident,
            "overlap": round(overlap, 3)}


def explain(result: dict) -> str:
    """One line saying why an answer was held back."""
    if result["supported"]:
        return "supported by the material"
    if result["bad_numbers"]:
        return (f"held back, {', '.join(result['bad_numbers'])} "
                f"{'appears' if len(result['bad_numbers']) == 1 else 'appear'} "
                f"nowhere in the material")
    if result["bad_identifiers"]:
        return (f"held back, {', '.join(result['bad_identifiers'])} appears "
                f"nowhere in the material")
    return (f"held back, only {result['overlap']:.0%} of the answer's content "
            f"appears in the material")


def gate(answer: str, passages: str, question: str = "",
         min_overlap: float = MIN_OVERLAP) -> tuple[str, dict]:
    """The answer, or a refusal in its place, plus why.

    The refusal is the same sentence `stratum ground` trains, so a caller
    matching on refusals sees one behaviour whether it was the model that
    declined or this check that made it.
    """
    from .grounded import REFUSAL

    result = check(answer, passages, question, min_overlap=min_overlap)
    return (answer if result["supported"] else REFUSAL), result
