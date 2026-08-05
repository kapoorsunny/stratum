"""
Who is allowed to see what, and where that knowledge is allowed to live.

The hard part of putting company data into a language model is not building
the model. It is that a model trained on everything will tell anybody
anything. You cannot filter a weight. Once a salary figure is in the
parameters, there is no query-time check that keeps it away from a
contractor, because the model does not retrieve it, it simply knows it.

So the decision has to be made before training, not after, and it is made
here.

Two ideas do all the work.

A COMPARTMENT is a set of documents that share an access policy. It usually
maps to something the company already has, a folder, a SharePoint site, a
department share. It is the unit of everything downstream, because it is the
smallest thing whose visibility is a single yes or no.

A TIER says where that compartment's knowledge is allowed to live. There are
three, and the choice is forced by two questions about the compartment
rather than by preference.

  COMPANY     everybody can see it, and it changes rarely
              -> it goes into the weights, in one shared adapter
  DEPARTMENT  a whole department shares it, and it changes rarely
              -> it goes into the weights, in that department's adapter
  RESTRICTED  narrow, or sensitive, or changes often
              -> it NEVER goes into the weights, only into the index

The tiering is what makes this deployable rather than merely correct.
Compartmenting everything sounds better until you count: a person holding
permissions to k compartments would need an adapter trained on that exact
combination, and there are two to the power n combinations. AC-LoRA names
this as its own open problem.

Real permissions are not arbitrary combinations though. They are a lattice.
Nobody holds a random twelve of fifty compartments, they hold public, plus
their department, plus their team, plus a couple of projects. So only the
broad stable tiers go into weights, where the count is small and
hierarchical, and everything narrow stays in the index where filtering a row
is easy and deleting one is instant.

That also puts each mechanism on the job it is good at. Training is good at
form, at how your organisation writes an answer. Retrieval is good at facts.
Volatile facts stay out of the weights, so there is nothing stale baked in
and nothing to unlearn when a document is withdrawn.
"""
from __future__ import annotations

import json
from pathlib import Path

TIERS = ("company", "department", "restricted")

# A compartment has to be seen by at least this many principals before its
# knowledge is allowed into shared weights. Below it, one person's access
# would put that data in a model other people load.
COMPANY_MIN_PRINCIPALS = 0.9   # as a fraction of all principals
DEPARTMENT_MIN_PRINCIPALS = 2


class AccessError(Exception):
    """Raised when a policy is malformed or a principal is unknown.

    Separate from ValueError so that a caller can tell a bad policy from a
    bad argument, and so nothing accidentally swallows it.
    """


class Compartment:
    """One set of documents sharing an access policy."""

    def __init__(self, name: str, tier: str = "restricted",
                 volatile: bool = False, description: str = ""):
        if tier not in TIERS:
            raise AccessError(
                f"Compartment '{name}' has tier '{tier}', which is not one of "
                f"{', '.join(TIERS)}.")
        self.name = name
        self.tier = tier
        self.volatile = volatile
        self.description = description

    def to_dict(self) -> dict:
        return {"name": self.name, "tier": self.tier,
                "volatile": self.volatile, "description": self.description}

    def __repr__(self) -> str:
        return f"Compartment({self.name!r}, tier={self.tier!r})"


class Principal:
    """A person or service, and the compartments they may see."""

    def __init__(self, name: str, compartments: list[str],
                 description: str = ""):
        self.name = name
        # A set, because every check downstream is a membership test and
        # duplicate grants in a policy file should not change behaviour.
        self.compartments = set(compartments)
        self.description = description

    def can_see(self, compartment: str) -> bool:
        return compartment in self.compartments

    def to_dict(self) -> dict:
        return {"name": self.name,
                "compartments": sorted(self.compartments),
                "description": self.description}

    def __repr__(self) -> str:
        return f"Principal({self.name!r}, {len(self.compartments)} compartments)"


class Policy:
    """Every compartment, every principal, and who may see what.

    This is deliberately a plain readable file rather than a service. An
    access policy that nobody can read is an access policy nobody can audit,
    and the first question in any security review is going to be show me
    exactly who can see this.
    """

    def __init__(self, compartments: list[Compartment],
                 principals: list[Principal]):
        self.compartments = {c.name: c for c in compartments}
        self.principals = {p.name: p for p in principals}
        self._check()

    def _check(self) -> None:
        """Refuse a policy that cannot mean what it says.

        Every failure here is one that would otherwise show up as a silent
        permission hole, so none of them are warnings.
        """
        if not self.compartments:
            raise AccessError("A policy needs at least one compartment.")
        if not self.principals:
            raise AccessError("A policy needs at least one principal.")

        for p in self.principals.values():
            unknown = p.compartments - set(self.compartments)
            if unknown:
                raise AccessError(
                    f"Principal '{p.name}' is granted {sorted(unknown)}, which "
                    f"do not exist. A grant to a compartment that does not "
                    f"exist is usually a typo in the name of one that does.")

        # A company tier compartment that some principals cannot see is the
        # dangerous case, because its knowledge goes into weights everybody
        # loads. Catching it here is the difference between a policy error
        # and a data leak.
        n = len(self.principals)
        for c in self.compartments.values():
            seen_by = self.principals_for(c.name)
            if c.tier == "company" and len(seen_by) < n:
                missing = sorted(set(self.principals) - set(seen_by))
                raise AccessError(
                    f"Compartment '{c.name}' is tier 'company', so its "
                    f"content would be trained into weights that every "
                    f"principal loads. But {missing} cannot see it.\n"
                    f" - Grant it to everyone, or\n"
                    f" - move it to tier 'department' or 'restricted'.")
            if c.tier == "department" and len(seen_by) < DEPARTMENT_MIN_PRINCIPALS:
                raise AccessError(
                    f"Compartment '{c.name}' is tier 'department' but only "
                    f"{len(seen_by)} principal(s) can see it. Training a "
                    f"shared adapter on data one person can see puts it in a "
                    f"model others may load. Use tier 'restricted'.")
            if c.tier in ("company", "department") and c.volatile:
                raise AccessError(
                    f"Compartment '{c.name}' is marked volatile but has tier "
                    f"'{c.tier}'. Volatile data must not go into weights, "
                    f"because withdrawing a document would mean retraining. "
                    f"Use tier 'restricted'.")

    def principals_for(self, compartment: str) -> list[str]:
        """Everyone who can see this compartment."""
        return sorted(p.name for p in self.principals.values()
                      if p.can_see(compartment))

    def visible_compartments(self, principal: str) -> set[str]:
        p = self.principals.get(principal)
        if p is None:
            raise AccessError(
                f"Unknown principal '{principal}'. Known principals are: "
                f"{', '.join(sorted(self.principals))}.")
        return set(p.compartments)

    def strata_for(self, principal: str) -> list[str]:
        """Which trained adapters this principal is allowed to load.

        Only the tiers that live in weights appear here. A restricted
        compartment has no adapter at all by construction, which is the
        whole point of the tier.
        """
        visible = self.visible_compartments(principal)
        return sorted(name for name in visible
                      if self.compartments[name].tier in ("company", "department"))

    def index_compartments(self, principal: str) -> set[str]:
        """Which compartments this principal's searches may touch.

        Every visible compartment, including the ones in weights, because a
        question can want a specific sentence out of a document whose gist
        the model already learned.
        """
        return self.visible_compartments(principal)

    def tier_of(self, compartment: str) -> str:
        c = self.compartments.get(compartment)
        if c is None:
            raise AccessError(f"Unknown compartment '{compartment}'.")
        return c.tier

    def tier_risk(self, tier: str) -> int:
        """How exposed a tier is, so two of them can be compared.

        company puts knowledge in an adapter everybody loads, department in
        one a group loads, restricted in nobody's weights at all. Higher is
        more exposed.
        """
        return {"restricted": 0, "department": 1, "company": 2}[tier]

    def suggest_tier(self, compartment: str) -> str:
        """The most exposed tier this compartment's visibility could justify.

        Compared against what the author declared, this is what lets
        `stratum access check` point at a compartment whose label is more
        generous than its readership.

        It is a ceiling and not an instruction. Declaring something
        restricted when its visibility would permit department is a choice
        to be careful, and being careful is never reported as a problem.
        """
        c = self.compartments[compartment]
        if c.volatile:
            return "restricted"
        seen = len(self.principals_for(compartment))
        total = len(self.principals)
        if total and seen >= total * COMPANY_MIN_PRINCIPALS:
            return "company"
        if seen >= DEPARTMENT_MIN_PRINCIPALS:
            return "department"
        return "restricted"

    def to_dict(self) -> dict:
        return {
            "compartments": [c.to_dict() for c in self.compartments.values()],
            "principals": [p.to_dict() for p in self.principals.values()],
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n",
                              encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "Policy":
        try:
            comps = [Compartment(**c) for c in d["compartments"]]
            prins = [Principal(**p) for p in d["principals"]]
        except (KeyError, TypeError) as e:
            raise AccessError(
                f"The policy file is missing something or has an unexpected "
                f"field: {e}. It needs a 'compartments' list and a "
                f"'principals' list.") from e
        return cls(comps, prins)

    @classmethod
    def load(cls, path: str) -> "Policy":
        p = Path(path)
        if not p.exists():
            raise AccessError(
                f"No policy file at {path}. Write one, or run "
                f"`stratum access init` to get a starting point.")
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise AccessError(f"The policy file at {path} is not valid JSON: {e}") from e
        return cls.from_dict(d)


def example_policy() -> Policy:
    """A small policy showing every tier, used by `stratum access init`.

    It is a real shape rather than a toy. Most companies end up with
    something everyone sees, a few department shares, and a long tail of
    things that must stay narrow.
    """
    return Policy(
        compartments=[
            Compartment("public", "company",
                        description="Standards, terminology, published material"),
            Compartment("engineering", "department",
                        description="Equipment manuals and design notes"),
            Compartment("safety", "department",
                        description="Hazard studies and operating procedures"),
            Compartment("finance", "restricted",
                        description="Contracts and commercial terms"),
            Compartment("hr", "restricted", volatile=True,
                        description="Personnel records, changes constantly"),
        ],
        principals=[
            Principal("contractor", ["public"],
                      description="Outside party, published material only"),
            Principal("engineer", ["public", "engineering"]),
            Principal("safety-officer", ["public", "safety"]),
            Principal("plant-manager", ["public", "engineering", "safety"]),
            Principal("executive", ["public", "engineering", "safety",
                                    "finance", "hr"]),
        ],
    )


def describe(policy: Policy) -> str:
    """A readable summary, because a policy nobody reads is a policy nobody
    audits."""
    lines = ["Compartments"]
    for c in policy.compartments.values():
        who = policy.principals_for(c.name)
        ceiling = policy.suggest_tier(c.name)
        flag = ("   <- TOO EXPOSED, the most it should be is "
                f"'{ceiling}'") if policy.tier_risk(c.tier) > policy.tier_risk(ceiling) else ""
        where = ("weights + index" if c.tier in ("company", "department")
                 else "index only")
        lines.append(f"  {c.name:<14} {c.tier:<11} {where:<16} "
                     f"{len(who)} principal(s){flag}")
        if c.volatile:
            lines.append(f"  {'':<14} volatile, so it stays out of the weights")

    lines.append("")
    lines.append("Principals")
    for p in policy.principals.values():
        strata = policy.strata_for(p.name)
        lines.append(f"  {p.name:<14} sees {len(p.compartments)} compartment(s), "
                     f"loads {len(strata)} adapter(s)")
        lines.append(f"  {'':<14} adapters: {', '.join(strata) if strata else 'none'}")
        blocked = sorted(set(policy.compartments) - p.compartments)
        lines.append(f"  {'':<14} blocked : {', '.join(blocked) if blocked else 'nothing'}")

    # The count that decides whether this is deployable. Merging more than
    # about three adapters degrades a model badly, which this project has
    # measured directly, so it gets said out loud rather than discovered.
    worst = max((len(policy.strata_for(p)) for p in policy.principals), default=0)
    lines.append("")
    lines.append(f"Most adapters any one principal loads: {worst}")
    if worst > 3:
        lines.append("  That is above the safe merge limit. Route between them "
                     "instead of merging,")
        lines.append("  or move some compartments to tier 'restricted'.")
    return "\n".join(lines)
