"""CLT-004: reachable-state refinement of a nonlinear observer closure."""
from dataclasses import dataclass
from vli.minimal_refinement import _express


@dataclass(frozen=True)
class FiberPlan:
    """Finite conditional witness code; source measurement remains external."""
    rows: tuple
    width: int
    linear_deficit: int

    def reconstruct(self, old_coordinate, witness):
        if type(witness) is not int or not 0 <= witness < 1 << self.width:
            raise ValueError("invalid witness")
        for key, values in self.rows:
            if key == old_coordinate:
                if witness >= len(values):
                    raise ValueError("unrealized witness")
                return values[witness]
        raise ValueError("unrealized old coordinate")


def form_fiber_plan(old, new):
    """Compare feature-space linear complement to minimal fixed-width fiber code.

    Requires independently verified old/new Grounds on the same declared
    physical finite domain; their allowed action alphabets may differ.
    This is not TDL-002 same-dynamics source authority.
    """
    if old.size != new.size:
        raise ValueError("different source domains")
    union = list(old.basis)
    for row in new.basis:
        if _express(row, tuple(union)) is None:
            union.append(row)
    deficit = len(union)-len(old.basis)
    fibers = {}
    for x in range(old.size):
        fibers.setdefault(old.encode(x),set()).add(new.encode(x))
    rows = tuple((k,tuple(sorted(v))) for k,v in sorted(fibers.items()))
    width = (max(len(v) for _,v in rows)-1).bit_length()
    return FiberPlan(rows,width,deficit)


def verify_plan(plan, old, new):
    """Re-derive exact fiber sets and test each reachable reconstruction."""
    if old.size != new.size:
        return False
    expected = {}
    for x in range(old.size):
        expected.setdefault(old.encode(x),set()).add(new.encode(x))
    if len(plan.rows) != len(expected) or len(dict(plan.rows)) != len(expected):
        return False
    for key, values in plan.rows:
        if tuple(sorted(expected.get(key,set()))) != values:
            return False
    if plan.width != (max(map(len,expected.values()))-1).bit_length():
        return False
    for x in range(old.size):
        old_z, new_z = old.encode(x), new.encode(x)
        witness = dict(plan.rows)[old_z].index(new_z)
        if plan.reconstruct(old_z,witness) != new_z:
            return False
    return True


def source_witness(plan, old, new, x):
    """Fixture source oracle; receiver does not need x after witness acquisition."""
    return dict(plan.rows)[old.encode(x)].index(new.encode(x))
