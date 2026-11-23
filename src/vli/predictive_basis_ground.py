"""PBG-001: exact rational predictive Ground for a finite stochastic source.

A source is a family T[a] of nonnegative rational matrices, with
sum_a T[a] 1 = 1. It assigns P(w)=alpha T[w] 1 to token words.
We compute the smallest invariant column space containing 1, then
form K[a] from T[a] B = B K[a].  A session operates only on its
r-dimensional row coordinate, not the original source states.

This is a constructive finite weighted-automaton experiment, NOT an
interpreter of a pretrained neural checkpoint. No neural execution is
hidden in the session or in the construction verifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json

SCHEMA = "vli.predictive-basis-ground.v1"
MAX_STATES = 64
MAX_ACTIONS = 16


def _r(value):
    if type(value) not in (int, str, Fraction):
        raise ValueError("Rational inputs must not be floating-point approximations")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Invalid rational input") from exc


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf8")


def _hash(value):
    return sha256(_canonical(value)).hexdigest()


def _q(x):
    return f"{x.numerator}/{x.denominator}"


def _matrix_payload(matrix):
    return [[_q(v) for v in row] for row in matrix]


def _matvec(matrix, vector):
    return tuple(sum((v*x for v, x in zip(row, vector)), Fraction()) for row in matrix)


def _rowmat(vector, matrix):
    return tuple(sum((vector[i]*matrix[i][j] for i in range(len(vector))), Fraction())
                 for j in range(len(matrix[0])))


def _coordinates(columns, target):
    """Exact, deterministic Gauss-Jordan solve B*c=target or return None."""
    n = len(target)
    r = len(columns)
    tableau = [[columns[j][i] for j in range(r)] + [target[i]] for i in range(n)]
    pivots = []
    pivot_row = 0
    for col in range(r):
        pivot = next((k for k in range(pivot_row, n) if tableau[k][col]), None)
        if pivot is None:
            continue
        tableau[pivot], tableau[pivot_row] = tableau[pivot_row], tableau[pivot]
        scale = tableau[pivot_row][col]
        tableau[pivot_row] = [v/scale for v in tableau[pivot_row]]
        for i in range(n):
            if i != pivot_row and tableau[i][col]:
                factor = tableau[i][col]
                tableau[i] = [v-factor*w for v, w in zip(tableau[i], tableau[pivot_row])]
        pivots.append(col)
        pivot_row += 1
    if any(not any(row[j] for j in range(r)) and row[r] for row in tableau):
        return None
    result = [Fraction()] * r
    for i, col in enumerate(pivots):
        result[col] = tableau[i][r]
    return tuple(result)


@dataclass(frozen=True)
class StochasticSource:
    """Finite hidden-state token generator; rows sum to 1 over ALL actions."""
    actions: tuple[str, ...]
    kernels: tuple[tuple[tuple[Fraction, ...], ...], ...]
    initial: tuple[Fraction, ...]

    def __post_init__(self):
        actions = tuple(self.actions)
        if (not 1 <= len(actions) <= MAX_ACTIONS or
                any(type(a) is not str or not a or len(a)>64 for a in actions) or
                len(set(actions)) != len(actions)):
            raise ValueError("Invalid token alphabet")
        initial = tuple(_r(v) for v in self.initial)
        n = len(initial)
        if not 1 <= n <= MAX_STATES or any(v < 0 for v in initial) or sum(initial) != 1:
            raise ValueError("Invalid initial source distribution")
        try:
            kernels = tuple(tuple(tuple(_r(v) for v in row) for row in matrix)
                            for matrix in self.kernels)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid stochastic source kernels") from exc
        if len(kernels) != len(actions) or any(
            len(mat) != n or any(len(row) != n for row in mat) for mat in kernels
        ) or any(v < 0 for mat in kernels for row in mat for v in row):
            raise ValueError("Invalid source transition geometry")
        if any(sum(sum(mat[i]) for mat in kernels) != 1 for i in range(n)):
            raise ValueError("Source token probabilities are not normalized")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "kernels", kernels)
        object.__setattr__(self, "initial", initial)

    @property
    def digest(self):
        return _hash({"actions":self.actions, "initial":[_q(v) for v in self.initial],
                      "kernels":[_matrix_payload(m) for m in self.kernels]})

    def word_probability(self, word):
        state = self.initial
        for name in word:
            if name not in self.actions:
                raise ValueError("Token outside source alphabet")
            state = _rowmat(state, self.kernels[self.actions.index(name)])
        return sum(state, Fraction())

    def next_distribution(self, word):
        state = self.initial
        for name in word:
            if name not in self.actions:
                raise ValueError("Token outside source alphabet")
            state = _rowmat(state, self.kernels[self.actions.index(name)])
        mass = sum(state, Fraction())
        if not mass:
            raise ValueError("Zero-probability prefix cannot be continued")
        return {a:sum(_rowmat(state, kernel), Fraction())/mass
                for a,kernel in zip(self.actions, self.kernels)}


@dataclass(frozen=True)
class PredictiveGround:
    actions: tuple[str, ...]
    transforms: tuple[tuple[tuple[Fraction, ...], ...], ...]
    initial: tuple[Fraction, ...]
    source_digest: str
    ground_digest: str

    @property
    def rank(self):
        return len(self.initial)

    def material(self):
        return {"schema":SCHEMA, "actions":self.actions, "rank":self.rank,
                "source_digest":self.source_digest,
                "initial":[_q(v) for v in self.initial],
                "transforms":[_matrix_payload(m) for m in self.transforms]}

    def check_identity(self):
        if _hash(self.material()) != self.ground_digest:
            raise ValueError("Ground contents differ from committed identity")


def _construct(source):
    n = len(source.initial)
    columns = [tuple(Fraction(1) for _ in range(n))]
    cursor = 0
    while cursor < len(columns):
        basis = columns[cursor]
        for matrix in source.kernels:
            candidate = _matvec(matrix, basis)
            if _coordinates(columns, candidate) is None:
                columns.append(candidate)
        cursor += 1
    r = len(columns)
    matrices = []
    for source_matrix in source.kernels:
        coeffs = []
        for col in columns:
            solved = _coordinates(columns, _matvec(source_matrix, col))
            if solved is None:
                raise AssertionError("Predictive basis not closed")
            coeffs.append(solved)
        matrices.append(tuple(tuple(coeffs[j][i] for j in range(r)) for i in range(r)))
    initial = tuple(sum((source.initial[i]*column[i] for i in range(n)), Fraction())
                    for column in columns)
    return tuple(columns), tuple(matrices), initial


def compile_predictive_ground(source):
    """Construct a minimal observable linear basis, without enumerating words."""
    if not isinstance(source, StochasticSource):
        raise ValueError("Source requires an exact stochastic transition law")
    columns, matrices, initial = _construct(source)
    material = {"schema":SCHEMA, "actions":source.actions, "rank":len(columns),
                "source_digest":source.digest, "initial":[_q(v) for v in initial],
                "transforms":[_matrix_payload(m) for m in matrices]}
    ground = PredictiveGround(source.actions, matrices, initial, source.digest, _hash(material))
    verify_predictive_ground(source, ground)
    return ground


def verify_predictive_ground(source, ground):
    """Reconstruct source-relative basis and all commutative diagrams independently.

    Source evidence is essential for qualification; sessions need only ground.
    Rehashing changed K matrices cannot bypass this verifier.
    """
    if not isinstance(source, StochasticSource) or not isinstance(ground, PredictiveGround):
        raise ValueError("Unqualified source or predictive Ground")
    ground.check_identity()
    if ground.source_digest != source.digest or ground.actions != source.actions:
        raise ValueError("Wrong source or token alphabet")
    columns, expected_transforms, expected_initial = _construct(source)
    if ground.rank != len(columns) or ground.transforms != expected_transforms or ground.initial != expected_initial:
        raise ValueError("Source-observation closure or minimal basis does not commute")
    for m, k in zip(source.kernels, ground.transforms):
        for j, col in enumerate(columns):
            lhs = _matvec(m, col)
            rhs = tuple(sum((columns[i][x]*k[i][j] for i in range(ground.rank)),Fraction())
                        for x in range(len(col)))
            if lhs != rhs:
                raise ValueError("Source to Virtual Layer transition mismatch")
    if ground.initial[0] != 1:
        raise ValueError("Incorrect predictive initial observation")
    return True


class PredictiveSession:
    """Persistent conditional-token observer using only the compiled Ground."""
    def __init__(self, ground):
        if not isinstance(ground, PredictiveGround):
            raise ValueError("Session requires a compiled predictive Ground")
        ground.check_identity()
        self.ground = ground
        self.state = ground.initial
        self.history = ()
        self.transition_count = 0

    def distribution(self):
        g = self.ground
        if self.state[0] != 1:
            raise ValueError("Session lost normalized predictive coordinate")
        values = {a:_rowmat(self.state, k)[0] for a,k in zip(g.actions,g.transforms)}
        if any(v < 0 for v in values.values()) or sum(values.values()) != 1:
            raise ValueError("Source-dependent predictive law violated normalization")
        return values

    def observe(self, token):
        if token not in self.ground.actions:
            raise ValueError("Unqualified token in virtual sequence")
        i = self.ground.actions.index(token)
        unnormalized = _rowmat(self.state, self.ground.transforms[i])
        p = unnormalized[0]
        if p <= 0:
            raise ValueError("Cannot continue a zero-probability token history")
        self.state = tuple(x/p for x in unnormalized)
        self.history += (token,)
        self.transition_count += 1
        return p

    def snapshot(self):
        self.ground.check_identity()
        body = {"schema":"vli.predictive-session.v1", "ground_digest":self.ground.ground_digest,
                "history":list(self.history), "state":[_q(v) for v in self.state]}
        return {**body, "snapshot_digest":_hash(body)}

    @classmethod
    def restore(cls, ground, snapshot, *, expected_digest):
        if (type(snapshot) is not dict or snapshot.get("snapshot_digest") != expected_digest or
                _hash({k:v for k,v in snapshot.items() if k!="snapshot_digest"}) != expected_digest or
                snapshot.get("schema") != "vli.predictive-session.v1" or
                snapshot.get("ground_digest") != ground.ground_digest):
            raise ValueError("Unpinned or altered predictive session")
        if type(snapshot.get("history")) is not list:
            raise ValueError("Missing certified causal sequence")
        session = cls(ground)
        for token in snapshot["history"]:
            session.observe(token)
        if snapshot["state"] != [_q(v) for v in session.state]:
            raise ValueError("Restored predictive state disagrees with independent replay")
        return session
