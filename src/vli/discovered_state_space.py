"""TIME-008: source-qualified discovery of rational recurrence processors.

No affine/geometric/quadratic/recurrence family is supplied. A finite signed
chronological source prefix selects the first uniquely identifiable exact
linear recurrence, and that order becomes the generated processor dimension.
A separate Cramer-rule verifier reconstructs this discovery without calling
the producer's Gaussian elimination. All predictions remain model-relative.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from itertools import permutations

from .ground import _digest, _require
from .temporal_intelligence import _fraction, _encode, verify_temporal_snapshot
from .authenticated_temporal_source import (
    verified_source_journal, _check_checkpoint_against_signed_source,
    _genesis, _extend,
)
from .intention_state_space import (
    _bounded, _q, _encode_mat, _encode_vec, navigate_relation, MAX_TARGET,
)

SCHEMA = "vl-i.time008.discovered-recurrence-ground.v1"
NAVIGATION = "vl-i.time008.discovered-navigation.v1"
MAX_ORDER = 4
MAX_EVENTS = 96


def _gauss(matrix, rhs):
    """Producer: exact row-reduced Gaussian elimination; singular => refuse."""
    n = len(rhs)
    augmented = [[_bounded(Fraction(a)) for a in row] +
                 [_bounded(Fraction(rhs[i]))] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if augmented[r][col]), None)
        if pivot is None:
            return None
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [_bounded(v / scale) for v in augmented[col]]
        for row in range(n):
            if row != col:
                factor = augmented[row][col]
                augmented[row] = [
                    _bounded(x - factor * y)
                    for x, y in zip(augmented[row], augmented[col])
                ]
    return tuple(row[-1] for row in augmented)


def _determinant(matrix):
    """Verifier: permutation formula, independent of Gaussian pivot logic."""
    n = len(matrix)
    answer = Fraction(0)
    for perm in permutations(range(n)):
        inversions = sum(perm[i] > perm[j]
                         for i in range(n) for j in range(i + 1, n))
        product = Fraction(-1 if inversions % 2 else 1)
        for i, j in enumerate(perm):
            product = _bounded(product * matrix[i][j])
        answer = _bounded(answer + product)
    return answer


def _cramer(matrix, rhs):
    denominator = _determinant(matrix)
    if not denominator:
        return None
    n = len(rhs)
    solution = []
    for col in range(n):
        replaced = [
            [rhs[i] if j == col else matrix[i][j] for j in range(n)]
            for i in range(n)
        ]
        solution.append(_bounded(_determinant(replaced) / denominator))
    return tuple(solution)


def _infer(values, training_count, max_order, solver):
    """Find the smallest full-rank order whose entire training prefix fits."""
    for order in range(1, min(max_order, training_count // 2) + 1):
        matrix = [
            [values[order + row - col - 1] for col in range(order)]
            for row in range(order)
        ]
        outputs = [values[order + row] for row in range(order)]
        coefficients = solver(matrix, outputs)
        if coefficients is None:
            continue
        if all(values[t] == sum(
                (coefficients[j] * values[t - j - 1]
                 for j in range(order)), Fraction(0))
               for t in range(order, training_count)):
            return order, coefficients
    raise ValueError("No identifiable exact recurrence under training/order budget")


def _predict(values, order, coefficients, training_count, validation_count):
    """Use model outputs, never true withheld values, as future regressors."""
    predicted = list(values[:order])
    for idx in range(order, len(values)):
        next_value = _bounded(sum(
            (coefficients[j] * predicted[idx - 1 - j]
             for j in range(order)), Fraction(0)))
        if next_value != values[idx]:
            phase = ("training" if idx < training_count else
                     "validation" if idx < training_count + validation_count
                     else "untouched test")
            raise ValueError("Discovered recurrence falsified by " + phase +
                             " observation " + str(idx))
        predicted.append(next_value)
    return predicted


def _authenticated(snapshot, *, journal_path, source_id, public_key,
                   expected_source_head, expected_source_count,
                   expected_ground_head, expected_ground_identity):
    packets = verified_source_journal(
        journal_path, expected_head=expected_source_head,
        expected_count=expected_source_count, source=source_id,
        public_key=public_key)
    verify_temporal_snapshot(snapshot, expected_head=expected_ground_head,
                             expected_identity=expected_ground_identity)
    _require(type(snapshot["events"]) is list and
             len(snapshot["events"]) == len(packets) and
             0 < len(packets) <= MAX_EVENTS,
             "Unsigned, missing or excessive environmental samples")
    _check_checkpoint_against_signed_source(snapshot, packets, source_id)
    _require(snapshot["contract"]["dimensions"] == 1,
             "TIME-008 requires scalar observations")
    values = tuple(_q(event["values"][0]) for event in snapshot["events"])
    times = tuple(_q(event["time"]) for event in snapshot["events"])
    _require(len(times) > 1 and
             times[1] > times[0] and
             all(times[i] - times[i-1] == times[1] - times[0]
                 for i in range(2, len(times))),
             "Irregular observation time grid; no uniform recurrence coordinate")
    return packets, values, times


def _prior_is_verified(previous, snapshot, packets, values, times, source_id,
                       public_key):
    """Only a single already-admitted first epoch can precede this revision."""
    _require(type(previous) is dict and previous.get("schema") == SCHEMA and
             previous.get("intention_space_digest") == _digest(
                 SCHEMA, {k:v for k,v in previous.items()
                          if k != "intention_space_digest"}),
             "Unqualified earlier Ground")
    old_count = previous["signed_source_count"]
    _require(previous["epoch_start"] == 0 and
             type(old_count) is int and 0 < old_count < len(packets) and
             previous["source_id"] == source_id and
             previous["source_public_key_sha256"] ==
                 sha256(bytes.fromhex(public_key)).hexdigest() and
             previous["ground_identity"] == snapshot["identity"] and
             previous["source_witnesses"] ==
                 [e["digest"] for e in snapshot["events"][:old_count]] and
             previous["ground_head"] ==
                 snapshot["events"][old_count-1]["digest"],
             "Previous Ground does not preserve source prefix")
    head = _genesis(source_id, public_key)
    for packet in packets[:old_count]:
        head = _extend(head, packet)
    _require(previous["signed_source_journal_head"] == head,
             "Earlier source pin cannot be reconstructed")
    order, coeffs = _infer(
        values[:old_count], previous["training_count"],
        previous["max_order"], _cramer)
    _predict(values[:old_count], order, coeffs, previous["training_count"],
             previous["validation_count"])
    _require(order == previous["dimension"] and
             [_encode(c) for c in coeffs] == previous["coefficients"] and
             previous["epoch_sample_count"] == old_count and
             previous["time_origin"] == _encode(times[0]) and
             previous["time_step"] == _encode(times[1]-times[0]),
             "Prior source-qualified processor was altered")
    return old_count


def _build(snapshot, packets, values, times, *, source_id, public_key,
           expected_source_head, training_count, validation_count, test_count,
           max_order, epoch_start, previous_space, solver):
    _require(type(max_order) is int and 1 <= max_order <= MAX_ORDER and
             all(type(x) is int for x in
                 (training_count, validation_count, test_count, epoch_start)) and
             training_count >= 2 and validation_count >= 2 and test_count >= 2 and
             0 <= epoch_start < len(values) and
             training_count + validation_count + test_count ==
                 len(values) - epoch_start,
             "Explicit chronological training/validation/test splits required")
    if previous_space is None:
        _require(epoch_start == 0, "First Ground must start with source genesis")
        prior_digest = None
    else:
        old_count = _prior_is_verified(
            previous_space, snapshot, packets, values, times,
            source_id, public_key)
        _require(epoch_start == old_count,
                 "Revision must begin strictly after preserved old source prefix")
        prior_digest = previous_space["intention_space_digest"]
    epoch_values = values[epoch_start:]
    order, coefficients = _infer(epoch_values, training_count, max_order, solver)
    _predict(epoch_values, order, coefficients, training_count, validation_count)
    first, delta = times[epoch_start], times[1]-times[0]
    matrix = (tuple(coefficients),) + tuple(
        tuple(Fraction(int(j == i-1)) for j in range(order))
        for i in range(1, order))
    seed = tuple(reversed(epoch_values[:order]))
    # Independent simple recurrence replay checks the exact observer output
    # at every source coordinate without hiding a withheld observation.
    body = {
        "schema": SCHEMA,
        "source_id": source_id,
        "signed_source_journal_head": expected_source_head,
        "source_public_key_sha256":
            sha256(bytes.fromhex(public_key)).hexdigest(),
        "signed_source_count": len(packets),
        "ground_identity": snapshot["identity"],
        "ground_head": snapshot["head"],
        "source_witnesses": [event["digest"] for event in snapshot["events"]],
        "previous_space_digest": prior_digest,
        "epoch_start": epoch_start,
        "epoch_sample_count": len(epoch_values),
        "training_count": training_count,
        "validation_count": validation_count,
        "test_count": test_count,
        "max_order": max_order,
        "family": "INFERRED_MINIMAL_FULL_RANK_LINEAR_RECURRENCE",
        "dimension": order,
        "coefficients": [_encode(c) for c in coefficients],
        "transition_operator": _encode_mat(matrix),
        "initial_state": _encode_vec(seed),
        "output_coordinate": 0,
        "time_origin": _encode(first),
        "time_step": _encode(delta),
        "seed_sample_coordinate": order-1,
        "qualified_observation_alphabet": [
            "current"] + ["lag:" + str(i) for i in range(1, order)],
        "heldout_validation": "EXACT_MATCH" ,
        "untouched_test": "EXACT_MATCH",
        "unseen_physical_future_is_known": False,
        "source_model_status":
            "FINITE_CHRONOLOGICAL_EMPIRICAL_FIT_NOT_EXTERNAL_LAW",
        "source_interpreter_grammar_is_external_seed": True,
    }
    return {**body, "intention_space_digest": _digest(SCHEMA, body)}


def construct_discovered_space(
        snapshot, *, journal_path, source_id, public_key,
        expected_source_head, expected_source_count,
        expected_ground_head, expected_ground_identity,
        training_count, validation_count, test_count, max_order=MAX_ORDER,
        epoch_start=0, previous_space=None):
    """Construct a new processor from signed evidence, not a chosen law label."""
    packets, values, times = _authenticated(
        snapshot, journal_path=journal_path, source_id=source_id,
        public_key=public_key, expected_source_head=expected_source_head,
        expected_source_count=expected_source_count,
        expected_ground_head=expected_ground_head,
        expected_ground_identity=expected_ground_identity)
    return _build(
        snapshot, packets, values, times, source_id=source_id,
        public_key=public_key, expected_source_head=expected_source_head,
        training_count=training_count, validation_count=validation_count,
        test_count=test_count, max_order=max_order, epoch_start=epoch_start,
        previous_space=previous_space, solver=_gauss)


def verify_discovered_space(
        snapshot, space, *, journal_path, source_id, public_key,
        expected_source_head, expected_source_count,
        expected_ground_head, expected_ground_identity, previous_space=None):
    """Independent Cramer reconstruction from the authenticated original bytes."""
    _require(type(space) is dict and space.get("schema") == SCHEMA,
             "No declared TIME-008 processor")
    packets, values, times = _authenticated(
        snapshot, journal_path=journal_path, source_id=source_id,
        public_key=public_key, expected_source_head=expected_source_head,
        expected_source_count=expected_source_count,
        expected_ground_head=expected_ground_head,
        expected_ground_identity=expected_ground_identity)
    rebuilt = _build(
        snapshot, packets, values, times, source_id=source_id,
        public_key=public_key, expected_source_head=expected_source_head,
        training_count=space["training_count"],
        validation_count=space["validation_count"],
        test_count=space["test_count"], max_order=space["max_order"],
        epoch_start=space["epoch_start"], previous_space=previous_space,
        solver=_cramer)
    _require(rebuilt == space,
             "Independent recurrence processor reconstruction disagrees")
    return True


def navigate_discovered(space, requested_time, *, output="current",
                        allow_projection=False, reflect=False, depth=2):
    """Two or more typed intention outputs may select one admitted coordinate."""
    _require(type(space) is dict and space.get("schema") == SCHEMA and
             space.get("intention_space_digest") == _digest(
                 SCHEMA, {k:v for k,v in space.items()
                          if k != "intention_space_digest"}) and
             type(requested_time) is str and
             type(output) is str and
             output in space["qualified_observation_alphabet"],
             "Unverified processor or unadmitted intention")
    start, delta = _q(space["time_origin"]), _q(space["time_step"])
    offset = (_q(requested_time)-start)/delta
    _require(offset.denominator == 1 and
             space["seed_sample_coordinate"] <= offset <= MAX_TARGET,
             "Requested coordinate outside admitted state-space")
    index = int(offset)
    lag = 0 if output == "current" else int(output.split(":")[1])
    _require(index >= lag and
             (index < space["epoch_sample_count"] or allow_projection is True),
             "Future model projection requires explicit authorization")
    matrix = tuple(tuple(_q(v) for v in row)
                   for row in space["transition_operator"])
    state = tuple(_q(v) for v in space["initial_state"])
    steps = index - space["seed_sample_coordinate"]
    value, vector, metrics = navigate_relation(matrix, state, steps, lag)
    calc = {"method": "COMPOSE_INFERRED_COMPANION_MATRIX",
            "source_steps_replayed": 0,
            "state_vector": _encode_vec(vector),
            "navigation_metrics": metrics}
    if reflect:
        from .reflective_rational import reflect_intention
        calc["source_reflection"] = reflect_intention(
            {**space, "output_coordinate": lag}, steps,
            expected_value=_encode(value), depth=depth)
    body = {
        "schema": NAVIGATION,
        "intention_space_digest": space["intention_space_digest"],
        "requested_time": requested_time,
        "logical_step_coordinate": index,
        "output": output,
        "value": _encode(value),
        "evidence_status": (
            "SOURCE_OBSERVED_COORDINATE" if
            index < space["epoch_sample_count"] else
            "UNVERIFIED_FUTURE_MODEL_PROJECTION"),
        "calculation": calc,
        "physical_future_observed": False,
    }
    return {**body, "navigation_digest": _digest(NAVIGATION, body)}


def verify_discovered_navigation(snapshot, space, result, *,
                                 previous_space=None, **source_contract):
    """Source-check Ground and independently reproduce interpreted realization."""
    verify_discovered_space(
        snapshot, space, previous_space=previous_space, **source_contract)
    _require(type(result) is dict and result.get("schema") == NAVIGATION,
             "Unqualified TIME-008 navigation receipt")
    calculation = result.get("calculation", {})
    original = navigate_discovered(
        space, result["requested_time"], output=result["output"],
        allow_projection=result.get("evidence_status") ==
            "UNVERIFIED_FUTURE_MODEL_PROJECTION",
        reflect="source_reflection" in calculation,
        depth=(len(calculation["source_reflection"]["reflection_receipts"])-1
               if "source_reflection" in calculation else 0))
    _require(original == result,
             "Reconstructed source-ground navigation does not match")
    return True
