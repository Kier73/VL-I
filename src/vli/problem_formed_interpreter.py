"""PFI-001: construct a problem-shaped interpreter from declared dynamics and intention.

This bridge composes NCR-006 (future-observation-minimal mathematical quotient)
with an independently auditable XOR microarchitecture and E-CPU's *existing*
content-addressed DAG. The host executes arithmetic; communication links are
not claimed to compute. Neither a ground digest nor a DAG commit grants
Hypervisor authority. Only fully declared stationary binary-affine laws are
supported in this first exact experiment.
"""
from __future__ import annotations

from hashlib import sha256
import json

SCHEMA = 'vl-i.problem-formed-interpreter.v1'
MIGRATION = 'vl-i.problem-formed-interpreter-succession.v1'
MAX_GATES = 20_000
MAX_REPEATS_BITS = 1024


def _require(ok, why):
    if not ok:
        raise ValueError(why)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def _hash(value):
    return sha256(_json(value)).hexdigest()


def _math():
    try:
        from navigable_compute.symbolic_quotient import construct_symbolic_quotient, navigate_symbolic
        from navigable_compute.symbolic_quotient_verify import verify_symbolic_quotient, verify_symbolic_result
    except ImportError as exc:
        raise RuntimeError('PFI-001 requires pinned Navigable-Compute NCR-006') from exc
    return construct_symbolic_quotient, navigate_symbolic, verify_symbolic_quotient, verify_symbolic_result


def _network(rows, bias, dimension):
    """Produce an explicit, executable dataflow circuit from a GF(2) affine map."""
    _require(type(dimension) is int and 0 <= dimension <= 64, 'Invalid quotient dimension')
    _require(type(rows) is tuple and all(type(v) is int and 0 <= v < 1 << dimension for v in rows),
             'Invalid affine-row representation')
    _require(type(bias) is int and 0 <= bias < 1 << len(rows), 'Invalid output bias')
    nodes, outputs, pairs, flips = [], [], {}, {}

    def xor(a, b):
        if a == 'zero':
            return b
        if b == 'zero':
            return a
        key = tuple(sorted((a, b)))
        if key not in pairs:
            name = 'g' + str(len(nodes))
            nodes.append({'id': name, 'op': 'xor', 'args': list(key)})
            pairs[key] = name
        return pairs[key]

    def flip(a):
        if a not in flips:
            name = 'g' + str(len(nodes))
            nodes.append({'id': name, 'op': 'flip', 'args': [a]})
            flips[a] = name
        return flips[a]

    for i, row in enumerate(rows):
        terms = ['x' + str(j) for j in range(dimension) if row & (1 << j)]
        out = 'zero'
        for term in terms:
            out = xor(out, term)
        if bias & (1 << i):
            out = flip(out)
        outputs.append(out)
        _require(len(nodes) <= MAX_GATES, 'Constructed microarchitecture exceeds gate budget')
    return {'input_bits': dimension, 'nodes': nodes, 'outputs': outputs,
            'primitive_set': sorted({node['op'] for node in nodes})}


def construct(source, *, max_dimension=64):
    """Form registers, machine operations and an observer from the problem itself.

    The only supplied architecture is the *source's causal law*. No quotient
    dimension, native operations, or observer decoder is provided by the caller.
    """
    build, _, verify, _ = _math()
    quotient = build(source, max_dimension=max_dimension)
    _require(verify(source, quotient), 'NCR independent mathematical verifier refused')
    transitions = [_network(op.rows, op.bias, quotient.dimension)
                   for op in quotient.operators]
    observation = _network(quotient.observer_decoder, 0, quotient.dimension)
    body = {'schema': SCHEMA, 'source': source.encode(),
            'source_sha256': source.digest, 'quotient': quotient.packet(),
            'source_bits': source.width, 'register_bits': quotient.dimension,
            'transitions': transitions, 'observer': observation,
            'primitive_set': sorted({p for graph in [*transitions, observation]
                                     for p in graph['primitive_set']}),
            'claim': 'finite-declared-affine-exact;no-external-host-authority'}
    body['digest'] = _hash(body)
    _require(verify_interpreter(source, quotient, body), 'Constructed interpreter did not verify')
    return quotient, body


def _symbolic_network(program):
    """Verify each generated gate as a symbolic affine function of its inputs.

    This checker does not call the generator; it propagates exact symbolic masks
    and constant terms through the supplied microarchitecture.
    """
    _require(type(program) is dict and set(program) ==
             {'input_bits', 'nodes', 'outputs', 'primitive_set'}, 'Malformed circuit')
    n = program['input_bits']
    _require(type(n) is int and 0 <= n <= 64, 'Invalid register size')
    _require(type(program['nodes']) is list and len(program['nodes']) <= MAX_GATES and
             type(program['outputs']) is list and
             type(program['primitive_set']) is list, 'Malformed circuit arrays')
    state = {'zero': (0, 0)}
    state.update({'x' + str(i): (1 << i, 0) for i in range(n)})
    used = set()
    for pos, node in enumerate(program['nodes']):
        _require(type(node) is dict and set(node) == {'id','op','args'} and
                 node['id'] == 'g' + str(pos) and node['id'] not in state and
                 type(node['args']) is list, 'Invalid circuit node identity')
        args = node['args']
        _require(all(type(a) is str and a in state for a in args),
                 'Non-topological circuit or unknown input')
        op = node['op']
        _require(op in ('xor','flip'), 'Unauthorized primitive')
        if op == 'xor':
            _require(len(args) == 2, 'XOR arity mismatch')
            x, y = (state[a] for a in args)
            value = (x[0] ^ y[0], x[1] ^ y[1])
        else:
            _require(len(args) == 1, 'FLIP arity mismatch')
            x = state[args[0]]
            value = (x[0], x[1] ^ 1)
        state[node['id']] = value
        used.add(op)
    _require(program['primitive_set'] == sorted(used), 'Primitive manifest altered')
    _require(all(type(out) is str and out in state for out in program['outputs']),
             'Unbound circuit output')
    return tuple(state[out][0] for out in program['outputs']), sum(
        (state[out][1] << i) for i, out in enumerate(program['outputs']))


def verify_interpreter(source, quotient, bundle):
    """Check source identity, independent NCR proof and every synthesized gate."""
    _, _, mathematical_check, _ = _math()
    try:
        if (type(bundle) is not dict or bundle.get('schema') != SCHEMA or
            bundle.get('digest') != _hash({k:v for k,v in bundle.items() if k != 'digest'}) or
            bundle.get('source') != source.encode() or
            bundle.get('source_sha256') != source.digest or
            bundle.get('quotient') != quotient.packet() or
            bundle.get('source_bits') != source.width or
            bundle.get('register_bits') != quotient.dimension or
            bundle.get('claim') != 'finite-declared-affine-exact;no-external-host-authority' or
            not mathematical_check(source, quotient)):
            return False
        ts = bundle['transitions']
        if type(ts) is not list or len(ts) != len(quotient.operators):
            return False
        seen = set()
        for program, operator in zip(ts, quotient.operators):
            rows, bias = _symbolic_network(program)
            if (program['input_bits'] != quotient.dimension or
                rows != operator.rows or bias != operator.bias):
                return False
            seen.update(program['primitive_set'])
        read_rows, read_bias = _symbolic_network(bundle['observer'])
        seen.update(bundle['observer']['primitive_set'])
        return (read_rows == quotient.observer_decoder and read_bias == 0 and
                bundle['observer']['input_bits'] == quotient.dimension and
                bundle['primitive_set'] == sorted(seen) and
                sum(len(g['nodes']) for g in ts) + len(bundle['observer']['nodes']) <=
                MAX_GATES * (len(ts) + 1))
    except (TypeError, ValueError, KeyError, IndexError, AttributeError, OverflowError):
        return False


def execute_network(program, input_state):
    """Run the *generated* gate network rather than directly using a source law."""
    n = program['input_bits']
    _require(type(input_state) is int and 0 <= input_state < 1 << n, 'Wrong register value')
    values = {'zero': 0}
    values.update({'x' + str(i): (input_state >> i) & 1 for i in range(n)})
    for node in program['nodes']:
        args = node['args']
        values[node['id']] = (values[args[0]] ^ values[args[1]]
                              if node['op'] == 'xor' else 1 ^ values[args[0]])
    return sum(values[name] << i for i, name in enumerate(program['outputs']))


def realize(source, quotient, bundle, initial, word, repetitions):
    """Choose execution *structure* by bounded operation cost and verify result.

    For short continuations, run the synthesized gate machine. For long ones,
    navigate its composed exact law via NCR-006 without repeated stepping.
    """
    _require(verify_interpreter(source, quotient, bundle), 'Unverified synthesized interpreter')
    _require(type(initial) is int and 0 <= initial < (1 << source.width) and
             type(word) is tuple and len(word) <= 1024 and
             all(type(i) is int and 0 <= i < len(bundle['transitions']) for i in word) and
             type(repetitions) is int and repetitions >= 0 and
             repetitions.bit_length() <= MAX_REPEATS_BITS,
             'Out-of-contract intention or source state')
    _, navigate, _, verify_result = _math()
    initial_native = quotient.project(initial)
    gate_cost = sum(len(bundle['transitions'][i]['nodes']) + quotient.dimension
                    for i in word)
    estimated_steps = gate_cost * repetitions
    algebra_budget = max(1, len(word)) * (quotient.dimension + 1) ** 2 + \
        max(1, repetitions.bit_length()) * (quotient.dimension + 1) ** 3
    if estimated_steps <= algebra_budget and repetitions * len(word) <= 5000:
        final_native = initial_native
        for _ in range(repetitions):
            for i in word:
                final_native = execute_network(bundle['transitions'][i], final_native)
        strategy = 'constructed-gate-interpreter'
        executed_gates = estimated_steps
    else:
        strategy = 'composed-native-ground-navigation'
        executed_gates = 0
        final_native = None
    oracle = navigate(source, quotient, initial, word, repetitions)
    _require(verify_result(source, quotient, oracle), 'Independent NCR distant-result verifier refused')
    if final_native is None:
        final_native = int(oracle.final_projection_decimal)
    _require(final_native == int(oracle.final_projection_decimal),
             'Generated interpreter disagrees with source mathematical realization')
    output = execute_network(bundle['observer'], final_native)
    _require(output == int(oracle.observed_decimal), 'Generated observer decoder failed')
    record = {'schema': 'vl-i.problem-formed-result.v1',
              'interpreter_digest': bundle['digest'], 'source_sha256': source.digest,
              'initial_decimal': str(initial), 'word': list(word),
              'repetitions_decimal': str(repetitions),
              'strategy': strategy, 'input_native_decimal': str(initial_native),
              'result_native_decimal': str(final_native), 'observed_decimal': str(output),
              'ncr_result_sha256': oracle.digest,
              'cost': {'constructed_gate_steps': executed_gates,
                       'estimated_direct_steps': estimated_steps,
                       'estimated_power_work': algebra_budget,
                       'ncr_operator_products': oracle.operator_products}}
    record['digest'] = _hash(record)
    return record


def verify_realization(source, quotient, bundle, receipt):
    """Recompute the scoped result independently of the supplied receipt."""
    try:
        if (type(receipt) is not dict or
            receipt.get('digest') != _hash({k:v for k,v in receipt.items() if k != 'digest'}) or
            receipt.get('interpreter_digest') != bundle['digest'] or
            receipt.get('source_sha256') != source.digest):
            return False
        expected = realize(source, quotient, bundle,
                           int(receipt['initial_decimal']),tuple(receipt['word']),
                           int(receipt['repetitions_decimal']))
        return receipt == expected
    except (ValueError, TypeError, KeyError, OverflowError):
        return False


def _span_express(row, basis):
    """Represent a covector in the *old* quotient space, or refuse."""
    pivot = {v.bit_length()-1: (v,i) for i,v in enumerate(basis) if v}
    target = row
    coord = 0
    for j in sorted(pivot, reverse=True):
        if target & (1 << j):
            target ^= pivot[j][0]
            coord ^= 1 << pivot[j][1]
    return coord if target == 0 else None


def certify_succession(old_source, old_quotient, old, new_source,
                        new_quotient, new, *, old_native_state, source_anchor=None):
    """Re-form an interpreter under a new intention, retaining exact old identity.

    If the new observation requires bits absent from the old registers, the
    new registers cannot be reconstructed from the old state alone: require an
    explicit source-state anchor and label it as declared, not authenticated.
    """
    _require(verify_interpreter(old_source, old_quotient, old) and
             verify_interpreter(new_source, new_quotient, new),
             'Missing admission of predecessor or successor')
    _require(old_source.width == new_source.width and
             old_source.generators == new_source.generators and
             old_source.origin == new_source.origin, 'Different source-law epoch')
    _require(type(old_native_state) is int and 0 <= old_native_state < 1 << old_quotient.dimension,
             'Missing retained processor state')
    coordinates = [_span_express(row, old_quotient.basis_rows)
                   for row in new_quotient.basis_rows]
    if all(v is not None for v in coordinates):
        new_native = sum(((row & old_native_state).bit_count() & 1) << i
                         for i,row in enumerate(coordinates))
        transition = 'mathematically-derived-quotient-map'
    else:
        _require(type(source_anchor) is int and
                 0 <= source_anchor < 1 << old_source.width and
                 old_quotient.project(source_anchor) == old_native_state,
                 'Refinement needs source-state anchor consistent with old coordinate')
        new_native = new_quotient.project(source_anchor)
        transition = 'declared-source-anchor-required'
    body = {'schema': MIGRATION, 'from_digest': old['digest'],
            'to_digest': new['digest'], 'old_native_decimal': str(old_native_state),
            'new_native_decimal': str(new_native), 'transition': transition,
            'map_rows_decimal': ([str(row) for row in coordinates]
                                 if all(v is not None for v in coordinates) else None),
            'source_anchor_decimal': (str(source_anchor) if transition ==
                                      'declared-source-anchor-required' else None),
            'scope': 'declared-source-state-only;no-independent-sensor-attestation'}
    body['digest'] = _hash(body)
    return body


def verify_succession(old_source, old_quotient, old, new_source,
                       new_quotient, new, certificate):
    try:
        if (certificate.get('digest') != _hash({k:v for k,v in certificate.items()
                                               if k != 'digest'}) or
            certificate['schema'] != MIGRATION):
            return False
        restored = certify_succession(old_source, old_quotient, old,
                                     new_source, new_quotient, new,
                                     old_native_state=int(certificate['old_native_decimal']),
                                     source_anchor=(int(certificate['source_anchor_decimal'])
                                        if certificate['source_anchor_decimal'] is not None else None))
        return restored == certificate
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def create_ecpu_dag(bundle, *, initial, word, repetitions):
    """Use native E-CPU content-addressed dependency DAG; no duplicate scheduler."""
    from mailcpu.computation_dag import ComputationNode, NavigableComputationDAG
    _require(type(initial) is int and type(repetitions) is int and
             type(word) is tuple, 'Malformed native execution request')
    common = dict(revision=1, operation='pfi.read-or-execute',
                  resource='/vl-i/problem-formed-interpreter',
                  context_digest=bundle['digest'])
    nodes = [
        ComputationNode(node_id='ground', dependencies=(),
                        payload={'interpreter_digest': bundle['digest']}, **common),
        ComputationNode(node_id='navigate', dependencies=('ground',),
                        payload={'initial_decimal': str(initial), 'word': list(word),
                                 'repetitions_decimal': str(repetitions)}, **common),
        ComputationNode(node_id='observe', dependencies=('navigate',),
                        payload={'intention':'read-declared-observer'}, **common),
    ]
    return NavigableComputationDAG(nodes)


def run_one_ecpu_frontier(source, quotient, bundle, dag):
    """Receiver-host-local, one node/quantum. Can be resumed from DAG snapshot.

    Mathematical computation occurs in this invoking process, NOT in a passive
    link; E-CPU DAG validates dependencies and commits but provides no automatic
    execution capability, Hypervisor authorization or mTLS transport here.
    """
    from mailcpu.computation_dag import NavigableComputationDAG
    _require(isinstance(dag, NavigableComputationDAG) and
             verify_interpreter(source, quotient, bundle), 'Invalid native host/ground')
    ready = dag.ready_nodes()
    if not ready:
        return False
    node = ready[0]
    _require(node.context_digest == bundle['digest'], 'Stale generated-processor identity')
    if node.node_id == 'ground':
        result = {'interpreter_digest': bundle['digest'],
                  'source_sha256': source.digest}
    elif node.node_id == 'navigate':
        _require(dag.result('ground')['interpreter_digest'] == bundle['digest'],
                 'Unpinned predecessor ground')
        payload = node.payload
        claimed = realize(source, quotient, bundle, int(payload['initial_decimal']),
                          tuple(payload['word']), int(payload['repetitions_decimal']))
        result = {'receipt': claimed}
    elif node.node_id == 'observe':
        claimed = dag.result('navigate')['receipt']
        _require(verify_realization(source, quotient, bundle, claimed),
                 'Native predecessor result failed independent re-evaluation')
        result = {'observed_decimal': claimed['observed_decimal'],
                  'receipt_digest': claimed['digest']}
    else:
        raise ValueError('Unexpected E-CPU compute node: ' + node.node_id)
    dag.commit(node.node_id, result)
    return True
