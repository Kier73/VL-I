"""SHAPE-010: monotone, counterexample-driven causal-state Ground expansion.

A fourth *executed and witnessed* native interlayer state extends an existing
SHAPE-009 affine plane to a three-dimensional simplex. This child Ground never
changes its parent. The learned law concerns layer-1 INPUT NORMALIZATION on
hypothetical simplex mixtures, not a new executed recurrent transition, a
full layer, next-token inference, or independent PyTorch parity.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import math
import struct

from .ground import _digest, _hex64, _require
from .causal_hidden_manifold import (
    CausalHiddenManifoldGround, _qualified_capture, _native_energy,
    _bytes, _energy as _parent_energy, EPS, TOL, MAX_EVENTS,
)
from .qwen_layer_chain import _f64_transfer_digest, SCHEMA as CHAIN

SCHEMA = "vl-i.shape010-causal-basis-growth.v1"
PROOF = "vl-i.shape010-three-variable-native-law.v1"
EVENT = "vl-i.shape010-observer-projection.v1"
POSITION = "vl-i.shape010-observer-position.v1"
CHECKPOINT = "vl-i.shape010-causal-growth-checkpoint.v1"
TRAIN = ((0., 0., 0.),
         (1., 0., 0.), (0., 1., 0.), (0., 0., 1.),
         (.5, 0., 0.), (0., .5, 0.), (0., 0., .5),
         (.5, .5, 0.), (.5, 0., .5), (0., .5, .5))
CHALLENGES = ((.17, .13, .11), (.31, .29, .17),
              (.08, .51, .27), (.25, .25, .25))
CANDIDATES = ("constant", "affine", "diagonal_quadratic", "full_quadratic")


def _domain(s, t, u):
    coords = (s, t, u)
    _require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                 for v in coords) and sum(coords) <= 1.,
             "Intention is outside source-qualified four-state simplex")
    return tuple(float(v) for v in coords)


def _vector(basis, p):
    a, b, c, d = basis
    s, t, u = p
    return [w + s * (x - w) + t * (y - w) + u * (z - w)
            for w, x, y, z in zip(a, b, c, d)]


def _energy(coeff, p):
    s, t, u = p
    a, b, c, d, e, f, g, h, i, j = coeff
    return (a*s*s + b*t*t + c*u*u + d*s*t + e*s*u + f*t*u
            + g*s + h*t + i*u + j)


def _fit(energies):
    """Infer ten coefficients solely from ten witnessed native energies."""
    e0 = energies[TRAIN[0]]
    axes = TRAIN[1:4]
    halves = TRAIN[4:7]
    diag = []
    linear = []
    for axis, half in zip(axes, halves):
        a = 2 * (energies[axis] - 2 * energies[half] + e0)
        diag.append(a)
        linear.append(energies[axis] - e0 - a)
    cross = []
    for p, (i, j) in zip(TRAIN[7:], ((0, 1), (0, 2), (1, 2))):
        cross.append(4 * (energies[p] - e0 - .25*diag[i] -
                          .25*diag[j] - .5*linear[i] - .5*linear[j]))
    return (*diag, *cross, *linear, e0)


def _algebra(basis):
    """Independent Gram identity; not supplied to the observation-based fit."""
    anchor = basis[0]
    ds = [[b-a for a, b in zip(anchor, row)] for row in basis[1:]]
    n = len(anchor)
    dot = lambda a, b: math.fsum(x*y for x, y in zip(a, b))/n
    return (dot(ds[0], ds[0]), dot(ds[1], ds[1]),
            dot(ds[2], ds[2]), 2*dot(ds[0], ds[1]),
            2*dot(ds[0], ds[2]), 2*dot(ds[1], ds[2]),
            2*dot(anchor, ds[0]), 2*dot(anchor, ds[1]),
            2*dot(anchor, ds[2]), dot(anchor, anchor))


def _candidate(coeff, kind):
    a, b, c, d, e, f, g, h, i, j = coeff
    if kind == "constant":
        return (0.,)*9 + (j,)
    if kind == "affine":
        return (0.,)*6 + (g, h, i, j)
    if kind == "diagonal_quadratic":
        return (a, b, c, 0., 0., 0., g, h, i, j)
    return coeff


def _scores(coeff, basis, weights, observations):
    scores = {}
    for kind in CANDIDATES:
        hypothesis = _candidate(coeff, kind)
        worst = 0.
        for p, native in observations:
            x = _vector(basis, p)
            energy = _energy(hypothesis, p)
            if not math.isfinite(energy) or energy + EPS <= 0:
                worst = float("inf")
                break
            scale = 1 / math.sqrt(energy + EPS)
            worst = max(worst, max(abs(z*(1+w)*scale-y)
                                   for z, w, y in zip(x, weights, native)))
        scores[kind] = worst
    return scores


def _read_whole(store, manifest, width):
    from mailcpu.computational_objects import ComputationalObjectManifest
    raw = store.read_bytes(ComputationalObjectManifest.from_dict(manifest),
                           max_bytes=65536)
    _require(len(raw) == width*8, "Unqualified source causal object width")
    return struct.unpack("<" + "d"*width, raw), raw


class ExpandedCausalGround:
    """Source revision 4 of a parent SHAPE-009 Ground, independently pinned."""

    def __init__(self, parent, fourth_report, *, expected_parent_checkpoint_digest):
        from mailcpu.structural_memory_pages import StructuralMemoryPageBacking
        from mailcpu.computational_objects import ComputationalObjectStore
        _require(isinstance(parent, CausalHiddenManifoldGround) and
                 _hex64(expected_parent_checkpoint_digest),
                 "An externally pinned SHAPE-009 source Ground is required")
        saved = parent.checkpoint()
        _require(saved["checkpoint_digest"] == expected_parent_checkpoint_digest,
                 "SHAPE-009 predecessor Ground changed")
        # Re-qualify the parent using its original E-CPU bytes and original chain,
        # not the mutable runtime object's assertions.
        self.parent_checkpoint = deepcopy(saved)
        self.parent = CausalHiddenManifoldGround.restore(
            saved, expected_head=expected_parent_checkpoint_digest)
        hidden, rec = _qualified_capture(
            fourth_report, self.parent.width, 3, self.parent.final_chain_head,
            model=self.parent.model, revision=self.parent.revision,
            index_ground_digest=self.parent.index_ground_digest)
        self.fourth_witness = rec
        self.rank_probe = self.parent.off_manifold(hidden)
        _require(self.rank_probe["relative_residual"] > 1e-7 and
                 not self.rank_probe["inside_declared_convex_hull"],
                 "Actual new causal position did not expand the affine rank")
        self.width = self.parent.width
        self.original_pretrained = self.parent.original_pretrained
        self.backing = StructuralMemoryPageBacking()
        self.store = ComputationalObjectStore(
            self.backing, chunk_size=4096, max_object_bytes=65536)
        payload = _bytes(hidden)
        manifest = self.store.put_bytes(payload)
        _require(self.store.verify(manifest), "New causal E-CPU object unavailable")
        self.fourth_object = manifest.to_dict()
        self.fourth_sha256 = sha256(payload).hexdigest()
        self._establish_source()
        basis = self._basis()
        weights = self.parent.norm_weights
        training = {}
        challenges = []
        for p in TRAIN:
            _, energy = _native_energy(_vector(basis, p), weights)
            training[p] = energy
        coeff = _fit(training)
        for p in CHALLENGES:
            y, energy = _native_energy(_vector(basis, p), weights)
            challenges.append((p, y))
        scores = _scores(coeff, basis, weights, challenges)
        algebra = _algebra(basis)
        _require(max(abs(x-y) for x, y in zip(coeff, algebra)) <= 3e-8 and
                 scores["full_quadratic"] <= TOL and
                 all(scores[k] > TOL for k in CANDIDATES[:-1]),
                 "Original observed states do not uniquely qualify degree-2 Ground")
        self.coefficients = tuple(coeff)
        self.proof = {
            "schema": PROOF, "source_root": self.source_root,
            "native_training_probes": [list(p) for p in TRAIN],
            "native_challenges": [list(p) for p in CHALLENGES],
            "observed_training_energies": [training[p] for p in TRAIN],
            "inferred_coefficients": list(coeff),
            "candidate_max_output_error": scores,
            "independent_gram_coefficients": list(algebra),
            "native_norm_observations": len(TRAIN) + len(CHALLENGES),
            "source_revision": 4,
            "admitted_scope": "hypothetical_convex_mixtures_of_four_executed_causal_states",
            "full_layer_or_future_token_claim": False,
        }
        self.proof["proof_digest"] = _digest(PROOF, self.proof)
        self._establish_ground()
        self.events = []
        self.head = self.genesis

    def _establish_source(self):
        self.source_root = _digest(SCHEMA, {
            "parent_ground": self.parent.ground_digest,
            "parent_checkpoint": self.parent_checkpoint["checkpoint_digest"],
            "parent_native_causal_head": self.parent.final_chain_head,
            "fourth_causal_witness": self.fourth_witness,
            "fourth_ECPU_manifest": self.fourth_object["manifest_digest"],
            "fourth_F64_sha256": self.fourth_sha256,
            "source_revision": 4})

    def _establish_ground(self):
        self.ground_digest = _digest(SCHEMA, {
            "source_root": self.source_root,
            "proof": self.proof["proof_digest"]})
        self.genesis = _digest(POSITION, {
            "ground": self.ground_digest, "observer": self.parent.observer,
            "genesis": "new_true_causal_state_enlarges_simplex"})

    def _basis(self):
        parent_rows = []
        for i in range(3):
            vector, _ = _read_whole(self.parent.store,
                                    self.parent.objects[f"causal_{i}"], self.width)
            parent_rows.append(vector)
        fourth, _ = _read_whole(self.store, self.fourth_object, self.width)
        return (*parent_rows, fourth)

    def _read_fourth(self, indices):
        from mailcpu.computational_objects import ComputationalObjectManifest
        m = ComputationalObjectManifest.from_dict(self.fourth_object)
        pages = {}
        out = []
        for index in indices:
            page = (index*8)//m.chunk_size
            if page not in pages:
                first = page*m.chunk_size
                pages[page] = self.store.read_range(
                    m, first, min(m.chunk_size, m.size_bytes-first))
            off = index*8-page*m.chunk_size
            out.append(struct.unpack("<d", pages[page][off:off+8])[0])
        return out, len(pages)

    def _form(self, p, indices):
        a, pa = self.parent._read("causal_0", indices)
        b, pb = self.parent._read("causal_1", indices)
        c, pc = self.parent._read("causal_2", indices)
        d, pd = self._read_fourth(indices)
        gain, pg = self.parent._read("gain", indices)
        energy = _energy(self.coefficients, p)
        _require(math.isfinite(energy) and energy + EPS > 0,
                 "Unqualified learned three-variable energy")
        s, t, u = p
        scale = 1 / math.sqrt(energy + EPS)
        values = [(w+s*(x-w)+t*(y-w)+u*(z-w))*g*scale
                  for w, x, y, z, g in zip(a, b, c, d, gain)]
        _require(all(math.isfinite(x) for x in values),
                 "Nonfinite learned observer projection")
        return values, pa+pb+pc+pd+pg

    def generate(self, s, t, u, indices, *, expected_head):
        p = _domain(s, t, u)
        _require(type(indices) in (tuple, list) and
                 1 <= len(indices) <= self.width and
                 all(type(x) is int and 0 <= x < self.width for x in indices) and
                 len(set(indices)) == len(indices) and
                 expected_head == self.head and len(self.events) < MAX_EVENTS,
                 "Unqualified projection or stale causal observer head")
        ix = list(indices)
        coordinate = {"ground": self.ground_digest, "source_revision": 4,
                      "native_causal_head": self.fourth_witness["chain_head"],
                      "mixture": list(p), "indices": ix, "previous": self.head}
        position = "state://vli/" + _digest(POSITION, coordinate)
        values, pages = self._form(p, ix)
        event = {"schema": EVENT, "previous": self.head, "position": position,
                 "mixture": list(p), "indices": ix,
                 "output_sha256": sha256(_bytes(values)).hexdigest(),
                 "verified_ECPU_pages": pages,
                 "native_calls_during_generation": 0,
                 "scope": "hypothetical_normalization_not_causal_continuation"}
        event["event_digest"] = _digest(EVENT, event)
        self.events.append(event)
        self.head = event["event_digest"]
        return {"position": position, "values": values,
                "event": deepcopy(event), "source_revision": 4}

    def native_reference(self, s, t, u, indices):
        from experiments.model_fabric.trained_subgraphs import centered_rms_norm
        p = _domain(s, t, u)
        _require(type(indices) in (tuple, list) and
                 len(indices) >= 1 and
                 all(type(i) is int and 0 <= i < self.width for i in indices),
                 "Invalid native reference observer projection")
        y = centered_rms_norm(_vector(self._basis(), p),
                              self.parent.norm_weights)
        return [y[i] for i in indices]

    def checkpoint(self):
        body = {"schema": CHECKPOINT, "parent_checkpoint":
                deepcopy(self.parent_checkpoint),
                "fourth_witness": deepcopy(self.fourth_witness),
                "rank_probe": deepcopy(self.rank_probe),
                "fourth_ECPU_backing": self.backing.snapshot(),
                "fourth_ECPU_object": deepcopy(self.fourth_object),
                "fourth_sha256": self.fourth_sha256,
                "source_root": self.source_root,
                "coefficients": list(self.coefficients),
                "proof": deepcopy(self.proof),
                "ground_digest": self.ground_digest,
                "events": deepcopy(self.events), "head": self.head}
        return {**body, "checkpoint_digest": _digest(CHECKPOINT, body)}

    @classmethod
    def restore(cls, saved, *, expected_head):
        from mailcpu.structural_memory_pages import StructuralMemoryPageBacking
        from mailcpu.computational_objects import ComputationalObjectStore
        _require(type(saved) is dict and _hex64(expected_head) and
                 saved.get("schema") == CHECKPOINT and
                 saved.get("checkpoint_digest") == expected_head and
                 _digest(CHECKPOINT, {k:v for k,v in saved.items()
                                      if k != "checkpoint_digest"}) == expected_head,
                 "Externally pinned SHAPE-010 checkpoint missing or altered")
        obj = cls.__new__(cls)
        obj.parent_checkpoint = deepcopy(saved["parent_checkpoint"])
        obj.parent = CausalHiddenManifoldGround.restore(
            obj.parent_checkpoint,
            expected_head=obj.parent_checkpoint["checkpoint_digest"])
        obj.width = obj.parent.width
        obj.original_pretrained = obj.parent.original_pretrained
        obj.fourth_witness = deepcopy(saved["fourth_witness"])
        obj.backing = StructuralMemoryPageBacking.from_snapshot(
            saved["fourth_ECPU_backing"])
        obj.store = ComputationalObjectStore(
            obj.backing, chunk_size=4096, max_object_bytes=65536)
        obj.fourth_object = deepcopy(saved["fourth_ECPU_object"])
        obj.fourth_sha256 = saved["fourth_sha256"]
        hidden, raw = _read_whole(obj.store, obj.fourth_object, obj.width)
        rec = obj.fourth_witness
        ws = rec["semantic_layer_witnesses"]
        _require(sha256(raw).hexdigest() == obj.fourth_sha256 and
                 _f64_transfer_digest(hidden) ==
                     rec["layer0_output_f64_digest"] ==
                     rec["layer1_input_f64_digest"] and
                 rec["position"] == 3 and
                 rec["previous_head"] == obj.parent.final_chain_head and
                 rec["native_complete_layers"] is True and
                 type(rec["original_token_id"]) is int and
                 type(ws) is list and len(ws) == 2 and
                 [w["layer"] for w in ws] == [0, 1] and
                 all(w["position"] == 3 and
                     w["complete_output_width"] == obj.width for w in ws) and
                 ws[0]["output_f64_digest"] == rec["layer0_output_f64_digest"] and
                 ws[1]["input_f64_digest"] == rec["layer1_input_f64_digest"] and
                 ws[1]["output_f64_digest"] == rec["layer1_output_f64_digest"] and
                 ws[0]["recurrent_state_digest"] ==
                     rec["layer0_recurrent_state_digest"] and
                 ws[1]["recurrent_state_digest"] ==
                     rec["layer1_recurrent_state_digest"] and
                 ws[1]["convolution_state_digest"] ==
                     rec["layer1_convolution_state_digest"],
                 "Fourth native causal state contradicts source ECPU object or receipts")
        semantic = {"schema": CHAIN, "model": obj.parent.model,
                    "revision": obj.parent.revision,
                    "index_ground_digest": obj.parent.index_ground_digest,
                    "position": 3, "original_token_id": rec["original_token_id"],
                    "previous": rec["previous_head"], "layer_witnesses": ws}
        _require(_digest(CHAIN, semantic) == rec["chain_head"],
                 "Fourth actual native causal chain cannot reconstruct")
        obj.rank_probe = obj.parent.off_manifold(hidden)
        _require(obj.rank_probe == saved["rank_probe"] and
                 obj.rank_probe["relative_residual"] > 1e-7 and
                 not obj.rank_probe["inside_declared_convex_hull"],
                 "Causal affine rank expansion cannot reconstruct")
        obj._establish_source()
        _require(obj.source_root == saved["source_root"],
                 "Causal source revision is not monotone")
        obj.coefficients = tuple(saved["coefficients"])
        obj.proof = deepcopy(saved["proof"])
        basis = obj._basis()
        algebra = _algebra(basis)
        training = {p:_energy(algebra,p) for p in TRAIN}
        fitted = _fit(training)
        _require(max(abs(a-b) for a,b in zip(algebra,obj.coefficients)) <= 3e-8
                 and max(abs(a-b) for a,b in zip(fitted,obj.coefficients)) <= 3e-8
                 and obj.proof["source_root"] == obj.source_root
                 and obj.proof["schema"] == PROOF
                 and obj.proof["source_revision"] == 4
                 and obj.proof["native_training_probes"] == [list(p) for p in TRAIN]
                 and obj.proof["native_challenges"] == [list(p) for p in CHALLENGES]
                 and obj.proof["native_norm_observations"] == 14
                 and obj.proof["admitted_scope"] ==
                    "hypothetical_convex_mixtures_of_four_executed_causal_states"
                 and obj.proof["full_layer_or_future_token_claim"] is False
                 and obj.proof["inferred_coefficients"] == list(obj.coefficients)
                 and all(abs(a-b) <= 3e-8 for a,b in
                     zip(obj.proof["observed_training_energies"],
                         [training[p] for p in TRAIN]))
                 and max(abs(a-b) for a,b in
                     zip(obj.proof["independent_gram_coefficients"],algebra)) <= 3e-8
                 and obj.proof["proof_digest"] ==
                    _digest(PROOF, {k:v for k,v in obj.proof.items()
                                    if k != "proof_digest"}),
                 "Observation-based expansion law or saved native probes invalid")
        # Recheck candidate falsification against an independent Gram-energy
        # reference without rerunning the original trained native operator.
        refs = []
        for p in CHALLENGES:
            x = _vector(basis,p)
            scale = 1/math.sqrt(_energy(algebra,p)+EPS)
            refs.append((p,[v*(1+w)*scale
                            for v,w in zip(x,obj.parent.norm_weights)]))
        scores = _scores(obj.coefficients, basis, obj.parent.norm_weights, refs)
        _require(all(abs(obj.proof["candidate_max_output_error"][k]-scores[k])
                     <= 3e-8 for k in CANDIDATES) and
                 scores["full_quadratic"] <= TOL and
                 all(scores[k] > TOL for k in CANDIDATES[:-1]),
                 "Recovered false-hypothesis counterexamples no longer hold")
        obj._establish_ground()
        _require(obj.ground_digest == saved["ground_digest"],
                 "Expanded causal Ground identity changed")
        obj.events = deepcopy(saved["events"])
        obj.head = obj.genesis
        for event in obj.events:
            _require(type(event) is dict and event.get("schema") == EVENT and
                     event.get("previous") == obj.head and
                     event.get("native_calls_during_generation") == 0 and
                     event.get("scope") ==
                        "hypothetical_normalization_not_causal_continuation" and
                     event.get("event_digest") == _digest(EVENT, {
                         k:v for k,v in event.items() if k != "event_digest"}),
                     "Recovered causal observer event forged")
            p = _domain(*event["mixture"])
            ix = event["indices"]
            _require(type(ix) is list and ix and
                     len(ix) == len(set(ix)) and
                     all(type(i) is int and 0 <= i < obj.width for i in ix),
                     "Recovered requested projection invalid")
            coordinate = {"ground": obj.ground_digest, "source_revision": 4,
                          "native_causal_head": obj.fourth_witness["chain_head"],
                          "mixture": list(p), "indices": ix,
                          "previous": obj.head}
            values, pages = obj._form(p, ix)
            _require(event["position"] == "state://vli/" +
                     _digest(POSITION, coordinate) and
                     event["verified_ECPU_pages"] == pages and
                     event["output_sha256"] == sha256(_bytes(values)).hexdigest(),
                     "Reconstructed observer response or position not reproducible")
            obj.head = event["event_digest"]
        _require(obj.head == saved["head"],
                 "Expanded causal observer head differs after recovery")
        return obj
