"""PBG-006: a source-qualified *finite neural trace ensemble* for PBG-005.

QwenCausalGDNChain executes original-topology trained numerical transitions.
SHAPE-009's _qualified_capture audits each complete two-layer causal report.
A declared observer turns full-F64 native outputs into categorical events.
An empirical uniform ensemble of independently acquired, finite trajectories
induces an *exact rational* absorbing trace law, learned with PBG-005 Hankel
queries. The law is over these witnessed tapes, NOT the complete neural source.

The inference reader sees only event-sequence probabilities, not original
weights, native recurrent state, or hidden ControlledSource matrices. Source
matrices are built separately from the observed tape corpus for qualification.
Neither token-ID labels nor arbitrary input vectors are a verified tokenizer.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as Q
import math
from typing import Any

from .source_hankel_ground import (
    QueryAtlas, SourceObservation, digest, encode,
    certify_source_equivalence,
)

SCHEMA = "vli.pbg006.native-causal-trace.v1"
MAX_TRAJECTORIES = 12
MAX_POSITIONS = 12
ALPHABET = ("act:step", "sense:readout:0", "sense:readout:1",
            "sense:readout:2")


@dataclass(frozen=True)
class NeuralObserver:
    """Fixed numeric readout: actual second-layer F64 coordinate >= threshold."""
    coordinate: int
    threshold: float = 0.0

    def __post_init__(self):
        if (type(self.coordinate) is not int or self.coordinate < 0 or
            type(self.threshold) not in (int, float) or
            not math.isfinite(self.threshold)):
            raise ValueError("Observer requires finite coordinate and threshold")
        object.__setattr__(self, "threshold", float(self.threshold))

    def classify(self, value: float) -> str:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Unqualified numerical source value")
        return "1" if value >= self.threshold else "0"

    def material(self):
        return {"coordinate": self.coordinate,
                "threshold_hex": self.threshold.hex(),
                "readout": "actual-second-layer-F64-ge-threshold"}


@dataclass(frozen=True)
class CausalTape:
    """One consecutive, independently admitted, original-topology causal run."""
    model: str
    revision: str
    index_ground_digest: str
    genesis: str
    observer: NeuralObserver
    inputs: tuple[tuple[int, tuple[float, ...]], ...]
    reports: tuple[dict, ...]
    labels: tuple[str, ...]
    witness_digest: str

    @property
    def length(self):
        return len(self.labels)

    def material(self):
        from .qwen_layer_chain import _f64_transfer_digest
        return {"schema": SCHEMA, "model": self.model, "revision": self.revision,
                "index_ground_digest": self.index_ground_digest,
                "genesis": self.genesis, "observer": self.observer.material(),
                "inputs": [[token, _f64_transfer_digest(vector)]
                           for token,vector in self.inputs],
                "native_reports": [digest(report) for report in self.reports],
                "labels": list(self.labels)}

    def verify(self):
        """Re-validate native causal report content and every exact F64 handoff."""
        from .causal_hidden_manifold import _qualified_capture
        from .qwen_layer_chain import _f64_transfer_digest
        if not (1 <= len(self.inputs) <= MAX_POSITIONS and
                len(self.inputs) == len(self.reports) == len(self.labels) and
                self.observer.coordinate < len(self.inputs[0][1]) and
                isinstance(self.model, str) and isinstance(self.revision, str) and
                isinstance(self.index_ground_digest, str) and
                len(self.index_ground_digest) == 64 and
                isinstance(self.genesis, str) and len(self.genesis) == 64):
            raise ValueError("Incomplete native causal tape")
        previous = self.genesis
        width = len(self.inputs[0][1])
        for index, ((token, vector), report, label) in enumerate(
                zip(self.inputs, self.reports, self.labels)):
            if (type(token) is not int or token < 0 or token >= 2**31 or
                    len(vector) != width or
                    not all(type(x) is float and math.isfinite(x) for x in vector)):
                raise ValueError("Unqualified original input geometry")
            _, native = _qualified_capture(
                report, width, index, previous,
                model=self.model, revision=self.revision,
                index_ground_digest=self.index_ground_digest)
            if (report["layer_witnesses"][0]["input_f64_digest"] !=
                    _f64_transfer_digest(vector) or
                    native["original_token_id"] != token):
                raise ValueError("Original numerical input or label was substituted")
            last = report["receiver_local_intermediates"][1]
            if (_f64_transfer_digest(last) != native["layer1_output_f64_digest"] or
                    self.observer.classify(float(last[self.observer.coordinate])) != label):
                raise ValueError("Native output observation was misclassified")
            previous = native["chain_head"]
        if digest(self.material()) != self.witness_digest:
            raise ValueError("Native causal witness fingerprint changed")
        return True


def capture_native_tape(chain, inputs, observer):
    """Execute a fresh, actual Qwen two-layer causal chain exactly once.

    Input vectors are caller-provisioned, not claimed to be tokenizer-derived
    pretrained token embeddings. A pre-existing receiver-local trained source
    supplies its own QwenCausalGDNChain, including its qualified range hosts.
    """
    from .qwen_layer_chain import QwenCausalGDNChain
    if not isinstance(chain, QwenCausalGDNChain) or chain.position != 0:
        raise ValueError("Fresh qualified two-layer native source required")
    inputs = tuple(inputs)
    if (not isinstance(observer, NeuralObserver) or
            not 1 <= len(inputs) <= MAX_POSITIONS or
            not 0 <= observer.coordinate < chain.width):
        raise ValueError("Invalid source readout or causal experiment")
    genesis = chain.head
    records, verified_inputs, labels = [], [], []
    for token, raw in inputs:
        if (type(token) is not int or not 0 <= token < 2**31 or
                not isinstance(raw, (tuple, list)) or len(raw) != chain.width or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in raw)):
            raise ValueError("Unqualified declared causal input")
        vector = tuple(float(x) for x in raw)
        native_result, report = chain.step(
            list(vector), token_id=token, capture_intermediates=True)
        if (tuple(native_result) != tuple(report["receiver_local_intermediates"][1])):
            raise ValueError("Trained chain result differs from receiver-captured output")
        verified_inputs.append((token, vector))
        records.append(report)
        labels.append(observer.classify(float(native_result[observer.coordinate])))
    draft = CausalTape(chain.model, chain.revision,
                       chain.index_ground_digest, genesis, observer,
                       tuple(verified_inputs), tuple(records), tuple(labels), "")
    tape = CausalTape(chain.model, chain.revision,
                      chain.index_ground_digest, genesis, observer,
                      tuple(verified_inputs), tuple(records), tuple(labels),
                      digest(draft.material()))
    tape.verify()
    return tape


def reframe_native_tape(tape, observer):
    """Observe a different coordinate/threshold from ALREADY admitted F64 bytes.

    No original neural transition is executed. This creates a different
    observer-relative source identity; it cannot extend the native causal path.
    """
    if (not isinstance(tape,CausalTape) or not tape.verify() or
            not isinstance(observer,NeuralObserver) or
            observer.coordinate >= len(tape.inputs[0][1])):
        raise ValueError("Unqualified observer reframe or incomplete original tape")
    labels=tuple(observer.classify(float(report["receiver_local_intermediates"][1]
                                    [observer.coordinate])) for report in tape.reports)
    draft=replace(tape,observer=observer,labels=labels,witness_digest="")
    reframed=replace(draft,witness_digest=digest(draft.material()))
    reframed.verify()
    return reframed


class WitnessedNeuralTraceEnsemble:
    """Exact finite law induced by actual witnessed causal model trajectories.

    First certify each original Qwen numerical transition through the existing
    SHAPE-009 source capture. Then construct the *empirical tape generator*.
    Its all-word certificate concerns precisely the finite, absorbing source
    defined by those tapes, not unobserved neural continuations.
    """
    def __init__(self, tapes):
        from .intention_predictive_ground import ControlledSource
        tapes = tuple(tapes)
        if (not 1 <= len(tapes) <= MAX_TRAJECTORIES or
                any(not isinstance(t, CausalTape) or not t.verify()
                    for t in tapes)):
            raise ValueError("At least one qualified causal trajectory required")
        parent=tapes[0]
        if any((t.model,t.revision,t.index_ground_digest,t.genesis,t.observer) !=
               (parent.model,parent.revision,parent.index_ground_digest,
                parent.genesis,parent.observer) for t in tapes):
            raise ValueError("Mixed model, observer, or causal genesis")
        if sum(t.length + 1 for t in tapes) > 64:
            raise ValueError("Native tape law exceeds bounded finite-source domain")
        self.tapes = tapes
        self.observer = parent.observer
        self.native_witness_digest = digest({
            "schema": SCHEMA, "model": parent.model,
            "revision": parent.revision,
            "index_ground_digest": parent.index_ground_digest,
            "genesis": parent.genesis, "observer": parent.observer.material(),
            "tapes": [t.witness_digest for t in tapes],
            "sampling": "uniform-one-of-observed-trajectories"})
        self.source = self._make_finite_source(ControlledSource)
        self.source_digest = self.source.digest

    def _make_finite_source(self, source_class):
        """A verifier-only explicit source law, generated from observed tapes."""
        symbols=[]
        offsets=[]
        for tape in self.tapes:
            offsets.append(len(symbols))
            symbols.extend((*tape.labels,"2"))
        n=len(symbols)
        initial=[Q() for _ in range(n)]
        for off in offsets: initial[off]=Q(1,len(offsets))
        actions=[]
        successor=[]
        for tape,off in zip(self.tapes,offsets):
            successor.extend(off+min(j+1,tape.length)
                             for j in range(tape.length+1))
        for i in range(n):
            row=[Q() for _ in range(n)]
            row[successor[i]]=Q(1)
            actions.append(tuple(row))
        sensor_outcomes=[]
        for label in ("0","1","2"):
            mat=[]
            for i,token in enumerate(symbols):
                row=[Q() for _ in range(n)]
                if token==label:row[i]=Q(1)
                mat.append(tuple(row))
            sensor_outcomes.append((label,tuple(mat)))
        return source_class(
            tuple(initial),tuple(Q(int(x=="2")) for x in symbols),
            (("step",tuple(actions),Q(1,100)),),
            (("readout",tuple(sensor_outcomes),Q(1,100)),))

    def probability(self, word):
        """Independent tape-walk oracle; never multiplies hidden source matrices."""
        word=tuple(word)
        if any(a not in ALPHABET for a in word):
            raise ValueError("Unknown neural trace event")
        accepted=0
        for tape in self.tapes:
            pos=0
            for event in word:
                if event=="act:step":
                    pos=min(pos+1,tape.length)
                elif event != "sense:readout:" + (
                        tape.labels[pos] if pos<tape.length else "2"):
                    break
            else:
                accepted+=1
        return Q(accepted,len(self.tapes))

    def read(self, words):
        """A source-only reader for QueryAtlas: no matrices are supplied."""
        return tuple(SourceObservation(
            tuple(word), self.probability(word), self.source_digest,
            digest({"schema":"vli.pbg006.neural-trace-reading.v1",
                    "native_witness":self.native_witness_digest,
                    "source":self.source_digest, "word":list(word),
                    "probability":encode(self.probability(word))})) for word in words)

    def atlas(self,max_queries=12000):
        return QueryAtlas(ALPHABET,self.source_digest,self.read,max_queries=max_queries)

    def certify(self,ground):
        """Independent exact tape-generator vs learned law qualification."""
        # The source law is derived from the receipt-verified tape ensemble;
        # the learned Ground was generated through only exact trace queries.
        if not all(t.verify() for t in self.tapes):
            raise ValueError("Native source lineage is not qualified")
        proof=certify_source_equivalence(self.source,ground)
        # Paired reachable-span equivalence is exact for the initial tape
        # ensemble. Source-global closure of every state is a strictly
        # stronger and often unnecessary property for a mixture of tapes.
        return {**proof,"native_witness_digest":self.native_witness_digest,
                "scope":"finite witnessed neural trace ensemble, not complete pretrained-model law"}

    def independently_reexecute(self, new_chain):
        """Optional live re-execution. Must be requested explicitly; expensive.

        This verifies the original numerical trajectory's reproducibility on a
        newly instantiated receiver-authorized native chain, when one is
        supplied per tape. No pretrained token embedding check is implied.
        """
        chains=tuple(new_chain() for _ in self.tapes)
        for tape,chain in zip(self.tapes,chains):
            other=capture_native_tape(chain,tape.inputs,self.observer)
            if (other.labels!=tape.labels or other.genesis!=tape.genesis or
                [r["chain_head"] for r in other.reports]!=
                [r["chain_head"] for r in tape.reports] or
                [r["layer_witnesses"][-1]["output_f64_digest"]
                 for r in other.reports]!=
                [r["layer_witnesses"][-1]["output_f64_digest"]
                 for r in tape.reports]):
                # Transport/range-host metadata can differ across receivers.
                # Require identical semantic causal heads and F64 output bytes.
                raise ValueError("Fresh numerical trained source disagrees with pinned witness")
        return True


class ReceiverQualifiedNeuralReader:
    """Compose PBG-005's native SAB/VGI quorum with a witnessed neural source.

    The E-CPU/VGI adapter commits exact source-trace readings, and this layer
    independently compares each result to the already native-qualified tape
    ensemble. The final evidence identity includes BOTH transport receipts and
    the specific neural witness. It never asks a learner for native matrices.
    """
    def __init__(self, ensemble, native_reader):
        from .source_hankel_fabric import NativeSourceReader
        if (not isinstance(ensemble,WitnessedNeuralTraceEnsemble) or
                not isinstance(native_reader,NativeSourceReader) or
                native_reader.source_digest!=ensemble.source_digest or
                native_reader.alphabet!=ALPHABET):
            raise ValueError("Neural and receiver-qualified query contracts disagree")
        self.ensemble=ensemble
        self.native_reader=native_reader

    def __call__(self, words):
        if not all(t.verify() for t in self.ensemble.tapes):
            raise ValueError("Original neural source is no longer qualified")
        records=self.native_reader(tuple(words))
        if len(records)!=len(words):
            raise ValueError("Native trace receiver omitted observations")
        checked=[]
        for word,record in zip(words,records):
            if (not isinstance(record,SourceObservation) or
                    record.word!=tuple(word) or
                    record.source_digest!=self.ensemble.source_digest or
                    record.probability!=self.ensemble.probability(word)):
                raise ValueError("Native receiver disagrees with original neural witnesses")
            checked.append(SourceObservation(
                tuple(word),record.probability,record.source_digest,
                digest({"schema":"vli.pbg006.receiver-native-neural.v1",
                        "word":list(word),"native_receiver":record.receipt_digest,
                        "neural_witness":self.ensemble.native_witness_digest})))
        return tuple(checked)
