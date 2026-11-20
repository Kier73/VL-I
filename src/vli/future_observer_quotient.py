"""NAV-003: bounded observer-relative equivalence of genuinely different Qwen states.

Different complete neural states may preserve the same declared observation
for ALL action histories within a finite, fully executed action tree. A
certificate proves only that exact finite horizon and exact action alphabet.
It is never an unrestricted bisimulation, a new full-neural-state prediction,
or evidence of language-token identity.

All first-time source transitions use NAV-001's qualified model workers;
the independent verifier uses saved numerical receipts and state pages.
"""
from __future__ import annotations

from copy import deepcopy
import math
import time

from .counterfactual_navigation import CounterfactualCausalGround, _pack, FOREST
from .ground import _digest, _hex64
from .exact_state_quotient import _implementation_fingerprint, _state_key

OBSERVER = "vl-i.future-observer-spec.v1"
CERTIFICATE = "vl-i.finite-future-observer-equivalence.v1"
NAVIGATION = "vl-i.finite-future-observer-navigation.v1"
MAX_HORIZON = 4
MAX_ACTIONS = 3
MAX_WITNESS_PAIRS = 120
EQUIVALENT = "CERTIFIED_FINITE_HORIZON"
REFUTED = "COUNTEREXAMPLE"
SCOPE = "finite_horizon_exhaustive_tested_action_alphabet_observer_projection"


def observer_spec(width, channels, *, observer="vli-future-observer"):
    """A fixed categorical readout of actually qualified numerical outputs.

    Each channel (coordinate, threshold) produces 0 below threshold, 1
    otherwise. This is a declared observer, NOT automatic model discovery.
    """
    if (type(width) is not int or not 1 <= width <= 32768 or
            type(observer) is not str or not 1 <= len(observer) <= 80 or
            type(channels) not in (list, tuple) or not 1 <= len(channels) <= 8):
        raise ValueError("Unqualified finite future observation intention")
    selected = []
    for item in channels:
        if (type(item) not in (tuple, list) or len(item) != 2 or
                type(item[0]) is not int or not 0 <= item[0] < width or
                type(item[1]) not in (float, int) or
                not math.isfinite(item[1])):
            raise ValueError("Observer coordinate/threshold is not admitted")
        selected.append({"index":item[0], "threshold":float(item[1])})
    if len({p["index"] for p in selected}) != len(selected):
        raise ValueError("Repeated observer coordinate")
    body = {"schema":OBSERVER, "width":width, "observer":observer,
            "channels":selected,
            "semantics":"bit_0_if_x_lt_threshold_else_1_binary64"}
    return {**body, "observer_digest":_digest(OBSERVER,body)}


def _check_observer(spec, width):
    if (type(spec) is not dict or spec.get("schema") != OBSERVER or
            spec.get("width") != width or not _hex64(spec.get("observer_digest"))):
        raise ValueError("Missing qualified fixed observer description")
    try:
        expected = observer_spec(
            spec["width"],
            [(channel["index"],channel["threshold"]) for channel in spec["channels"]],
            observer=spec["observer"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Malformed future observer projection") from exc
    if expected != spec:
        raise ValueError("Altered future-observer interpretation")


def _readout(spec, output):
    if len(output) != spec["width"]:
        raise ValueError("Wrong complete qualified future numerical readout")
    return [int(output[p["index"]] >= p["threshold"]) for p in spec["channels"]]


def action_alphabet(width, actions):
    """Explicit finite action alphabet; labels do not prove embedding origin."""
    if (type(actions) not in (list,tuple) or
            not 1 <= len(actions) <= MAX_ACTIONS):
        raise ValueError("Unqualified finite counterfactual action alphabet")
    result = []
    for action in actions:
        if (type(action) not in (list,tuple) or len(action) != 3):
            raise ValueError("Malformed action name/token/vector")
        name, token, hidden = action
        if (type(name) is not str or not 1 <= len(name) <= 48 or
                type(token) is not int or not 0 <= token < 2**31):
            raise ValueError("Invalid bounded action identifier")
        raw, digest = _pack(hidden)
        if len(raw) != width*8:
            raise ValueError("Action hidden vector width changed")
        result.append({"name":name,"token_id":token,"input_f64_digest":digest})
    if len({a["name"] for a in result}) != len(result):
        raise ValueError("Duplicate ambiguous action identifier")
    if len({(a["token_id"],a["input_f64_digest"]) for a in result}) != len(result):
        raise ValueError("Duplicate numerical-and-token transition")
    return result


def _head_at_prefix(ground, count):
    if type(count) is not int or count < 0 or count > len(ground.nodes):
        raise ValueError("Invalid source-committed forest prefix count")
    head = _digest(FOREST, {"root":ground.root_digest,"observer":ground.observer})
    for node in list(ground.nodes.values())[:count]:
        head = _digest(FOREST,{"previous":head,"node_digest":node["node_digest"]})
    return head


def _root(ground, coordinate, externally_pinned_digest):
    if (coordinate not in ground.nodes or
            ground.nodes[coordinate]["node_digest"] != externally_pinned_digest):
        raise ValueError("Untrusted or stale causal-state root pin")
    ground._inspect(ground.nodes[coordinate],verify_checkpoint=True)
    return ground.nodes[coordinate]


def _make_row(ground, observer, action, path, left_parent, right_parent,
              left_node, right_node):
    """One mutually qualified pair of real executed trained transitions."""
    a = ground._inspect(left_node,verify_checkpoint=True)
    b = ground._inspect(right_node,verify_checkpoint=True)
    if (left_node["parent"] != left_parent or
            right_node["parent"] != right_parent or
            left_node["token_id"] != action["token_id"] or
            right_node["token_id"] != action["token_id"] or
            left_node["input_f64_digest"] != action["input_f64_digest"] or
            right_node["input_f64_digest"] != action["input_f64_digest"] or
            left_node["depth"] != right_node["depth"]):
        raise ValueError("Incompatible actual future transition evidence")
    pa,pb = _readout(observer,a),_readout(observer,b)
    return {"path":list(path), "left_parent":left_parent,
            "right_parent":right_parent,
            "left_coordinate":left_node["coordinate"],
            "right_coordinate":right_node["coordinate"],
            "left_node_digest":left_node["node_digest"],
            "right_node_digest":right_node["node_digest"],
            "left_output_f64_digest":left_node["output_f64_digest"],
            "right_output_f64_digest":right_node["output_f64_digest"],
            "left_observation":pa,"right_observation":pb,
            "equal":pa==pb}


class FiniteFutureObserverGround:
    """Exhaustive, finite-horizon relation and independently rechecked oracle.

    Construction performs genuine trained arithmetic for *both* different
    source states on every qualified action prefix; it never assumes an
    unobserved branch is equivalent. Only after exhaustive certification
    can an observer query another coordinate for the SAME explicit action
    vectors and a path of at most the certified horizon.
    """

    def __init__(self, ground):
        if not isinstance(ground,CounterfactualCausalGround):
            raise ValueError("An existing source-qualified NAV-001 forest is required")
        self.ground = ground
        self.last_work = None

    def _roots(self, left, right, left_pin, right_pin, implementation):
        ground=self.ground
        if left == right:
            raise ValueError("Distinct causal histories are required")
        a,b = _root(ground,left,left_pin),_root(ground,right,right_pin)
        if a["depth"] != b["depth"]:
            raise ValueError("Future states must share a causal position")
        key_a,bytes_a = _state_key(ground,a,implementation)
        key_b,bytes_b = _state_key(ground,b,implementation)
        if key_a == key_b:
            raise ValueError("These states are already exact Markov equivalent; use NAV-002")
        return (key_a,key_b,bytes_a+bytes_b)

    @staticmethod
    def _extent(action_count, horizon):
        if (type(horizon) is not int or not 1 <= horizon <= MAX_HORIZON or
                type(action_count) is not int or not 1 <= action_count <= MAX_ACTIONS):
            raise ValueError("Unqualified requested proof horizon or alphabet")
        total = sum(action_count**k for k in range(1,horizon+1))
        if total > MAX_WITNESS_PAIRS:
            raise ValueError("Requested model frontier exceeds bounded proof work")
        return total

    def prove(self,left,right,*,expected_left_digest,expected_right_digest,
              expected_head,observer,actions,horizon):
        """Qualify ALL future action histories or emit the shortest counterexample."""
        g=self.ground
        if expected_head != g.head:
            raise ValueError("Stale original model forest head before proof")
        g._audit()
        _check_observer(observer,g.signature["width"])
        alphabet=action_alphabet(g.signature["width"],actions)
        count=self._extent(len(alphabet),horizon)
        fingerprint=_implementation_fingerprint()
        initial_keys=self._roots(left,right,expected_left_digest,
                                 expected_right_digest,fingerprint)
        if len(g.nodes)+2*count > g.max_nodes:
            raise ValueError("Insufficient state-page capacity for exhaustive proof frontier")
        start_head=g.head
        start_count=len(g.nodes)
        start_ops=g.executed_transitions
        start_reuse=g.reused_transitions
        start=time.perf_counter_ns()
        rows=[]
        queue=[([],left,right)]
        mismatch=None
        while queue:
            nxt=[]
            for path,lparent,rparent in queue:
                for action in alphabet:
                    new_path=path+[action["name"]]
                    vector=next(data[2] for data in actions if data[0]==action["name"])
                    lpin=(g.nodes[lparent]["node_digest"])
                    lvalues,lnode,_=g.advance(lparent,vector,
                         token_id=action["token_id"],
                         expected_parent_digest=lpin,expected_head=g.head)
                    rpin=(g.nodes[rparent]["node_digest"])
                    rvalues,rnode,_=g.advance(rparent,vector,
                         token_id=action["token_id"],
                         expected_parent_digest=rpin,expected_head=g.head)
                    row=_make_row(g,observer,action,new_path,lparent,rparent,
                                  lnode,rnode)
                    rows.append(row)
                    if not row["equal"]:
                        mismatch=deepcopy(row)
                        break
                    if len(new_path)<horizon:
                        nxt.append((new_path,lnode["coordinate"],rnode["coordinate"]))
                if mismatch is not None:
                    break
            if mismatch is not None:
                break
            queue=nxt
        verdict=REFUTED if mismatch is not None else EQUIVALENT
        if mismatch is None and len(rows)!=count:
            raise ValueError("Incomplete exhaustive trained future observation tree")
        body={
            "schema":CERTIFICATE,
            "source_root":g.source_root,
            "operator_fingerprint":fingerprint,
            "observer":deepcopy(observer),
            "alphabet":alphabet,
            "left_root":left,"right_root":right,
            "left_root_digest":expected_left_digest,
            "right_root_digest":expected_right_digest,
            "initial_exact_state_keys":list(initial_keys[:2]),
            "root_state_bytes_compared":initial_keys[2],
            "horizon":horizon,"declared_frontier_pairs":count,
            "forest_head_before":start_head,"forest_node_count_before":start_count,
            "forest_head_after":g.head,
            "verdict":verdict,
            "rows":rows,"first_counterexample":mismatch,
            "complete_exhaustive_frontier":mismatch is None,
            "new_trained_transitions_for_evidence":g.executed_transitions-start_ops,
            "reused_trained_transitions_for_evidence":g.reused_transitions-start_reuse,
            "original_decoder_completed":False,
            "claim_scope":SCOPE,
        }
        certificate={**body,"certificate_digest":_digest(CERTIFICATE,body)}
        self.last_work={"wall_ns":time.perf_counter_ns()-start,
                        "new_trained_transitions":g.executed_transitions-start_ops,
                        "state_pair_witnesses":len(rows),
                        "source_f64_full_state_bytes_anchored":initial_keys[2]}
        self.verify(certificate,expected_digest=certificate["certificate_digest"],
                    expected_head=g.head,
                    expected_left_digest=expected_left_digest,
                    expected_right_digest=expected_right_digest)
        return certificate

    def verify(self,certificate,*,expected_digest,expected_head,
               expected_left_digest,expected_right_digest):
        """Independently reconstruct full BFS witnesses from retained model pages.

        This is source/state evidence validation, not another trained model
        forward run. A signed hash by itself never establishes future parity:
        both sides of every declared action prefix are checked.
        """
        g=self.ground
        if (type(certificate) is not dict or not _hex64(expected_digest) or
                certificate.get("certificate_digest") != expected_digest or
                _digest(CERTIFICATE,{k:v for k,v in certificate.items()
                           if k!="certificate_digest"})!=expected_digest or
                expected_head!=g.head or
                certificate.get("forest_head_after") != expected_head or
                certificate.get("source_root") != g.source_root or
                certificate.get("claim_scope")!=SCOPE or
                certificate.get("original_decoder_completed") is not False):
            raise ValueError("Unpinned, stale or altered future-equivalence certificate")
        g._audit()
        observer=certificate["observer"]
        _check_observer(observer,g.signature["width"])
        alphabet=certificate["alphabet"]
        if (type(alphabet) is not list or not alphabet or
                any(type(a) is not dict or set(a)!={"name","token_id","input_f64_digest"}
                    or type(a["name"]) is not str or not a["name"] or len(a["name"])>48
                    or type(a["token_id"]) is not int or not 0<=a["token_id"]<2**31
                    or not _hex64(a["input_f64_digest"]) for a in alphabet) or
                len({a["name"] for a in alphabet})!=len(alphabet) or
                len({(a["token_id"],a["input_f64_digest"]) for a in alphabet})!=len(alphabet)):
            raise ValueError("Altered finite action alphabet")
        horizon=certificate["horizon"]
        extent=self._extent(len(alphabet),horizon)
        if (certificate.get("declared_frontier_pairs")!=extent or
                not isinstance(certificate.get("rows"),list) or
                certificate.get("initial_exact_state_keys") is None or
                certificate.get("complete_exhaustive_frontier") not in (True,False)):
            raise ValueError("Incomplete or unbounded observer-equivalence frontier")
        fingerprint=_implementation_fingerprint()
        if certificate["operator_fingerprint"]!=fingerprint:
            raise ValueError("Different mathematical interpreter implementation")
        left,right=certificate["left_root"],certificate["right_root"]
        if (certificate["left_root_digest"] != expected_left_digest or
                certificate["right_root_digest"] != expected_right_digest):
            raise ValueError("Stale root external source pins")
        keys=self._roots(left,right,expected_left_digest,expected_right_digest,
                         fingerprint)
        if (certificate["initial_exact_state_keys"]!=list(keys[:2]) or
                certificate["root_state_bytes_compared"]!=keys[2] or
                certificate["forest_head_before"] !=
                  _head_at_prefix(g,certificate["forest_node_count_before"])):
            raise ValueError("Altered source model state or original forest prefix")
        index=0
        queue=[([],left,right)]
        first_mismatch=None
        while queue:
            nxt=[]
            for path,lparent,rparent in queue:
                for action in alphabet:
                    if index>=len(certificate["rows"]):
                        raise ValueError("Missing qualified future continuation witness")
                    row=certificate["rows"][index]
                    index+=1
                    new_path=path+[action["name"]]
                    l_coord=g._coordinate(lparent,action["token_id"],action["input_f64_digest"])
                    r_coord=g._coordinate(rparent,action["token_id"],action["input_f64_digest"])
                    if l_coord not in g.nodes or r_coord not in g.nodes:
                        raise ValueError("Claimed future never executed in source model")
                    actual=_make_row(g,observer,action,new_path,lparent,rparent,
                        g.nodes[l_coord],g.nodes[r_coord])
                    if actual!=row:
                        raise ValueError("Counterexample or forged future observation receipt")
                    if not row["equal"]:
                        first_mismatch=row
                        break
                    if len(new_path)<horizon:
                        nxt.append((new_path,l_coord,r_coord))
                if first_mismatch is not None:
                    break
            if first_mismatch is not None:
                break
            queue=nxt
        if index!=len(certificate["rows"]):
            raise ValueError("Extra unadmitted future observation witnesses")
        verdict=REFUTED if first_mismatch is not None else EQUIVALENT
        if (certificate["verdict"]!=verdict or
                certificate["first_counterexample"]!=first_mismatch or
                certificate["complete_exhaustive_frontier"]!=(first_mismatch is None) or
                (first_mismatch is None and index!=extent)):
            raise ValueError("False future-equivalence or counterexample verdict")
        if (type(certificate["new_trained_transitions_for_evidence"]) is not int or
                type(certificate["reused_trained_transitions_for_evidence"]) is not int or
                certificate["new_trained_transitions_for_evidence"]<0 or
                certificate["reused_trained_transitions_for_evidence"]<0):
            raise ValueError("Unqualified trained future evidence costs")
        return {"verdict":verdict,"witness_pairs_verified":index,
                "horizon":horizon,"observer_digest":observer["observer_digest"],
                "counterexample":deepcopy(first_mismatch)}

    def navigate(self,certificate,history,*,anchor,expected_certificate_digest,
                 expected_head,expected_anchor_digest):
        """Give ONLY a proven projected observation for virtual history labels.

        A query action is (alphabet action name, NEW token-ID label, exact
        hidden vector). It must match its admitted numerical input bytes.
        Re-labeled token IDs are NOT original tokenizer-to-embedding proof.
        """
        if certificate.get("verdict")!=EQUIVALENT:
            raise ValueError("Refuted relation cannot predict a future")
        other_pin=certificate["right_root_digest"] if anchor=="left" else certificate["left_root_digest"]
        self.verify(certificate,expected_digest=expected_certificate_digest,
                    expected_head=expected_head,
                    expected_left_digest=(expected_anchor_digest if anchor=="left" else other_pin),
                    expected_right_digest=(expected_anchor_digest if anchor=="right" else other_pin))
        if (anchor not in ("left","right") or type(history) not in (tuple,list) or
                not 1<=len(history)<=certificate["horizon"]):
            raise ValueError("Unqualified virtual future navigation request")
        names={a["name"]:a for a in certificate["alphabet"]}
        prior=certificate["left_root"] if anchor=="left" else certificate["right_root"]
        previous=expected_anchor_digest
        bypath={tuple(r["path"]):r for r in certificate["rows"]}
        path=[]
        states=[]
        for idx,action in enumerate(history):
            if not isinstance(action,(tuple,list)) or len(action)!=3:
                raise ValueError("Malformed future query action")
            name,token,hidden=action
            if (type(name) is not str or name not in names or
                    type(token) is not int or not 0<=token<2**31):
                raise ValueError("Unsupported future query action or token label")
            raw,digest=_pack(hidden)
            if len(raw)!=self.ground.signature["width"]*8 or digest!=names[name]["input_f64_digest"]:
                raise ValueError("Future action is outside certified numerical input alphabet")
            path.append(name)
            witness=bypath[tuple(path)]
            coordinate=self.ground._coordinate(prior,token,digest)
            observation=witness["left_observation"]
            if coordinate in self.ground.nodes:
                exact=self.ground._inspect(self.ground.nodes[coordinate],
                                           verify_checkpoint=True)
                if _readout(certificate["observer"],exact)!=observation:
                    raise ValueError("Known source transition contradicts projected future")
            record={"step":idx,"path":list(path),
                    "coordinate":coordinate,"previous":previous,
                    "input_f64_digest":digest,"intended_token_label":token,
                    "observation":observation,
                    "left_trained_witness":witness["left_node_digest"],
                    "right_trained_witness":witness["right_node_digest"],
                    "new_trained_transitions":0,
                    "entire_numerical_state_reconstructed":False}
            states.append(record)
            previous=_digest(NAVIGATION,record)
            prior=coordinate
        body={"schema":NAVIGATION,
              "certificate_digest":expected_certificate_digest,
              "external_forest_head":expected_head,
              "anchor":anchor,"anchor_digest":expected_anchor_digest,
              "states":states,
              "scope":"observer_projection_only_within_certified_finite_horizon"}
        return {**body,"navigation_digest":_digest(NAVIGATION,body)}

    def verify_navigation(self,certificate,receipt,*,expected_navigation_digest,
                          expected_certificate_digest,expected_head,
                          expected_anchor_digest):
        if (type(receipt) is not dict or
                receipt.get("navigation_digest")!=expected_navigation_digest or
                not _hex64(expected_navigation_digest) or
                _digest(NAVIGATION,{k:v for k,v in receipt.items()
                          if k!="navigation_digest"})!=expected_navigation_digest or
                not isinstance(receipt.get("states"),list)):
            raise ValueError("Unpinned or altered projected future navigation receipt")
        byname={x["name"]:x for x in certificate["alphabet"]}
        try:
            actions=[(x["path"][-1],x["intended_token_label"],
                      None) for x in receipt["states"]]
            # Inputs are intentionally not stored in the short coordinate
            # proof; verify by digest against a qualified alphabet and
            # independently reconstruct the signed observational path below.
            names={x["name"]:x for x in certificate["alphabet"]}
            if any(x["input_f64_digest"]!=names[name]["input_f64_digest"]
                   for x,(name,_,_) in zip(receipt["states"],actions)):
                raise ValueError("Future action input changed")
            self.verify(certificate,expected_digest=expected_certificate_digest,
                expected_head=expected_head,
                expected_left_digest=(expected_anchor_digest
                    if receipt["anchor"]=="left" else certificate["left_root_digest"]),
                expected_right_digest=(expected_anchor_digest
                    if receipt["anchor"]=="right" else certificate["right_root_digest"]))
            if (certificate["verdict"]!=EQUIVALENT or
                    receipt["certificate_digest"]!=expected_certificate_digest or
                    receipt["external_forest_head"]!=expected_head or
                    receipt["anchor_digest"]!=expected_anchor_digest):
                raise ValueError("Refuted/stale projected future navigation")
            anchor=receipt["anchor"]
            if anchor not in ("left","right") or not 1<=len(actions)<=certificate["horizon"]:
                raise ValueError("Future navigation extent changed")
            old=certificate["left_root"] if anchor=="left" else certificate["right_root"]
            prior=expected_anchor_digest
            path=[]
            rows={tuple(x["path"]):x for x in certificate["rows"]}
            for idx,(step,(name,token,_)) in enumerate(zip(receipt["states"],actions)):
                path.append(name)
                row=rows[tuple(path)]
                d=names[name]["input_f64_digest"]
                coordinate=self.ground._coordinate(old,token,d)
                record={"step":idx,"path":list(path),
                        "coordinate":coordinate,"previous":prior,
                        "input_f64_digest":d,"intended_token_label":token,
                        "observation":row["left_observation"],
                        "left_trained_witness":row["left_node_digest"],
                        "right_trained_witness":row["right_node_digest"],
                        "new_trained_transitions":0,
                        "entire_numerical_state_reconstructed":False}
                if record != step:
                    raise ValueError("Projected observer navigation receipt is contradictory")
                if coordinate in self.ground.nodes:
                    exact=self.ground._inspect(self.ground.nodes[coordinate],
                                               verify_checkpoint=True)
                    if _readout(certificate["observer"],exact)!=record["observation"]:
                        raise ValueError("Known executed observation contradicts virtual future")
                old,prior=coordinate,_digest(NAVIGATION,record)
            if receipt.get("scope")!="observer_projection_only_within_certified_finite_horizon":
                raise ValueError("Invalid projection claim scope")
        except (KeyError,TypeError,IndexError) as exc:
            raise ValueError("Malformed projected observer future proof") from exc
        return [deepcopy(x["observation"]) for x in receipt["states"]]
