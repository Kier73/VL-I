"""TIME-010: execute an independently admitted rational Ground in native E-CPU.

Unlike TIME-009's equal-observation join, this operator actually computes
the entire discovered companion-matrix law inside E-CPU's receiver-local
VirtualLayerRuntime and Hypervisor. The handler is locally registered, pure,
and reconstructs exact rational arithmetic from receiver-qualified source
coefficients. A remote workload cannot supply executable code or rights.

The claim layer reuses E-CPU's majority-backed TransferableExecutionClaimStore.
Only a CURRENT assigned host can publish the result; previously running
pure calculations may finish but cannot be semantically accepted. There is
NO IPS-008 external resource effect in this experiment and no claim that a
hash or a local event fingerprint cryptographically attests a physical host.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from pathlib import Path

from .ground import _digest, _require
from .temporal_intelligence import _fraction, _encode
from .discovered_state_space import (
    SCHEMA as SOURCE_SCHEMA, verify_discovered_space, navigate_discovered,
)
from .problem_formed_claims import _atomic_json, _read_json

SCHEMA="vl-i.time010.native-source-qualified-rational-processor.v1"
CANDIDATE="vl-i.time010.native-rational-compute-candidate.v1"
PUBLICATION="vl-i.time010.quorum-native-rational-publication.v1"
OPERATOR="vli.rational_companion"
MAX_BITS=4096
MAX_STEP=1048576


def _native():
    """Load the original E-CPU native execution and quorum classes."""
    from mailcpu import (
        Capability, HeadlessOS, HypervisorAction, LexicalDecisionModel,
        HypervisorPolicy, OperatorRegistry, OperatorSpec, SystemIdentity,
        SystemSeed, SystemSeedEnvelope, VirtualBudget, VirtualGroundRef,
        VirtualLayerRuntime, VirtualLayerWorkload, VirtualOperatorRef,
        RecoveryLineageAuthority, RecoveryLineageCoordinator,
        SeedCommitMembership, TransferableExecutionClaimStore,
    )
    return locals()


def _check(q):
    """Reject excessive exact rational intermediate bit work, not just outputs."""
    _require(type(q) is Fraction and
             q.numerator.bit_length() <= MAX_BITS and
             q.denominator.bit_length() <= MAX_BITS,
             "Native rational arithmetic exceeded bounded exact bit budget")
    return q


def _reference_input(space, time, output, allow_projection):
    """Independently qualify a requested coordinate before making a workload."""
    expected=navigate_discovered(space,time,output=output,
                                 allow_projection=allow_projection)
    k=expected["logical_step_coordinate"] - space["seed_sample_coordinate"]
    _require(type(k) is int and 0<=k<=MAX_STEP,
             "Native source-relative coordinate exceeds bounded step budget")
    lag=0 if output=="current" else int(output.split(":")[1])
    return expected, {
        "source_ground_digest":space["intention_space_digest"],
        "signed_source_head":space["signed_source_journal_head"],
        "source_observer_head":space["ground_head"],
        "time":time, "source_step":k, "lag":lag,
        "evidence_status":expected["evidence_status"],
    }


def _native_compute(matrix, initial, count, output_index):
    """Own exact rational evaluator with logarithmic binary matrix composition.

    This source implementation is distinct from TIME-007's producer and
    verifier interpreters. It runs ONLY inside the E-CPU operator handler.
    """
    dim=len(initial)
    _require(1<=dim<=4 and len(matrix)==dim and
             all(len(row)==dim for row in matrix) and
             type(count) is int and 0<=count<=MAX_STEP and
             type(output_index) is int and 0<=output_index<dim,
             "Malformed received native mathematical task")
    def multiply(X,Y):
        result=[]
        for row in X:
            out=[]
            for col in zip(*Y):
                out.append(_check(sum(
                    (_check(a*b) for a,b in zip(row,col)),Fraction())))
            result.append(tuple(out))
        return tuple(result)
    result=tuple(tuple(Fraction(int(i==j)) for j in range(dim))
                 for i in range(dim))
    power=matrix
    products=0
    n=count
    while n:
        if n%2:
            result=multiply(result,power)
            products+=1
        n//=2
        if n:
            power=multiply(power,power)
            products+=1
    final=tuple(_check(sum(
        (_check(a*b) for a,b in zip(row,initial)),Fraction()))
        for row in result)
    return {
        "exact_value":_encode(final[output_index]),
        "exact_state":[_encode(q) for q in final],
        "matrix_products":products,
        "source_transitions_replayed":0,
        "source_samples_materialized":0,
        "evaluator":"NATIVE_ECPU_EXACT_BINARY_MATRIX_OPERATOR",
    }


def _spec():
    """Retain the same locally required E-CPU compute right as other pure operators."""
    m=_native()
    return m["OperatorSpec"](
        operator_id=OPERATOR, version=1,
        description="Execute Cramer-qualified source-derived rational state transition",
        input_kind="object",output_kind="object",substrates=("cpu",),
        required_rights=frozenset({"compute"}),max_input_bytes=4096,
        max_output_bytes=16384,base_energy=1,energy_per_byte=1,
        deterministic=True,effect_mode="pure",
    )


def _admit(snapshot,space,source_contract,expected_space_digest,previous_space):
    """Bind a processor to the original independently signed source evidence."""
    _require(type(space) is dict and space.get("schema")==SOURCE_SCHEMA and
             type(expected_space_digest) is str and
             space.get("intention_space_digest")==expected_space_digest and
             type(source_contract) is dict,
             "External policy does not admit this source-qualified processor")
    verify_discovered_space(
        snapshot,space,previous_space=previous_space,**source_contract)
    dim=space["dimension"]
    _require(type(dim) is int and 1<=dim<=4,
             "Native processor has an unsupported dimension")
    matrix=tuple(tuple(_check(_fraction(q)) for q in row)
                 for row in space["transition_operator"])
    initial=tuple(_check(_fraction(q)) for q in space["initial_state"])
    _require(len(matrix)==len(initial)==dim and
             all(len(row)==dim for row in matrix),
             "Source-qualified processor matrix shape changed")
    return matrix,initial


def execute_native_rational(snapshot,space,*,source_contract,
        expected_space_digest,intended_time,output="current",
        allow_projection=False,host="rational-host-a",grant_compute=True,
        previous_space=None):
    """Actually run a source-verified mathematical law via native E-CPU Hypervisor.

    Native E-CPU grants compute only from the receiver kernel. A source claim,
    processor digest, or copied candidate cannot grant execution capabilities.
    """
    matrix,initial=_admit(snapshot,space,source_contract,
                          expected_space_digest,previous_space)
    reference,task=_reference_input(space,intended_time,output,allow_projection)
    _require(type(host) is str and 1<=len(host)<=64 and
             host[0].isalnum() and host.isascii() and
             all(c.isalnum() or c in "-_." for c in host),
             "Unadmitted E-CPU receiver host identity")
    m=_native()
    operating_system=m["HeadlessOS"]()
    operating_system.hypervisor.classifier=m["LexicalDecisionModel"]()
    operating_system.hypervisor.policy=m["HypervisorPolicy"](
        embedding_review_confidence=1.0)
    identity=m["SystemIdentity"]("time010-"+host,1)
    envelope=m["SystemSeedEnvelope"](
        identity,"time010-processor","time010-rational-ground",
        m["SystemSeed"].from_system(operating_system))
    _require(operating_system.seed_admission.admit(envelope).accepted,
             "Native receiver refused its own admitted System Ground")
    virtual_ground=m["VirtualGroundRef"].from_envelope(envelope)
    invoked=[]
    def handler(value,workload):
        """Receiver-local native handler: no external code or remote operator."""
        _require(value==task and
                 workload.ground.to_dict()==virtual_ground.to_dict(),
                 "Native hosted operation changed source coordinates or Ground")
        result=_native_compute(matrix,initial,value["source_step"],value["lag"])
        invoked.append(_digest(SCHEMA+".activation",{
            "ground":expected_space_digest,"task":value,"result":result,
            "receiver":host}))
        return {
            **result,"source_ground_digest":expected_space_digest,
            "source_signed_head":space["signed_source_journal_head"],
            "source_observer_head":space["ground_head"],
            "coordinate":intended_time,
            "evidence_status":reference["evidence_status"],
        }
    registry=m["OperatorRegistry"]()
    spec=_spec()
    registry.register(spec,handler)
    runtime=m["VirtualLayerRuntime"](
        registry,host,frozenset({"cpu"}))
    permission=(m["Capability"](
        "/virtual-layer/operators",frozenset({"compute"}))
        if grant_compute else m["Capability"]("/safe",frozenset({"read"})))
    pid=operating_system.environment.kernel.spawn(
        "rational-ground-worker",capabilities=(permission,),
        parent_pid=operating_system.environment.init_pid)
    workload=m["VirtualLayerWorkload"](
        workload_id="time010-"+expected_space_digest[:20]+"-"+str(task["source_step"]),
        system=identity,ground=virtual_ground,
        intention={"goal":{"ground":expected_space_digest,"time":intended_time,
                            "output":output,"semantic_language":"rational-companion"}},
        operators=(m["VirtualOperatorRef"](OPERATOR,1,"cpu"),),
        input_value=task,budget=m["VirtualBudget"](
            200_000,200_000,20000),logical_time=task["source_step"]+1,
        eligible_nodes=(host,),max_hops=1)
    receipt=runtime.execute_operator(
        operating_system,pid,"time010-source","time010-native-rational",
        task["source_step"]+1,task["source_step"]+1,workload,OPERATOR)
    _require(receipt.action is m["HypervisorAction"].ALLOW and
             receipt.executed and len(invoked)==1 and
             receipt.effect is not None and
             receipt.effect["kind"]=="virtual_layer.operator_result" and
             receipt.effect["operator_contract_digest"]==spec.digest,
             "Receiver-local Hypervisor refused admitted rational execution")
    native_result=dict(receipt.effect["result"])
    _require(native_result["exact_value"]==reference["value"] and
             native_result["exact_state"]==
                 reference["calculation"]["state_vector"] and
             native_result["source_transitions_replayed"]==0 and
             operating_system.environment.kernel.journal.verify(),
             "Actual native rational execution differs from independent source oracle")
    body={
        "schema":SCHEMA,
        "source_ground_digest":expected_space_digest,
        "source_signed_journal_head":space["signed_source_journal_head"],
        "source_observer_head":space["ground_head"],
        "native_operator_id":OPERATOR,
        "native_operator_spec_digest":spec.digest,
        "native_virtual_ground":virtual_ground.to_dict(),
        "host":host,"coordinate":intended_time,"output":output,
        "evidence_status":reference["evidence_status"],
        "native_result":native_result,
        "native_effect":receipt.effect,
        "native_event_fingerprint":receipt.fingerprint,
        "receiver_handler_invocations":len(invoked),
        "local_kernel_journal_verified":True,
        "independent_source_navigation_digest":reference["navigation_digest"],
        "native_mathematical_law_executed":True,
        "native_effect_mode":"pure",
        "external_resource_effect_committed":False,
        "remote_hardware_attested":False,
    }
    return {**body,"execution_digest":_digest(SCHEMA,body)}


class FencedNativeRationalProcessor:
    """Reassign and accept only current pure native rational results via E-CPU.

    This one-request sequential owner protocol uses real independently stored
    E-CPU recovery authority journals and TransferableExecutionClaimStore.
    It has no second homemade quorum or bearer authority. Pure work may run
    on multiple hosts; only the current claim epoch can commit its result.
    A revision of source semantics requires a separate explicitly bound
    claim/processor namespace; this does not imply atomic semantic cutover.
    """

    def __init__(self,root,snapshot,space,*,source_contract,
                 expected_space_digest,intended_time,output="current",
                 allow_projection=False,previous_space=None,
                 job="time010-experiment"):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.snapshot,self.space,self.source_contract=snapshot,space,source_contract
        self.expected_space_digest=expected_space_digest
        self.time,self.output=intended_time,output
        self.allow_projection=allow_projection
        self.previous_space=previous_space
        self.job=job
        _admit(snapshot,space,source_contract,expected_space_digest,previous_space)
        self.reference,self.input=_reference_input(
            space,intended_time,output,allow_projection)
        self.fingerprint=_digest(SCHEMA+".request",{
            "job":job,"source_ground":expected_space_digest,
            "source_head":space["signed_source_journal_head"],
            "time":intended_time,"output":output,
            "future_opt_in":allow_projection,
        })
        identity_path=self.root/"native-processor-identity.json"
        if identity_path.exists():
            _require(_read_json(identity_path)=={
                "fingerprint":self.fingerprint,
                "schema":SCHEMA},
                "Recovered processor has a different mathematical or source identity")
        else:
            _atomic_json(identity_path,{
                "fingerprint":self.fingerprint,"schema":SCHEMA})
        m=_native()
        membership=m["SeedCommitMembership"](
            "time010-"+self.fingerprint[:18],1,
            ("A","B","C"),"A",1,"0"*64)
        authorities={a:m["RecoveryLineageAuthority"](
            membership,a,self.root/"quorum"/(a+".jsonl"))
            for a in membership.member_ids}
        self.coordinator=m["RecoveryLineageCoordinator"](
            membership,authorities)
        self.store=m["TransferableExecutionClaimStore"](
            self.root/"claims",self.coordinator)
        self.spec=_spec()

    def _binding(self):
        """Bind request, source and operator identity to one native claim."""
        return {
            "parent_request_digest":_digest(
                SCHEMA+".parent",{"source":self.expected_space_digest,
                                 "journal_head":self.space["signed_source_journal_head"]}),
            "child_request_digest":_digest(
                SCHEMA+".child",{"request":self.fingerprint,"input":self.input}),
            "address_digest":_digest(
                SCHEMA+".address",{"operator":self.spec.digest,
                                   "requested_output":self.output,"time":self.time}),
            "operator_spec":self.spec,
            "computational_ground_fingerprint":self.expected_space_digest,
        }

    @staticmethod
    def _actor(peer,host):
        """Describe one bounded processor execution locus and implementation."""
        _require(type(peer) is str and type(host) is str and
                 1<=len(peer)<=64 and 1<=len(host)<=64 and
                 all(s.isascii() and s[0].isalnum() and
                     all(c.isalnum() or c in "-_." for c in s)
                     for s in (peer,host)),
                 "Unqualified claim owner or computational host")
        return {
            "owner_peer_id":peer,"owner_host_id":host,
            "implementation_fingerprint":_digest(
                SCHEMA+".host",{"operator":OPERATOR,
                                "implementation":"native-rational-v1",
                                "host":host}),
        }

    def claim(self,*,peer,host):
        """Acquire an E-CPU native claim, never a VL-I self-issued grant."""
        return self.store.create(**self._binding(),**self._actor(peer,host))

    def _fresh(self,claim_id,peer,host):
        """Ask E-CPU's original quorum coordinator to validate current owner."""
        token=self.store.get(claim_id)
        token.verify_for_target(
            self.coordinator,**self._binding(),**self._actor(peer,host))
        return token

    def calculate(self,claim_id,*,peer,host,grant_compute=True):
        """Run native E-CPU Hypervisor under the CURRENT owner; do not publish."""
        token=self._fresh(claim_id,peer,host)
        native=execute_native_rational(
            self.snapshot,self.space,source_contract=self.source_contract,
            expected_space_digest=self.expected_space_digest,
            intended_time=self.time,output=self.output,
            allow_projection=self.allow_projection,host=host,
            grant_compute=grant_compute,previous_space=self.previous_space)
        body={
            "schema":CANDIDATE,"claim_id":claim_id,
            "assignment_epoch":token.claim.current_assignment.epoch,
            "peer":peer,"host":host,
            "request_fingerprint":self.fingerprint,
            "result_exact_rational":native["native_result"]["exact_value"],
            "native_execution":native,
        }
        return {**body,"candidate_digest":_digest(CANDIDATE,body)}

    def reassign(self,claim_id,*,peer,host,failure_evidence_digest):
        """Rehome a pure processor via the original E-CPU quorum fence."""
        ticket=self.store.get(claim_id)
        return self.store.reassign(
            claim_id,expected_epoch=ticket.claim.current_assignment.epoch,
            **self._actor(peer,host),
            failure_evidence_digest=failure_evidence_digest)

    def _validate(self,candidate,token):
        """Independently reproduce exact native output under trusted source law."""
        _require(type(candidate) is dict and
                 set(candidate)=={
                     "schema","claim_id","assignment_epoch","peer","host",
                     "request_fingerprint","result_exact_rational",
                     "native_execution","candidate_digest"} and
                 candidate["schema"]==CANDIDATE and
                 candidate["candidate_digest"]==_digest(
                     CANDIDATE,{k:v for k,v in candidate.items()
                                if k!="candidate_digest"}) and
                 candidate["request_fingerprint"]==self.fingerprint and
                 candidate["claim_id"]==token.claim.claim_id and
                 candidate["assignment_epoch"]==
                     token.claim.current_assignment.epoch and
                 candidate["peer"]==
                     token.claim.current_assignment.owner_peer_id and
                 candidate["host"]==
                     token.claim.current_assignment.owner_host_id,
                 "Unqualified or reassigned native processor candidate")
        native=candidate["native_execution"]
        _require(type(native) is dict and
                 native.get("schema")==SCHEMA and
                 native.get("execution_digest")==_digest(
                     SCHEMA,{k:v for k,v in native.items()
                             if k!="execution_digest"}) and
                 native.get("source_ground_digest")==self.expected_space_digest and
                 native.get("source_signed_journal_head")==
                     self.space["signed_source_journal_head"] and
                 native.get("host")==candidate["host"] and
                 native.get("coordinate")==self.time and
                 native.get("output")==self.output and
                 native.get("native_operator_spec_digest")==self.spec.digest and
                 native.get("native_result",{}).get("exact_value")==
                     self.reference["value"] and
                 candidate["result_exact_rational"]==self.reference["value"] and
                 native.get("native_mathematical_law_executed") is True and
                 native.get("receiver_handler_invocations")==1 and
                 native.get("local_kernel_journal_verified") is True,
                 "Native mathematical result or execution evidence disagrees")
        # An event fingerprint is only a local integrity claim. Re-execute
        # under the native Hypervisor, with actual compute rights, before
        # accepting a caller-controlled candidate in the quorum.
        replay=execute_native_rational(
            self.snapshot,self.space,source_contract=self.source_contract,
            expected_space_digest=self.expected_space_digest,
            intended_time=self.time,output=self.output,
            allow_projection=self.allow_projection,host=candidate["host"],
            grant_compute=True,previous_space=self.previous_space)
        _require(replay==native,
                 "Fresh native Hypervisor replay rejects an untrusted candidate")

    def publish(self,candidate):
        """Commit a current owner result through E-CPU's durable quorum."""
        _require(type(candidate) is dict and
                 type(candidate.get("claim_id")) is str,
                 "Native publication lacks typed candidate owner")
        token=self.store.get(candidate["claim_id"])
        self._fresh(candidate["claim_id"],candidate["peer"],candidate["host"])
        self._validate(candidate,token)
        # The publisher verifies current quorum owner again before commit.
        finished=self.store.complete(
            candidate["claim_id"],
            expected_epoch=candidate["assignment_epoch"],
            **self._actor(candidate["peer"],candidate["host"]),
            result_digest=_digest(SCHEMA+".value",{
                "rational":candidate["result_exact_rational"]}),
            child_receipt_digest=candidate["candidate_digest"])
        _require(finished.claim.status.value=="completed",
                 "E-CPU refused native rational result publication")
        _atomic_json(self.root/"accepted-native-result.json",candidate)
        return self.verify_published(candidate["claim_id"])

    def verify_published(self,claim_id):
        """Recover and validate a real, current majority-accepted pure result."""
        token=self.store.get(claim_id)
        token.verify_current(self.coordinator)
        _require(token.claim.status.value=="completed",
                 "Native rational result not yet majority-published")
        candidate=_read_json(self.root/"accepted-native-result.json")
        self._validate(candidate,token)
        _require(token.claim.result_digest==_digest(
                     SCHEMA+".value",{"rational":candidate["result_exact_rational"]}) and
                 token.claim.child_receipt_digest==candidate["candidate_digest"],
                 "Native quorum result differs from receiver-executed source law")
        body={
            "schema":PUBLICATION,
            "source_ground_digest":self.expected_space_digest,
            "source_journal_head":self.space["signed_source_journal_head"],
            "request_fingerprint":self.fingerprint,
            "native_claim_id":claim_id,
            "native_claim_generation":token.manifest.generation,
            "assignment_epoch":token.claim.current_assignment.epoch,
            "current_host":token.claim.current_assignment.owner_host_id,
            "native_candidate_digest":candidate["candidate_digest"],
            "result_exact_rational":candidate["result_exact_rational"],
            "source_navigation_digest":self.reference["navigation_digest"],
            "source_law_executed_by_native_hypervisor":True,
            "fenced_pure_result_published":True,
            "external_effect_performed":False,
            "multiple_physical_executions_excluded":False,
        }
        return {**body,"publication_digest":_digest(PUBLICATION,body)}
