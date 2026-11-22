"""PFI-005: a checked SFI-003 interpreter revision on PFI-004's native fabric.

The source semantics are invariant, the generated register space is not.
Run the old and revised interpreters under separate qualified native Ground
identities. Reuse the existing E-CPU claim store, Hypervisor, DAG and fencing,
rather than introducing a second authority or treating a content digest as one.

This is a bounded, pure, local-infrastructure experiment. It is not a
transparent cross-revision handoff of in-flight claims: outstanding work must
be reissued under the new program identity and current authority.
"""
from __future__ import annotations
import json

from .problem_formed_sat import DirectSatFabric

SOURCE_SCHEMA="vli.pfi005.proven-morphic-revision.v1"
WITNESS_SCHEMA="vli.pfi005.source-witness-translation.v1"


def _imports():
    from morphic_sat import (compile_ground, normalize, digest, canonical,
                            execute_partition, decode)
    from morphic_verifier import (verify, check_partition, _independent_decode)
    from recursive_sat_compiler import refine, audit_refinement
    return locals()


def _require(test,reason):
    if not test:
        raise ValueError(reason)


def _canonical_copy(obj):
    return json.loads(json.dumps(obj,sort_keys=True,allow_nan=False))


def create_revision_source(clauses,n,*,candidate_budget=4000):
    """Construct a revision *using the existing SAT interpreter*, not a new one."""
    M=_imports()
    source=M["normalize"](clauses,n)
    baseline,_=M["compile_ground"](source,n,max_candidates=candidate_budget)
    refined,report=M["refine"](source,n,candidate_budget=candidate_budget)
    M["audit_refinement"](source,n,refined,report)
    _require(report["outcome"]=="IMPROVED_BY_OWN_SAT_INTERPRETER",
             "source does not justify a new interpreter architecture")
    _require(len(refined["registers"])<len(baseline["registers"]),
             "no independently verified register reduction")
    body={"schema":SOURCE_SCHEMA,"n":n,"source_clauses":[list(c) for c in source],
          "baseline":baseline,"refined":refined,"meta_report":report}
    body=_canonical_copy(body)
    body["digest"]=M["digest"](body)
    verify_revision_source(body)
    return body


def verify_revision_source(receipt):
    """Source/semantic verification before either interpreter may execute."""
    M=_imports()
    if type(receipt) is not dict or set(receipt)!={"schema","n","source_clauses",
            "baseline","refined","meta_report","digest"}:
        raise ValueError("incomplete revision/source record")
    _require(receipt["schema"]==SOURCE_SCHEMA
             and receipt["digest"]==M["digest"]({k:v for k,v in receipt.items() if k!="digest"}),
             "wrong revision record identity")
    n=receipt["n"]
    source=M["normalize"](receipt["source_clauses"],n)
    _require(receipt["source_clauses"]==[list(c) for c in source],
             "noncanonical or altered revision source")
    before=receipt["baseline"];after=receipt["refined"];report=receipt["meta_report"]
    _require(M["verify"](before,max_free=256)["proven"]
             and M["verify"](after,max_free=256)["proven"],
             "source compiler supplied unproved interpreter")
    _require(before["source_clauses"]==receipt["source_clauses"]
             and after["source_clauses"]==receipt["source_clauses"],
             "revision changed original CNF")
    recreated,_=M["compile_ground"](source,n,
                                    max_candidates=report.get("candidate_budget",4000))
    _require(recreated==before,"claimed previous interpreter differs from baseline")
    proof=M["audit_refinement"](source,n,after,report)
    _require(report.get("outcome")=="IMPROVED_BY_OWN_SAT_INTERPRETER"
             and proof["verified_register_reduction"]>=1,
             "refinement is not an independently justified improvement")
    return proof


def _form(receipt,*,intention="#SAT",maximum_free=12,partition_size=8,revision="refined"):
    M=_imports()
    verify_revision_source(receipt)
    _require(intention=="#SAT", "PFI-005 currently certifies complete counting only")
    _require(revision in ("baseline","refined"),"unknown interpreter revision")
    _require(type(maximum_free) is int and 0<=maximum_free<=20
             and type(partition_size) is int and 1<=partition_size<=65536,
             "invalid finite program budget")
    ground=receipt[revision]
    d=len(ground["registers"]);total=1<<d
    count=(total+partition_size-1)//partition_size
    _require(d<=maximum_free and count<=128,
             "formed interpreter cannot cover its full coordinate domain")
    source_sha=M["digest"]({"n":receipt["n"],"clauses":receipt["source_clauses"]})
    program={"schema":"vli.pfi005.morphic-sat-program.v1",
             "mode":"constructed-search","revision":revision,
             "source_sha256":source_sha,"source_variables":receipt["n"],
             "source_clauses":receipt["source_clauses"],
             "intention":intention,"free_dimension":d,
             "ground":ground,"revision_source_sha256":receipt["digest"],
             "baseline_ground_sha256":receipt["baseline"]["digest"],
             "partition_size":partition_size,
             "budgets":{"maximum_free":maximum_free,"partition_size":partition_size},
             "partitions":[{"index":i,"start":start,
                            "stop":min(total,start+partition_size)}
                           for i,start in enumerate(range(0,total,partition_size))]}
    program["digest"]=M["digest"](program)
    return program,{"generated_registers":d,"generated_gates":len(ground["instructions"]),
                    "partitions":count,"coordinates":total,
                    "original_source_sha256":source_sha}


def morphic_runtime(*,revision="refined"):
    """Seven-call adapter used by the *existing* DirectSatFabric.

    Only the mathematical producer/checker is specialized. All E-CPU claim
    publication and checkpoint validation stay in the original PFI-004 class.
    """
    M=_imports()
    def form(receipt,*,intention="#SAT",maximum_free=12,partition_size=8):
        return _form(receipt,intention=intention,maximum_free=maximum_free,
                     partition_size=partition_size,revision=revision)

    def verify_program(program,source_receipt):
        _require(type(program) is dict and
                 program.get("digest")==M["digest"]({k:v for k,v in program.items()
                                                   if k!="digest"}),
                 "altered generated processor identity")
        limits=program.get("budgets")
        _require(type(limits) is dict,"missing original finite program budget")
        regenerated,_=form(source_receipt,intention=program.get("intention"),
                           maximum_free=limits.get("maximum_free"),
                           partition_size=limits.get("partition_size"))
        _require(regenerated==program,"interpreter does not reconstruct")
        return True

    def execute_partition(program,i):
        _require(type(i) is int and 0<=i<len(program["partitions"]),
                 "partition not in formed processor")
        return M["execute_partition"](program["ground"],i,intent="COUNT",
                budget=1<<program["free_dimension"],
                partition_size=program["partition_size"])

    def check_segment(program,result):
        _require(result.get("intent")=="COUNT",
                 "source counting requires complete segment evaluation")
        return M["check_partition"](program["ground"],result,
                    maximum=1<<program["free_dimension"],
                    partition_size=program["partition_size"])

    def aggregate(program,results,*,source_receipt=None):
        _require(source_receipt is not None,
                 "source receipt is required to admit a final model count")
        verify_program(program,source_receipt)
        if type(results) is not list or len(results)!=len(program["partitions"]):
            raise ValueError("not all original source coordinate intervals are covered")
        indexed={}
        for result in results:
            check_segment(program,result)
            i=result["index"]
            _require(i not in indexed,"duplicate segment is not coverage")
            indexed[i]=result
        _require(set(indexed)==set(range(len(program["partitions"]))),
                 "missing source partition")
        count=sum(indexed[i]["models"] for i in sorted(indexed))
        witnesses=[int(indexed[i]["witness_decimal"]) for i in sorted(indexed)
                   if indexed[i]["witness_decimal"] is not None]
        bits=([((witnesses[0]>>j)&1) for j in range(program["source_variables"])]
              if witnesses else None)
        return {"status":"SAT" if count else "UNSAT","count":count,
                "witness":bits,"evaluated":sum(x["checked"] for x in indexed.values()),
                "interpreter":program["digest"],"revision":program["revision"]}

    return form,verify_program,execute_partition,check_segment,aggregate,M["digest"],M["canonical"]


def translate_source_witness(receipt,bits):
    """Re-express a *source-validated* model in both independent register spaces."""
    M=_imports()
    verify_revision_source(receipt)
    n=receipt["n"]
    _require(type(bits) is int and 0<=bits<(1<<n),
             "invalid original-variable assignment")
    for clause in receipt["source_clauses"]:
        _require(any(((bits>>(abs(l)-1))&1)==int(l>0) for l in clause),
                 "cannot translate an assignment violating the original CNF")
    coordinates={}
    for revision in ("baseline","refined"):
        ground=receipt[revision]
        z=sum(((bits>>(v-1))&1)<<j for j,v in enumerate(ground["registers"]))
        _require(M["_independent_decode"](ground,z)==bits,
                 "source witness fails the new interpreter's independent decoder")
        coordinates[revision]=str(z)
    body={"schema":WITNESS_SCHEMA,
          "source_sha256":M["digest"]({"n":n,"clauses":receipt["source_clauses"]}),
          "revision_source_sha256":receipt["digest"],
          "source_assignment_decimal":str(bits),
          "baseline_ground_sha256":receipt["baseline"]["digest"],
          "refined_ground_sha256":receipt["refined"]["digest"],
          "coordinates":coordinates}
    body["digest"]=M["digest"](body)
    return body


def verify_translated_witness(receipt,certificate):
    M=_imports()
    _require(type(certificate) is dict and certificate.get("schema")==WITNESS_SCHEMA
             and certificate.get("digest")==M["digest"]({k:v for k,v in certificate.items()
                                                         if k!="digest"}),
             "bad translated-witness envelope")
    observed=translate_source_witness(receipt,int(certificate["source_assignment_decimal"]))
    _require(certificate==observed,"forged source-witness coordinate translation")
    return True


class RevisedSatFabric(DirectSatFabric):
    """Reuse native claim/rehome/fencing through a checked SFI-003 runtime."""
    def __init__(self,root,source_receipt,*,revision="refined",intention="#SAT",
                 maximum_free=12,partition_size=8,job="sat-revised"):
        super().__init__(root,source_receipt,intention=intention,
                         maximum_free=maximum_free,partition_size=partition_size,
                         job=job,runtime_adapter=lambda: morphic_runtime(revision=revision))
