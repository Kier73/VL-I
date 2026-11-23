"""PBG-004: evidence-conditioned structural memory over qualified predictive Grounds.

Do not confuse a merely surprising observation with a source-law counterexample.
Each candidate is itself a PBG-002 exact source-qualified predictive Ground.
Online Bayesian updating operates in the native coordinates of each Ground;
no original source transition matrices are used in inference or planning.

The accepted evidence boundary is PBG-003's actual E-CPU SAB-001 execution and
VGI strict finalization. This module cannot authenticate a physical sensor,
certify an unmodeled candidate law, or make an actuator crash-atomic.
"""
from __future__ import annotations

from dataclasses import dataclass, InitVar
from fractions import Fraction as Q
from functools import lru_cache
import re

from .intention_predictive_ground import (
    ControlledSource, Decision, IntentionGround, IntentionSession,
    MAX_HORIZON, compile_intention_ground, verify_intention_ground,
)
from .predictive_basis_ground import _hash, _q, _r, _rowmat

SCHEMA = "vl-i.pbg004.evidence-adaptive-ground.v1"
NAME = re.compile(r"[A-Za-z][A-Za-z0-9._-]{0,63}\Z")
MAX_HYPOTHESES = 8


class GroundExpansionRequired(ValueError):
    """Qualified evidence has zero likelihood in every current candidate law."""


def _catalogue(ground):
    return (ground.actions(), tuple((name, tuple(labels))
        for name,labels in ground.sensors().items()))


@dataclass(frozen=True)
class QualifiedLaw:
    """A source-checked hypothesis. Online code retains only its reduced Ground."""
    name: str
    ground: IntentionGround
    prior: Q
    goal_digest: str
    qualification_source: InitVar[ControlledSource | None] = None

    @classmethod
    def from_source(cls, name, source, prior):
        if not isinstance(source, ControlledSource):
            raise ValueError("Candidate must have a fully declared exact source")
        ground = compile_intention_ground(source)
        if not verify_intention_ground(source, ground):
            raise ValueError("Candidate does not commute with its source law")
        return cls(name, ground, _r(prior), _hash([_q(g) for g in source.goal]), source)

    def __post_init__(self, qualification_source):
        if (not isinstance(qualification_source, ControlledSource) or
                not verify_intention_ground(qualification_source, self.ground) or
                self.goal_digest != _hash([_q(g) for g in qualification_source.goal])):
            raise ValueError("Hypothesis requires independently verified source and goal")
        if (type(self.name) is not str or NAME.fullmatch(self.name) is None or
                not isinstance(self.ground,IntentionGround) or
                type(self.goal_digest) is not str or
                not re.fullmatch(r"[0-9a-f]{64}", self.goal_digest)):
            raise ValueError("Invalid qualified hypothesis identity")
        self.ground.check_identity()
        p=_r(self.prior)
        if p<=0:
            raise ValueError("Hypothesis prior mass must be positive")
        object.__setattr__(self,"prior",p)


@dataclass(frozen=True)
class RevisionPlan:
    epoch: str
    scheduled: object


class AdaptiveGroundCouncil:
    """Retain rival predictive laws; accept only qualified external event records.

    One independent *contract* Ground connects to PBG-003's E-CPU/VGI bridge.
    It must assign nonzero probability to every reading used in the experiment;
    its role is evidence transport, not privileged hypothesis truth.

    Candidates have strictly positive declared prior masses; observations
    multiply each model's unnormalized mass by its conditional likelihood.
    Actions advance material predictions, but change no model likelihood.
    """
    def __init__(self,contract_ground, laws):
        if not isinstance(contract_ground, IntentionGround):
            raise ValueError("Independent observer contract Ground required")
        contract_ground.check_identity()
        laws=tuple(laws)
        if (not 1<=len(laws)<=MAX_HYPOTHESES or
                any(not isinstance(x,QualifiedLaw) for x in laws) or
                len({x.name for x in laws})!=len(laws) or
                len({x.ground.source_digest for x in laws})!=len(laws) or
                len({x.goal_digest for x in laws})!=1 or
                any(_catalogue(x.ground)!=_catalogue(contract_ground) for x in laws)):
            raise ValueError("Candidate goals and operational contracts must agree")
        self.anchor=IntentionSession(contract_ground)
        self.laws={x.name:x for x in laws}
        self.states={x.name:IntentionSession(x.ground) for x in laws}
        self.mass={x.name:x.prior for x in laws}
        self.events=[]
        self.pending=None
        self._seen=set()
        self.revision=0

    @property
    def posterior(self):
        total=sum(self.mass.values(),Q())
        if total<=0:raise GroundExpansionRequired("No supported candidate law")
        return {k:self.mass[k]/total for k in self.laws}

    @property
    def active(self):
        weights=self.posterior
        return max(self.laws, key=lambda k:weights[k])

    @property
    def goal_probability(self):
        return sum((self.posterior[k]*self.states[k].goal_probability()
                    for k in self.laws),Q())

    @property
    def epoch(self):
        return _hash({"schema":SCHEMA,"revision":self.revision,
                      "contract":self.anchor.ground.ground_digest,
                      "anchor":self.anchor.snapshot()["snapshot_digest"],
                      "models":[(k,x.ground.ground_digest,x.goal_digest,_q(x.prior),
                                 _q(self.mass[k]),
                                 [_q(v) for v in self.states[k].state])
                                for k,x in self.laws.items()],
                      "events":self.events,
                      "pending":None if self.pending is None else self.pending[1]["digest"]})

    def _simulate(self,kind,name,label):
        proposed={}
        masses={}
        for k,law in self.laws.items():
            state=self.states[k]
            prior=self.mass[k]
            if prior==0:
                proposed[k]=None
                masses[k]=Q()
                continue
            follow=IntentionSession(law.ground)
            follow.state=state.state
            follow.history=list(state.history)
            if kind=="act":
                follow.apply(name)
                masses[k]=prior
            elif kind=="sense":
                chances=follow.sensor_distribution(name)
                if label not in chances:raise ValueError("Undeclared sensor symbol")
                p=chances[label]
                masses[k]=prior*p
                if p>0:follow.observe(name,label)
                else:follow=None
            else:raise ValueError("Unknown event kind")
            proposed[k]=follow
        return proposed,masses

    def propose(self,runtime,kind,name,endpoints,*,nonce,coordinate):
        """Issue a fenced E-CPU plan bound to current candidate posterior epoch."""
        if self.pending is not None:raise GroundExpansionRequired("Evidence awaits new Ground")
        from .goal_evidence_fabric import prepare
        scheduled=prepare(self.anchor,runtime,kind,name,endpoints,
                          nonce=nonce,coordinate=coordinate)
        if nonce in self._seen:raise ValueError("Previously committed evidence nonce")
        return RevisionPlan(self.epoch,scheduled)

    def _apply(self,plan,record,*,check_epoch=True):
        if check_epoch and plan.epoch!=self.epoch:
            raise ValueError("Stale candidate/Ground revision epoch")
        if record["nonce"] in self._seen or record["previous"]!=self.anchor.snapshot():
            raise ValueError("Repeated or noncausal source evidence")
        kind,name=record["kind"],record["operation"]
        label=str(record["symbol"]) if kind=="sense" else ""
        proposed,masses=self._simulate(kind,name,label)
        if sum(masses.values(),Q())==0:
            return None
        next_anchor=IntentionSession.restore(self.anchor.ground,record["after"],
                            expected_digest=record["after"]["snapshot_digest"])
        before=self.epoch
        for k in self.laws:
            if proposed[k] is not None:self.states[k]=proposed[k]
        self.mass=masses
        self.anchor=next_anchor
        self.events.append({"kind":kind,"operation":name,"label":label,
                            "evidence_digest":record["digest"],"nonce":record["nonce"],
                            "intent_digest":record["intent_digest"],
                            "vgi_ledger_digest":record["vgi_ledger_digest"]})
        self._seen.add(record["nonce"])
        self.revision+=1
        return {"status":"REVISION_ACCEPTED","before_epoch":before,
                "after_epoch":self.epoch,"active":self.active,
                "posterior":{k:_q(p) for k,p in self.posterior.items()},
                "goal_probability":_q(self.goal_probability),
                "eliminated":[k for k,p in self.mass.items() if p==0],
                "evidence_digest":record["digest"]}

    def accept(self,plan,record,runtime):
        """Re-verify E-CPU operation and VGI finalization without re-execution."""
        from .goal_evidence_fabric import verify_evidence_record
        if not isinstance(plan,RevisionPlan) or self.pending is not None:
            raise GroundExpansionRequired("A previous observation awaits Ground extension")
        if plan.epoch!=self.epoch:
            raise ValueError("Stale hypothesis revision")
        if (not verify_evidence_record(self.anchor.ground,plan.scheduled,runtime,record)
                or record["previous"]!=self.anchor.snapshot()):
            raise ValueError("E-CPU/VGI evidence record was not independently qualified")
        result=self._apply(plan,record)
        if result is None:
            # The evidence is qualified, but every current law calls it impossible.
            # Retain the actual event for source-qualified Ground extension;
            # do not hallucinate a posterior, forget the reading, or re-execute.
            self.pending=(plan,record)
            return {"status":"EXPANSION_REQUIRED","evidence_digest":record["digest"],
                    "previous_epoch":plan.epoch}
        return result

    def dispatch(self,plan,runtime):
        """Execute via PBG-003 on a disposable session, then verify/commit.

        External actuation may have happened even if later verification refuses;
        the source host must independently provide crash-durable completion.
        """
        from .goal_evidence_fabric import execute
        if not isinstance(plan,RevisionPlan) or plan.epoch!=self.epoch or self.pending:
            raise ValueError("Stale or blocked candidate epoch")
        shadow=IntentionSession.restore(self.anchor.ground,plan.scheduled.previous,
                          expected_digest=plan.scheduled.previous["snapshot_digest"])
        record=execute(plan.scheduled,shadow,runtime)
        return self.accept(plan,record,runtime)

    def admit_source(self,name,source,prior):
        """Add a separately declared and independently checked new source law.

        Replays *only compiled Virtual Ground operations* over accepted event
        descriptions to catch up the new candidate. No material action is rerun.
        No unconstrained law is inferred from one finite observation.
        """
        if len(self.laws)>=MAX_HYPOTHESES:raise ValueError("Hypothesis capacity reached")
        law=QualifiedLaw.from_source(name,source,prior)
        if (name in self.laws or law.ground.source_digest in
                {x.ground.source_digest for x in self.laws.values()} or
                law.goal_digest!=next(iter(self.laws.values())).goal_digest or
                _catalogue(law.ground)!=_catalogue(self.anchor.ground)):
            raise ValueError("Candidate alters intention or operation identity")
        state=IntentionSession(law.ground)
        mass=law.prior
        for event in self.events:
            if mass==0:break
            if event["kind"]=="act":
                state.apply(event["operation"])
            else:
                p=state.sensor_distribution(event["operation"])[event["label"]]
                mass*=p
                if p:state.observe(event["operation"],event["label"])
        self.laws[name]=law
        self.states[name]=state
        self.mass[name]=mass
        self.revision+=1
        result={"status":"HYPOTHESIS_ADMITTED","candidate":name,
                "historical_likelihood_weight":_q(mass)}
        if self.pending is not None:
            plan,record=self.pending
            self.pending=None
            outcome=self._apply(plan,record,check_epoch=False)
            if outcome is None:
                self.pending=(plan,record)
                result["status"]="EXPANSION_STILL_REQUIRED"
            else:
                result={**outcome,"status":"GROUNDED_AFTER_EXPANSION",
                        "added":name,"no_external_execution_repeated":True}
        return result

    def plan(self,horizon=2):
        """Source-free, exact Bayesian Bellman planning across live Grounds."""
        if self.pending is not None:raise GroundExpansionRequired("Cannot plan from unexplained evidence")
        if type(horizon) is not int or not 0<=horizon<=MAX_HORIZON:
            raise ValueError("Unsupported planning horizon")
        active=tuple(k for k in self.laws if self.mass[k]>0)
        weights=tuple(self.posterior[k] for k in active)
        states=tuple(self.states[k].state for k in active)
        actions=self.anchor.ground.actions()
        sensors=self.anchor.ground.sensors()
        matrices={k:{(typ,name,label):m for typ,name,label,m,cost in self.laws[k].ground.operators}
                  for k in active}
        @lru_cache(None)
        def best(ws,zs,h):
            goal=sum((w*z[1] for w,z in zip(ws,zs)),Q())
            candidate=Decision('stop','',goal,goal,Q())
            if h==0:return candidate
            for name,cost in actions:
                following=best(ws,tuple(_rowmat(z,matrices[k][('act',name,'')])
                                    for k,z in zip(active,zs)),h-1)
                option=Decision('act',name,following.success-following.cost-cost,
                                following.success,following.cost+cost)
                if option.value>candidate.value:candidate=option
            for name,outs in sensors.items():
                success,cost=Q(),outs[0][1]
                branches=[]
                for label,_ in outs:
                    raw=[_rowmat(z,matrices[k][('sense',name,label)])
                         for k,z in zip(active,zs)]
                    likelihood=sum((w*v[0] for w,v in zip(ws,raw)),Q())
                    if not likelihood:continue
                    new_ws=tuple(w*v[0]/likelihood for w,v in zip(ws,raw))
                    new_zs=tuple(tuple(x/v[0] for x in v) if v[0] else z
                                 for v,z in zip(raw,zs))
                    following=best(new_ws,new_zs,h-1)
                    success+=likelihood*following.success
                    cost+=likelihood*following.cost
                    branches.append((label,likelihood,following.kind,following.name))
                option=Decision('sense',name,success-cost,success,cost,tuple(branches))
                if option.value>candidate.value:candidate=option
            return candidate
        return best(weights,states,horizon)

    def checkpoint(self):
        """Source-free mathematical checkpoint; external root must be pinned."""
        if self.pending is not None:
            raise GroundExpansionRequired("Pending evidence requires its qualified receipt to recover")
        body={"schema":SCHEMA,"epoch":self.epoch,"revision":self.revision,
              "contract_ground":self.anchor.ground.ground_digest,
              "anchor":self.anchor.snapshot(),"events":list(self.events),
              "candidates":[{"name":k,"ground":law.ground.ground_digest,
                             "goal":law.goal_digest,"prior":_q(law.prior),
                             "mass":_q(self.mass[k]),
                             "state":[_q(v) for v in self.states[k].state]}
                            for k,law in self.laws.items()]}
        return {**body,"checkpoint_digest":_hash(body)}

    @classmethod
    def restore(cls,contract_ground,laws,snapshot,*,expected_digest):
        """Reconstruct all virtual states from history; does not re-attest TLS."""
        if (type(snapshot) is not dict or snapshot.get("checkpoint_digest")!=expected_digest
            or _hash({k:v for k,v in snapshot.items() if k!="checkpoint_digest"})!=expected_digest
            or snapshot.get("schema")!=SCHEMA or type(snapshot.get("events")) is not list):
            raise ValueError("Unpinned or changed structural memory")
        council=cls(contract_ground,laws)
        for event in snapshot['events']:
            if (type(event) is not dict or set(event)!={"kind","operation","label",
                    "evidence_digest","intent_digest","vgi_ledger_digest","nonce"} or
                    not isinstance(event['evidence_digest'],str) or
                    len(event['evidence_digest'])!=64 or
                    (event['kind']=='sense' and not isinstance(event['vgi_ledger_digest'],str)) or
                    (event['kind']=='act' and event['vgi_ledger_digest'] is not None)):
                raise ValueError("Malformed source evidence history")
            if event['kind']=='act':
                council.anchor.apply(event['operation'])
            elif event['kind']=='sense':
                council.anchor.observe(event['operation'],event['label'])
            else:raise ValueError("Unsupported historical event")
            states,masses=council._simulate(event['kind'],event['operation'],event['label'])
            if sum(masses.values(),Q())==0:
                raise GroundExpansionRequired("History is impossible under supplied source laws")
            for k,state in states.items():
                if state is not None:council.states[k]=state
            council.mass=masses
            if event["nonce"] in council._seen:
                raise ValueError("Replayed operation nonce in checkpoint")
            council._seen.add(event["nonce"])
            council.events.append(event)
            council.revision+=1
        council.revision=snapshot['revision']
        if (council.epoch!=snapshot['epoch'] or
            council.anchor.snapshot()!=snapshot['anchor'] or
            council.checkpoint()['candidates']!=snapshot['candidates'] or
            council.checkpoint()['events']!=snapshot['events']):
            raise ValueError("Independent predictive replay disagrees with checkpoint")
        return council
