"""PBG-002: goal-conditioned predictive Ground with sensing and intervention.

A fully supplied finite controlled stochastic source is compiled into the
smallest column space containing constant 1 and goal indicator g and closed
under every controlled transition and sensor-outcome kernel: M B = B K.

Online interpretation operates only the compiled r-dimensional state. Passive
observations condition a belief; controlled actions change its predicted world
state. No API here executes or attests an external physical intervention.
The planner uses exact rational probabilities/costs and bounded lookahead.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from itertools import product

from .predictive_basis_ground import (
    _coordinates, _hash, _matvec, _matrix_payload, _q, _r, _rowmat,
    MAX_STATES,
)

SCHEMA = "vli.intention-predictive-ground.v1"
MAX_OPERATIONS = 16
MAX_HORIZON = 4


def _matrix(raw, n):
    try:
        matrix = tuple(tuple(_r(x) for x in row) for row in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid controlled source kernel") from exc
    if len(matrix) != n or any(len(row) != n or any(v < 0 for v in row) for row in matrix):
        raise ValueError("Invalid controlled source kernel geometry")
    return matrix


@dataclass(frozen=True)
class ControlledSource:
    """actions=(name, stochastic_matrix, cost); sensors=(name, outcome kernels, cost).

    Outcome kernels are sub-stochastic; summing over outcomes yields a
    stochastic matrix. For passive sensors this sum MUST be the identity.
    """
    initial: tuple[Q, ...]
    goal: tuple[Q, ...]
    actions: tuple[tuple[str, tuple, Q], ...]
    sensors: tuple[tuple[str, tuple[tuple[str, tuple], ...], Q], ...]

    def __post_init__(self):
        initial = tuple(_r(x) for x in self.initial)
        n = len(initial)
        if not 1 <= n <= MAX_STATES or any(x < 0 for x in initial) or sum(initial) != 1:
            raise ValueError("Invalid initial source distribution")
        goal = tuple(_r(x) for x in self.goal)
        if len(goal) != n or any(x not in (0, 1) for x in goal):
            raise ValueError("Goal must be a declared binary event")
        if not (0 < sum(goal) < n):
            raise ValueError("Fixture requires a nonconstant goal projection")
        actions = []
        for name, raw, raw_cost in self.actions:
            matrix = _matrix(raw, n)
            if any(sum(row) != 1 for row in matrix):
                raise ValueError("Intervention must be stochastic")
            actions.append((name, matrix, _r(raw_cost)))
        sensors = []
        for name, outcomes, raw_cost in self.sensors:
            os = [(label, _matrix(mat, n)) for label, mat in outcomes]
            if not os or len({label for label, _ in os}) != len(os):
                raise ValueError("Sensor outcomes must be distinct and nonempty")
            if any(any(sum(mat[i][j] for label, mat in os) != int(i == j)
                       for j in range(n)) for i in range(n)):
                raise ValueError("Sensor must not change material state")
            sensors.append((name, tuple(os), _r(raw_cost)))
        names = [n for n, _, _ in actions+sensors]
        if (not names or len(names) > MAX_OPERATIONS or len(set(names)) != len(names)
            or any(type(name) is not str or not 1 <= len(name) <= 50 for name in names)
            or any(cost < 0 for _, _, cost in actions+sensors)
            or any(type(label) is not str or not label for _,os,_ in sensors for label,_ in os)
            or sum(len(os) for _,os,_ in sensors)+len(actions) > MAX_OPERATIONS):
            raise ValueError("Invalid or oversized controlled operation catalogue")
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "goal", goal)
        object.__setattr__(self, "actions", tuple(actions))
        object.__setattr__(self, "sensors", tuple(sensors))

    @property
    def digest(self):
        return _hash({"initial": [_q(x) for x in self.initial],
                      "goal": [_q(x) for x in self.goal],
                      "actions": [(name, _matrix_payload(m), _q(c)) for name,m,c in self.actions],
                      "sensors": [(name,[(label,_matrix_payload(m)) for label,m in os],_q(c))
                                  for name,os,c in self.sensors]})


def _operators(source):
    ops=[]
    for name,mat,cost in source.actions:
        ops.append(("act",name,"",mat,cost))
    for name,outcomes,cost in source.sensors:
        for label,mat in outcomes:
            ops.append(("sense",name,label,mat,cost))
    return ops


def _construct(source):
    n=len(source.initial)
    columns=[tuple(Q(1) for _ in range(n)), tuple(source.goal)]
    ops=_operators(source)
    cursor=0
    while cursor < len(columns):
        for _,_,_,mat,_ in ops:
            candidate=_matvec(mat,columns[cursor])
            if _coordinates(columns,candidate) is None:
                columns.append(candidate)
        cursor+=1
    r=len(columns)
    reduced=[]
    for kind,name,label,mat,cost in ops:
        solved=[]
        for col in columns:
            c=_coordinates(columns,_matvec(mat,col))
            if c is None:
                raise AssertionError("Source observation closure was incomplete")
            solved.append(c)
        k=tuple(tuple(solved[j][i] for j in range(r)) for i in range(r))
        reduced.append((kind,name,label,k,cost))
    initial=tuple(sum((x*y for x,y in zip(source.initial,col)),Q()) for col in columns)
    return tuple(columns),tuple(reduced),initial


@dataclass(frozen=True)
class IntentionGround:
    source_digest: str
    ground_digest: str
    initial: tuple[Q, ...]
    operators: tuple[tuple[str,str,str,tuple,Q], ...]

    @property
    def rank(self):
        return len(self.initial)

    def material(self):
        return {"schema":SCHEMA,"source_digest":self.source_digest,
                "initial":[_q(x) for x in self.initial],
                "operators":[(kind,name,label,_matrix_payload(mat),_q(cost))
                             for kind,name,label,mat,cost in self.operators]}

    def check_identity(self):
        if _hash(self.material()) != self.ground_digest:
            raise ValueError("Altered intention Ground content")

    def actions(self):
        return tuple((name,cost) for kind,name,label,mat,cost in self.operators if kind=="act")

    def sensors(self):
        result={}
        for kind,name,label,mat,cost in self.operators:
            if kind=="sense":
                result.setdefault(name,[]).append((label,cost))
        return {k:tuple(v) for k,v in result.items()}


def compile_intention_ground(source):
    if not isinstance(source,ControlledSource):
        raise ValueError("Exact declared controlled source required")
    _,ops,initial=_construct(source)
    tmp=IntentionGround(source.digest,"",initial,ops)
    ground=IntentionGround(source.digest,_hash(tmp.material()),initial,ops)
    verify_intention_ground(source,ground)
    return ground


def verify_intention_ground(source,ground):
    if not isinstance(source,ControlledSource) or not isinstance(ground,IntentionGround):
        raise ValueError("Invalid source or Ground certificate")
    ground.check_identity()
    columns,operations,initial=_construct(source)
    if (source.digest!=ground.source_digest or initial!=ground.initial or
        ground.operators!=operations or len(columns)!=ground.rank):
        raise ValueError("Source/goal/reduced operator commutation disagrees")
    # Independent direct operator audit, including every column of the goal basis.
    for original,compiled in zip(_operators(source), ground.operators):
        mat,k=original[3],compiled[3]
        for j in range(ground.rank):
            lhs=_matvec(mat,columns[j])
            rhs=tuple(sum((columns[x][i]*k[x][j] for x in range(ground.rank)),Q())
                      for i in range(len(source.initial)))
            if lhs!=rhs:
                raise ValueError("Altered goal-predictive transition")
    return True


@dataclass(frozen=True)
class Decision:
    kind: str
    name: str
    value: Q
    success: Q
    cost: Q
    branches: tuple=()


class IntentionSession:
    """Model-internal belief and planning; actions are PREDICTIONS, not effects."""
    def __init__(self,ground):
        if not isinstance(ground,IntentionGround):
            raise ValueError("Compiled intention Ground is required")
        ground.check_identity()
        self.ground=ground
        self.state=ground.initial
        self.history=[]

    def goal_probability(self):
        return self.state[1]

    def _operator(self,kind,name,label=""):
        for entry in self.ground.operators:
            if entry[:3]==(kind,name,label):
                return entry
        raise ValueError("Unqualified virtual intention operation")

    def sensor_distribution(self,name):
        outcomes=self.ground.sensors().get(name)
        if outcomes is None:
            raise ValueError("Unknown virtual sensor")
        result={}
        for label,_ in outcomes:
            result[label]=_rowmat(self.state,self._operator("sense",name,label)[3])[0]
        if any(x<0 for x in result.values()) or sum(result.values())!=1:
            raise ValueError("Qualified virtual sensor probability was violated")
        return result

    def apply(self,name):
        """Apply a *predicted* intervention; external effect needs E-CPU receipt."""
        mat=self._operator("act",name)[3]
        next_state=_rowmat(self.state,mat)
        if next_state[0]!=1:
            raise ValueError("Intervention was not stochastic")
        self.state=next_state
        self.history.append(("act",name,""))
        return self.goal_probability()

    def observe(self,name,label):
        """Condition on a supplied reading, without attesting its provenance."""
        mat=self._operator("sense",name,label)[3]
        next_state=_rowmat(self.state,mat)
        p=next_state[0]
        if p<=0:
            raise ValueError("Impossible observation cannot be admitted")
        self.state=tuple(x/p for x in next_state)
        self.history.append(("sense",name,label))
        return p

    def plan(self,horizon=2):
        """Exact bounded decision-tree search: P(goal)-expected action/sense cost.

        At each node the planner can STOP, model an intervention, or choose
        a sensor and condition each outcome, then select future operations.
        Costs are declared abstract success-equivalent units, not latency.
        """
        if type(horizon) is not int or not 0<=horizon<=MAX_HORIZON:
            raise ValueError("Unqualified lookahead horizon")
        ground=self.ground
        actions=ground.actions()
        sensors=ground.sensors()
        op={(kind,name,label):mat for kind,name,label,mat,cost in ground.operators}
        @lru_cache(None)
        def best(state,h):
            stop=Decision("stop","",state[1],state[1],Q())
            if h==0:
                return stop
            candidate=stop
            for name,cost in actions:
                next_state=_rowmat(state,op[("act",name,"")])
                if next_state[0]!=1:
                    raise ValueError("Invalid compiled causal transition")
                following=best(next_state,h-1)
                obj=Decision("act",name,following.success-(following.cost+cost),
                             following.success,following.cost+cost)
                if obj.value>candidate.value:
                    candidate=obj
            for name,outcomes in sensors.items():
                sensor_cost=outcomes[0][1]
                success,cost=Q(),sensor_cost
                branches=[]
                for label,_ in outcomes:
                    next_state=_rowmat(state,op[("sense",name,label)])
                    p=next_state[0]
                    if p<0:
                        raise ValueError("Negative outcome probability")
                    if not p:
                        continue
                    posterior=tuple(x/p for x in next_state)
                    future=best(posterior,h-1)
                    success+=p*future.success
                    cost+=p*future.cost
                    branches.append((label,p,future.kind,future.name))
                obj=Decision("sense",name,success-cost,success,cost,tuple(branches))
                if obj.value>candidate.value:
                    candidate=obj
            return candidate
        return best(self.state,horizon)

    def snapshot(self):
        body={"schema":"vli.intention-session.v1","ground_digest":self.ground.ground_digest,
              "history":[list(e) for e in self.history],
              "state":[_q(x) for x in self.state]}
        return {**body,"snapshot_digest":_hash(body)}

    @classmethod
    def restore(cls,ground,snapshot,*,expected_digest):
        if (type(snapshot) is not dict or snapshot.get("snapshot_digest")!=expected_digest
            or _hash({k:v for k,v in snapshot.items() if k!="snapshot_digest"})!=expected_digest
            or snapshot.get("schema")!="vli.intention-session.v1"
            or snapshot.get("ground_digest")!=ground.ground_digest
            or type(snapshot.get("history")) is not list):
            raise ValueError("Unpinned predictive intention session")
        session=cls(ground)
        for kind,name,label in snapshot["history"]:
            if kind=="act" and label=="":
                session.apply(name)
            elif kind=="sense":
                session.observe(name,label)
            else:
                raise ValueError("Invalid restored intention transition")
        if [_q(x) for x in session.state]!=snapshot.get("state"):
            raise ValueError("Altered state refused by independent virtual replay")
        return session
