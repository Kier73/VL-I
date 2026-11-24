"""PBG-005: observer-relative predictive law derived from exact source probes.

Learning sees only source-fingerprint-bound trace weights P(word); no hidden
source matrices are used during the construction. Candidate exact-rational
finite linear representations use Hankel prefix/suffix bases. Counterexamples
can expand the basis monotonically. A separate source-native commutation audit
qualifies infinite-horizon interpretation for a *declared* finite source.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from hashlib import sha256
import json
from itertools import product

SCHEMA = 'vli.source-hankel-ground.v1'
MAX_ALPHABET = 12
MAX_RANK = 16
MAX_WORD = 32


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def digest(x):
    return sha256(canonical(x)).hexdigest()


def rational(x):
    if type(x) not in (str, int, Q):
        raise ValueError('Exact rational source observation required')
    return Q(x)


def encode(x):
    return f'{x.numerator}/{x.denominator}'


def matvec(mat,vec):
    return tuple(sum((a*b for a,b in zip(row,vec)),Q()) for row in mat)


def rowmat(row,mat):
    return tuple(sum((row[k]*mat[k][j] for k in range(len(row))),Q()) for j in range(len(mat[0])))


def matrixmul(left,right):
    return tuple(tuple(sum((left[i][k]*right[k][j] for k in range(len(left[0]))),Q())
                       for j in range(len(right[0]))) for i in range(len(left)))


def solve(matrix, rhs):
    """Exact square solve. Reject singular or inconsistent matrices."""
    n=len(matrix)
    if n==0 or len(rhs)!=n or any(len(row)!=n for row in matrix):
        raise ValueError('Non-square Hankel basis')
    table=[list(row)+[rhs[i]] for i,row in enumerate(matrix)]
    for c in range(n):
        pivot=next((i for i in range(c,n) if table[i][c]),None)
        if pivot is None:
            raise ValueError('Singular Hankel basis')
        table[c],table[pivot]=table[pivot],table[c]
        norm=table[c][c]
        table[c]=[x/norm for x in table[c]]
        for i in range(n):
            if i!=c and table[i][c]:
                mul=table[i][c]
                table[i]=[a-mul*b for a,b in zip(table[i],table[c])]
    return tuple(row[-1] for row in table)


def rank(matrix):
    if not matrix:
        return 0
    n=len(matrix[0]);rows=[list(x) for x in matrix]
    if any(len(row)!=n for row in rows):
        raise ValueError('Ragged matrix')
    r=0
    for c in range(n):
        pivot=next((i for i in range(r,len(rows)) if rows[i][c]),None)
        if pivot is None:
            continue
        rows[r],rows[pivot]=rows[pivot],rows[r]
        v=rows[r][c]
        rows[r]=[x/v for x in rows[r]]
        for i in range(r+1,len(rows)):
            if rows[i][c]:
                m=rows[i][c]
                rows[i]=[a-m*b for a,b in zip(rows[i],rows[r])]
        r+=1
        if r==len(rows):
            break
    return r


def _strings(items):
    return [list(w) for w in items]


@dataclass(frozen=True)
class SourceObservation:
    word: tuple[str,...]
    probability: Q
    source_digest: str
    receipt_digest: str

    def __post_init__(self):
        if not isinstance(self.word,tuple) or any(type(x) is not str or not x for x in self.word):
            raise ValueError('Invalid source word')
        p=rational(self.probability)
        if not 0<=p<=1 or len(self.word)>MAX_WORD:
            raise ValueError('Invalid source observation mass')
        for h in (self.source_digest,self.receipt_digest):
            if type(h) is not str or len(h)!=64 or any(x not in '0123456789abcdef' for x in h):
                raise ValueError('Invalid source qualification identity')
        object.__setattr__(self,'probability',p)


@dataclass(frozen=True)
class LearnedGround:
    alphabet: tuple[str,...]
    prefixes: tuple[tuple[str,...],...]
    suffixes: tuple[tuple[str,...],...]
    initial: tuple[Q,...]
    terminal: tuple[Q,...]
    transforms: tuple[tuple[tuple[Q,...],...],...]
    source_digest: str
    observations_digest: str
    ground_digest: str
    qualified: bool=False

    @property
    def rank(self):
        return len(self.initial)

    def material(self):
        return {'schema':SCHEMA,'alphabet':self.alphabet,'prefixes':_strings(self.prefixes),
                'suffixes':_strings(self.suffixes),'initial':list(map(encode,self.initial)),
                'terminal':list(map(encode,self.terminal)),
                'transforms':[[list(map(encode,row)) for row in m] for m in self.transforms],
                'source_digest':self.source_digest,'observations_digest':self.observations_digest}

    def check_identity(self):
        if digest(self.material())!=self.ground_digest:
            raise ValueError('Altered learned Ground identity')

    def probability(self,word):
        self.check_identity()
        if len(word)>MAX_WORD:
            raise ValueError('Word exceeds finite query bound')
        z=self.initial
        for a in word:
            try: i=self.alphabet.index(a)
            except ValueError as exc:raise ValueError('Unknown predicted event') from exc
            z=rowmat(z,self.transforms[i])
        return sum((a*b for a,b in zip(z,self.terminal)),Q())

    def next_state(self,z,event,*,conditional=True):
        nextrow=rowmat(z,self.transforms[self.alphabet.index(event)])
        p=sum((a*b for a,b in zip(nextrow,self.terminal)),Q())
        if conditional:
            if p<=0:
                raise ValueError('Impossible or inadmissible event')
            return tuple(x/p for x in nextrow),p
        return nextrow,p


class QueryAtlas:
    """Source observations are acquired through an injected qualified reader.

    reader(words) returns SourceObservation records for requested unobserved
    words. The atlas caches qualified facts and rejects changed source identity.
    No trust arises from a caller-constructed unkeyed digest: a production
    reader must independently bind source, authorization, and receipt lineage.
    """
    def __init__(self,alphabet,source_digest,reader,max_queries=1500):
        alphabet=tuple(alphabet)
        if (not 1<=len(alphabet)<=MAX_ALPHABET or len(set(alphabet))!=len(alphabet)
            or any(type(a) is not str or not a for a in alphabet)
            or type(source_digest) is not str or len(source_digest)!=64
            or not callable(reader) or type(max_queries) is not int or max_queries<1):
            raise ValueError('Invalid finite source-reader contract')
        self.alphabet=alphabet
        self.source_digest=source_digest
        self.reader=reader
        self.max_queries=max_queries
        self.cache={}
        self.receipts={}
        self.calls=0

    def observe_many(self,words):
        requested=tuple(dict.fromkeys(tuple(w) for w in words))
        if any(len(w)>MAX_WORD or any(a not in self.alphabet for a in w) for w in requested):
            raise ValueError('Word outside declared source alphabet')
        missing=tuple(w for w in requested if w not in self.cache)
        if missing:
            if len(self.cache)+len(missing)>self.max_queries:
                raise ValueError('Source query budget exhausted')
            records=tuple(self.reader(missing))
            if len(records)!=len(missing):
                raise ValueError('Source reader omitted or added a query')
            staged=[]
            for word,record in zip(missing,records):
                if (not isinstance(record,SourceObservation) or record.word!=word
                    or record.source_digest!=self.source_digest):
                    raise ValueError('Source receipt does not bind exact requested word')
                staged.append(record)
            for record in staged:
                self.cache[record.word]=record.probability
                self.receipts[record.word]=record.receipt_digest
            self.calls+=1
        return tuple(self.cache[w] for w in requested)

    def probability(self,word):
        return self.observe_many((word,))[0]

    @property
    def evidence_digest(self):
        return digest({'source':self.source_digest,
                       'events':[(list(w),encode(self.cache[w]),self.receipts[w])
                                 for w in sorted(self.cache)]})


class HankelLearner:
    """Adaptive exact Hankel learner. External counterexample grows the basis.

    Compact learned law is a candidate until independent source verifier
    checks commutation. Finite agreement alone is not proof of all futures.
    """
    def __init__(self,atlas,max_rank=MAX_RANK):
        if not isinstance(atlas,QueryAtlas):
            raise ValueError('Requires qualified source observer')
        self.atlas=atlas
        self.max_rank=max_rank
        self.prefix_candidates=[()]
        self.suffix_candidates=[()]
        self.counterexamples=[]
        self.ground=None
        self.revisions=[]

    def _expand(self,collection,frontier):
        existing=set(collection)
        for word in frontier:
            if len(word)<=MAX_WORD and word not in existing:
                collection.append(word);existing.add(word)

    def build(self):
        atlas=self.atlas
        # Expand each already-witnessed coordinate by one observation.
        rows=list(self.prefix_candidates)
        cols=list(self.suffix_candidates)
        self._expand(rows,(p+(a,) for p in self.prefix_candidates for a in atlas.alphabet))
        self._expand(cols,((a,)+s for s in self.suffix_candidates for a in atlas.alphabet))
        # Batch the rectangular Hankel evidence instead of one transport event per scalar.
        word_matrix=[[r+s for s in cols] for r in rows]
        atlas.observe_many(w for row in word_matrix for w in row)
        full=[[atlas.cache[w] for w in row] for row in word_matrix]
        chosen_rows=[]
        for i,row in enumerate(full):
            if rank([full[k] for k in chosen_rows]+[row])>len(chosen_rows):
                chosen_rows.append(i)
        if len(chosen_rows)>self.max_rank:
            raise ValueError('Observed rank exceeds bounded interpretation')
        chosen_cols=[]
        for j in range(len(cols)):
            projected=[[full[i][k] for k in chosen_cols+[j]] for i in chosen_rows]
            if rank(projected)>len(chosen_cols):
                chosen_cols.append(j)
            if len(chosen_cols)==len(chosen_rows):
                break
        if len(chosen_cols)!=len(chosen_rows):
            raise AssertionError('Hankel basis pivot failure')
        P=tuple(rows[i] for i in chosen_rows)
        S=tuple(cols[j] for j in chosen_cols)
        H=tuple(tuple(atlas.cache[p+s] for s in S) for p in P)
        r=len(P)
        atlas.observe_many(p+(a,)+s for p in P for a in atlas.alphabet for s in S)
        transforms=[]
        for a in atlas.alphabet:
            ha=tuple(tuple(atlas.cache[p+(a,)+s] for s in S) for p in P)
            # H * K = H_a => K's columns use one independent rational solve.
            columns=[solve(H,tuple(ha[i][j] for i in range(r))) for j in range(r)]
            transforms.append(tuple(tuple(columns[j][i] for j in range(r)) for i in range(r)))
        initial=tuple(atlas.probability(s) for s in S)
        terminal=solve(H,tuple(atlas.probability(p) for p in P))
        tmp=LearnedGround(atlas.alphabet,P,S,initial,terminal,tuple(transforms),
                          atlas.source_digest,atlas.evidence_digest,'')
        ground=LearnedGround(atlas.alphabet,P,S,initial,terminal,tuple(transforms),
                             atlas.source_digest,atlas.evidence_digest,digest(tmp.material()))
        if ground.probability(())!=1:
            raise ValueError('Source did not normalize empty word to one')
        # A full rank is not implied by finite samples; caller must challenge.
        self.prefix_candidates=list(P)
        self.suffix_candidates=list(S)
        self.ground=ground
        self.revisions.append(ground.ground_digest)
        return ground

    def counterexample(self,witness):
        if self.ground is None:
            raise ValueError('Compile the provisional law first')
        witness=tuple(witness)
        if not witness or len(witness)>MAX_WORD:
            raise ValueError('Counterexample has invalid length')
        actual=self.atlas.probability(witness)
        prediction=self.ground.probability(witness)
        if actual==prediction:
            raise ValueError('Not a source-distinguishing counterexample')
        old=self.ground
        self.counterexamples.append((witness,actual,prediction,self.atlas.receipts[witness]))
        self._expand(self.prefix_candidates,(witness[:i] for i in range(len(witness)+1)))
        self._expand(self.suffix_candidates,(witness[i:] for i in range(len(witness)+1)))
        result=self.build()
        if result.rank<=old.rank:
            # A counterexample may require another distinction not yet resolved.
            # Never publish a false rank-increase or all-horizon certificate.
            if result.probability(witness)!=actual:
                raise ValueError('Counterexample did not resolve into the observed Ground')
        if result.probability(witness)!=actual:
            raise ValueError('Refined Ground still misses accepted evidence')
        return result


class HankelObserverSession:
    """Source-free replay after a separate exact finite-source qualification.

    Qualification must be issued by the receiving verifier. A raw, caller-
    supplied hash is not a cryptographic certificate of source authenticity.
    """
    def __init__(self,ground,source):
        proof=certify_source_equivalence(source,ground)
        if proof['status']!='EQUIVALENT_ALL_WORDS':
            raise ValueError('Learned Ground requires independently verified source equivalence')
        self.ground=ground
        self.qualification=proof
        self.state=ground.initial
        self.history=[]

    def mass(self,event):
        if event not in self.ground.alphabet:
            raise ValueError('Operation is not admitted by the Ground')
        _,p=self.ground.next_state(self.state,event,conditional=False)
        return p

    def apply(self,operation):
        event='act:'+operation
        updated,p=self.ground.next_state(self.state,event,conditional=False)
        if p!=1:
            raise ValueError('Non-stochastic intervention in learned representation')
        self.state=updated
        self.history.append(('act',operation,''))

    def observe(self,sensor,label):
        event='sense:'+sensor+':'+str(label)
        updated,p=self.ground.next_state(self.state,event,conditional=True)
        self.state=updated
        self.history.append(('sense',sensor,str(label)))
        return p

    def goal_probability(self,goal_sensor='read-goal',success_label='1'):
        return self.mass('sense:'+goal_sensor+':'+success_label)

    def replay_events(self,events):
        """Pure structural-memory replay of already qualified event *labels*.

        This method does not reverify E-CPU/VGI receipts; the caller must use
        PBG-004's accepted event chain, and pin that chain independently.
        """
        for event in events:
            kind=event['kind']; name=event['operation']
            if kind=='act': self.apply(name)
            elif kind=='sense': self.observe(name,event['label'])
            else: raise ValueError('Unqualified historical event kind')
        return self

    def snapshot(self):
        body={'schema':'vli.hankel-observer-session.v1',
              'source':self.ground.source_digest,'ground':self.ground.ground_digest,
              'proof':digest(self.qualification),
              'state':list(map(encode,self.state)),
              'history':[[kind,name,label] for kind,name,label in self.history]}
        return {**body,'snapshot_digest':digest(body)}

def _source_operators(source,alphabet):
    """Independent verifier-only adapter to declared ControlledSource."""
    from .intention_predictive_ground import ControlledSource
    if not isinstance(source,ControlledSource):
        raise ValueError('Full-source audit requires an exact declared source law')
    source_ops={}
    for name,m,c in source.actions:
        source_ops['act:'+name]=m
    for name,outcomes,c in source.sensors:
        for label,m in outcomes:
            source_ops['sense:'+name+':'+label]=m
    if set(source_ops)!=set(alphabet):
        raise ValueError('Learned operational alphabet does not match declared source')
    return source_ops


def source_trace_probability(source,word):
    """Exact source-native oracle, never used inside HankelLearner."""
    ops=_source_operators(source,tuple(
        ['act:'+a for a,_,_ in source.actions]+[
        'sense:'+name+':'+label for name,outs,_ in source.sensors for label,_ in outs]))
    state=source.initial
    for item in word:
        state=rowmat(state,ops[item])
    return sum(state,Q())


def verify_source_closure(source,ground):
    """Independent source-global commutation certificate for fully supplied class.

    The learner does not see these matrices. The verifier checks M_a B=B K_a
    for each operation a, as well as terminal and initial readouts. Finite
    samples alone do not imply this certificate.
    """
    if not isinstance(ground,LearnedGround):
        raise ValueError('Not a learned Ground')
    ground.check_identity()
    if source.digest!=ground.source_digest:
        raise ValueError('Source identity changed')
    ops=_source_operators(source,ground.alphabet)
    n=len(source.initial);ones=tuple(Q(1) for _ in range(n))
    columns=[]
    for suffix in ground.suffixes:
        v=ones
        for a in reversed(suffix):
            v=matvec(ops[a],v)
        columns.append(v)
    if rank([[col[i] for col in columns] for i in range(n)])!=ground.rank:
        raise ValueError('Observed suffixes are not an independent source basis')
    alpha=tuple(sum((a*col[i] for i,a in enumerate(source.initial)),Q()) for col in columns)
    if alpha!=ground.initial:
        raise ValueError('Ground initial observation disagrees with source')
    for i in range(n):
        if sum((columns[j][i]*ground.terminal[j] for j in range(ground.rank)),Q())!=1:
            raise ValueError('Source terminal observation cannot be reconstructed')
    for a,k in zip(ground.alphabet,ground.transforms):
        for j,col in enumerate(columns):
            observed=matvec(ops[a],col)
            reduced=tuple(sum((columns[q][i]*k[q][j] for q in range(ground.rank)),Q())
                          for i in range(n))
            if observed!=reduced:
                raise ValueError('Source-global transition closure fails for '+a)
    return True


def certify_source_equivalence(source,ground):
    """Exact all-word equivalence or a *minimal-length* source counterexample.

    Independent verifier only. The paired reachable row-space for the block
    diagonal source/learner automaton has dimension <= n_source + n_ground.
    Exploration stops when no new independent joint row can be reached, not
    after a guessed finite horizon. This is a real linear-algebra certificate
    for the *declared exact finite linear source*, not neural-source discovery.
    """
    if not isinstance(ground,LearnedGround):
        raise ValueError('Requires a learned Ground')
    ground.check_identity()
    if source.digest!=ground.source_digest:
        raise ValueError('Source identity mismatch')
    ops=_source_operators(source,ground.alphabet)
    n=len(source.initial);r=ground.rank
    queue=[((),tuple(source.initial),ground.initial)]
    spanning=[];witnesses=[];cursor=0
    while cursor<len(queue):
        word,source_row,ground_row=queue[cursor]
        cursor+=1
        joint=tuple(source_row)+tuple(ground_row)
        if rank(spanning+[joint])==len(spanning):
            continue
        spanning.append(joint)
        witnesses.append(word)
        actual=sum(source_row,Q())
        predicted=sum((x*y for x,y in zip(ground_row,ground.terminal)),Q())
        if actual!=predicted:
            return {'status':'COUNTEREXAMPLE','word':word,'actual':actual,
                    'predicted':predicted,'reachable_rank':len(spanning),
                    'source_digest':source.digest,'ground_digest':ground.ground_digest,
                    'witness_count':len(witnesses)}
        if len(spanning)>n+r:
            raise AssertionError('Paired reachable span exceeds source+learner dimension')
        for a,k in zip(ground.alphabet,ground.transforms):
            queue.append((word+(a,),rowmat(source_row,ops[a]),rowmat(ground_row,k)))
    return {'status':'EQUIVALENT_ALL_WORDS','word':None,
            'reachable_rank':len(spanning),'source_digest':source.digest,
            'ground_digest':ground.ground_digest,'witness_count':len(witnesses),
            'certificate_kind':'exact-paired-reachable-linear-span'}


def find_counterexample(source,ground,max_length=None):
    """Counterexample oracle from full finite source; no brute-force word search.

    Optional max_length is a *reporting* cap: use the full proof first, then
    refuse to call a longer sequence a bounded witness. No match means exact
    all-future equivalence, not merely finite corpus agreement.
    """
    proof=certify_source_equivalence(source,ground)
    if proof['status']=='EQUIVALENT_ALL_WORDS':
        return None
    if max_length is not None and len(proof['word'])>max_length:
        raise ValueError('Shortest source counterexample exceeds requested horizon')
    return proof['word'],proof['actual'],proof['predicted']
