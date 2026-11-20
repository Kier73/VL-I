"""TIME-007: construct and independently verify an exact rational self-interpreter.

The same five source syntax forms can express the task program and a
source-level interpreter of those very forms. The program computes matrix
powers via binary composition, no materialized future trajectory or hidden
'matrix_power' primitive. A separate continuation-stack verifier reproduces
every result and expression cost.

The *honest external seed* consists of bounded typed tuple construction,
selection/equality, exact rational addition/multiplication, and integer
odd/half. The reflector is synthesized from a fixed declared grammar; its
host bootstrap is not generated. This is bounded metainterpretation, not
self-hosted hardware, infinite reflection, or unrestricted language invention.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from math import factorial, gcd

from .ground import _canonical, _digest, _require
from .temporal_intelligence import _fraction, _encode

SCHEMA="vl-i.intention-constructed-reflective-rational-language.v1"
FORMS=("lit","arg","if","prim","call")
SEED=("at","cons","eq","rest","qadd","qmul","odd","half")
MAX_RATIONAL_BITS=4096
MAX_FUEL=8_000_000


def lit(v):return ("lit",v)
def arg(n):return ("arg",n)
def choose(t,y,n):return ("if",t,y,n)
def primitive(name,*inputs):return ("prim",name,tuple(inputs))
def call(name,*inputs):return ("call",name,tuple(inputs))


def _pair(q):
    return (q.numerator,q.denominator)


def _fraction_pair(x):
    if (type(x) is not tuple or len(x)!=2 or
            any(type(v) is not int for v in x) or x[1]<=0 or
            abs(x[0]).bit_length()>MAX_RATIONAL_BITS or
            x[1].bit_length()>MAX_RATIONAL_BITS or
            gcd(x[0],x[1])!=1):
        raise ValueError("Malformed or oversized canonical rational pair")
    return Fraction(x[0],x[1])


def _qadd(a,b):
    return _pair(_guard(_fraction_pair(a)+_fraction_pair(b)))


def _qmul(a,b):
    return _pair(_guard(_fraction_pair(a)*_fraction_pair(b)))


def _guard(q):
    if (q.numerator.bit_length()>MAX_RATIONAL_BITS or
            q.denominator.bit_length()>MAX_RATIONAL_BITS):
        raise ValueError("Reflected rational exceeds fixed bit resource")
    return q


def _apply_primitive(name,params):
    if name=="at" and len(params)==2:
        seq,index=params
        if type(seq) is tuple and type(index) is int and 0<=index<len(seq):
            return seq[index]
    if name=="rest" and len(params)==1 and type(params[0]) is tuple:
        return params[0][1:]
    if name=="cons" and len(params)==2 and type(params[1]) is tuple:
        return (params[0],)+params[1]
    if name=="eq" and len(params)==2:
        return type(params[0]) is type(params[1]) and params[0]==params[1]
    if name=="qadd" and len(params)==2:return _qadd(*params)
    if name=="qmul" and len(params)==2:return _qmul(*params)
    if name=="odd" and len(params)==1 and type(params[0]) is int and params[0]>=0:
        return (params[0]&1)==1
    if name=="half" and len(params)==1 and type(params[0]) is int and params[0]>=0:
        return params[0]//2
    raise ValueError("Operation absent from declared physical arithmetic seed")


def host_run(program,entry,argv,*,fuel=MAX_FUEL):
    """Small independently testable recursive host bootstrap, no eval/apply."""
    _require(type(program) is tuple and 1<=len(program)<=16 and
             type(argv) is tuple and type(entry) is str and
             type(fuel) is int and 0<fuel<=MAX_FUEL,
             "Invalid finite rational interpreter input")
    defs={}
    for row in program:
        _require(type(row) is tuple and len(row)==2 and
                 type(row[0]) is str and row[0].isidentifier() and
                 row[0] not in defs,
                 "Invalid named rational function table")
        defs[row[0]]=row[1]
    _require(entry in defs,"Unknown expression program entry")
    cost=0

    def evaluate(node,frame):
        nonlocal cost
        cost+=1
        if cost>fuel:
            raise ValueError("Host interpretation exceeded bounded expression fuel")
        if type(node) is not tuple or not node:
            raise ValueError("Malformed expression")
        tag=node[0]
        if tag=="lit" and len(node)==2:
            return node[1]
        if tag=="arg" and len(node)==2 and type(node[1]) is int and 0<=node[1]<len(frame):
            return frame[node[1]]
        if tag=="if" and len(node)==4:
            value=evaluate(node[1],frame)
            if type(value) is not bool:
                raise ValueError("Non-Boolean source branch")
            return evaluate(node[2] if value else node[3],frame)
        if tag in ("prim","call") and len(node)==3 and type(node[1]) is str and type(node[2]) is tuple:
            values=tuple(evaluate(t,frame) for t in node[2])
            if tag=="prim":
                return _apply_primitive(node[1],values)
            if node[1] not in defs:
                raise ValueError("Source called an unknown function")
            return evaluate(defs[node[1]],values)
        raise ValueError("Unknown rational expression syntax")

    try:
        result=evaluate(defs[entry],argv)
    except (RecursionError,IndexError) as exc:
        raise ValueError("Unqualified or overly deep finite interpreter") from exc
    return result,cost


def _independent_primitive(name,values):
    """Independent typed reductions, not a call into producer _apply_primitive."""
    if name=="eq" and len(values)==2:
        return type(values[0])==type(values[1]) and values[0]==values[1]
    if name=="at" and len(values)==2 and type(values[0]) is tuple and type(values[1]) is int:
        if 0<=values[1]<len(values[0]):return values[0][values[1]]
    if name=="cons" and len(values)==2 and type(values[1]) is tuple:
        return (values[0],)+values[1]
    if name=="rest" and len(values)==1 and type(values[0]) is tuple:
        return values[0][1:]
    if name in ("qadd","qmul") and len(values)==2:
        operands=tuple(_fraction_pair(x) for x in values)
        q=sum(operands,Fraction(0)) if name=="qadd" else operands[0]*operands[1]
        return _pair(_guard(q))
    if name in ("half","odd") and len(values)==1 and type(values[0]) is int and values[0]>=0:
        k=values[0]
        return k//2 if name=="half" else k%2!=0
    raise ValueError("Independent verifier refused malformed primitive")


def verify_run(program,entry,argv,*,fuel=MAX_FUEL):
    """Completely iterative continuation VM; no recursive producer evaluation."""
    _require(type(program) is tuple and 1<=len(program)<=16
             and type(entry) is str and type(argv) is tuple
             and type(fuel) is int and 0<fuel<=MAX_FUEL,
             "Independent verifier input is not admitted")
    defs={}
    for row in program:
        if type(row) is not tuple or len(row)!=2 or type(row[0]) is not str or row[0] in defs:
            raise ValueError("Malformed independent function table")
        defs[row[0]]=row[1]
    _require(entry in defs,"Independent verifier missing entrypoint")
    work=[("node",defs[entry],argv)]
    values=[]
    steps=0
    while work:
        item=work.pop()
        if item[0]=="node":
            steps+=1
            if steps>fuel:raise ValueError("Independent verifier fuel exhausted")
            node,frame=item[1],item[2]
            if type(node) is not tuple or not node:raise ValueError("Malformed independent AST")
            tag=node[0]
            if tag=="lit" and len(node)==2:
                values.append(node[1])
            elif tag=="arg" and len(node)==2 and type(node[1]) is int and 0<=node[1]<len(frame):
                values.append(frame[node[1]])
            elif tag=="if" and len(node)==4:
                work.append(("branch",node[2],node[3],frame))
                work.append(("node",node[1],frame))
            elif tag in ("call","prim") and len(node)==3 and type(node[1]) is str and type(node[2]) is tuple:
                work.append(("invoke",tag,node[1],len(node[2])))
                for child in reversed(node[2]):
                    work.append(("node",child,frame))
            else:raise ValueError("Independent source AST rejected")
        elif item[0]=="branch":
            test=values.pop()
            if type(test) is not bool:raise ValueError("Nonboolean reference branch")
            work.append(("node",item[1] if test else item[2],item[3]))
        elif item[0]=="invoke":
            _,kind,name,n=item
            if n>len(values):raise ValueError("Insufficient argument values")
            frame=tuple(values[-n:]) if n else ()
            if n:del values[-n:]
            if kind=="prim":
                values.append(_independent_primitive(name,frame))
            else:
                if name not in defs:raise ValueError("Unbound reference function")
                work.append(("node",defs[name],frame))
        else:
            raise ValueError("Invalid continuation")
    _require(len(values)==1,"Independent interpreter did not reduce to a value")
    return values[0],steps


def _take(node,index):
    return primitive("at",node,lit(index))


def _tuple(items):
    result=lit(())
    for item in reversed(items):
        result=primitive("cons",item,result)
    return result


def _qsum(expressions):
    result=lit((0,1))
    for term in expressions:
        result=primitive("qadd",result,term)
    return result


def _compile_matrix_multiply(d):
    A,B=arg(0),arg(1)
    rows=[]
    for i in range(d):
        cells=[]
        for j in range(d):
            cells.append(_qsum(tuple(
                primitive("qmul",_take(_take(A,i),k),_take(_take(B,k),j))
                for k in range(d))))
        rows.append(_tuple(cells))
    return _tuple(rows)


def _compile_matrix_vector(d):
    A,V=arg(0),arg(1)
    return _tuple([_qsum(tuple(
        primitive("qmul",_take(_take(A,i),j),_take(V,j))
        for j in range(d))) for i in range(d)])


def _identity(d):
    return tuple(tuple((int(i==j),1) for j in range(d))
                 for i in range(d))


def compile_intention_program(d,output_index):
    """Construct task processor source from DIMENSION and output intention."""
    _require(type(d) is int and 1<=d<=3 and
             type(output_index) is int and 0<=output_index<d,
             "Unadmitted source-derived matrix task")
    n,A,acc=arg(0),arg(1),arg(2)
    power=choose(primitive("eq",n,lit(0)),acc,
        call("power",primitive("half",n),
             call("matmul",A,A),
             choose(primitive("odd",n),
                    call("matmul",acc,A),acc)))
    main=_take(call("matvec",call("power",arg(0),arg(1),lit(_identity(d))),
                    arg(2)),output_index)
    return (("matmul",_compile_matrix_multiply(d)),
            ("matvec",_compile_matrix_vector(d)),
            ("power",power),
            ("main",main))


def _codebook(seed):
    available=sorted(FORMS,key=lambda name:sha256(
        (seed+":"+name).encode("ascii")).hexdigest())
    coord=int(seed[:16],16)%factorial(len(FORMS))
    order=[]
    for count in range(len(available),0,-1):
        digit,coord=divmod(coord,factorial(count-1))
        order.append(available.pop(digit))
    return tuple((name,index) for index,name in enumerate(order))


def _encode_source(program,book):
    codes=dict(book)
    def encode(node):
        _require(type(node) is tuple and node and node[0] in FORMS,
                 "Unadmitted reflective source form")
        tag=node[0]
        if tag=="lit":
            _require(len(node)==2,"Malformed quoted source")
            return (codes[tag],node[1])
        if tag=="arg":
            _require(len(node)==2 and type(node[1]) is int,"Malformed source address")
            return (codes[tag],node[1])
        if tag=="if":
            _require(len(node)==4,"Malformed source condition")
            return (codes[tag],encode(node[1]),encode(node[2]),encode(node[3]))
        _require(len(node)==3 and type(node[1]) is str and type(node[2]) is tuple,
                 "Malformed source call")
        return (codes[tag],node[1],tuple(encode(child) for child in node[2]))
    return tuple((name,encode(expr)) for name,expr in program)


def compile_self_interpreter(book):
    """Generate same-language source-level eval, no host eval or apply primitive."""
    codes=dict(book)
    defs,name=arg(0),arg(1)
    first=_take(defs,0)
    lookup=choose(primitive("eq",_take(first,0),name),
                  _take(first,1),call("lookup",primitive("rest",defs),name))
    xs,frame,program=arg(0),arg(1),arg(2)
    evaluate_list=choose(
        primitive("eq",xs,lit(())),lit(()),
        primitive("cons",
            call("evaluate",_take(xs,0),frame,program),
            call("evaluate_list",primitive("rest",xs),frame,program)))
    op,params=arg(0),arg(1)
    dispatch=_take(lit(()),0) # fail-closed source for unknown external op
    for name in reversed(SEED):
        arity=1 if name in ("rest","odd","half") else 2
        implementation=primitive(name,*(_take(params,i) for i in range(arity)))
        dispatch=choose(primitive("eq",op,lit(name)),implementation,dispatch)
    node,env,defs=arg(0),arg(1),arg(2)
    tag=_take(node,0)
    options={
      "lit":_take(node,1),
      "arg":primitive("at",env,_take(node,1)),
      "if":choose(call("evaluate",_take(node,1),env,defs),
                  call("evaluate",_take(node,2),env,defs),
                  call("evaluate",_take(node,3),env,defs)),
      "prim":call("perform",_take(node,1),
                  call("evaluate_list",_take(node,2),env,defs)),
      "call":call("invoke",defs,_take(node,1),
                  call("evaluate_list",_take(node,2),env,defs)),
    }
    evaluator=_take(lit(()),0)
    for syntax in reversed(FORMS):
        evaluator=choose(primitive("eq",tag,lit(codes[syntax])),
                         options[syntax],evaluator)
    invoke=call("evaluate",
                call("lookup",arg(0),arg(1)),arg(2),arg(0))
    run=call("invoke",arg(0),arg(1),arg(2))
    return (("lookup",lookup),
            ("evaluate_list",evaluate_list),
            ("perform",dispatch),
            ("evaluate",evaluator),
            ("invoke",invoke),
            ("run",run))


def _matrix_input(space):
    A=tuple(tuple(_pair(_fraction(v)) for v in row)
            for row in space["transition_operator"])
    V=tuple(_pair(_fraction(v)) for v in space["initial_state"])
    return A,V


def reflect_intention(space,step,*,expected_value,depth=2):
    """Construct task and metainterpreter from admitted state-space intention.

    Source-level reflection is independently checked against the iterative
    continuation machine, not merely compared to the source model checksum.
    """
    _require(type(space) is dict and
             space.get("schema")=="vl-i.intention-derived-rational-state-space.v1"
             and type(step) is int and 0<=step<=1_048_576
             and type(depth) is int and 0<=depth<=2,
             "Unadmitted reflective state-space intention")
    _fraction(expected_value)
    d=space["dimension"]
    output=space["output_coordinate"]
    book=_codebook(space["intention_space_digest"])
    program=compile_intention_program(d,output)
    interpreter=compile_self_interpreter(book)
    matrix,initial=_matrix_input(space)
    input_data=(step,matrix,initial)
    results=[]
    for level in range(depth+1):
        if level==0:
            host_program,entry,args=program,"main",input_data
        else:
            encoded_task=_encode_source(program,book)
            inner=(encoded_task,"main",input_data)
            if level==1:
                host_program,entry,args=interpreter,"run",inner
            else:
                encoded_self=_encode_source(interpreter,book)
                host_program,entry,args=interpreter,"run",(
                    encoded_self,"run",inner)
        try:
            value,steps=host_run(host_program,entry,args)
        except ValueError as exc:
            # Limit of the recursive host is a real resource refusal.
            raise ValueError("Reflective source execution was not admitted") from exc
        checked,other=verify_run(host_program,entry,args)
        _require(value==checked and steps==other and
                 type(value) is tuple and len(value)==2,
                 "Independent reflective interpreter disagrees")
        _require(_encode(_fraction_pair(value))==expected_value,
                 "Reflective state-space processor disagrees with mathematical Ground")
        results.append({
            "depth":level,"expression_steps":steps,
            "result_exact_rational":expected_value,
            "host_and_independent_continuation_agree":True,
        })
    body={
        "schema":SCHEMA,
        "intention_space_digest":space["intention_space_digest"],
        "selected_coordinate":step,
        "state_dimension":d,
        "source_program":program,
        "source_program_sha256":_digest(SCHEMA+".program",program),
        "self_interpreter_sha256":_digest(SCHEMA+".interpreter",interpreter),
        "source_codebook":book,
        "declared_external_primitive_seed":list(SEED),
        "generated_mathematical_language":(
            "matrix-composition-exponentiation-by-squaring"),
        "reflection_receipts":results,
        "trajectory_states_constructed":0,
        "primitive_arithmetic_is_external_seed":True,
        "reflector_is_unbounded_hardware":False,
    }
    return {**body,"reflection_digest":_digest(SCHEMA,body)}
