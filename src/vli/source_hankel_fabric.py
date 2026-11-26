"""PBG-005: receiver-qualified batched scalar trace observations over E-CPU/VGI.

Source kernels remain behind two receiver-provisioned handlers. A trace query
observes exact rational *model probabilities*, not a physical trajectory.
The SAB receipt verifies receiver-executed code; VGI finalizes equality of
separate source-identified reports. Neither implies independent physical truth
or remote TLS identity. It is not an actuator or a pretrained model reader.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
import secrets

from .source_hankel_ground import SourceObservation, digest, encode, rational, MAX_WORD

INPUT='pbg005.trace.request.v1'
CHANNEL='pbg005.trace.channel'
PROTOCOL='pbg005.trace.protocol/1'
GROUND_ID='pbg005.source-trace.v1'
STATE='source-trace'
MAX_BATCH=128


@dataclass(frozen=True)
class ProbeEndpoint:
    source_id: str
    locus: str
    ingress: str='start'
    collector: str='collector'


def operation_id(source_id):
    return f'pbg005.read-trace.{source_id}.v1'


def output_kind(source_id):
    return f'pbg005.trace.result.{source_id}'


def _octal_digest(x):
    """Losslessly encode SHA-256 as 86 Z/8Z symbols for VGI's typed outcome."""
    return [int(c) for c in format(int(x,16),'086o')]


def read_source_batch(runtime,endpoints,source_digest,alphabet,words,*,nonce,coordinate):
    from mailcpu.substrate_bootstrap import (AdaptiveBootstrap,BootstrapTask,
                                             verify_bootstrap_report,digest as ecpu_hash)
    from vgi.channels import ChannelContract,ChannelExchangeLedger,ChannelMessage
    from vgi.encounters import EncounterProvenance
    from vgi.evidence import TransitionEvidenceLedger,TransitionEvidencePolicy,TransportFrameReceipt
    endpoints=tuple(endpoints)
    words=tuple(tuple(x) for x in words)
    alphabet=tuple(alphabet)
    if (not isinstance(runtime,AdaptiveBootstrap) or len(endpoints)!=2
        or len({e.source_id for e in endpoints})!=2
        or len({e.locus for e in endpoints})!=2
        or any(not isinstance(e,ProbeEndpoint) for e in endpoints)
        or not words or len(words)>MAX_BATCH or len(set(words))!=len(words)
        or any(len(w)>MAX_WORD or any(a not in alphabet for a in w) for w in words)
        or type(nonce) is not str or len(nonce)!=64
        or any(a not in '0123456789abcdef' for a in nonce)
        or type(coordinate) is not int or coordinate<0):
        raise ValueError('Invalid source-batch or receiving fabric contract')
    question={'source_digest':source_digest,'alphabet':alphabet,'words':[list(w) for w in words]}
    question_digest=digest(question)
    context=digest({'schema':'pbg005.source-batch.context.v1',
                    'source_digest':source_digest,'question':question_digest,
                    'nonce':nonce,'coordinate':coordinate,
                    'inventory':runtime.inventory.fingerprint,
                    'policy':runtime.policy.fingerprint})
    reports=[];planned=[]
    for endpoint in endpoints:
        task=BootstrapTask(
            intention_id='probe-'+endpoint.source_id,
            ground_digest=context,source=endpoint.ingress,input_kind=INPUT,
            target=endpoint.collector,output_kind=output_kind(endpoint.source_id),
            value={'source_digest':source_digest,'question_digest':question_digest,
                   'words':[list(w) for w in words],'nonce':nonce,
                   'source_id':endpoint.source_id},
            max_quanta=100000,max_steps=16,max_bytes=100000,require_oracle=True)
        if any(k[1]==operation_id(endpoint.source_id) and k[0]!=endpoint.locus
               for k in runtime.realizers):
            raise ValueError('Exact source oracle operation is bound at another locus')
        plan=runtime.plan(task,coordinate=coordinate)
        realizes=[step for step in plan.steps if step.kind=='realize']
        if (len(realizes)!=1 or realizes[0].locus!=endpoint.locus or
            realizes[0].operation!=operation_id(endpoint.source_id) or
            realizes[0].produces!=output_kind(endpoint.source_id)):
            raise ValueError('Unqualified E-CPU oracle host substitution')
        planned.append((task,plan,endpoint))
    for task,plan,endpoint in planned:
        if runtime.plan(task,coordinate=coordinate).fingerprint!=plan.fingerprint:
            raise ValueError('Native path changed before reading source')
        report=runtime.execute(task,plan,coordinate=coordinate)
        if not verify_bootstrap_report(report,task,plan):
            raise ValueError('Invalid source observation E-CPU report')
        value=report['output']
        expected={'request_digest':ecpu_hash(task.value),'source_id':endpoint.source_id,
                  'source_digest':source_digest,'question_digest':question_digest}
        if (type(value) is not dict or set(value)!=set(expected)|{'values'} or
            any(value.get(k)!=v for k,v in expected.items()) or
            type(value['values']) is not list or len(value['values'])!=len(words)):
            raise ValueError('Wrong source, question or trace response geometry')
        for x in value['values']:
            if type(x) is not str or '/' not in x or encode(rational(x))!=x:
                raise ValueError('Noncanonical exact source response')
            if not 0<=Q(x)<=1:
                raise ValueError('Source response violates probability domain')
        reports.append(report)
    # Each source must contribute its *own* exact-value digest to VGI.
    # Never pre-normalize divergent readings into the first source's claim.
    labels_by_source=[_octal_digest(digest({
        'question':question_digest,'values':r['output']['values']})) for r in reports]
    contract=ChannelContract(CHANNEL,PROTOCOL,
                             observation_schema='vgi.transition-observation/1')
    exchange=ChannelExchangeLedger()
    event='probe-'+question_digest[:16]
    messages=[]
    for i,(report,endpoint) in enumerate(zip(reports,endpoints)):
        message=ChannelMessage('pbg005-'+str(i)+'-'+nonce[:20],CHANNEL,PROTOCOL,
            'observation',{'status':'known','source_state':STATE,
                           'action':event,'target':STATE,'output':labels_by_source[i],
                           'ground_fingerprint':report['report_digest']},
            EncounterProvenance(endpoint.source_id,
                                'ecpu-report://'+report['report_digest'],collector='pbg005'))
        messages.append(message)
        exchange=exchange.record_observation(contract,message)
    encounters=exchange.to_encounter_ledger()
    ledger=TransitionEvidenceLedger(TransitionEvidencePolicy(
        GROUND_ID,context,tuple(e.source_id for e in endpoints),2,True))
    for i,message in enumerate(messages,1):
        frame=TransportFrameReceipt.capture(message,transport_id='receiver-local-sab',sequence=i)
        ledger=ledger.ingest_channel_message(exchange,message.message_id,frame,encounters)
    ledger=ledger.finalize(STATE,event)
    ledger=TransitionEvidenceLedger.from_dict(ledger.to_dict())
    decision=ledger.decision(STATE,event)
    if decision.status!='AUTHORIZED':
        raise ValueError('VGI strict source quorum refused divergent source observations: '+decision.status)
    observed=ledger.transition_observation(STATE,event)
    first=reports[0]['output']['values']
    if (len(observed.outcomes)!=1 or
        list(observed.outcomes[0].output)!=labels_by_source[0] or
        any(r['output']['values']!=first for r in reports[1:])):
        raise ValueError('Qualified source reports disagree on exact trace values')
    # Verify E-CPU report lineage inside VGI evidence; a quorum by itself is
    # not an assertion that the original source is semantically correct.
    records=tuple(x for x in ledger.evidence if x.event==(STATE,event))
    if len(records)!=2:
        raise ValueError('Missing VGI source receipt')
    for i,(e,message,report,endpoint) in enumerate(zip(records,messages,reports,endpoints),1):
        frame=TransportFrameReceipt.capture(message,transport_id='receiver-local-sab',sequence=i)
        if (e.message_id!=message.message_id or e.source!=endpoint.source_id or
            e.frame_id!=frame.frame_id or e.payload_digest!=frame.payload_digest or
            e.source_model_fingerprint!=report['report_digest'] or
            e.context_digest!=context or e.outcomes[0].output!=tuple(labels_by_source[i-1])):
            raise ValueError('VGI event cannot be traced to native E-CPU report')
    return tuple(SourceObservation(w,Q(v),source_digest,
        digest({'context':context,'word':w,'value':v,'ecpu':tuple(r['report_digest'] for r in reports),
                'vgi':ledger.fingerprint,'decision':decision.digest})) for w,v in zip(words,first)),{
                    'context_digest':context,'question_digest':question_digest,
                    'ledger_digest':ledger.fingerprint,'report_digests':tuple(r['report_digest'] for r in reports),
                    'source_ids':tuple(e.source_id for e in endpoints),
                    'quorum':decision.status,'word_count':len(words)}


class NativeSourceReader:
    """A source-qualified membership-query adapter; batching saves transport."""
    def __init__(self,runtime,endpoints,source_digest,alphabet,*,coordinate=3,
                 nonce_factory=None,batch_size=MAX_BATCH):
        if not 1<=batch_size<=MAX_BATCH:
            raise ValueError('Invalid native batch size')
        self.runtime=runtime
        self.endpoints=tuple(endpoints)
        self.source_digest=source_digest
        self.alphabet=tuple(alphabet)
        self.coordinate=coordinate
        self.nonce_factory=nonce_factory or (lambda:secrets.token_hex(32))
        self.batch_size=batch_size
        self.lineage=[]

    def __call__(self,words):
        result=[]
        for offset in range(0,len(words),self.batch_size):
            portion=words[offset:offset+self.batch_size]
            records,report=read_source_batch(self.runtime,self.endpoints,self.source_digest,
                self.alphabet,portion,nonce=self.nonce_factory(),coordinate=self.coordinate)
            self.lineage.append(report)
            result.extend(records)
        return tuple(result)
