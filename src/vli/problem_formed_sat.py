"""PFI-SAT-001: direct-SAT problem-formed segments on native fenced E-CPU.

The same source-CNF Ground reconstructs the machine, typed packet graph, and
operator before each execution.  Exactly one accepted native claim epoch may
publish a pure segment, even when a prior host calculated the right result.
Three on-disk quorum authorities share a filesystem in this bounded test;
this is not independently networked mTLS execution or effect-resource fencing.
"""
from __future__ import annotations

from pathlib import Path
from hashlib import sha256

from .problem_formed_claims import _atomic_json,_read_json


def _sat():
    try:
        from self_forming_sat import form,verify_program,execute_partition,check_segment,aggregate,digest,canonical
    except ImportError as exc:
        raise RuntimeError('SFI-001 pinned In-Formation-Space experiment is required') from exc
    return form,verify_program,execute_partition,check_segment,aggregate,digest,canonical


def _native():
    from mailcpu import (RecoveryLineageAuthority,RecoveryLineageCoordinator,
                         SeedCommitMembership,TransferableExecutionClaimStore,
                         OperatorSpec,Capability,HeadlessOS,HypervisorAction,
                         LexicalDecisionModel,HypervisorPolicy,OperatorRegistry,
                         SystemIdentity,SystemSeed,SystemSeedEnvelope,VirtualBudget,
                         VirtualGroundRef,VirtualLayerRuntime,VirtualLayerWorkload,
                         VirtualOperatorRef)
    from mailcpu.computation_dag import ComputationNode,NavigableComputationDAG
    from mailcpu.owned_network import OwnedNetworkFabric,NetworkAddress
    return locals()


def _need(ok,why):
    if not ok:raise ValueError(why)


def _hash(obj):
    return sha256(_sat()[-1](obj)).hexdigest()


def _stage_spec():
    O=_native()['OperatorSpec']
    return O('pfi.sat_source_segment',1,
             'Execute exact source CNF predicate over a source-verified formed coordinate partition',
             'object','object',('cpu',),required_rights=frozenset({'compute'}),
             base_energy=1,energy_per_byte=1,max_input_bytes=8192,
             max_output_bytes=8192,deterministic=True,effect_mode='pure')


class DirectSatFabric:
    """Native E-CPU FIFO, DAG, Hypervisor, and majority-published SAT partitions."""
    SCHEMA='vli.problem-formed-sat-fabric.v1'
    CANDIDATE='vli.sat-stage-candidate.v1'

    def __init__(self,root,source_receipt, *,intention='#SAT',
                 maximum_free=12,partition_size=8,job='sat-formed',runtime_adapter=None):
        self._sat_runtime = _sat if runtime_adapter is None else runtime_adapter
        _need(callable(self._sat_runtime), 'SAT runtime adapter is not callable')
        form,verify_program,*_= self._sat_runtime()
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.receipt=source_receipt
        self.program,self.construction=form(source_receipt,intention=intention,
                         maximum_free=maximum_free,partition_size=partition_size)
        _need(self.program['mode']=='constructed-search',
              'Host execution only for source-verified, admitted finite search')
        _need(verify_program(self.program,source_receipt),
              'Formed interpreter lacks exact source Ground')
        _need(type(job) is str and 1<=len(job)<=48 and job.isascii() and
              all(c.isalnum() or c in '_-' for c in job),
              'Invalid source-bound work identity')
        self.job=job
        self.identity=_hash({'job':job,'source':self.program['source_sha256'],
                 'program':self.program['digest'],'intention':intention})
        m=_native();membership=m['SeedCommitMembership']('sat-formed-authority',1,
                       ('A','B','C'),'A',1,'0'*64)
        authorities={x:m['RecoveryLineageAuthority'](membership,x,
                 self.root/'authorities'/(x+'.jsonl')) for x in membership.member_ids}
        self.coordinator=m['RecoveryLineageCoordinator'](membership,authorities)
        self.store=m['TransferableExecutionClaimStore'](self.root/'claims',self.coordinator)
        self.spec=_stage_spec()
        self.checkpoint=self.root/'processor.json'
        self._setup()
        if self.checkpoint.exists():
            self._restore(_read_json(self.checkpoint))
        else:self._save()

    def _setup(self):
        m=_native();Part=self.program['partitions'];self.fabric=m['OwnedNetworkFabric'](queue_limit=2)
        self.fabric.create_namespace('origin')
        self.receipts={}
        self.ports={}
        for part in Part:
            i=part['index'];name='sat'+str(i)
            self.fabric.create_namespace(name)
            tx=self.fabric.create_interface('origin','tx'+str(i),mtu=8192)
            rx=self.fabric.create_interface(name,'rx',mtu=8192)
            self.fabric.connect('satlink'+str(i),tx,rx)
            self.ports[i]=(tx,rx)
        Node=m['ComputationNode'];DAG=m['NavigableComputationDAG']
        common={'revision':1,'operation':'source-bound SAT partition',
                'resource':'/vli/problem-formed-sat','context_digest':self.program['digest']}
        nodes=[Node(node_id='dispatch',dependencies=(),payload={'job':self.identity},**common)]
        nodes.extend(Node(node_id='part-'+str(p['index']),dependencies=('dispatch',),
                  payload={'start':p['start'],'stop':p['stop'],'job':self.identity},**common)
                  for p in Part)
        nodes.append(Node(node_id='finish',dependencies=tuple('part-'+str(i)
            for i in range(len(Part))),payload={'job':self.identity},**common))
        self.dag=DAG(nodes)

    def _packet(self,i):
        p=self.program['partitions'][i]
        data={'schema':'vli.sat-segment-packet.v1','job':self.identity,
              'program':self.program['digest'],'source':self.program['source_sha256'],
              'part_index':i,'start':p['start'],'stop':p['stop']}
        data['sha256']=_hash(data)
        return data

    def _wire(self,i):
        _,_,_,_,_,_,canon=self._sat_runtime()
        return canon(self._packet(i))

    def _save(self):
        state={'job':self.identity,'program':self.program['digest'],
               'dag':self.dag.snapshot(),'fabric':self.fabric.snapshot(),
               'receipts':self.receipts}
        _atomic_json(self.checkpoint,{'schema':self.SCHEMA,'state':state,'sha256':_hash(state)})

    def _restore(self,envelope):
        _need(envelope.get('schema')==self.SCHEMA and
              envelope.get('sha256')==_hash(envelope.get('state')),
              'Incorrect source-bound SAT checkpoint')
        s=envelope['state'];_need(s['job']==self.identity and
             s['program']==self.program['digest'],'Changed SAT interpreter epoch')
        m=_native();d=m['NavigableComputationDAG'].from_snapshot(s['dag'])
        f=m['OwnedNetworkFabric'].from_snapshot(s['fabric'])
        _need([n.to_dict() for n in d.nodes]==[n.to_dict() for n in self.dag.nodes],
              'Checkpoint changed the source-derived DAG')
        if d.committed_nodes:
            _need('dispatch' in d.committed_nodes,'Segment committed before dispatch')
        for i in range(len(self.program['partitions'])):
            node='part-'+str(i)
            expected_frame=self._wire(i)
            rx=self.ports[i][1]
            if node in d.committed_nodes:
                _need(str(i) in s['receipts'],'Committed stage has no receipt')
                result=s['receipts'][str(i)]['result']
                self._sat_runtime()[3](self.program,result)
                stage_receipt=s['receipts'][str(i)]
                committed=d.result(node)
                _need(committed.get('result_sha256')==_hash(result) and
                      stage_receipt['result_sha256']==_hash(result) and
                      committed.get('claim_id')==stage_receipt.get('claim_id') and
                      committed.get('epoch')==stage_receipt.get('assignment_epoch'),
                      'Committed result changed')
                ticket=self.store.get(committed['claim_id'])
                ticket.verify_current(self.coordinator)
                _need(ticket.claim.status.value=='completed' and
                      ticket.claim.computational_ground_fingerprint==self.program['digest'] and
                      ticket.claim.parent_request_digest==self.identity and
                      ticket.claim.current_assignment.epoch==committed['epoch'] and
                      ticket.claim.current_assignment.owner_host_id==stage_receipt['host'] and
                      ticket.claim.result_digest==_hash(result),
                      'Restored SAT stage lacks current majority publication')
                _need(f.peek(rx) is None,'Committed partition still has queued frame')
            elif 'dispatch' in d.committed_nodes:
                frame=f.peek(rx)
                _need(frame is not None and frame.source==self.ports[i][0] and
                      frame.payload==expected_frame,
                      'Pending source-bound SAT frame missing or modified')
            else:_need(f.peek(rx) is None,'Undispatched work has queued frame')
        _need(set(s['receipts'])==set(n.split('-')[1] for n in d.committed_nodes
               if n.startswith('part-')),'Extraneous SAT receipt')
        self.fabric=f;self.dag=d;self.receipts=s['receipts']

    def dispatch(self):
        _need('dispatch' not in self.dag.committed_nodes,'Already dispatched')
        for i in range(len(self.program['partitions'])):
            tx,rx=self.ports[i]
            self.fabric.send(tx,rx,self._wire(i),0)
        self.dag.commit('dispatch',{'job':self.identity,'frames':len(self.program['partitions'])})
        self._save()

    def _ready(self,i):
        _need(type(i) is int and 0<=i<len(self.program['partitions']) and
              'part-'+str(i) in {n.node_id for n in self.dag.ready_nodes()},
              'SAT partition not ready or already committed')
        frame=self.fabric.peek(self.ports[i][1])
        _need(frame is not None and frame.payload==self._wire(i),
              'SAT receiver did not receive exact original source-bound frame')
        return self.program['partitions'][i]

    def _bind(self,i):
        p=self._ready(i)
        return {'parent_request_digest':self.identity,
                'child_request_digest':_hash({'program':self.program['digest'],'part':p}),
                'address_digest':_hash({'operator':self.spec.digest,'part':p,
                           'source':self.program['source_sha256']}),
                'operator_spec':self.spec,
                'computational_ground_fingerprint':self.program['digest']}

    @staticmethod
    def _actor(peer,host):
        _need(type(peer) is str and type(host) is str and 1<=len(peer)<=48 and
              1<=len(host)<=48 and all(x.isascii() and
              all(c.isalnum() or c in '-_.' for c in x) for x in (peer,host)),
              'Unqualified SAT computational host identity')
        return {'owner_peer_id':peer,'owner_host_id':host,
                'implementation_fingerprint':_hash({'local-sat-impl':'source-cnfeval-v1',
                                                        'host':host})}

    def claim(self,i,*,peer,host):
        return self.store.create(**self._bind(i),**self._actor(peer,host))

    def _check_target(self,i,claim_id,peer,host):
        token=self.store.get(claim_id)
        token.verify_for_target(self.coordinator,**self._bind(i),**self._actor(peer,host))
        return token

    def _host_execute(self,i,host,*,grant_compute=True):
        M=_native();spec=self.spec;program=self.program;call=[]
        os=M['HeadlessOS']()
        os.hypervisor.classifier=M['LexicalDecisionModel']()
        os.hypervisor.policy=M['HypervisorPolicy'](embedding_review_confidence=1.0)
        sid=M['SystemIdentity']('sat-formed-'+host,1)
        envelope=M['SystemSeedEnvelope'](sid,'formed-sat','formed-sat',M['SystemSeed'].from_system(os))
        _need(os.seed_admission.admit(envelope).accepted,'Receiver Ground admission refused')
        def run(value,workload):
            _need(value=={'program':program['digest'],'part':i},
                  'Receiver-bound workload changed')
            call.append(i)
            return self._sat_runtime()[2](program,i)
        registry=M['OperatorRegistry']();registry.register(spec,run)
        runtime=M['VirtualLayerRuntime'](registry,host,frozenset({'cpu'}))
        cap=M['Capability']('/virtual-layer/operators',frozenset({'compute'})) if grant_compute else M['Capability']('/safe',frozenset({'read'}))
        pid=os.environment.kernel.spawn('sat-formed-receiver',capabilities=(cap,),parent_pid=os.environment.init_pid)
        workload=M['VirtualLayerWorkload'](
            workload_id='sat-'+str(i),system=sid,ground=M['VirtualGroundRef'].from_envelope(envelope),
            intention={'goal':{'SAT_source':program['source_sha256'],'SAT_intention':program['intention'],
                               'part':i,'program':program['digest']}},
            operators=(M['VirtualOperatorRef']('pfi.sat_source_segment',1,'cpu'),),
            input_value={'program':program['digest'],'part':i},
            budget=M['VirtualBudget'](10_000_000,100_000,100_000),logical_time=i+1,
            eligible_nodes=(host,),max_hops=1)
        event=runtime.execute_operator(os,pid,'sfi-source','pfi.sat-segment',i+1,
                  i+1,workload,'pfi.sat_source_segment')
        _need(event.action is M['HypervisorAction'].ALLOW and event.executed and
              len(call)==1 and event.effect is not None,
              'SAT receiver Hypervisor refused execution')
        result=dict(event.effect['result']);self._sat_runtime()[3](program,result)
        return result,{'host':host,'part':i,'event_fingerprint':event.fingerprint,
                       'result_sha256':_hash(result),'hypervisor_action':'ALLOW',
                       'local_kernel_journal_verified':os.environment.kernel.journal.verify()}

    def calculate(self,i,claim_id,*,peer,host,grant_compute=True):
        tok=self._check_target(i,claim_id,peer,host)
        res,rr=self._host_execute(i,host,grant_compute=grant_compute)
        body={'schema':self.CANDIDATE,'job':self.identity,'program':self.program['digest'],
              'source':self.program['source_sha256'],'part':i,
              'claim_id':claim_id,'assignment_epoch':tok.claim.current_assignment.epoch,
              'peer':peer,'host':host,'result':res,'receiver':rr}
        body['sha256']=_hash(body)
        return body

    def reassign(self,i,claim_id,*,peer,host,failure_evidence_digest):
        tok=self.store.get(claim_id)
        self._bind(i)
        return self.store.reassign(claim_id,
            expected_epoch=tok.claim.current_assignment.epoch,**self._actor(peer,host),
            failure_evidence_digest=failure_evidence_digest)

    def _check_candidate(self,c,token,i):
        actor=token.claim.current_assignment
        expected={'schema':self.CANDIDATE,'job':self.identity,
          'program':self.program['digest'],'source':self.program['source_sha256'],
          'part':i,'claim_id':token.claim.claim_id,'assignment_epoch':actor.epoch,
          'peer':actor.owner_peer_id,'host':actor.owner_host_id}
        _need(type(c) is dict and set(c)==set(expected)|{'receiver','result','sha256'}
              and all(c[k]==v for k,v in expected.items()) and
              c['sha256']==_hash({k:v for k,v in c.items() if k!='sha256'}),
              'SAT candidate identity, host or epoch altered')
        self._sat_runtime()[3](self.program,c['result'])
        r=c['receiver']
        _need(type(r) is dict and r.get('host')==actor.owner_host_id and
              r.get('part')==i and r.get('hypervisor_action')=='ALLOW' and
              r.get('local_kernel_journal_verified') is True and
              r.get('result_sha256')==_hash(c['result']) and
              type(r.get('event_fingerprint')) is str and len(r['event_fingerprint'])==64,
              'No qualified SAT receiver-local execution evidence')

    def publish(self,candidate):
        i=candidate.get('part');self._bind(i)
        tok=self.store.get(candidate['claim_id'])
        if tok.claim.status.value=='completed':return self.reconcile(i,candidate['claim_id'])
        self._check_candidate(candidate,tok,i)
        self._check_target(i,candidate['claim_id'],candidate['peer'],candidate['host'])
        path=self.root/'candidates'/(candidate['claim_id']+'.json')
        _atomic_json(path,candidate)
        self.store.complete(candidate['claim_id'],expected_epoch=candidate['assignment_epoch'],
                            **self._actor(candidate['peer'],candidate['host']),
                            result_digest=_hash(candidate['result']),
                            child_receipt_digest=candidate['sha256'])
        return self.reconcile(i,candidate['claim_id'])

    def reconcile(self,i,claim_id):
        self._bind(i)
        token=self.store.get(claim_id)
        _need(token.claim.status.value=='completed',
              'No quorum-published SAT claim to reconcile')
        token.verify_current(self.coordinator)
        candidate=_read_json(self.root/'candidates'/(claim_id+'.json'))
        self._check_candidate(candidate,token,i)
        _need(token.claim.result_digest==_hash(candidate['result']) and
              token.claim.child_receipt_digest==candidate['sha256'],
              'Durable candidate and majority result disagree')
        # Pure work may be independently replayed by a recovered receiver.
        replay,rr=self._host_execute(i,candidate['host'])
        _need(replay==candidate['result'],'Recovered SAT partition disagrees')
        frame=self.fabric.receive(self.ports[i][1])
        _need(frame is not None and frame.payload==self._wire(i),
              'SAT source-frame replay differs')
        self.dag.commit('part-'+str(i),{'result_sha256':_hash(replay),
                                      'claim_id':claim_id,'epoch':candidate['assignment_epoch']})
        self.receipts[str(i)]={'result':replay,'result_sha256':_hash(replay),
              'claim_id':claim_id,'host':candidate['host'],
              'assignment_epoch':candidate['assignment_epoch'],
              'receiver':rr}
        self._save()
        return {'status':'published','part':i,'host':candidate['host'],
                'epoch':candidate['assignment_epoch'],'interpreter':self.program['digest']}

    def finish(self):
        _need({n.node_id for n in self.dag.ready_nodes()}=={'finish'},
              'All SAT partitions must be published before final reduction')
        summary=self._sat_runtime()[4](self.program,[self.receipts[str(i)]['result']
                            for i in range(len(self.program['partitions']))],
                            source_receipt=self.receipt)
        self.dag.commit('finish',summary)
        self._save()
        return summary

    def result(self):
        _need(self.dag.is_complete,'SAT machine not fully published')
        return self.dag.result('finish')
