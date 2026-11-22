"""PFI-002: construct active logical observer-hosts from a formed interpreter.

Uses real E-CPU OwnedNetworkFabric queues, E-CPU native computation DAG and
receiver-local VirtualLayerRuntime/Hypervisor authorization. A generated XOR
processor is partitioned by a declared per-host gate budget; each stage is an
actual receiver-executed operation. The portable checkpoint can be passed to
another Python process, which must rebind trusted operators and obtain its
own local capability. A virtual E-CPU link never computes by itself.

The bounded demonstration is one generated transition plus one observer read;
not a general host scheduler, real network hop, Byzantine quorum, or general
program-to-substrate compiler.
"""
from __future__ import annotations

import json
from hashlib import sha256

from .problem_formed_interpreter import verify_interpreter, execute_network

SCHEMA = 'vl-i.problem-formed-observer-fabric.v1'
PACKET = 'vl-i.formed-fabric-frame.v1'
MAX_PACKET = 8192
MAX_STAGES = 48


def _canon(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _hash(data):
    return sha256(_canon(data)).hexdigest()


def _need(predicate, message):
    if not predicate:
        raise ValueError(message)


def _source_modules():
    from mailcpu.computation_dag import ComputationNode, NavigableComputationDAG
    from mailcpu.owned_network import OwnedNetworkFabric, NetworkAddress
    return ComputationNode, NavigableComputationDAG, OwnedNetworkFabric, NetworkAddress


def _machine_stage(program, start, stop, final=False, kind='transition'):
    """One interpreter-derived piece of a complete topological gate circuit."""
    return {'kind': kind, 'gates': program['nodes'][start:stop],
            'final': final, 'outputs': program['outputs'] if final else [],
            'input_bits': program['input_bits']}


def _form_stages(bundle, transition, gates_per_host):
    _need(type(gates_per_host) is int and 1 <= gates_per_host <= 64,
          'Host capacity must be a bounded positive gate count')
    _need(type(transition) is int and 0 <= transition < len(bundle['transitions']),
          'Invalid generated instruction index')
    program = bundle['transitions'][transition]
    gates = len(program['nodes'])
    steps = []
    for start in range(0, max(1, gates), gates_per_host):
        end = min(start + gates_per_host, gates)
        steps.append(_machine_stage(program, start, end, final=end == gates))
    observer = bundle['observer']
    count = len(observer['nodes'])
    for start in range(0, max(1, count), gates_per_host):
        end = min(start + gates_per_host, count)
        steps.append(_machine_stage(observer, start, end,
                                    final=end == count, kind='observer'))
    _need(2 <= len(steps) <= MAX_STAGES,
          'Problem requires more active hosts than allocated')
    return steps


def _apply_stage(stage, values):
    """Execute exactly this generated gate segment; no arbitrary code payload."""
    _need(type(values) is dict and all(type(k) is str and type(v) is int and
              v in (0, 1) for k,v in values.items()), 'Malformed wire-state bytes')
    state = dict(values)
    for gate in stage['gates']:
        _need(gate['op'] in ('xor', 'flip') and all(a in state for a in gate['args']),
              'Invalid or missing predecessor wire')
        if gate['op'] == 'xor':
            state[gate['id']] = state[gate['args'][0]] ^ state[gate['args'][1]]
        else:
            state[gate['id']] = 1 ^ state[gate['args'][0]]
    if stage['final']:
        value = sum(state[name] << i for i, name in enumerate(stage['outputs']))
        if stage['kind'] == 'transition':
            return {'zero':0, **{'x'+str(i):(value >> i)&1
                                for i in range(stage['input_bits'])}}
        return {'result': value}
    return state


def _frame(epoch, job, next_index, wires, previous_sha):
    body = {'schema':PACKET, 'epoch':epoch, 'job':job,
            'next_index':next_index, 'wires':wires, 'previous_sha256':previous_sha}
    body['sha256'] = _hash(body)
    return body


def _decode(raw):
    _need(type(raw) is bytes and 0 < len(raw) <= MAX_PACKET, 'Invalid E-CPU payload length')
    try:
        doc = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique,
                         parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    except (ValueError, UnicodeError) as exc:
        raise ValueError('Invalid typed E-CPU frame') from exc
    _need(type(doc) is dict and set(doc) ==
          {'schema','epoch','job','next_index','wires','previous_sha256','sha256'} and
          _canon(doc) == raw and doc['schema'] == PACKET and
          type(doc['sha256']) is str and
          doc['sha256'] == _hash({k:v for k,v in doc.items() if k!='sha256'}),
          'Malformed or changed typed E-CPU frame')
    return doc


def _unique(pairs):
    out = {}
    for key,value in pairs:
        if key in out:
            raise ValueError('Duplicate frame field')
        out[key] = value
    return out


def _local_hypervisor_execute(stage, wires, *, epoch, index, host,
                               bundle_digest, grant_compute=True):
    """Native Virtual Layer + real receiver-local E-CPU Hypervisor boundary."""
    from mailcpu import (
        Capability, HeadlessOS, HypervisorAction, LexicalDecisionModel,
        HypervisorPolicy, OperatorRegistry, OperatorSpec, SystemIdentity,
        SystemSeed, SystemSeedEnvelope, VirtualBudget, VirtualGroundRef,
        VirtualLayerRuntime, VirtualLayerWorkload, VirtualOperatorRef,
    )
    _need(type(host) is str and 1 <= len(host) <= 48 and
          host.replace('-','').replace('_','').isalnum(), 'Invalid host identity')
    operating_system = HeadlessOS()
    operating_system.hypervisor.classifier = LexicalDecisionModel()
    operating_system.hypervisor.policy = HypervisorPolicy(embedding_review_confidence=1.0)
    identity = SystemIdentity('formed-fabric-'+host, 1)
    seed = SystemSeed.from_system(operating_system)
    envelope = SystemSeedEnvelope(identity, 'formed-fabric', 'formed-fabric', seed)
    _need(operating_system.seed_admission.admit(envelope).accepted,
          'Receiver failed to admit local system Ground')
    ground = VirtualGroundRef.from_envelope(envelope)
    calls = []
    def handler(value, workload):
        # Handler is bound locally from the source-admitted compiled stage,
        # never loaded from a network frame or a remote operator declaration.
        _need(value['epoch'] == epoch and value['stage'] == index and
              value['bundle_digest'] == bundle_digest and
              value['wires'] == wires,
              'Receiver detected input/epoch mismatch')
        calls.append(index)
        return _apply_stage(stage, dict(value['wires']))
    registry = OperatorRegistry()
    registry.register(OperatorSpec(
        operator_id='pfi.formed_segment', version=1,
        description='Execute source-qualified generated Boolean gate segment',
        input_kind='object', output_kind='object', substrates=('cpu',),
        required_rights=frozenset({'compute'}), max_input_bytes=MAX_PACKET,
        max_output_bytes=MAX_PACKET, base_energy=1,
        energy_per_byte=1, deterministic=True, effect_mode='pure'), handler)
    runtime = VirtualLayerRuntime(registry, host, frozenset({'cpu'}))
    pid = operating_system.environment.kernel.spawn(
        'formed-fabric-receiver',
        capabilities=(Capability('/virtual-layer/operators', frozenset({'compute'}))
                      if grant_compute else Capability('/safe',frozenset({'read'})),),
        parent_pid=operating_system.environment.init_pid)
    workload = VirtualLayerWorkload(
        workload_id='pfi-fabric-'+str(index), system=identity, ground=ground,
        intention={'goal':{'source_bound_interpreter':bundle_digest,'stage':index}},
        operators=(VirtualOperatorRef('pfi.formed_segment',1,'cpu'),),
        input_value={'epoch':epoch,'stage':index,'bundle_digest':bundle_digest,
                     'wires':wires},
        budget=VirtualBudget(100_000,100_000,1000),logical_time=index+1,
        eligible_nodes=(host,),max_hops=1)
    receipt=runtime.execute_operator(operating_system,pid,'pfi-fabric-source',
                                    'pfi.formed-fabric',index+1,index+1,
                                    workload,'pfi.formed_segment')
    _need(receipt.action is HypervisorAction.ALLOW and receipt.executed and
          len(calls) == 1 and receipt.effect is not None and
          receipt.effect['kind'] == 'virtual_layer.operator_result',
          'Receiver-local Hypervisor refused this operation')
    result=dict(receipt.effect['result'])
    _need(result == _apply_stage(stage, wires),
          'Hosted result disagrees with explicit generated gate semantics')
    return result, {'host':host,'stage':index,'event_fingerprint':receipt.fingerprint,
                    'result_sha256':_hash(result), 'hypervisor_action':'ALLOW',
                    'local_kernel_journal_verified':operating_system.environment.kernel.journal.verify()}


class FormedFabric:
    """A reconstructible execution of a problem-shaped active-host pipeline."""
    def __init__(self, source, quotient, bundle, *, initial, transition=0,
                 gates_per_host=2, job='formed-task-001'):
        Node, DAG, Fabric, Address = _source_modules()
        _need(verify_interpreter(source,quotient,bundle), 'Unqualified problem-formed Ground')
        _need(type(initial) is int and 0 <= initial < 1 << source.width,
              'Source coordinate outside described problem')
        _need(type(job) is str and 1 <= len(job) <= 48 and job.isascii() and
              all(c.isalnum() or c in '-_' for c in job), 'Invalid task identity')
        self.source, self.quotient, self.bundle = source,quotient,bundle
        self.initial, self.transition, self.gates_per_host, self.job = initial,transition,gates_per_host,job
        self.steps = _form_stages(bundle, transition, gates_per_host)
        self.epoch = _hash({'problem':source.digest, 'interpreter':bundle['digest'],
                            'transition':transition, 'host_gate_budget':gates_per_host,
                            'task':job,'initial_decimal':str(initial)})
        self.fabric=Fabric(queue_limit=2)
        names=['origin']+['stage-'+str(i) for i in range(len(self.steps))]+['sink']
        for name in names: self.fabric.create_namespace(name)
        for idx in range(len(names)-1):
            left=self.fabric.create_interface(names[idx],'tx',mtu=MAX_PACKET)
            right=self.fabric.create_interface(names[idx+1],'rx',mtu=MAX_PACKET)
            self.fabric.connect('link-'+str(idx),left,right)
        # E-CPU DAG enforces the same generated-stage predecessor order.
        common={'revision':1,'operation':'pfi.generated-host-transition',
                'resource':'/vl-i/formed-fabric','context_digest':bundle['digest']}
        nodes=[Node(node_id='dispatch',dependencies=(),payload={'epoch':self.epoch},**common)]
        for i in range(len(self.steps)):
            nodes.append(Node(node_id='step-'+str(i),
                              dependencies=('dispatch',) if i==0 else ('step-'+str(i-1),),
                              payload={'epoch':self.epoch,'stage_index':i,
                                       'gate_count':len(self.steps[i]['gates'])}, **common))
        nodes.append(Node(node_id='finish',dependencies=('step-'+str(len(self.steps)-1),),
                          payload={'epoch':self.epoch},**common))
        self.dag=DAG(nodes)
        self.receipts=[]
        self.bytes_transmitted=0

    def _reference_packets(self):
        """Canonical expected state at every queue boundary, rebuilt from source."""
        p=self.quotient.project(self.initial)
        states={'zero':0, **{'x'+str(i):(p>>i)&1 for i in range(self.quotient.dimension)}}
        prior='0'*64
        packets=[]
        for i in range(len(self.steps)+1):
            packet=_frame(self.epoch,self.job,i,states,prior)
            packets.append(packet)
            if i<len(self.steps):
                states=_apply_stage(self.steps[i],states)
                prior=packet['sha256']
        # This check makes final observation semantics independent of gate layout.
        expected=self.quotient.read(self.quotient.operators[self.transition].apply(p))
        _need(packets[-1]['wires']=={'result':expected},
              'Generated host graph disagrees with source-defined observation')
        return packets

    def _receiver(self, index):
        _,_,_,Address=_source_modules()
        return Address('stage-'+str(index),'rx') if index<len(self.steps) else Address('sink','rx')

    def _sender(self, index):
        _,_,_,Address=_source_modules()
        return Address('origin','tx') if index==0 else Address('stage-'+str(index-1),'tx')

    def _send(self, packet, index):
        raw=_canon(packet)
        _need(len(raw)<=MAX_PACKET, 'Generated packet exceeds E-CPU route MTU')
        self.fabric.send(self._sender(index),self._receiver(index),raw,index)
        self.bytes_transmitted+=len(raw)

    def advance(self, *, host='receiver-a', grant_compute=True):
        """One atomic-in-snapshot quantum. Denied worker does not consume frame."""
        expected=self._reference_packets()
        ready=self.dag.ready_nodes()
        if not ready: return False
        node=ready[0].node_id
        _need(node in ['dispatch','finish']+[f'step-{i}' for i in range(len(self.steps))],
              'Unrecognized E-CPU frontier')
        if node=='dispatch':
            self._send(expected[0],0)
            self.dag.commit(node,{'epoch':self.epoch,'output_sha256':expected[0]['sha256']})
            return True
        index=len(self.steps) if node=='finish' else int(node.split('-')[1])
        receiver=self._receiver(index)
        queued=self.fabric.peek(receiver,max_size=MAX_PACKET)
        _need(queued is not None and queued.source==self._sender(index) and
              queued.logical_time==index,
              'Missing or stale E-CPU route frame')
        actual=_decode(queued.payload)
        _need(actual == expected[index], 'Received Ground differs from source-derived state')
        if node=='finish':
            frame=self.fabric.receive(receiver,max_size=MAX_PACKET)
            _need(frame is not None and frame.payload==queued.payload,'Frame changed before receipt')
            result=actual['wires']['result']
            self.dag.commit('finish',{'epoch':self.epoch,'observed_decimal':str(result),
                                      'input_sha256':actual['sha256']})
            return True
        stage=self.steps[index]
        output, authority=_local_hypervisor_execute(stage,actual['wires'],
            epoch=self.epoch,index=index,host=host,bundle_digest=self.bundle['digest'],
            grant_compute=grant_compute)
        _need(output == expected[index+1]['wires'],
              'Receiver host returned a semantically incorrect state')
        frame=self.fabric.receive(receiver,max_size=MAX_PACKET)
        _need(frame is not None and frame.payload==queued.payload,'Frame changed before receipt')
        self._send(expected[index+1],index+1)
        self.dag.commit(node,{'epoch':self.epoch,'input_sha256':actual['sha256'],
                              'output_sha256':expected[index+1]['sha256'],
                              'receiver':authority})
        self.receipts.append(authority)
        return True

    def snapshot(self):
        state={'epoch':self.epoch,'job':self.job,'initial_decimal':str(self.initial),
               'transition':self.transition,'gates_per_host':self.gates_per_host,
               'interpreter_digest':self.bundle['digest'],
               'fabric':self.fabric.snapshot(), 'dag':self.dag.snapshot(),
               'receipts':self.receipts,'bytes_transmitted':self.bytes_transmitted}
        return {'schema':SCHEMA,'state':state,'sha256':_hash(state)}

    @classmethod
    def restore(cls, source, quotient, bundle, checkpoint):
        _,DAG,Fabric,_=_source_modules()
        _need(type(checkpoint) is dict and set(checkpoint)=={'schema','state','sha256'} and
              checkpoint['schema']==SCHEMA and checkpoint['sha256']==_hash(checkpoint['state']),
              'Invalid checkpoint envelope')
        body=checkpoint['state']
        inst=cls(source,quotient,bundle,initial=int(body['initial_decimal']),
                 transition=body['transition'],gates_per_host=body['gates_per_host'],job=body['job'])
        _need(body['epoch']==inst.epoch and body['interpreter_digest']==bundle['digest'],
              'Stale interpreter epoch')
        inst.dag=DAG.from_snapshot(body['dag'])
        inst.fabric=Fabric.from_snapshot(body['fabric'])
        inst.receipts=body['receipts']
        inst.bytes_transmitted=body['bytes_transmitted']
        # A rehashed forged checkpoint must not be able to redefine the job's
        # stages or invent semantic intermediate values. Reconstruct all expected
        # native DAG nodes, commits, pending queue and byte-accounting.
        baseline=cls(source,quotient,bundle,initial=inst.initial,
                     transition=inst.transition,gates_per_host=inst.gates_per_host,job=inst.job)
        _need([n.to_dict() for n in inst.dag.nodes]==[n.to_dict() for n in baseline.dag.nodes],
              'Native DAG substituted or changed')
        committed=list(inst.dag.committed_nodes)
        done=[n for n in ['dispatch']+[f'step-{i}' for i in range(len(inst.steps))]+['finish']
              if n in committed]
        _need(len(done)==len(committed) and set(done)==set(committed) and
              done==(['dispatch']+[f'step-{i}' for i in range(len(done)-1)]
                     if len(done)<=len(inst.steps)+1 else
                     ['dispatch']+[f'step-{i}' for i in range(len(inst.steps))]+['finish']),
              'Non-prefix host execution checkpoint')
        expected=inst._reference_packets()
        step_commits=[n for n in done if n.startswith('step-')]
        _need(len(inst.receipts)==len(step_commits) and
              len(inst.fabric.links())==len(inst.steps)+1 and
              len(inst.fabric.namespaces())==len(inst.steps)+2,
              'Changed topology or host receipt count')
        for idx,name in enumerate(step_commits):
            result=inst.dag.result(name)
            _need(result['epoch']==inst.epoch and
                  result['input_sha256']==expected[idx]['sha256'] and
                  result['output_sha256']==expected[idx+1]['sha256'] and
                  result['receiver']==inst.receipts[idx],
                  'Semantically forged native predecessor commit')
        if done:
            _need(inst.dag.result('dispatch')=={'epoch':inst.epoch,
                    'output_sha256':expected[0]['sha256']},
                  'Changed original dispatch')
        if 'finish' in done:
            _need(inst.dag.result('finish')=={'epoch':inst.epoch,
                  'observed_decimal':str(expected[-1]['wires']['result']),
                  'input_sha256':expected[-1]['sha256']},
                  'Changed terminal observer commit')
        next_index=max(0,len(done)-1)
        queues=inst.fabric.snapshot()['queues']
        if done and 'finish' not in done:
            _need(len(queues)==1 and len(queues[0]['frames'])==1 and
                  _decode(inst.fabric.peek(inst._receiver(next_index)).payload)==expected[next_index],
                  'Unmatched in-flight semantic E-CPU frame')
        else:
            _need(not queues, 'Unexpected queued frame at checkpoint')
        _need(inst.bytes_transmitted==sum(len(_canon(expected[i]))
                 for i in range(len(done)) if i <= len(inst.steps)),
              'Fabric payload accounting is inconsistent')
        return inst

    def result(self):
        _need(self.dag.is_complete,'No committed observer realization')
        return {'epoch':self.epoch,'interpreter_digest':self.bundle['digest'],
                'observer_result':int(self.dag.result('finish')['observed_decimal']),
                'logical_hosts':len(self.steps)+2,
                'executed_host_stages':len(self.steps),
                'host_identifiers':[r['host'] for r in self.receipts],
                'fabric_bytes':self.bytes_transmitted,
                'native_dag_digest':self.dag.snapshot()['digest'],
                'scope':'software E-CPU owned fabric; receiver-local hypervisor; no physical network'}
