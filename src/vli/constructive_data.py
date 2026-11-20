"""INTERACT-001: interpretation -> qualified data mutation -> constructed output.

An intention addresses a future *state position*, not a scalar or an output
cache. The Ground executes bounded typed operations over observed records,
then builds JSON/CSV/Markdown/Python and optionally streams those views through
VL-I's existing E-CPU / VLD ResponseFormation. No hidden model forward is used.
"""
from __future__ import annotations

import ast
import csv
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
import math
import re

from .ground import _digest, _require, _hex64

SOURCE = 'vl-i.constructive-table-ground.v1'
QUERY = 'vl-i.constructive-query.v1'
POSITION = 'vl-i.constructive-position.v1'
EVENT = 'vl-i.constructive-interaction.v1'
CHECKPOINT = 'vl-i.constructive-session.v1'
NAME = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,63}\Z')
KINDS = {'string', 'integer', 'number', 'boolean'}
OPS = {'add', 'sub', 'mul', 'div'}
FILTERS = {'eq', 'ne', 'gt', 'ge', 'lt', 'le'}
MAX_BYTES = 1024 * 1024
MAX_ROWS = 512
MAX_STEPS = 24
MAX_EVENTS = 32


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',', ':'),ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def _unique(pairs):
    out = {}
    for key,value in pairs:
        if key in out:
            raise ValueError('Duplicate JSON field: ' + key)
        out[key] = value
    return out


def _no_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def _typed(kind, value):
    if kind == 'string': return type(value) is str and len(value.encode('utf-8')) <= 2048
    if kind == 'boolean': return type(value) is bool
    if kind == 'integer': return type(value) is int and abs(value) <= 10**15
    if kind == 'number':
        return type(value) in (int, float) and math.isfinite(value) and abs(value) <= 10**15
    return False


def _name(value):
    return type(value) is str and NAME.fullmatch(value) is not None


class ConstructiveTableGround:
    """Ground of bounded real input rows with source-bound structural semantics.

    Data lives in the caller-supplied source, never a synthesized prediction.
    The manifest is a description/digest, not a copy of each row.
    """
    def __init__(self, raw: bytes):
        _require(type(raw) is bytes and 2 <= len(raw) <= MAX_BYTES,
                 'Invalid source extent')
        data = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique,
                          parse_constant=_no_constant)
        _require(type(data) is dict and set(data) == {'schema','columns','rows'} and
                 data['schema'] == 'vl-i.typed-table.v1' and
                 type(data['columns']) is list and 1 <= len(data['columns']) <= 32 and
                 type(data['rows']) is list and len(data['rows']) <= MAX_ROWS,
                 'Unqualified typed record Ground')
        cols = {}
        for entry in data['columns']:
            _require(type(entry) is dict and set(entry) == {'name','type'} and
                     _name(entry['name']) and entry['type'] in KINDS and
                     entry['name'] not in cols, 'Invalid field structure')
            cols[entry['name']] = entry['type']
        for row in data['rows']:
            _require(type(row) is dict and set(row) == set(cols) and
                     all(_typed(cols[k], v) for k,v in row.items()),
                     'Row does not satisfy inferred/declarative field laws')
        self.columns, self.rows = cols, deepcopy(data['rows'])
        self.source = raw
        body = {'schema':SOURCE,'source_sha256':sha256(raw).hexdigest(),
                'source_bytes':len(raw),'columns':data['columns'],
                'row_count':len(self.rows),'scope':'typed_observed_rows_not_model_weights'}
        self.manifest = {**body,'ground_digest':_digest(SOURCE,body)}

    @staticmethod
    def from_rows(columns, rows):
        raw = canonical({'schema':'vl-i.typed-table.v1',
                         'columns':[{'name':name,'type':kind}
                                    for name,kind in columns.items()], 'rows':rows})
        return ConstructiveTableGround(raw)


def _operand(operand, types):
    if type(operand) is dict and set(operand) == {'field'}:
        name = operand['field']
        _require(name in types and types[name] in ('number','integer'),
                 'Non-numeric source operator')
        return ('field',name)
    _require(type(operand) in (int,float) and math.isfinite(operand)
             and abs(operand) <= 10**15, 'Unqualified numeric constant')
    return ('literal',operand)


def query_intention(ground, *, filters=(), derive=(), select=(), sort=None,
                    descending=False, limit=100, observer='observer'):
    """Declare a *computable* result instead of a catalogued read-only view."""
    _require(isinstance(ground,ConstructiveTableGround) and _name(observer)
             and type(limit) is int and 1 <= limit <= MAX_ROWS
             and type(descending) is bool and type(filters) in (tuple,list)
             and type(derive) in (tuple,list) and len(filters)+len(derive)<=MAX_STEPS,
             'Invalid bounded constructive intention')
    types=dict(ground.columns)
    derivations=[]
    for step in derive:
        _require(type(step) is dict and set(step)=={'name','left','op','right'}
                 and _name(step['name']) and step['name'] not in types
                 and step['op'] in OPS, 'Invalid constructed field law')
        left=_operand(step['left'],types)
        right=_operand(step['right'],types)
        _require(left[0]=='field', 'Derived left operand must be an observed field')
        types[step['name']]='number'
        derivations.append(deepcopy(step))
    predicates=[]
    for clause in filters:
        _require(type(clause) is dict and set(clause)=={'field','op','value'}
                 and clause['field'] in types and clause['op'] in FILTERS
                 and _typed(types[clause['field']],clause['value']),
                 'Unqualified typed selection predicate')
        _require(clause['op'] in ('eq','ne') or
                 types[clause['field']] in ('integer','number'),
                 'Ordered predicate requires numeric data')
        predicates.append(deepcopy(clause))
    selected=list(select) if select else list(types)
    _require(selected and len(selected)==len(set(selected))
             and all(k in types for k in selected), 'Invalid requested output columns')
    _require(sort is None or (sort in selected and types[sort] in ('string','integer','number')),
             'Sorting requires a visible ordered field')
    record={'schema':QUERY,'ground_digest':ground.manifest['ground_digest'],
            'filters':predicates,'derive':derivations,'select':selected,
            'sort':sort,'descending':descending,'limit':limit,
            'observer':observer,'scope':'typed_interaction_not_free_text_understanding'}
    record['intent_digest']=_digest(QUERY,record)
    return record


def _check_intent(ground, intent):
    _require(type(intent) is dict and intent.get('schema')==QUERY and
             _hex64(intent.get('intent_digest')) and
             _digest(QUERY,{k:v for k,v in intent.items() if k!='intent_digest'})==intent['intent_digest']
             and intent.get('ground_digest')==ground.manifest['ground_digest'],
             'Unadmitted or changed constructive intention')
    expected=query_intention(ground,filters=intent['filters'],derive=intent['derive'],
          select=intent['select'],sort=intent['sort'],descending=intent['descending'],
          limit=intent['limit'],observer=intent['observer'])
    _require(expected==intent, 'Constructive intent failed re-admission')


def position(ground, intent, *, previous_head):
    _check_intent(ground,intent)
    _require(_hex64(previous_head),'Missing causal state-position root')
    record={'schema':POSITION,'ground_digest':ground.manifest['ground_digest'],
            'intent_digest':intent['intent_digest'],'previous_head':previous_head,
            'expected_fields':list(intent['select']),
            'scope':'address_before_computation_not_materialized_result'}
    root=_digest(POSITION,record)
    return {**record,'position_digest':root,'coordinate':'state://vli/'+root}


def _get(row, operand):
    return row[operand['field']] if type(operand) is dict else operand


def _calculate(op, x, y):
    if op=='div':
        _require(y!=0,'Undefined division in qualified Ground')
        value=x/y
    elif op=='mul':value=x*y
    elif op=='add':value=x+y
    else:value=x-y
    _require(math.isfinite(value) and abs(value)<=10**15,
             'Constructed result not representable')
    return value


def _passes(row,clause):
    a,b=row[clause['field']],clause['value']
    return {'eq':lambda:a==b,'ne':lambda:a!=b,'gt':lambda:a>b,
            'ge':lambda:a>=b,'lt':lambda:a<b,'le':lambda:a<=b}[clause['op']]()


def evaluate(ground,intent):
    """Execute selected declarative relations against actual observed records."""
    _check_intent(ground,intent)
    selected=[]
    for incoming in ground.rows:
        row=dict(incoming)
        for step in intent['derive']:
            row[step['name']]=_calculate(step['op'],_get(row,step['left']),
                                         _get(row,step['right']))
        if all(_passes(row,cond) for cond in intent['filters']):
            selected.append({name:row[name] for name in intent['select']})
    if intent['sort'] is not None:
        selected.sort(key=lambda row:row[intent['sort']],reverse=intent['descending'])
    return selected[:intent['limit']]


def compile_python(intent):
    """Synthesize a *pure runnable function* from typed laws, never from free code."""
    expr={'add':'+','sub':'-','mul':'*','div':'/'}
    comp={'eq':'==','ne':'!=','gt':'>','ge':'>=','lt':'<','le':'<='}
    operand=lambda obj: 'row['+repr(obj['field'])+']' if type(obj) is dict else repr(obj)
    lines=['# Generated by VL-I from a qualified constructive Ground.','def construct(rows):',
           '    out = []','    for source in rows:','        row = dict(source)']
    for step in intent['derive']:
        left=operand(step['left']);right=operand(step['right'])
        if step['op']=='div':
            lines+=['        if '+right+' == 0: raise ValueError("Undefined division")']
        lines+=['        row['+repr(step['name'])+'] = '+left+' '+expr[step['op']]+' '+right]
    for cond in intent['filters']:
        lines+=['        if not (row['+repr(cond['field'])+'] '+comp[cond['op']]
                +' '+repr(cond['value'])+'): continue']
    items=', '.join(repr(name)+': row['+repr(name)+']' for name in intent['select'])
    lines+=['        out.append({'+items+'})']
    if intent['sort'] is not None:
        lines+=['    out.sort(key=lambda row: row['+repr(intent['sort'])
                +'], reverse='+str(intent['descending'])+')']
    lines+=['    return out[:'+str(intent['limit'])+']','']
    source='\n'.join(lines)
    ast.parse(source)
    return source


def _format(ground,intent,rows):
    """Construct actual requested outputs; no static response prose is claimed."""
    keys=intent['select']
    buffer=StringIO(newline='')
    writer=csv.writer(buffer,lineterminator='\n')
    writer.writerow(keys)
    for row in rows:writer.writerow([row[k] for k in keys])
    markdown=['# Constructed data result','',
              'Ground: `'+ground.manifest['ground_digest']+'`','',
              '| '+' | '.join(keys)+' |','| '+' | '.join('---' for _ in keys)+' |']
    for row in rows:
        markdown.append('| '+' | '.join(str(row[k]).replace('|','\\|').replace('\n',' ')
                                          for k in keys)+' |')
    paragraph=('Constructed '+str(len(rows))+' matching record(s) from a '
       +str(len(ground.rows))+'-record Ground, with fields '+', '.join(keys)+'. '
       +'The result is derived from observed values and typed transformations; '
       +'it does not assert that a pretrained model performed inference.')
    code=compile_python(intent)
    outputs={'text':paragraph, 'page':'\n'.join(markdown)+'\n',
             'code':code,'json':canonical(rows).decode('utf-8'),
             'csv':buffer.getvalue(),'tokens':list(paragraph.encode('utf-8'))}
    return outputs


class InteractiveDataSession:
    """Single-writer state over observed bytes, with optimistic causal updates."""
    def __init__(self,source: bytes,*,observer='observer'):
        _require(_name(observer),'Unqualified observer identity')
        self.initial=ConstructiveTableGround(source)
        self.ground=self.initial
        self.observer=observer
        self.genesis=_digest(CHECKPOINT,{'initial_ground':self.initial.manifest['ground_digest'],
                                        'observer':observer})
        self.head=self.genesis
        self.events=[]

    def interact(self, intent, *, expected_head, render=False, views=('text','page','code','tokens')):
        """Locate -> execute -> render -> commit; publish only after each check."""
        _require(expected_head==self.head and len(self.events)<MAX_EVENTS,
                 'Stale causal branch or exhausted turn budget')
        _check_intent(self.ground,intent)
        pos=position(self.ground,intent,previous_head=self.head)
        rows=evaluate(self.ground,intent)
        outputs=_format(self.ground,intent,rows)
        serialized=canonical({'rows':rows,'outputs':outputs})
        _require(len(serialized)<=256*1024,'Rendered result exceeds declared capacity')
        rendered=None
        if render:
            _require(type(views) is tuple and views and len(set(views))==len(views) and
                     set(views).issubset({'text','page','code','tokens'}),
                     'Invalid output view topology')
            from .response_state import (SourceSlot, ResponseSourceGround, ResponseFormation,
                                         compile_intention as render_intent)
            slots=tuple(SourceSlot(v,v,(canonical(outputs[v]) if v=='tokens'
                                            else outputs[v].encode('utf-8'))) for v in views)
            source=ResponseSourceGround(slots)
            ri=render_intent('Construct the selected qualified result',slots=views,
                             observer=self.observer)
            frame=ResponseFormation(source,ri)
            for _ in range(20):
                if frame.dag.is_complete:break
                frame.advance(max_nodes=16)
            _require(frame.dag.is_complete,'VLD response DAG did not finish')
            rendered=frame.observe()
        event={'schema':EVENT,'sequence':len(self.events),'previous_head':self.head,
               'position':pos,'intent':deepcopy(intent),
               'result_sha256':sha256(canonical(rows)).hexdigest(),
               'outputs_sha256':sha256(canonical(outputs)).hexdigest(),
               'rows_returned':len(rows),'rows_scanned':len(self.ground.rows),
               'requested_views':list(views) if render else [],
               'render_digest':rendered['observation_digest'] if rendered else None,
               'scope':'actual_typed_record_computation_and_constructed_views'}
        event['event_digest']=_digest(EVENT,event)
        self.head=event['event_digest']
        self.events.append(event)
        return {'coordinate':pos['coordinate'],'rows':rows,'views':outputs,
                'rendered':rendered,'receipt':deepcopy(event)}

    def update(self, *, where, assign, expected_head):
        """An authorized, type-preserving *actual data edit*, not a tool proposal."""
        _require(expected_head==self.head and len(self.events)<MAX_EVENTS,
                 'Stale or exhausted data revision')
        _require(type(where) is dict and set(where)=={'field','value'} and
                 where['field'] in self.ground.columns and
                 _typed(self.ground.columns[where['field']],where['value']) and
                 type(assign) is dict and 1<=len(assign)<=8 and
                 all(k in self.ground.columns and _typed(self.ground.columns[k],v)
                     for k,v in assign.items()),'Unqualified typed data edit')
        changed=deepcopy(self.ground.rows)
        affected=0
        for row in changed:
            if row[where['field']]==where['value']:
                row.update(assign)
                affected+=1
        _require(affected>0,'Data action did not address an existing row')
        # Build and independently re-admit the entire new data Ground first.
        candidate=ConstructiveTableGround.from_rows(self.ground.columns,changed)
        event={'schema':EVENT,'sequence':len(self.events),'previous_head':self.head,
               'before_ground':self.ground.manifest['ground_digest'],
               'after_ground':candidate.manifest['ground_digest'],
               'where':deepcopy(where),'assign':deepcopy(assign),'rows_changed':affected,
               'scope':'type_preserving_caller_authorized_ground_mutation'}
        event['event_digest']=_digest(EVENT,event)
        self.events.append(event)
        self.head=event['event_digest']
        self.ground=candidate
        return deepcopy(event)

    def checkpoint(self):
        obj={'schema':CHECKPOINT,'genesis':self.genesis,'observer':self.observer,
             'initial_source_sha256':sha256(self.initial.source).hexdigest(),
             'current_ground_digest':self.ground.manifest['ground_digest'],
             'head':self.head,'events':deepcopy(self.events)}
        obj['checkpoint_digest']=_digest(CHECKPOINT,obj)
        return obj

    @classmethod
    def restore(cls, source,checkpoint,*,expected_head):
        """Replay every semantic effect from pinned source bytes, not cached output."""
        _require(type(checkpoint) is dict and checkpoint.get('schema')==CHECKPOINT and
                 checkpoint.get('checkpoint_digest')==expected_head and
                 _hex64(expected_head) and _digest(CHECKPOINT,{k:v for k,v in checkpoint.items()
                   if k!='checkpoint_digest'})==expected_head and
                 type(checkpoint.get('events')) is list and
                 len(checkpoint['events'])<=MAX_EVENTS,
                 'Missing trusted or unaltered causal checkpoint')
        restored=cls(source,observer=checkpoint['observer'])
        _require(restored.genesis==checkpoint['genesis'] and
                 sha256(source).hexdigest()==checkpoint['initial_source_sha256'],
                 'Checkpoint refers to different source data')
        for old in checkpoint['events']:
            _require(type(old) is dict and old.get('previous_head')==restored.head,
                     'Tampered event sequence or causal predecessor')
            if 'intent' in old:
                result=restored.interact(old['intent'],expected_head=restored.head,
                    render=old.get('render_digest') is not None,
                    views=tuple(old.get('requested_views', [])))
                # Rebuild exact VLD response projection when the original did so.
                fresh=result['receipt']
                compare=('schema','sequence','previous_head','position','intent',
                         'result_sha256','outputs_sha256','rows_returned','rows_scanned',
                         'requested_views','render_digest','scope')
                _require(all(fresh[k]==old[k] for k in compare),
                         'Restored construction no longer reproduces output')
                restored.events[-1]=deepcopy(old)
                restored.head=old['event_digest']
            else:
                receipt=restored.update(where=old['where'],assign=old['assign'],
                                         expected_head=restored.head)
                _require(receipt==old,'Restored update differs from original')
            _require(_digest(EVENT,{k:v for k,v in old.items()
                                  if k!='event_digest'})==old['event_digest'],
                     'Altered event digest')
        _require(restored.head==checkpoint['head'] and
                 restored.ground.manifest['ground_digest']==checkpoint['current_ground_digest'],
                 'Replayed state differs from trusted checkpoint')
        return restored
