"""Construct and *change* useful artifacts from pinned large-model evidence.

This is interaction over observed data, not 27B/DeepSeek native inference.
E-CPU and VLD render the generated outputs through their existing response DAG.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path

from vli.ground import load_json
from vli.constructive_data import ConstructiveTableGround, InteractiveDataSession, query_intention
from vli.large_model_comparison import SOURCES


def run(upstream, *, use_renderer=True):
    path=Path(upstream)/'experiments/model_fabric/datasets/2026-11-19-large-model-frontier-observations.json'
    record=load_json(path)
    if record.get('schema')!='vl64.architecture-scale-observations.v1':
        raise ValueError('Wrong pinned observation schema')
    rows=[]
    for route in sorted(SOURCES):
        cfg=SOURCES[route]
        reported=record['models'][cfg['archive_key']]['report']
        for field, value in (
            ('model',cfg['model']),('revision',cfg['revision']),
            ('advertised_payload_bytes',cfg['advertised_bytes']),
            ('shards',cfg['shards']),('layers',cfg['layers']),
            ('mapped_tensors',cfg['tensors']),('index_sha256',cfg['index_sha256'])):
            if reported.get(field)!=value:
                # Model names in cfg bind original Qwen and DeepSeek; no alias substitution.
                raise ValueError('Pinned large model structure changed: '+field)
        rows.append({'route':route,'model':cfg['model'],'layers':cfg['layers'],
                     'weight_bytes':cfg['advertised_bytes'],'shards':cfg['shards'],
                     'mapped_tensors':cfg['tensors'],'review_state':'unexamined'})
    columns={'route':'string','model':'string','layers':'integer',
             'weight_bytes':'integer','shards':'integer','mapped_tensors':'integer',
             'review_state':'string'}
    source=ConstructiveTableGround.from_rows(columns,rows)
    session=InteractiveDataSession(source.source,observer='vli_research')

    def ask():
        return query_intention(session.ground,
           derive=({'name':'weight_gib','left':{'field':'weight_bytes'},
                    'op':'div','right':1073741824},
                   {'name':'bytes_per_shard','left':{'field':'weight_bytes'},
                    'op':'div','right':{'field':'shards'}}),
           select=('model','layers','weight_bytes','weight_gib',
                   'bytes_per_shard','review_state'),
           sort='weight_bytes',descending=True,limit=2,observer='vli_research')
    initial=session.interact(ask(),expected_head=session.head,render=use_renderer)
    update=session.update(where={'field':'route','value':'qwen-27b'},
               assign={'review_state':'selected_for_next_experiment'},
               expected_head=session.head)
    after=session.interact(ask(),expected_head=session.head,render=use_renderer)
    snapshot=session.checkpoint()
    restored=InteractiveDataSession.restore(source.source,snapshot,
                                    expected_head=snapshot['checkpoint_digest'])
    if restored.head!=session.head:
        raise ValueError('Recovered description-native data interaction diverged')
    # Real result changes, but original model weight size remains intact.
    if not (initial['rows'][1]['review_state']=='unexamined' and
            after['rows'][1]['review_state']=='selected_for_next_experiment'):
        raise ValueError('Constructive edit was not reflected in rendered output')
    report={'schema':'vl-i.constructive-data-demo.v1',
        'source_report_sha256':sha256(path.read_bytes()).hexdigest(),
        'source_ground_digest':source.manifest['ground_digest'],
        'pinned_model_revisions':{r:SOURCES[r]['revision'] for r in SOURCES},
        'initial_coordinate':initial['coordinate'],
        'after_edit_coordinate':after['coordinate'],
        'source_update_receipt':update,
        'initial_rows':initial['rows'],'constructed_rows':after['rows'],
        'rendered_views':after['rendered'],
        'checkpoint_head':snapshot['checkpoint_digest'],
        'reconstructed_data_state_verified':True,
        'generated_source_code_sha256':sha256(after['views']['code'].encode()).hexdigest(),
        'scope':'real_archived_model_metadata_interaction_not_pretrained_inference'}
    return report,after['views']


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--upstream',required=True)
    parser.add_argument('--out-dir',default='artifacts/constructive-data')
    parser.add_argument('--no-response-dag',action='store_true')
    args=parser.parse_args(argv)
    report,outputs=run(args.upstream,use_renderer=not args.no_response_dag)
    folder=Path(args.out_dir);folder.mkdir(parents=True,exist_ok=True)
    for key,extension in [('page','md'),('code','py'),('json','json'),
                          ('csv','csv'),('text','txt')]:
        (folder/('constructed_result.'+extension)).write_text(outputs[key],encoding='utf-8')
    (folder/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print('CONSTRUCTIVE_DATA_STATUS',report['scope'])
    print('CONSTRUCTIVE_CHECKPOINT',report['checkpoint_head'])

if __name__=='__main__':main()
