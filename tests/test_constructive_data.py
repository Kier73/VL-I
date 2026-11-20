"""Executable construction, artifact building, mutation and strict causal evidence."""
from __future__ import annotations

import ast
import copy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from vli.constructive_data import (ConstructiveTableGround, InteractiveDataSession,
   query_intention, evaluate, compile_python, position, canonical)


def observed_table():
    return ConstructiveTableGround.from_rows(
        {'model':'string','family':'string','layers':'integer',
         'weight_bytes':'integer','status':'string'}, [
        {'model':'Qwen3.8-27B','family':'qwen3_5','layers':64,
         'weight_bytes':55562855904,'status':'partial'},
        {'model':'DeepSeek V4.1 Flash','family':'deepseek_v41','layers':40,
         'weight_bytes':510286023000,'status':'partial'},
        {'model':'Non-model source','family':'metadata','layers':0,
         'weight_bytes':400,'status':'catalogue'},
    ])


def qualified_query(ground):
    return query_intention(ground,
        derive=({'name':'weight_gib','left':{'field':'weight_bytes'},
                 'op':'div','right':1073741824},),
        filters=({'field':'layers','op':'gt','value':0},),
        select=('model','layers','weight_bytes','weight_gib','status'),
        sort='weight_bytes', descending=True, limit=2)


class TestConstructiveData(unittest.TestCase):
    def setUp(self):
        self.g=observed_table()
        self.session=InteractiveDataSession(self.g.source,observer='independent_a')

    def test_actual_model_scale_is_not_changed(self):
        result=evaluate(self.g,qualified_query(self.g))
        self.assertEqual([z['model'] for z in result],
                         ['DeepSeek V4.1 Flash','Qwen3.8-27B'])
        self.assertEqual([z['weight_bytes'] for z in result],
                         [510286023000,55562855904])
        self.assertAlmostEqual(result[0]['weight_gib'],510286023000/1073741824)

    def test_position_is_before_result_and_not_a_numeric_output(self):
        i=qualified_query(self.g)
        p=position(self.g,i,previous_head=self.session.head)
        self.assertTrue(p['coordinate'].startswith('state://vli/'))
        self.assertNotIn('weight_gib',p)
        result=self.session.interact(i,expected_head=self.session.head)
        self.assertEqual(p['coordinate'],result['coordinate'])
        self.assertEqual(result['receipt']['rows_scanned'],3)

    def test_constructed_outputs_are_actual_data_not_a_catalogue(self):
        r=self.session.interact(qualified_query(self.g),expected_head=self.session.head)
        self.assertIn('DeepSeek V4.1 Flash',r['views']['page'])
        self.assertIn('Qwen3.8-27B',r['views']['csv'])
        self.assertEqual(json.loads(r['views']['json']),r['rows'])
        self.assertEqual(r['views']['tokens'],list(r['views']['text'].encode()))
        self.assertIn('def construct(rows):',r['views']['code'])
        ast.parse(r['views']['code'])

    def test_generated_python_constructs_same_result_in_other_process(self):
        i=qualified_query(self.g)
        code=compile_python(i)
        proc=subprocess.run([sys.executable,'-c',
           code+'\nimport sys,json\nprint(json.dumps(construct(json.load(sys.stdin))))'],
           input=json.dumps(self.g.rows),text=True,capture_output=True)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertEqual(json.loads(proc.stdout),evaluate(self.g,i))

    def test_interaction_updates_actual_ground_and_recomputes(self):
        before=self.session.head
        receipt=self.session.update(where={'field':'model','value':'Qwen3.8-27B'},
                     assign={'status':'ready'}, expected_head=before)
        self.assertNotEqual(receipt['before_ground'],receipt['after_ground'])
        self.assertEqual(receipt['rows_changed'],1)
        i=qualified_query(self.session.ground)
        r=self.session.interact(i,expected_head=self.session.head)
        self.assertEqual(r['rows'][1]['status'],'ready')
        self.assertNotEqual(r['coordinate'],position(self.g,qualified_query(self.g),
                          previous_head=before)['coordinate'])

    def test_restart_recovers_changed_data_and_constructs_next_view(self):
        self.session.interact(qualified_query(self.g),expected_head=self.session.head)
        self.session.update(where={'field':'model','value':'DeepSeek V4.1 Flash'},
                            assign={'status':'qualified'},expected_head=self.session.head)
        cp=self.session.checkpoint()
        restored=InteractiveDataSession.restore(self.g.source,cp,
                                expected_head=cp['checkpoint_digest'])
        self.assertEqual(restored.head,self.session.head)
        self.assertEqual(restored.ground.manifest,self.session.ground.manifest)
        result=restored.interact(qualified_query(restored.ground),expected_head=restored.head)
        self.assertEqual(result['rows'][0]['status'],'qualified')

    def test_reject_stale_head_and_no_partial_edit(self):
        source=self.session.ground.manifest
        with self.assertRaisesRegex(ValueError,'Stale'):
            self.session.update(where={'field':'model','value':'Qwen3.8-27B'},
                         assign={'status':'ready'},expected_head='0'*64)
        self.assertEqual(self.session.ground.manifest,source)
        with self.assertRaisesRegex(ValueError,'Stale'):
            self.session.interact(qualified_query(self.g),expected_head='0'*64)

    def test_reject_missing_source_row(self):
        with self.assertRaisesRegex(ValueError,'existing row'):
            self.session.update(where={'field':'model','value':'Unseen'},
                assign={'status':'ready'},expected_head=self.session.head)

    def test_reject_changed_column_type(self):
        with self.assertRaisesRegex(ValueError,'data edit'):
            self.session.update(where={'field':'model','value':'Qwen3.8-27B'},
                assign={'layers':'sixty-four'},expected_head=self.session.head)

    def test_reject_malformed_source_duplicate_json_key(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            ConstructiveTableGround(b'{"schema":"vl-i.typed-table.v1","rows":[],"rows":[],"columns":[]}')

    def test_reject_nonfinite_float(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            ConstructiveTableGround(b'{"schema":"vl-i.typed-table.v1","columns":[],"rows":[NaN]}')

    def test_reject_wrong_filter_type(self):
        with self.assertRaisesRegex(ValueError,'typed selection'):
            query_intention(self.g,filters=({'field':'layers','op':'eq','value':'64'},))

    def test_reject_unsupported_derived_operation(self):
        with self.assertRaisesRegex(ValueError,'field law'):
            query_intention(self.g,derive=({'name':'foo','left':{'field':'layers'},
                                          'op':'eval','right':1},))

    def test_reject_zero_division(self):
        i=query_intention(self.g,derive=({'name':'oops','left':{'field':'layers'},
                                        'op':'div','right':0},))
        with self.assertRaisesRegex(ValueError,'Undefined division'):
            self.session.interact(i,expected_head=self.session.head)
        self.assertEqual(self.session.events,[])

    def test_reject_altered_intention_digest(self):
        intent=qualified_query(self.g)
        intent['limit']=1
        with self.assertRaisesRegex(ValueError,'changed'):
            self.session.interact(intent,expected_head=self.session.head)

    def test_reject_corrupted_checkpoint_without_data_edit(self):
        self.session.interact(qualified_query(self.g),expected_head=self.session.head)
        cp=self.session.checkpoint()
        forged=copy.deepcopy(cp)
        forged['events'][0]['rows_returned']=123
        with self.assertRaisesRegex(ValueError,'checkpoint'):
            InteractiveDataSession.restore(self.g.source,forged,
                                            expected_head=cp['checkpoint_digest'])

    def test_reject_mismatched_source_when_restoring(self):
        self.session.interact(qualified_query(self.g),expected_head=self.session.head)
        cp=self.session.checkpoint()
        changed=copy.deepcopy(self.g.rows)
        changed[0]['weight_bytes']+=1
        other=ConstructiveTableGround.from_rows(self.g.columns,changed)
        with self.assertRaisesRegex(ValueError,'different source'):
            InteractiveDataSession.restore(other.source,cp,
                                            expected_head=cp['checkpoint_digest'])

    def test_reject_unapproved_sort_and_output(self):
        with self.assertRaisesRegex(ValueError,'Sorting'):
            query_intention(self.g,select=('model',),sort='weight_bytes')

    def test_zero_matches_generates_real_empty_table(self):
        q=query_intention(self.g,filters=({'field':'family','op':'eq','value':'none'},),
                         select=('model','weight_bytes'))
        report=self.session.interact(q,expected_head=self.session.head)
        self.assertEqual(report['rows'],[])
        self.assertEqual(report['views']['csv'],'model,weight_bytes\n')

    def test_reject_unrecognized_record_columns(self):
        wrong={'schema':'vl-i.typed-table.v1','columns':[{'name':'x','type':'number'}],
               'rows':[{'x':1,'y':3}]}
        with self.assertRaisesRegex(ValueError,'Row'):
            ConstructiveTableGround(canonical(wrong))

    def test_cross_process_recovery_from_pinned_source_and_checkpoint(self):
        self.session.interact(qualified_query(self.g),expected_head=self.session.head)
        self.session.update(where={'field':'model','value':'Qwen3.8-27B'},
                            assign={'status':'reviewed'},expected_head=self.session.head)
        cp=self.session.checkpoint()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'source.json').write_bytes(self.g.source)
            (root/'checkpoint.json').write_text(json.dumps(cp))
            script='''from vli.constructive_data import InteractiveDataSession,query_intention
from pathlib import Path
import json,sys
root=Path(sys.argv[1])
source=(root/'source.json').read_bytes()
record=json.loads((root/'checkpoint.json').read_text())
session=InteractiveDataSession.restore(source,record,expected_head=sys.argv[2])
intent=query_intention(session.ground,select=('model','status'),sort='model')
result=session.interact(intent,expected_head=session.head)
print(json.dumps({'head':session.head,'rows':result['rows']}))
'''
            proc=subprocess.run([sys.executable,'-c',script,str(root),cp['checkpoint_digest']],
                                capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            payload=json.loads(proc.stdout)
            self.assertEqual(next(r['status'] for r in payload['rows'] if r['model']=='Qwen3.8-27B'),
                             'reviewed')

    def test_real_vld_response_dag_constructs_requested_views(self):
        try:
            import mailcpu.data_ground
            import vld.navigation.dag
        except ImportError:
            self.skipTest('E-CPU/VLD pinned predecessors not installed locally')
        i=qualified_query(self.g)
        result=self.session.interact(i,expected_head=self.session.head,render=True,
                       views=('text','page','code','tokens'))
        observed=result['rendered']
        self.assertEqual(observed['state'],'RENDERED')
        self.assertEqual(set(x['kind'] for x in observed['projections']),
                         {'text','page','code','tokens'})
        self.assertEqual(observed['projections'][0]['content'],result['views']['text'])
        cp=self.session.checkpoint()
        restored=InteractiveDataSession.restore(self.g.source,cp,
                              expected_head=cp['checkpoint_digest'])
        self.assertEqual(restored.head,self.session.head)

    def test_generated_json_csv_and_python_have_same_values(self):
        query=qualified_query(self.g)
        result=self.session.interact(query,expected_head=self.session.head)
        import csv,io
        parsed=list(csv.DictReader(io.StringIO(result['views']['csv'])))
        self.assertEqual(len(parsed),len(result['rows']))
        for row,original in zip(parsed,result['rows']):
            self.assertEqual(row['model'],original['model'])
            self.assertEqual(int(row['weight_bytes']),original['weight_bytes'])

    def test_reject_wrong_derived_reference(self):
        with self.assertRaisesRegex(ValueError,'Non-numeric'):
            query_intention(self.g,derive=({'name':'fail','left':{'field':'model'},
                                            'op':'mul','right':4},))

    def test_rendered_result_is_not_committed_on_unknown_format(self):
        i=qualified_query(self.g)
        with self.assertRaisesRegex(ValueError,'Invalid output view'):
            self.session.interact(i,expected_head=self.session.head,render=True,
                                  views=('file_execute',))
        self.assertEqual(self.session.events,[])

    def test_pinned_large_model_evidence_runs_real_constructive_demo(self):
        upstream=Path(__file__).resolve().parents[1]/'upstream'/'vl64'
        if not upstream.is_dir():
            self.skipTest('Immutable VL-64 evidence checkout absent locally')
        from examples.constructive_data_demo import run
        report, views=run(upstream,use_renderer=True)
        self.assertTrue(report['reconstructed_data_state_verified'])
        self.assertEqual({row['weight_bytes'] for row in report['constructed_rows']},
                         {55562855904,510286023000})
        self.assertNotEqual(report['initial_coordinate'],report['after_edit_coordinate'])
        self.assertEqual(report['rendered_views']['state'],'RENDERED')
        self.assertIn('def construct(rows):',views['code'])

    def test_generated_program_uses_literal_escaping_not_injected_code(self):
        hostile="evil');__import__('os').system('id');#"
        t=ConstructiveTableGround.from_rows({'name':'string','x':'number'},
             [{'name':hostile,'x':1}])
        q=query_intention(t,filters=({'field':'name','op':'eq','value':hostile},),
                          select=('name','x'))
        code=compile_python(q)
        ast.parse(code)
        self.assertEqual(evaluate(t,q)[0]['name'],hostile)


if __name__=='__main__': unittest.main()
