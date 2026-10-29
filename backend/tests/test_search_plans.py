import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.routers import plans


class SearchPlanTests(unittest.TestCase):
    def test_strict_constraint_validation(self):
        with self.assertRaises(ValidationError):
            plans.SearchPlan(query='q', source_queries={'ocr':['x']})
        with self.assertRaises(ValidationError):
            plans.SearchPlan(query='q', source_queries={'asr':['a','b']}, source_query_weights={'asr':[1]})
        with self.assertRaises(ValidationError):
            plans.SearchPlan(query='q', restrict_to_parent=True)

    def test_sequential_pools_persist_and_feedback_does_not_replace_parent(self):
        async def bridge(cwd, tools, message, history, model):
            data, _ = tools.execute('register_plans', {'plans':[
                {'label':'first','query':'first','mode':'caption'},
                {'label':'second','query':'second','mode':'asr'}]})
            yield {'type':'plans', 'plans':data['plans']}
            # Producer must finish remaining queued plans even if Codex ends here.
            data, _ = tools.execute('run_next_plan', {})
            yield {'type':'pool', **data}

        calls = []
        def execute(plan, *args):
            calls.append(plan.query)
            return [{'video_id':'v','keyframe_id':'1'}], [{'id':'v_1','video_id':'v','frame_id':1}]

        app = FastAPI()
        app.include_router(plans.router)
        with tempfile.TemporaryDirectory() as tmp, patch.object(plans,'ROOT',Path(tmp)), \
             patch.object(plans,'execute',execute), \
             patch.object(plans,'get_dataset_resources',return_value=(SimpleNamespace(),{},SimpleNamespace())), \
             patch('src.mock_codex.stream_codex',bridge), TestClient(app) as client:
            response = client.post('/api/agent/chat',json={'dataset':'v3c','message':'find'})
            self.assertEqual(response.status_code,200)
            events = [json.loads(line) for line in response.text.splitlines()]
            identifier = events[0]['session_id']
            pools = [e for e in events if e['type']=='pool']
            self.assertEqual(len(pools),2)
            self.assertEqual(calls,['first','second'])
            original_id = pools[0]['id']
            response = client.post('/api/agent/chat',json={'dataset':'v3c','message':'refine',
                'session_id':identifier,'parent_pool_id':original_id})
            self.assertEqual(response.status_code,200)
            saved = client.get(f'/api/agent/sessions/{identifier}').json()
            self.assertEqual(len(saved['plans']),4)
            self.assertEqual(saved['plans'][0]['id'],original_id)
            self.assertEqual(saved['plans'][2]['plan']['parent_pool_id'],original_id)
            self.assertEqual(client.post('/api/agent/chat',json={'dataset':'lhe','message':'q',
                'session_id':identifier}).status_code,409)

    def test_failed_strategy_does_not_block_later_pool(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(plans,'ROOT',Path(tmp)):
            session = {'id':'test','dataset':'v3c','video_ids':[], 'plans':[], 'history':[]}
            plans.save(session)
            tools = plans.RealTools(session,SimpleNamespace(),SimpleNamespace(),[])
            tools.execute('register_plans',{'plans':[{'query':'bad'},{'query':'good'}]})
            with patch.object(plans,'execute',side_effect=[RuntimeError('missing source'),([],[])]):
                first, _ = tools.execute('run_next_plan',{})
                second, _ = tools.execute('run_next_plan',{})
            self.assertEqual(first['status'],'failed')
            self.assertEqual(second['status'],'completed')


if __name__ == '__main__':
    unittest.main()
