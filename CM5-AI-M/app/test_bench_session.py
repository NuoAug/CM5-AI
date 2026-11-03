"""有界实机批次、人工评分分母、异步trace与SSE结果采集；不访问真实设备。"""
import json
import base64
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
from flask import Flask, Response, g, jsonify, request, stream_with_context
from bench_session import BenchError, BenchSessions, register_test_session_api
from trace import TraceStore


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.manager=BenchSessions(Path(self.tmp.name),'unit-test')
        self.addCleanup(self.manager.close)

    def started(self):
        self.manager.start({'label':'测试批次'})
        self.manager.case({'case_id':'A01','category':'语音','prompt':'小冰，前面有什么','expected':'实际场景'})
        return self.manager.context()

    def events(self):
        _,chunks=self.manager.export()
        return json.loads(''.join(chunks))

    def test_empty_status_and_missing_case_are_explicit(self):
        self.assertIsNone(self.manager.status()['session'])
        with self.assertRaises(BenchError):self.manager.case({'case_id':'A01'})
        self.manager.start({})
        with self.assertRaises(BenchError):self.manager.review({'case_id':'A01'})

    def test_request_keeps_case_when_user_selects_next_before_completion(self):
        old=self.started()
        self.manager.case({'case_id':'A02'})
        self.manager.interaction(old,{'final_transcript':'识别原话'},{'answer':'旧题答案','trace_id':'old'},200)
        record=next(e for e in self.events()['events'] if e['type']=='interaction')['payload']
        self.assertEqual(record['case_id'],'A01')
        self.assertEqual(record['case_definition_at_request']['expected'],'实际场景')
        self.assertIsNone(record['physical_first_sound_ms'])
        self.assertGreaterEqual(record['server_http_duration_ms'],0)

    def test_export_does_not_include_audio_images_or_credentials(self):
        context=self.started()
        self.manager.interaction(context,{'audio':'sensitive','image':'sensitive','authorization':'secret','text':'问题'},
                                 {'answer':'回答','image':'bad','password':'bad','trace_id':'t'},200)
        data=self.events()
        encoded=json.dumps(data,ensure_ascii=False)
        self.assertNotIn('sensitive',encoded)
        self.assertNotIn('secret',encoded)
        self.assertIn('问题',encoded)
        self.assertIn('回答',encoded)

    def test_manual_accuracy_counts_only_scored_latest_cases(self):
        self.started()
        self.manager.review({'recognition_correct':True,'image_correct':None})
        self.manager.case({'case_id':'A02'})
        self.manager.review({'recognition_correct':False,'image_correct':None})
        self.manager.case({'case_id':'A03'})
        summary=self.manager.status()['summary']
        self.assertEqual(summary['manual_quality']['recognition_correct']['scored_cases'],2)
        self.assertEqual(summary['manual_quality']['recognition_correct']['case_accuracy'],.5)
        self.assertIsNone(summary['manual_quality']['image_correct']['case_accuracy'])
        self.manager.review({'case_id':'A02','recognition_correct':True})
        self.assertEqual(self.manager.status()['summary']['manual_quality']['recognition_correct']['case_accuracy'],1)
        self.assertEqual(self.manager.status()['summary']['unreviewed_cases'],1)

    def test_review_rejects_boolean_as_rating_and_numbers_as_truth(self):
        self.started()
        for body in ({'fluency':True},{'fluency':0},{'stability':6},{'recognition_correct':1},{'image_correct':'true'},{'outcome':'success'}):
            with self.subTest(body=body), self.assertRaises(BenchError):self.manager.review(body)
        self.assertEqual(self.manager.status()['session']['counts']['reviews'],0)

    def test_duplicate_case_is_revision_not_new_unique_case(self):
        self.started()
        self.manager.case({'case_id':'A01','prompt':'修订问题'})
        counts=self.manager.status()['session']['counts']
        self.assertEqual(counts['cases'],1)
        self.assertEqual(counts['case_events'],2)

    def test_trace_from_different_dialogue_session_joins_active_test_case(self):
        context=self.started()
        trace={'trace_id':'hardware-1','meta':{'session_id':'device-dialogue'},'marks':[{'stage':'answer'}]}
        self.manager.observe_trace(trace,context)
        self.manager.case({'case_id':'A02'})
        self.manager.observe_trace(dict(trace,marks=[{'stage':'play_start'}]))
        snapshots=[e['payload'] for e in self.events()['events'] if e['type']=='trace']
        self.assertEqual([s['case_id'] for s in snapshots],['A01','A01'])
        self.assertIn('not acoustic',snapshots[-1]['timing_basis'])

    def test_duplicate_trace_is_not_persisted_twice(self):
        context=self.started()
        for _ in range(3):self.manager.observe_trace({'trace_id':'x','marks':[]},context)
        self.assertEqual(self.manager.status()['session']['counts']['trace_snapshots'],1)

    def test_stop_keeps_inflight_and_late_playback_but_no_new_context(self):
        context=self.started()
        self.manager.observe_trace({'trace_id':'t','marks':[]},context)
        self.manager.stop({})
        self.assertIsNone(self.manager.context())
        self.manager.interaction(context,{}, {'answer':'在途结果','trace_id':'t'},200)
        self.manager.observe_trace({'trace_id':'t','marks':[{'stage':'play_start'}]})
        self.assertEqual(self.manager.status()['session']['counts']['interactions'],1)
        self.assertEqual(self.manager.status()['session']['counts']['trace_snapshots'],2)
        self.assertFalse(self.manager.status()['session']['active'])

    def test_two_batches_never_reassign_old_request(self):
        old=self.started()
        self.manager.stop({})
        self.manager.start({'label':'第二批'})
        self.manager.case({'case_id':'B01'})
        self.manager.interaction(old,{}, {'answer':'旧轮'},200)
        self.assertEqual(self.manager.status()['session']['counts']['interactions'],0)
        self.assertEqual(self.manager.status(old['test_session_id'])['session']['counts']['interactions'],1)

    def test_restart_marks_unfinished_batch_interrupted(self):
        self.started()
        other=BenchSessions(self.tmp.name,'unit-test')
        self.addCleanup(other.close)
        self.assertEqual(other.status()['session']['status'],'interrupted')
        self.assertFalse(other.status()['session']['active'])
        self.assertIsNone(other.status()['summary']['evidence_limits']['human_barge_in_verified'])

    def test_frame_evidence_preserves_original_bytes_and_deduplicates(self):
        context=self.started()
        raw=b'actual image byte fixture'
        resolved={'image':'data:image/png;base64,'+base64.b64encode(raw).decode(),'frame_id':'f1','frame_context':{'used_frame_id':'f1'}}
        self.manager.frame_evidence(context,resolved)
        self.manager.frame_evidence(context,resolved)
        meta=self.manager.status()['session']
        self.assertEqual(meta['frame_bytes'],len(raw))
        self.assertEqual(len(meta['frame_assets']),1)
        asset=context['frame_asset']
        location=Path(self.tmp.name)/context['test_session_id']/asset['relative_path']
        self.assertEqual(location.read_bytes(),raw)
        self.manager.interaction(context,{}, {'answer':'图像答案'},200)
        record=self.manager.status()['recent_interactions'][-1]
        self.assertEqual(record['frame_asset']['sha256'],asset['sha256'])
        self.assertNotIn(base64.b64encode(raw).decode(),json.dumps(self.events()))

    def test_frame_budget_does_not_discard_model_response(self):
        context=self.started()
        self.manager._state()['meta']['frame_limit_bytes']=1
        self.manager.frame_evidence(context,{'image':base64.b64encode(b'longer').decode(),'frame_id':'f1'})
        self.assertTrue(self.manager.status()['session']['frame_logging_limited'])
        self.assertNotIn('frame_asset',context)
        self.manager.interaction(context,{}, {'answer':'正常回答'},200)
        self.assertEqual(self.manager.status()['recent_interactions'][-1]['response']['answer'],'正常回答')

    def test_export_rejects_path_traversal(self):
        self.started()
        for bad in ['../secret','/tmp','test-20261103-123456-00000000/..']:
            with self.assertRaises(BenchError):self.manager.export(bad)

    def test_log_capacity_is_visible_and_cannot_fake_saved_review(self):
        self.manager.max_events=2
        self.started()
        with self.assertRaises(BenchError):self.manager.review({'recognition_correct':True})
        data=self.manager.status()
        self.assertTrue(data['summary']['logging_limited'])
        self.assertEqual(data['session']['counts']['reviews'],0)

    def test_disk_reopen_and_streaming_export_preserve_records(self):
        context=self.started()
        self.manager.interaction(context,{'text':'第一问'},{'answer':'第一答'},200)
        self.manager.stop({})
        sid=self.manager.status()['session']['test_session_id']
        other=BenchSessions(self.tmp.name,'unit-test')
        self.addCleanup(other.close)
        _,parts=other.export(sid)
        data=json.loads(''.join(parts))
        self.assertEqual(data['session']['status'],'stopped')
        self.assertTrue(any(e['type']=='interaction' for e in data['events']))


class ResourceTests(unittest.TestCase):
    def test_resource_snapshot_uses_separate_sampler_and_records_overhead(self):
        with tempfile.TemporaryDirectory() as tmp:
            sampled=threading.Event()
            def resources():
                sampled.set()
                return {'rss_bytes':12345,'physical_first_sound_ms':None}
            manager=BenchSessions(tmp,'test',resource_getter=resources)
            try:
                manager.start({})
                self.assertTrue(sampled.wait(1))
                # close joins the sampler after its already-started snapshot append.
                manager.close()
                self.assertEqual(manager.status()['session']['counts']['resource_samples'],1)
                last=manager.status()['last_resource']
                self.assertEqual(last['values']['rss_bytes'],12345)
                self.assertGreaterEqual(last['sample_duration_ms'],0)
            finally:manager.close()

    def test_resource_exception_is_observable_and_does_not_end_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            called=threading.Event()
            def broken():
                called.set()
                raise RuntimeError('fixture resource probe failed')
            manager=BenchSessions(tmp,'test',resource_getter=broken)
            try:
                manager.start({})
                self.assertTrue(called.wait(1))
                manager.close()
                self.assertTrue(manager.status()['session']['active'])
                self.assertIn('fixture resource probe failed',manager.status()['last_resource']['error'])
            finally:manager.close()


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.manager=BenchSessions(self.tmp.name,'unit-test');self.addCleanup(self.manager.close)
        self.app=Flask(__name__)
        self.trace=TraceStore(Path(self.tmp.name)/'trace.jsonl')
        @self.app.post('/api/voice/final')
        def voice():
            body=request.get_json()
            self.trace.start('hardware-trace','voice-final',session_id='device')
            self.trace.mark('hardware-trace','asr_end',text=body['final_transcript'])
            self.trace.finish('hardware-trace')
            return jsonify(answer='模型答案',trace_id='hardware-trace',text=body['final_transcript'])
        @self.app.post('/api/talk')
        def talk():
            def events():
                yield 'data: '+json.dumps({'sentence':'第一段'})+'\n\n'
                yield 'data: '+json.dumps({'done':True,'answer':'完整答案','frame_id':'f1'})+'\n\n'
            return Response(stream_with_context(events()),mimetype='text/event-stream')
        register_test_session_api(self.app,self.manager,self.trace)
        self.client=self.app.test_client()
        self.client.post('/api/test/session/start',json={})
        self.client.post('/api/test/case',json={'case_id':'A01','prompt':'实际提问'})

    def test_json_voice_captures_asr_answer_and_async_trace(self):
        result=self.client.post('/api/voice/final',json={'session_id':'device','final_transcript':'语音原话','image':'not logged'})
        self.assertEqual(result.status_code,200)
        self.trace.mark('hardware-trace','play_start',source='device')
        data=self.client.get('/api/test/session/export').get_json()
        event=next(e['payload'] for e in data['events'] if e['type']=='interaction')
        self.assertEqual(event['request']['final_transcript'],'语音原话')
        self.assertEqual(event['response']['answer'],'模型答案')
        self.assertNotIn('image',event['request'])
        trace=[e for e in data['events'] if e['type']=='trace']
        self.assertEqual(len(trace),2)
        self.assertEqual(trace[-1]['payload']['trace']['marks'][-1]['stage'],'play_start')

    def test_sse_final_is_captured_without_buffering_output(self):
        response=self.client.post('/api/talk',json={'question':'问题'},buffered=False)
        iterator=iter(response.response)
        first=next(iterator)
        self.assertIn(b'sentence',first)
        list(iterator)
        response.close()
        last=self.manager.status()['recent_interactions'][-1]
        self.assertEqual(last['response']['answer'],'完整答案')
        self.assertFalse(last['stream_incomplete'])

    def test_started_case_stays_bound_when_sse_finishes_later(self):
        response=self.client.post('/api/talk',json={'question':'旧题'},buffered=False)
        self.manager.case({'case_id':'A02'})
        response.get_data();response.close()
        self.assertEqual(self.manager.status()['recent_interactions'][-1]['case_id'],'A01')

    def test_malformed_control_body_is_rejected(self):
        response=self.client.post('/api/test/case',json=['A01'])
        self.assertEqual(response.status_code,400)

    def test_empty_array_does_not_start_new_batch(self):
        self.client.post('/api/test/session/stop',json={})
        response=self.client.post('/api/test/session/start',json=[])
        self.assertEqual(response.status_code,400)
        self.assertFalse(self.manager.status()['session']['active'])

    def test_export_is_json_attachment(self):
        response=self.client.get('/api/test/session/export')
        self.assertEqual(response.status_code,200)
        self.assertIn('attachment;',response.headers['Content-Disposition'])
        self.assertEqual(response.get_json()['schema_version'],'pi-test-session.v1')


if __name__=='__main__':unittest.main()
