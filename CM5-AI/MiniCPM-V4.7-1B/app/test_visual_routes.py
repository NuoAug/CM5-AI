"""真实HTTP入口的帧/对象/OCR证据回归，推理与主控均明确替身。"""
import base64
import io
import json
import threading
import unittest
from unittest import mock
from PIL import Image
import server
from visual_context import VisualContextStore


def png(color):
    buf = io.BytesIO()
    Image.new('RGB', (80, 60), color).save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


class VisualRoutesTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()
        self.store = VisualContextStore()
        self.patches = [mock.patch.object(server, 'VISUAL_CONTEXT', self.store),
                        mock.patch.object(server.VISUAL_BRIDGE, 'store', self.store),
                        mock.patch.object(server, 'vlm_health', return_value=True),
                        mock.patch.object(server, 'budget_guard', return_value=True)]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.provider = mock.Mock()
        self.provider.acquire_frame.return_value = {'status': 'error', 'error': {'code': 'DEPENDENCY_UNAVAILABLE', 'message': '测试替身不可用'}}
        patch = mock.patch.object(server, 'MAIN_CONTROL', self.provider)
        patch.start()
        self.addCleanup(patch.stop)
        server.SESSIONS.clear()
        self.sid = self.id()

    def eyes(self, **kwargs):
        return self.client.post('/api/eyes', json=dict(session_id=self.sid, **kwargs))

    def upload(self, frame, color, **extra):
        with mock.patch.object(server, 'call_vlm', return_value={'answer': '画面是一只杯子。'}):
            response = self.eyes(task='scene', image=png(color), frame_id=frame, **extra)
        self.assertEqual(response.status_code, 200, response.get_json())
        return response.get_json()

    def action(self, action, **body):
        response = self.client.post('/api/session/visual/action', json=dict(session_id=self.sid, action=action, **body))
        self.assertEqual(response.status_code, 200, response.get_json())
        return response.get_json()

    def test_followup_reuses_image_and_unknown_capture_time(self):
        first = self.upload('a', 'red')
        with mock.patch.object(server, 'call_vlm', return_value={'answer': '它是红色。'}) as model:
            response = self.eyes(task='followup', question='它是什么颜色')
        data = response.get_json()
        self.assertEqual(response.status_code, 200, data)
        self.assertEqual(data['frame_id'], first['frame_id'])
        self.assertTrue(data['has_image'])
        self.assertIsNone(data['frame_context']['used_frame']['captured_mono_ms'])
        self.assertTrue(model.call_args.args[0])
        self.provider.acquire_frame.assert_not_called()

    def test_refresh_failure_never_reuses_old_image(self):
        self.upload('a', 'red')
        with mock.patch.object(server, 'call_vlm') as model:
            response = self.eyes(task='followup', question='现在呢，重新看', new_frame=True)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()['code'], 'DEPENDENCY_UNAVAILABLE')
        model.assert_not_called()
        self.provider.acquire_frame.assert_called_once()
        self.assertEqual(self.store.current_frame(self.sid)['frame_id'], 'a')

    def test_chat_never_calls_frame_provider_or_retains_supplied_image(self):
        with mock.patch.object(server, 'call_vlm', return_value={'answer': '我们聊一聊。'}):
            response = self.eyes(task='chat', question='今天心情不好', image=png('red'))
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.store.snapshot(self.sid)['current_frame_id'])
        self.provider.acquire_frame.assert_not_called()

    def test_comparison_passes_two_ordered_images(self):
        self.upload('a', 'red', slot='A')
        self.upload('b', 'blue', slot='B')
        with mock.patch.object(server, 'call_vlm', return_value={'answer': 'A红，B蓝。'}) as model:
            response = self.eyes(task='compare', item='两只杯子')
        self.assertEqual(response.status_code, 200, response.get_json())
        images = model.call_args.args[0]
        self.assertIsInstance(images, list)
        self.assertEqual(len(images), 2)
        colors = [Image.open(io.BytesIO(base64.b64decode(encoded))).getpixel((0, 0)) for encoded in images]
        self.assertGreater(colors[0][0], 200)
        self.assertGreater(colors[1][2], 200)
        self.assertEqual([f['frame_id'] for f in response.get_json()['frame_context']['comparison_frames']], ['a', 'b'])
        self.action('swap')
        with mock.patch.object(server, 'call_vlm', return_value={'answer': 'A蓝，B红。'}):
            response = self.eyes(task='compare', item='两只杯子')
        self.assertEqual([f['frame_id'] for f in response.get_json()['frame_context']['comparison_frames']], ['b', 'a'])

    def test_ocr_evidence_is_from_full_original_frame_and_replaced(self):
        self.upload('label', 'white')
        lines = [{'text': '会员价12.50元/瓶', 'score': .96, 'box': [[2,2],[50,2],[50,12],[2,12]]}]
        with mock.patch.object(server, 'ocr_engine', return_value='mock'), mock.patch.object(server, 'parse_ocr_result', return_value=lines):
            for _ in range(3):
                response = self.eyes(task='text')
                self.assertEqual(response.status_code, 200, response.get_json())
        evidence = response.get_json()['ocr_evidence']
        self.assertEqual(evidence['association'], 'ambiguous')
        price = evidence['fields']['price'][0]
        self.assertEqual(price['value'], '12.50')
        self.assertEqual(price['source']['frame_id'], 'label')
        self.assertEqual(len(self.store.ocr_for(self.sid, 'label')), 1)

    def test_target_correction_updates_prompt_without_moving_ocr(self):
        self.upload('correction-frame', 'white')
        one = self.action('bind', frame_id='correction-frame', roi_xyxy_px=[0,0,30,30], name='左侧杯子')['result']
        two = self.action('bind', frame_id='correction-frame', roi_xyxy_px=[40,0,80,30], name='右侧杯子')['result']
        self.action('correct', object_id=one['visual_target_id'], frame_id='correction-frame')
        captured = []
        def fake_model(image, prompt, *args, **kwargs):
            captured.append(server.visual_prompt(prompt))
            return {'answer': '这只杯子是白色。'}
        with mock.patch.object(server, 'call_vlm', side_effect=fake_model):
            response = self.eyes(task='followup', question='我说的是左边这只')
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertIn(one['visual_target_id'], captured[0])
        self.assertNotIn(two['visual_target_id'], captured[0])
        self.assertIn('[0, 0, 30, 30]', captured[0])

    def test_invalid_stream_image_cannot_leak_admission_slots(self):
        active = server.ACTIVE['heavy']
        for _ in range(server.HEAVY_LIMIT + 2):
            response = self.client.post('/api/talk', json={'session_id': self.sid, 'task': 'scene', 'image': 'invalid'})
            self.assertEqual(response.status_code, 400, response.get_json())
            self.assertEqual(server.ACTIVE['heavy'], active)

    def test_selected_roi_crops_only_inference_copy(self):
        image = Image.new('RGB', (80,60), 'red')
        image.paste('blue', (40,0,80,60))
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        encoded = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
        with mock.patch.object(server, 'call_vlm', return_value={'answer': '两个杯子。'}):
            response = self.eyes(task='scene', image=encoded, frame_id='crop-frame')
        self.assertEqual(response.status_code, 200)
        target = self.action('bind', frame_id='crop-frame', roi_xyxy_px=[40,0,80,60], name='右边')['result']
        with mock.patch.object(server, 'call_vlm', return_value={'answer': '蓝色。'}) as model:
            response = self.eyes(task='followup', question='它什么颜色')
        pixels = Image.open(io.BytesIO(base64.b64decode(model.call_args.args[0])))
        self.assertEqual(pixels.size, (40,60))
        self.assertGreater(pixels.getpixel((20,20))[2], 200)
        inputs = response.get_json()['frame_context']['vlm_inputs']
        self.assertEqual(inputs[0]['roi_xyxy_px'], [40,0,80,60])
        self.assertEqual(inputs[0]['transform_to_original'], [1.0,0,40,0,1.0,0,0,0,1])
        self.assertEqual(self.store.frame(self.sid, 'crop-frame')['width_px'], 80)

    def test_stream_done_carries_used_frame(self):
        self.upload('sse-frame', 'red')
        with mock.patch.object(server, 'stream_vlm', return_value=iter(['这是红色杯子。'])):
            response = self.client.post('/api/talk', json={'session_id': self.sid, 'task': 'followup', 'question': '它是什么颜色'}, buffered=True)
        data = [json.loads(line[6:]) for line in response.get_data(as_text=True).splitlines() if line.startswith('data: ')]
        done = next(item for item in data if item.get('done'))
        self.assertEqual(done['frame_id'], 'sse-frame')
        self.assertEqual(done['frame_context']['used_frame_id'], 'sse-frame')

    def test_cancelled_late_provider_cannot_replace_new_photo(self):
        from test_visual_bridge import picture, envelope
        started, release = threading.Event(), threading.Event()
        raw = picture('red')
        result = envelope('provider-old', raw)
        def acquire(**kwargs):
            started.set()
            if not release.wait(3):
                raise TimeoutError('test barrier')
            return result
        self.provider.acquire_frame.side_effect = acquire
        self.provider.read_frame_asset.return_value = raw
        old = []
        def work():
            with server.app.test_client() as client:
                old.append(client.post('/api/eyes', json={'session_id':self.sid, 'request_id':'old-provider', 'task':'scene'}))
        worker = threading.Thread(target=work, daemon=True)
        with mock.patch.object(server, 'call_vlm', return_value={'answer':'新图片。'}):
            worker.start()
            self.assertTrue(started.wait(1))
            self.upload('new-blue', 'blue')
            release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(old[0].status_code, 499, old[0].get_json())
        self.assertEqual(self.store.current_frame(self.sid)['frame_id'], 'new-blue')
        self.assertNotIn('provider-old', [f['frame_id'] for f in self.store.snapshot(self.sid)['frames']])

    def test_cancelled_late_ocr_cannot_commit_evidence(self):
        self.upload('ocr-old', 'white')
        started, release = threading.Event(), threading.Event()
        def ocr(_):
            started.set()
            if not release.wait(3):
                raise TimeoutError('test barrier')
            return 'mock'
        lines = [{'text':'12.50元', 'score':.9, 'box':[[2,2],[50,2],[50,12],[2,12]]}]
        old = []
        def work():
            with server.app.test_client() as client:
                old.append(client.post('/api/eyes', json={'session_id':self.sid, 'request_id':'old-ocr', 'task':'text'}))
        worker = threading.Thread(target=work, daemon=True)
        with mock.patch.object(server, 'ocr_engine', side_effect=ocr), mock.patch.object(server, 'parse_ocr_result', return_value=lines):
            worker.start()
            self.assertTrue(started.wait(1))
            self.upload('ocr-new', 'blue')
            release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(old[0].status_code, 499, old[0].get_json())
        self.assertEqual(self.store.ocr_for(self.sid, 'ocr-old'), [])
        self.assertEqual(self.store.current_frame(self.sid)['frame_id'], 'ocr-new')

    def test_two_images_reach_actual_request_payload(self):
        class Response:
            def raise_for_status(self): pass
            def json(self): return {'choices': [{'message': {'content': '结果'}, 'finish_reason': 'stop'}]}
            def close(self): pass
        with server.app.test_request_context('/api/eyes'), mock.patch.object(server.requests, 'post', return_value=Response()) as post:
            server.call_vlm(['YQ==', 'Yg=='], '比较')
        content = post.call_args.kwargs['json']['messages'][-1]['content']
        self.assertEqual([p['image_url']['url'] for p in content if p['type']=='image_url'], ['data:image/jpeg;base64,YQ==', 'data:image/jpeg;base64,Yg=='])


if __name__ == '__main__':
    unittest.main()
