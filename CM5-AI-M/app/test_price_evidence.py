"""价格 OCR 证据与文字/语音入口会话路由回归，全部使用替身，不接真模型/设备。"""
import base64
import io
import unittest
from unittest import mock

from PIL import Image

import server
from visual_context import VisualContextStore


def png(color):
    buf = io.BytesIO()
    Image.new('RGB', (80, 60), color).save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def price_lines(text, box=((2, 2), (60, 2), (60, 20), (2, 20))):
    return [{'text': text, 'score': .95, 'box': [list(point) for point in box]}]


class PriceEvidenceTests(unittest.TestCase):
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
        provider = mock.Mock()
        provider.acquire_frame.return_value = {'status': 'error', 'error': {'code': 'DEPENDENCY_UNAVAILABLE', 'message': '测试替身不可用'}}
        patch = mock.patch.object(server, 'MAIN_CONTROL', provider)
        patch.start()
        self.addCleanup(patch.stop)
        server.SESSIONS.clear()
        self.sid = self.id()

    def upload(self, frame, color, **extra):
        with mock.patch.object(server, 'call_vlm', return_value={'answer': '画面里有一个商品。'}):
            response = self.client.post('/api/eyes', json=dict(session_id=self.sid, task='scene', image=png(color), frame_id=frame, **extra))
        self.assertEqual(response.status_code, 200, response.get_json())
        return response.get_json()

    def ask(self, text, **body):
        response = self.client.post('/api/ask', json=dict(session_id=self.sid, text=text, **body))
        self.assertEqual(response.status_code, 200, response.get_json())
        return response.get_json()

    def visual_action(self, action, **body):
        response = self.client.post('/api/session/visual/action', json=dict(session_id=self.sid, action=action, **body))
        self.assertEqual(response.status_code, 200, response.get_json())
        return response.get_json()

    def test_price_answers_from_ocr_without_inventing_attribution(self):
        self.upload('shelf', 'white')
        lines = price_lines('会员价12.50元/瓶')
        with mock.patch.object(server, 'ocr_engine', return_value='mock'), \
                mock.patch.object(server, 'parse_ocr_result', return_value=lines):
            data = self.ask('右边那包多少钱')
        self.assertEqual(data['intent'], 'read_price')
        self.assertEqual(data['source'], 'ocr')
        self.assertIn('12.50', data['answer'])
        self.assertIn('没有把它归到具体商品', data['answer'])
        evidence = data['ocr_evidence']
        self.assertEqual(evidence['association'], 'ambiguous')
        self.assertEqual(evidence['fields']['price'][0]['value'], '12.50')
        self.assertEqual(evidence['fields']['price'][0]['source']['frame_id'], 'shelf')

    def test_price_binds_only_to_confirmed_object_in_same_frame(self):
        self.upload('shelf', 'white')
        bound = self.visual_action('bind', frame_id='shelf', roi_xyxy_px=[0, 0, 80, 60], name='这包商品')['result']
        self.visual_action('correct', object_id=bound['visual_target_id'], frame_id='shelf')
        lines = price_lines('会员价12.50元')
        with mock.patch.object(server, 'ocr_engine', return_value='mock'), \
                mock.patch.object(server, 'parse_ocr_result', return_value=lines):
            data = self.ask('这个多少钱')
        self.assertEqual(data['source'], 'ocr')
        self.assertIn('这件商品的价签是12.50 元', data['answer'])
        self.assertNotIn('没有把它归到具体商品', data['answer'])
        self.assertEqual(data['ocr_evidence']['association'], 'confirmed')

    def test_two_slot_price_compare_uses_ocr_amounts(self):
        self.upload('a', 'red', slot='A')
        self.upload('b', 'blue', slot='B')
        with mock.patch.object(server, 'ocr_engine', return_value='mock'), \
                mock.patch.object(server, 'parse_ocr_result',
                                  side_effect=[price_lines('促销价8.00元'), price_lines('6.50元/瓶')]):
            data = self.ask('这两个哪个更便宜')
        self.assertEqual(data['intent'], 'compare')
        self.assertEqual(data['source'], 'ocr')
        self.assertIn('8.00', data['answer'])
        self.assertIn('6.50', data['answer'])
        self.assertIn('B 更便宜', data['answer'])

    def test_price_compare_without_slots_asks_for_slots(self):
        self.upload('only', 'white')
        with mock.patch.object(server, 'ocr_engine', return_value='mock'):
            response = self.client.post('/api/ask', json=dict(session_id=self.sid, text='这两个哪个更便宜'))
        self.assertEqual(response.status_code, 404, response.get_json())
        self.assertEqual(response.get_json()['code'], 'NO_FRAME')
        self.assertIn('A', response.get_json()['error'])

    def test_text_entry_after_visual_turn_keeps_followup(self):
        self.upload('cached', 'red')
        with mock.patch.object(server, 'call_vlm', return_value={'answer': '我觉得挺好看。'}) as model:
            data = self.ask('那你觉得呢', image=png('red'), frame_id='cached')
        self.assertEqual(data['intent'], 'followup')
        self.assertEqual(data['source'], 'vlm')
        self.assertTrue(model.call_args.args[0])

    def test_text_entry_after_chat_turn_does_not_use_cached_photo(self):
        self.upload('cached', 'red')
        with mock.patch.object(server, 'call_vlm', return_value={'answer': '我在听。'}):
            self.ask('你好')
        with mock.patch.object(server, 'call_vlm', return_value={'answer': '我觉得挺好。'}) as model:
            data = self.ask('那你觉得呢', image=png('blue'), frame_id='cached')
        self.assertEqual(data['intent'], 'chat')
        self.assertIsNone(model.call_args.args[0])

    def test_voice_entry_after_visual_turn_keeps_followup(self):
        self.upload('cached', 'red')
        with mock.patch.object(server, 'call_vlm', return_value={'answer': '我觉得挺好看。'}) as model:
            response = self.client.post('/api/voice/final', json={
                'is_final': True, 'wake_detected': True,
                'final_transcript': '小冰，那你觉得呢', 'session_id': self.sid})
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()['intent'], 'followup')
        self.assertTrue(model.call_args.args[0])

    def test_voice_entry_after_chat_turn_does_not_use_cached_photo(self):
        self.upload('cached', 'red')
        with mock.patch.object(server, 'call_vlm', return_value={'answer': '我在听。'}):
            first = self.client.post('/api/voice/final', json={
                'is_final': True, 'wake_detected': True,
                'final_transcript': '小冰，陪我聊聊今天的压力', 'session_id': self.sid})
        self.assertEqual(first.status_code, 200, first.get_json())
        self.assertEqual(first.get_json()['intent'], 'chat')
        with mock.patch.object(server, 'call_vlm', return_value={'answer': '我觉得挺好。'}) as model:
            second = self.client.post('/api/voice/final', json={
                'is_final': True, 'wake_detected': True,
                'final_transcript': '小冰，那你觉得呢', 'session_id': self.sid})
        self.assertEqual(second.status_code, 200, second.get_json())
        self.assertEqual(second.get_json()['intent'], 'chat')
        self.assertIsNone(model.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
