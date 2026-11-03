"""流式ASR最终话语入口及边生成边播报的整合验证。"""
import unittest
from unittest import mock
import server


class FinalVoiceTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()

    def test_partial_is_rejected_before_control(self):
        with mock.patch.object(server.MAIN_CONTROL, 'control_intent') as control:
            response = self.client.post('/api/voice/final', json={'is_final': False, 'final_transcript': '开始引导'})
        self.assertEqual(response.status_code, 400)
        control.assert_not_called()

    def test_final_does_not_decode_audio_again(self):
        with mock.patch.object(server, 'asr_transcribe') as offline:
            response = self.client.post('/api/voice/final', json={'is_final': True, 'wake_detected': True, 'final_transcript': '小冰，不要开始引导', 'session_id': 'final-negated'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['intent'], 'clarify')
        offline.assert_not_called()

    def test_voice_chat_segments_are_queued_before_model_completion(self):
        observed = []
        def stream(*args, **kwargs):
            yield '这是第一段完整回答，先说说你的感受。'
            observed.append(server.OUTPUT.stats()['last_event']['source'])
            yield '我们接着慢慢聊。'
        with mock.patch.object(server, 'vlm_health', return_value=True), \
                mock.patch.object(server, 'stream_vlm', side_effect=stream), \
                mock.patch.object(server, 'asr_transcribe') as offline:
            response = self.client.post('/api/voice/final', json={
                'is_final': True, 'wake_detected': True, 'final_transcript': '小冰，陪我聊聊今天的压力',
                'session_id': 'final-chat', 'stream_speech': True})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(observed, ['voice'])
        self.assertIn('我们接着', response.json['answer'])
        offline.assert_not_called()

    def test_cancelled_final_does_not_become_voice_error(self):
        def stream(*args, **kwargs):
            server.OUTPUT.cancel()
            yield '不应入队。'
        with mock.patch.object(server, 'vlm_health', return_value=True), mock.patch.object(server, 'stream_vlm', side_effect=stream):
            response = self.client.post('/api/voice/final', json={
                'is_final': True, 'wake_detected': True, 'final_transcript': '小冰，陪我聊聊',
                'session_id': 'final-cancel', 'stream_speech': True})
        self.assertEqual(response.json['code'], 'CANCELLED')
        self.assertEqual(response.status_code, 499)
