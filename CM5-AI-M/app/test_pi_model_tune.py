"""CM5 tuning regressions using explicit test doubles for the model and hardware."""
import base64
import io
import unittest
from unittest import mock

from PIL import Image

import server
from turns import TurnCancelled


def image(color):
    data = io.BytesIO()
    Image.new('RGB', (80, 60), color).save(data, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode()


class PiModelTuneTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()
        for name, kwargs in (
            ('budget_guard', {'return_value': True}),
            ('vlm_health', {'return_value': True}),
            ('push_answer_segments', {}),
        ):
            patch = mock.patch.object(server, name, **kwargs)
            patch.start()
            self.addCleanup(patch.stop)

    def test_near_black_frame_bypasses_model(self):
        with mock.patch.object(server, 'call_vlm') as model:
            response = self.client.post('/api/eyes', json={
                'task': 'scene', 'session_id': self.id(), 'image': image('#070707'),
            })
        self.assertEqual(response.status_code, 422)
        data = response.get_json()
        self.assertEqual(data['code'], 'LOW_QUALITY_FRAME')
        self.assertEqual(data['input_quality'][0]['reason'], 'near_black')
        model.assert_not_called()

    def test_detail_question_reaches_model_and_filters_action_claim(self):
        question = '只说杯盖是什么颜色'
        with mock.patch.object(server, 'call_vlm', return_value={
            'answer': '前方没有明显障碍，可以走过去。',
        }) as model:
            response = self.client.post('/api/eyes', json={
                'task': 'detail', 'session_id': self.id(), 'image': image('red'),
                'question': question,
            })
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertIn(question, model.call_args.args[1])
        self.assertEqual(response.get_json()['answer'], server.VISUAL_ACTION_REPLY)

    def test_closed_response_during_cancel_is_a_cancelled_turn(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        with server.app.test_request_context('/api/talk'):
            turn = server.begin_turn(self.id(), self.id())

            def interrupted(**kwargs):
                turn.cancelled.set()
                raise AttributeError("'NoneType' object has no attribute 'read'")

            response.iter_lines.side_effect = interrupted
            with mock.patch.object(server.requests, 'post', return_value=response):
                with self.assertRaises(TurnCancelled):
                    list(server.stream_vlm('测试'))
            self.assertIsNone(turn.backend_exit_confirmed)
            turn.finish_compute()

