"""独立设备模式真实HTTP路由接线；相机/模型是明确构造替身，不作实机验收。"""
import base64
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from PIL import Image
import server
from bench_session import BenchSessions
from standalone_frame_provider import DeviceTestVisualContextStore, StandaloneFrameProvider


class PiCameraRoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.manager=BenchSessions(self.tmp.name,'fixture-test');self.addCleanup(self.manager.close)
        self.manager.start({'label':'constructed camera routing'})
        self.manager.case({'case_id':'V01','prompt':'颜色和对象'})
        original=server.TEST_SESSIONS
        self.store=DeviceTestVisualContextStore(allow_standalone_device=True)
        self.camera=mock.Mock()
        self.camera.status.return_value={'available':True,'running':True}
        self.camera.capture_still.side_effect=lambda:Image.new('RGB',(80,60),'red')
        self.provider=StandaloneFrameProvider(self.camera)
        self.control=mock.Mock()
        patches=[mock.patch.object(server,'IMAGE_SOURCE','standalone-device-test'),
                 mock.patch.object(server,'VISUAL_CONTEXT',self.store),
                 mock.patch.object(server.VISUAL_BRIDGE,'store',self.store),
                 mock.patch.object(server.VISUAL_BRIDGE,'main_control_getter',lambda:self.provider),
                 mock.patch.object(server,'MAIN_CONTROL',self.control),
                 mock.patch.object(server,'TEST_SESSIONS',self.manager),
                 mock.patch.object(original,'context',self.manager.context),
                 mock.patch.object(original,'interaction',self.manager.interaction),
                 mock.patch.object(original,'observe_trace',self.manager.observe_trace),
                 mock.patch.object(server,'budget_guard',return_value=True),
                 mock.patch.object(server,'vlm_health',return_value=True)]
        for patch in patches:
            patch.start();self.addCleanup(patch.stop)
        self.client=server.app.test_client()

    def test_no_upload_uses_device_camera_and_retains_actual_frame_evidence(self):
        with mock.patch.object(server,'call_vlm',return_value={'answer':'红色区域。'}):
            response=self.client.post('/api/eyes',json={'session_id':'fixture-web','task':'scene'})
        data=response.get_json()
        self.assertEqual(response.status_code,200,data)
        self.camera.capture_still.assert_called_once()
        self.control.acquire_frame.assert_not_called()
        frame=data['frame_context']['used_frame']
        self.assertEqual(frame['evidence_mode'],'development_live')
        self.assertEqual(frame['producer']['role'],'standalone_device')
        self.assertEqual(frame['source_kind'],'camera')
        self.assertIsNone(frame['sensor_capture_mono_ms'])
        meta=self.manager.status()['session']
        self.assertEqual(len(meta['frame_assets']),1)
        asset=next(iter(meta['frame_assets'].values()))
        retained=Path(self.tmp.name)/meta['test_session_id']/asset['relative_path']
        with Image.open(retained) as retained_image:
            self.assertEqual(retained_image.size,(80,60))
        record=self.manager.status()['recent_interactions'][-1]
        self.assertEqual(record['case_id'],'V01')
        self.assertEqual(record['frame_asset']['sha256'],asset['sha256'])

    def test_followup_reuses_device_frame_without_new_capture(self):
        with mock.patch.object(server,'call_vlm',return_value={'answer':'红色。'}):
            first=self.client.post('/api/eyes',json={'session_id':'fixture-follow','task':'scene'}).get_json()
            second=self.client.post('/api/eyes',json={'session_id':'fixture-follow','task':'followup','question':'它什么颜色'}).get_json()
        self.assertEqual(first['frame_id'],second['frame_id'])
        self.assertEqual(self.camera.capture_still.call_count,1)
        self.assertEqual(self.manager.status()['session']['frame_bytes'],next(iter(self.manager.status()['session']['frame_assets'].values()))['bytes'])

    def test_native_voice_captures_its_own_camera_frame_not_web_object(self):
        with mock.patch.object(server,'call_vlm',return_value={'answer':'红色杯子。'}) as model:
            first=self.client.post('/api/eyes',json={'session_id':'fixture-web-object','task':'scene'}).get_json()
            bind=self.client.post('/api/session/visual/action',json={'session_id':'fixture-web-object','action':'bind','frame_id':first['frame_id'],'roi_xyxy_px':[0,0,20,20],'name':'web-selected'}).get_json()
            self.assertTrue(bind['ok'])
            response=self.client.post('/api/voice/final',json={'session_id':'fixture-device','final_transcript':'小冰，前面有什么','is_final':True,'wake_detected':True})
            result=response.get_json()
            self.assertEqual(response.status_code,200,result)
            encoded=model.call_args.args[0]
            self.assertEqual(Image.open(io.BytesIO(base64.b64decode(encoded))).size,(80,60))
        self.assertEqual(self.camera.capture_still.call_count,2)
        self.assertNotEqual(result['frame_id'],first['frame_id'])
        self.assertIsNone(result['frame_context']['selected_object_id'])
        self.control.acquire_frame.assert_not_called()
        records=self.manager.status()['recent_interactions']
        self.assertEqual([r['case_id'] for r in records],['V01','V01'])
        self.assertEqual(records[-1]['request']['final_transcript'],'小冰，前面有什么')


if __name__=='__main__':unittest.main()
