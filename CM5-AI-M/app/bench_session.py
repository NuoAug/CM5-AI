"""树莓派实机测试批次：本地有界日志、人工评价及软件观测；不授予硬件验收结论。"""
from collections import OrderedDict, deque
from copy import deepcopy
from datetime import datetime, timezone
import codecs
import base64
import binascii
import json
import hashlib
import math
from pathlib import Path
import re
import threading
import time
import uuid


SCHEMA = "pi-test-session.v1"
TEST_ID = re.compile(r"^test-\d{8}-\d{6}-[0-9a-f]{8}$")
CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
REQUEST_PATHS = frozenset(("/api/voice", "/api/voice/final", "/api/eyes", "/api/ask", "/api/talk", "/api/ocr", "/api/vlm", "/api/device/speak"))
REQUEST_FIELDS = ("text", "question", "item", "task", "final_transcript", "session_id", "utterance_id", "request_id", "frame_id", "captured_at", "image_source", "new_frame", "is_final", "wake_detected")
RESPONSE_FIELDS = ("answer", "error", "code", "source", "intent", "text", "cmd", "frame_id", "trace_id", "latency_ms", "finish_reason", "truncated", "has_image", "cancelled", "done", "spoken", "fields", "ocr_evidence")
EVIDENCE_LIMITS = {
    "software_timing_only": True,
    "physical_first_sound_ms": None,
    "human_barge_in_verified": None,
    "recognition_accuracy_basis": "manual_case_reviews_only; planned prompt is not verified spoken ground truth",
    "clock_basis": "server monotonic for HTTP duration; trace client wall clocks are not synchronized or certified",
}


class BenchError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def small(value, depth=0):
    """不接受原始音频/图片/任意大返回；日志内单个值有明确上限。"""
    if depth > 7:
        return "[depth limited]"
    if isinstance(value, dict):
        return {str(k)[:128]: small(v, depth+1) for k, v in list(value.items())[:128]
                if str(k).lower() not in {"audio", "image", "image_vlm", "authorization", "password", "api_key", "vlm_api_key"}}
    if isinstance(value, (list, tuple)):
        return [small(v, depth+1) for v in value[:128]]
    if isinstance(value, str):
        return value[:8192]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value if value is None or isinstance(value, (int, float, bool)) else str(value)[:512]


def text(body, name, max_len=4096):
    value = body.get(name, "")
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > max_len:
        raise BenchError(f"{name}必须为不超过{max_len}字符的文字")
    return value.strip()


class BenchSessions:
    def __init__(self, root, version, *, resource_getter=None, sample_seconds=5,
                 max_events=20000, max_bytes=64*1024*1024):
        self.root = Path(root)
        self.version = version
        self.resource_getter = resource_getter
        self.sample_seconds = max(1, sample_seconds)
        self.max_events, self.max_bytes = max_events, max_bytes
        self._lock = threading.RLock()
        self._sessions = OrderedDict()
        self._active_id = None
        self._last_id = None
        self._trace_context = OrderedDict()
        self._stop = threading.Event()
        self._thread = None
        self._recover()

    def _directory(self, session_id):
        if not isinstance(session_id, str) or not TEST_ID.fullmatch(session_id):
            raise BenchError("测试批次ID无效")
        path = self.root / session_id
        if path.resolve().parent != self.root.resolve():
            raise BenchError("测试路径不在本轮目录内")
        return path

    def _recover(self):
        marker = self.root / "active.json"
        if not marker.exists():
            return
        try:
            sid = json.loads(marker.read_text(encoding="utf8"))["test_session_id"]
            path = self._directory(sid) / "session.json"
            data = json.loads(path.read_text(encoding="utf8"))
            data.update(active=False, status="interrupted", ended_wall_ms=int(time.time()*1000),
                        recovery_note="服务上次退出时批次未结束，不计作稳定性通过")
            self._sessions[sid] = self._load_state(data)
            self._last_id = sid
            self._persist(self._sessions[sid])
            marker.unlink()
        except (OSError, ValueError, KeyError, TypeError):
            # 不删除无法确认属于合法批次的文件；下一次start明确拒绝覆盖。
            pass

    @staticmethod
    def _load_state(meta):
        return {"meta": meta, "recent_interactions": deque(maxlen=20), "recent_reviews": deque(maxlen=20),
                "last_resource": None, "late_until": 0, "trace_hashes": OrderedDict()}

    def _persist(self, state):
        directory = self._directory(state['meta']['test_session_id'])
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / "session.json.tmp"
        tmp.write_text(json.dumps(state['meta'], ensure_ascii=False, indent=2), encoding="utf8")
        tmp.replace(directory / "session.json")

    def _append(self, state, kind, payload):
        meta = state['meta']
        event = {"schema_version": SCHEMA, "type": kind, "observed_wall_ms": int(time.time()*1000),
                 "test_session_id": meta['test_session_id'], "payload": small(payload)}
        raw = (json.dumps(event, ensure_ascii=False, separators=(',', ':'))+'\n').encode('utf8')
        if len(raw) > 128*1024:
            event['payload'] = {"omitted": True, "reason": "event payload exceeded128KiB", "case_id": payload.get('case_id')}
            raw = (json.dumps(event, ensure_ascii=False)+'\n').encode('utf8')
        if meta['event_count'] >= self.max_events or meta['logged_bytes'] + len(raw) > self.max_bytes:
            meta['logging_limited'] = True
            self._persist(state)
            return False
        try:
            with (self._directory(meta['test_session_id']) / "events.jsonl").open('ab') as f:
                f.write(raw)
        except OSError as exc:
            meta['logging_error'] = type(exc).__name__+': '+str(exc)[:256]
            raise
        meta['event_count'] += 1
        meta['logged_bytes'] += len(raw)
        key = {"case": "case_events", "review": "reviews", "interaction": "interactions", "resource": "resource_samples", "trace": "trace_snapshots"}.get(kind)
        if key:
            meta['counts'][key] += 1
        self._persist(state)
        return True

    def _state(self, session_id=None):
        sid = session_id or self._active_id or self._last_id
        if not sid:
            raise BenchError("尚无测试批次，请先开始测试", 409)
        if sid not in self._sessions:
            path = self._directory(sid) / 'session.json'
            try:
                meta = json.loads(path.read_text(encoding='utf8'))
            except FileNotFoundError as exc:
                raise BenchError('未找到该测试批次', 404) from exc
            self._sessions[sid] = self._load_state(meta)
        self._sessions.move_to_end(sid)
        return self._sessions[sid]

    def start(self, body):
        label, notes = text(body, 'label', 128), text(body, 'notes', 2000)
        with self._lock:
            if self._active_id:
                raise BenchError('已有进行中的测试批次，请先结束', 409)
            if (self.root / 'active.json').exists():
                raise BenchError('发现无法恢复的批次标记，请保留文件并检查日志', 409)
            sid = 'test-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
            meta = {"schema_version": SCHEMA, "test_session_id": sid, "label": label or '树莓派实机测试',
                    "notes": notes, "version": self.version, "active": True, "status": "active",
                    "started_wall_ms": int(time.time()*1000), "ended_wall_ms": None,
                    "current_case_id": None, "cases": {}, "latest_reviews": {},
                    "counts": {k: 0 for k in ('cases','case_events','reviews','interactions','resource_samples','trace_snapshots')},
                    "event_count": 0, "logged_bytes": 0, "logging_limited": False,
                    "frame_bytes": 0, "frame_assets": {}, "frame_limit_bytes": 64*1024*1024,
                    "evidence_limits": dict(EVIDENCE_LIMITS), "resource_sample_seconds": self.sample_seconds}
            state = self._load_state(meta)
            self._sessions[sid] = state
            self._active_id = self._last_id = sid
            self._persist(state)
            (self.root/'active.json').write_text(json.dumps({'test_session_id':sid}), encoding='utf8')
            self._append(state, 'session_start', {'label':meta['label']})
            while len(self._sessions) > 4:
                self._sessions.popitem(last=False)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._sample_loop, daemon=True, name='pi-test-resources')
                self._thread.start()
            return self.status()

    def case(self, body):
        case_id = text(body, 'case_id', 64)
        if not CASE_ID.fullmatch(case_id):
            raise BenchError('题号使用字母、数字及 . _ : -，最多64字符')
        case = {'case_id':case_id, 'category':text(body,'category',64), 'prompt':text(body,'prompt'),
                'expected':text(body,'expected'), 'notes':text(body,'notes',2000)}
        with self._lock:
            if not self._active_id:
                raise BenchError('请先开始测试批次',409)
            state = self._state()
            cases = state['meta']['cases']
            if case_id not in cases and len(cases) >= 1000:
                raise BenchError('单批最多1000个题号，请开始新批次',409)
            cases[case_id] = case
            state['meta']['counts']['cases'] = len(cases)
            state['meta']['current_case_id'] = case_id
            self._append(state,'case',case)
            return self.status()

    def review(self, body):
        with self._lock:
            state = self._state(body.get('test_session_id'))
            cid = text(body,'case_id',64) or state['meta']['current_case_id']
            if not cid or cid not in state['meta']['cases']:
                raise BenchError('请先设置有效题号',409)
            review = {'case_id':cid, 'trace_id':text(body,'trace_id',128)}
            for key in ('recognition_correct','image_correct'):
                value = body.get(key)
                if value is not None and type(value) is not bool:
                    raise BenchError(key+'必须为true、false或null')
                review[key] = value
            for key in ('fluency','stability','latency_rating'):
                value = body.get(key)
                if value is not None and (type(value) is not int or not 1 <= value <= 5):
                    raise BenchError(key+'必须为1至5整数或null')
                review[key] = value
            for key, limit in (('notes',2000),('heard_transcript',4096),('spoken_reference',4096),('observed_answer',8192),('error_category',128)):
                review[key] = text(body,key,limit)
            outcome = body.get('outcome')
            if outcome is not None and outcome not in ('pass','fail','untested','indeterminate'):
                raise BenchError('outcome无效')
            review['outcome'] = outcome
            if not self._append(state,'review',review):
                raise BenchError('日志已达上限，评价未保存；请新建批次',409)
            state['meta']['latest_reviews'][cid] = review
            state['recent_reviews'].append(review)
            self._persist(state)
            return self.status(state['meta']['test_session_id'])

    def stop(self, body):
        notes = text(body,'notes',2000)
        with self._lock:
            if not self._active_id:
                raise BenchError('没有进行中的测试批次',409)
            state = self._state()
            self._append(state,'session_stop',{'notes':notes})
            state['meta'].update(active=False,status='stopped',ended_wall_ms=int(time.time()*1000),stop_notes=notes)
            state['late_until'] = time.monotonic()+30
            self._active_id = None
            self._persist(state)
            (self.root/'active.json').unlink(missing_ok=True)
            return self.status(state['meta']['test_session_id'])

    def context(self):
        with self._lock:
            if not self._active_id:
                return None
            meta = self._state()['meta']
            cid = meta['current_case_id']
            return {'test_session_id':meta['test_session_id'], 'case_id':cid,
                    'case_definition':deepcopy(meta['cases'].get(cid)), 'request_start_mono':time.monotonic()}

    def frame_evidence(self, context, resolved):
        """仅批次请求保存实际使用的原图；预览不录像，JSON只记资产引用。"""
        if not context:
            return
        value = resolved.get('image')
        if not isinstance(value, str) or not value:
            return
        media_type = value[5:].split(';',1)[0] if value.startswith('data:') else 'application/octet-stream'
        extension = {'image/png':'png','image/jpeg':'jpg','image/webp':'webp','image/bmp':'bmp','image/gif':'gif','image/tiff':'tiff'}.get(media_type,'image')
        encoded = value.split(',',1)[-1] if value.startswith('data:') else value
        if len(encoded) > 12*1024*1024:
            return
        try:
            content = base64.b64decode(encoded,validate=True)
        except (ValueError,binascii.Error):
            return
        if not 0 < len(content) <= 8*1024*1024:
            return
        digest = hashlib.sha256(content).hexdigest()
        with self._lock:
            state = self._state(context['test_session_id'])
            meta = state['meta']
            assets = meta.setdefault('frame_assets',{})
            if digest not in assets:
                if meta.get('frame_bytes',0)+len(content)>meta.get('frame_limit_bytes',64*1024*1024):
                    meta['frame_logging_limited'] = True
                    self._persist(state)
                    return
                directory = self._directory(meta['test_session_id'])/'frames'
                directory.mkdir(exist_ok=True)
                path = directory/(digest+'.'+extension)
                path.write_bytes(content)
                assets[digest] = {'relative_path':'frames/'+path.name,'sha256':digest,'bytes':len(content),'media_type':media_type}
                meta['frame_bytes'] = meta.get('frame_bytes',0)+len(content)
            context['frame_asset'] = deepcopy(assets[digest])
            self._append(state,'frame',{'case_id':context['case_id'],'frame_id':resolved.get('frame_id'),
                                       'asset':assets[digest], 'frame_context':resolved.get('frame_context')})

    def interaction(self, context, request_data, response_data, status, *, incomplete=False):
        if not context:
            return
        with self._lock:
            state = self._state(context['test_session_id'])
            response_data = {k:response_data[k] for k in RESPONSE_FIELDS+('frame_context',) if k in response_data}
            record = {'case_id':context['case_id'], 'case_definition_at_request':context['case_definition'],
                      'request':small(request_data), 'response':small(response_data), 'http_status':status,
                      'server_http_duration_ms':round((time.monotonic()-context['request_start_mono'])*1000,1),
                      'stream_incomplete':incomplete, 'physical_first_sound_ms':None, 'frame_asset':context.get('frame_asset')}
            self._append(state,'interaction',record)
            state['recent_interactions'].append(record)
            tid = response_data.get('trace_id') or request_data.get('request_id')
            if tid:
                self._remember_trace(str(tid),context)

    def _remember_trace(self, trace_id, context):
        self._trace_context[trace_id] = context
        self._trace_context.move_to_end(trace_id)
        while len(self._trace_context) > 2000:
            self._trace_context.popitem(last=False)

    def observe_trace(self, data, context=None):
        with self._lock:
            tid = str(data.get('trace_id') or '')
            context = context or self._trace_context.get(tid)
            if not tid or not context:
                return
            state = self._state(context['test_session_id'])
            if not state['meta']['active'] and time.monotonic() > state['late_until']:
                return
            self._remember_trace(tid,context)
            snapshot = small(data)
            digest = hashlib.sha256(json.dumps(snapshot,sort_keys=True,ensure_ascii=False).encode('utf8')).hexdigest()
            if state['trace_hashes'].get(tid)==digest:
                return
            state['trace_hashes'][tid] = digest
            while len(state['trace_hashes'])>200:
                state['trace_hashes'].popitem(last=False)
            self._append(state,'trace',{'case_id':context['case_id'], 'trace':snapshot,
                                      'timing_basis':'software events; browser timestamps not clock synchronized; play_start is not acoustic first sound'})

    def _sample_loop(self):
        while not self._stop.is_set():
            with self._lock:
                sid = self._active_id
            if sid and self.resource_getter:
                started=time.monotonic()
                try:
                    value=small(self.resource_getter())
                    record={'sample_duration_ms':round((time.monotonic()-started)*1000,1), 'values':value}
                    kind='resource'
                except Exception as exc:
                    record={'error':type(exc).__name__+': '+str(exc)[:256]};kind='resource_error'
                with self._lock:
                    state=self._state(sid)
                    if state['meta']['active']:
                        try:
                            self._append(state,kind,record)
                        except OSError as exc:
                            state['meta']['logging_error'] = type(exc).__name__+': '+str(exc)[:256]
                        state['last_resource']=record
            self._stop.wait(self.sample_seconds)

    @staticmethod
    def summary(meta):
        reviews=list(meta['latest_reviews'].values())
        quality={}
        for key in ('recognition_correct','image_correct'):
            measured=[r[key] for r in reviews if type(r.get(key)) is bool]
            quality[key]={'scored_cases':len(measured),'correct_cases':sum(measured),
                          'case_accuracy':sum(measured)/len(measured) if measured else None,
                          'basis':'latest manual case review; not per-utterance CER or independent quality acceptance'}
        return {'manual_quality':quality, 'reviewed_cases':len(reviews),
                'unreviewed_cases':len(meta['cases'])-len(reviews), 'logging_limited':meta['logging_limited'],
                'evidence_limits':dict(EVIDENCE_LIMITS)}

    def status(self, session_id=None):
        with self._lock:
            if not session_id and not self._active_id and not self._last_id:
                return {'ok':True,'session':None,'summary':None}
            state=self._state(session_id)
            meta=deepcopy(state['meta'])
            return {'ok':True,'session':meta,'summary':self.summary(meta),
                    'recent_interactions':deepcopy(list(state['recent_interactions'])),
                    'recent_reviews':deepcopy(list(state['recent_reviews'])), 'last_resource':deepcopy(state['last_resource'])}

    def export(self, session_id=None):
        with self._lock:
            state=self._state(session_id)
            meta=deepcopy(state['meta'])
            path=self._directory(meta['test_session_id'])/'events.jsonl'
            byte_limit=path.stat().st_size if path.exists() else 0
        def generate():
            yield '{"schema_version":'+json.dumps(SCHEMA)+',"session":'+json.dumps(meta,ensure_ascii=False)+',"summary":'+json.dumps(self.summary(meta),ensure_ascii=False)+',"events":['
            first=True
            if byte_limit:
                with path.open('rb') as f:
                    while f.tell()<byte_limit:
                        line=f.readline()
                        if not line:break
                        event=json.loads(line.decode('utf8'))
                        if not first:yield ','
                        first=False
                        yield json.dumps(event,ensure_ascii=False)
            yield ']}'
        return meta['test_session_id'],generate()

    def close(self):
        self._stop.set()
        if self._thread:self._thread.join(timeout=1)


def register_test_session_api(app, manager, traces=None):
    import atexit
    from flask import Response, g, has_request_context, jsonify, request
    atexit.register(manager.close)

    if traces is not None:
        def trace_snapshot(data):
            context=getattr(g,'pi_test_context',None) if has_request_context() else None
            manager.observe_trace(data,context)
        traces.subscribe(trace_snapshot)

    @app.before_request
    def pi_test_before():
        if request.path in REQUEST_PATHS:
            g.pi_test_context=manager.context()
            body=request.get_json(silent=True) or {}
            if not isinstance(body,dict):body={}
            g.pi_test_request={k:body[k] for k in REQUEST_FIELDS if k in body}
            g.pi_test_request.update(path=request.path, has_uploaded_image=bool(body.get('image') or body.get('image_vlm')),has_uploaded_audio=bool(body.get('audio')))

    @app.after_request
    def pi_test_after(response):
        context=getattr(g,'pi_test_context',None)
        if not context:return response
        request_data=getattr(g,'pi_test_request',{})
        def record(data,incomplete=False):
            try:manager.interaction(context,request_data,data,response.status_code,incomplete=incomplete)
            except OSError as exc:app.logger.error('测试结果写入失败: %s',exc)
        if response.is_json:
            data=response.get_json(silent=True)
            if not isinstance(data,dict):data={}
            frame=getattr(g,'visual_frame_context',None)
            if frame:data=dict(data,frame_context={k:frame[k] for k in ('used_frame_id','used_frame','selected_object_id','comparison_frames','vlm_inputs') if k in frame})
            reason=getattr(g,'vlm_finish_reason',None)
            if reason:data=dict(data,finish_reason=reason,truncated=reason=='length')
            record(data)
        elif response.mimetype=='text/event-stream':
            iterator=response.response
            def stream():
                decoder=codecs.getincrementaldecoder('utf8')('replace');buffer='';result={};complete=False
                try:
                    for part in iterator:
                        chunk=part if isinstance(part,str) else decoder.decode(part)
                        buffer+=chunk
                        while '\n\n' in buffer:
                            packet,buffer=buffer.split('\n\n',1)
                            for line in packet.splitlines():
                                if line.startswith('data: '):
                                    try:item=json.loads(line[6:])
                                    except ValueError:continue
                                    if isinstance(item,dict) and (item.get('done') or 'error' in item):
                                        result=small(item);complete=bool(item.get('done'))
                        if len(buffer)>128*1024:buffer=''
                        yield part
                finally:
                    try:
                        closer=getattr(iterator,'close',None)
                        if closer:closer()
                    finally:record(result,incomplete=not complete)
            response.response=stream()
        else:record({'error':'non-JSON interaction response'})
        return response

    def perform(action):
        try:
            body=request.get_json(silent=True)
            if body is None:body={}
            if not isinstance(body,dict):raise BenchError('请求必须是JSON对象')
            return jsonify(action(body))
        except BenchError as exc:return jsonify(ok=False,error=str(exc)),exc.status
        except OSError:return jsonify(ok=False,error='测试结果磁盘写入失败，请保留服务日志并检查空间'),507

    app.add_url_rule('/api/test/session/start','pi_test_start',lambda:perform(manager.start),methods=['POST'])
    app.add_url_rule('/api/test/case','pi_test_case',lambda:perform(manager.case),methods=['POST'])
    app.add_url_rule('/api/test/review','pi_test_review',lambda:perform(manager.review),methods=['POST'])
    app.add_url_rule('/api/test/session/stop','pi_test_stop',lambda:perform(manager.stop),methods=['POST'])

    @app.get('/api/test/session/status')
    def pi_test_status():
        try:return jsonify(manager.status(request.args.get('test_session_id')))
        except BenchError as exc:return jsonify(ok=False,error=str(exc)),exc.status

    @app.get('/api/test/session/export')
    def pi_test_export():
        try:
            sid,stream=manager.export(request.args.get('test_session_id'))
            return Response(stream,mimetype='application/json',headers={'Content-Disposition':'attachment; filename="'+sid+'.json"'})
        except BenchError as exc:return jsonify(ok=False,error=str(exc)),exc.status
