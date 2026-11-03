"""真实无声故障的进程/管道注入：空闲kill、spawn取消与连续普通回答。"""
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock
import wave
from tts_worker import PiperWorker, TTSInterrupted
from output import OutputBus
from device_io import DeviceSink


def wav():
    out=io.BytesIO()
    with wave.open(out,'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(b'\0\0'*1600)
    return out.getvalue()


class Pipe:
    def __init__(self,proc):self.proc=proc;self.closed=False
    def write(self,content):
        if self.proc.block_write:
            self.proc.writing.set()
            if not self.proc.release.wait(2):raise TimeoutError('fixture write')
        if self.proc.returncode is not None:raise BrokenPipeError('dead Piper')
        data=json.loads(content.decode('utf8'))
        Path(data['output_file']).write_bytes(wav())
        return len(content)
    def flush(self):
        if self.proc.returncode is not None:raise BrokenPipeError('dead Piper')
    def close(self):self.closed=True


class Process:
    def __init__(self,*args,**kwargs):
        self.returncode=None;self.stdin=Pipe(self);self.kill_calls=0
        self.block_write=False;self.writing=threading.Event();self.release=threading.Event()
    def poll(self):return self.returncode
    def kill(self):
        self.kill_calls+=1;self.returncode=-9;self.release.set()
    def terminate(self):self.returncode=-15;self.release.set()
    def wait(self,timeout=None):return self.returncode


class PiperCancellationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);binary=root/'piper';model=root/'voice.onnx';binary.touch();model.touch()
        self.worker=PiperWorker(binary,model,root/'tmp')
        self.addCleanup(self.worker.stop)
        self.processes=[]
        def spawn(*args,**kwargs):
            proc=Process();self.processes.append(proc);return proc
        patch=mock.patch('tts_worker.subprocess.Popen',side_effect=spawn)
        patch.start();self.addCleanup(patch.stop)

    def test_idle_interrupt_never_reuses_dead_pipe(self):
        self.worker.synth('第一句')
        old=self.processes[-1]
        self.assertTrue(self.worker.interrupt())
        self.assertEqual(old.poll(),-9)
        self.assertFalse(self.worker.status()['alive'])
        data,_=self.worker.synth('取消后新一句')
        self.assertGreater(len(data),1024)
        self.assertIsNot(self.processes[-1],old)
        self.assertEqual(self.worker.status()['synth_count'],2)

    def test_already_exited_idle_process_is_replaced_before_write(self):
        self.worker.synth('预热')
        old=self.processes[-1];old.returncode=-7
        self.worker.synth('新句')
        self.assertIsNot(self.processes[-1],old)
        self.assertTrue(old.stdin.closed)
        self.assertTrue(self.worker.status()['alive'])

    def test_interrupt_during_write_stays_cancelled_and_next_request_recovers(self):
        self.worker.synth('预热')
        proc=self.processes[-1];proc.block_write=True;errors=[]
        def work():
            try:self.worker.synth('被取消的句子')
            except Exception as exc:errors.append(exc)
        thread=threading.Thread(target=work);thread.start()
        self.assertTrue(proc.writing.wait(.5))
        self.worker.interrupt();thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(errors[0],TTSInterrupted)
        data,_=self.worker.synth('独立新句')
        self.assertGreater(len(data),1024)
        self.assertEqual(self.worker.status()['synth_count'],2)

    def test_cancel_during_spawn_cannot_publish_late_process(self):
        spawning,release=threading.Event(),threading.Event();errors=[];spawned=[]
        def slow_spawn(*a,**k):
            p=Process();spawned.append(p);spawning.set();release.wait(1);return p
        with mock.patch('tts_worker.subprocess.Popen',side_effect=slow_spawn):
            thread=threading.Thread(target=lambda:self.capture_synth(errors));thread.start()
            self.assertTrue(spawning.wait(.5));self.worker.interrupt();release.set();thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(errors[0],TTSInterrupted)
        self.assertFalse(self.worker.status()['alive'])
        self.assertEqual(spawned[0].poll(),-9)
        self.worker.synth('后续句子')
        self.assertTrue(self.worker.status()['alive'])

    def capture_synth(self,errors):
        try:self.worker.synth('旧句子')
        except Exception as exc:errors.append(exc)

    def test_status_does_not_wait_for_synthesis_lock(self):
        self.worker.synth('预热');proc=self.processes[-1];proc.block_write=True;errors=[]
        thread=threading.Thread(target=lambda:self.capture_synth(errors));thread.start()
        self.assertTrue(proc.writing.wait(.5))
        started=time.monotonic();self.worker.status()
        self.assertLess(time.monotonic()-started,.1)
        self.worker.interrupt();thread.join(1)

    def test_real_preempt_path_speaks_two_new_answers_after_manual_warmup(self):
        self.worker.synth('先手动合成')
        played=[];first=threading.Event();second=threading.Event()
        class Player:
            def feed(self,data,cancel):
                if not cancel.is_set():
                    played.append(data)
                    (first if len(played)==1 else second).set()
                return True
            def stop(self):pass
            def end_stream(self):pass
        bus=OutputBus()
        sink=DeviceSink(lambda text:self.worker.synth(text)[0],Player(),cancel_synth=self.worker.interrupt)
        bus.attach_sink(sink)
        try:
            bus.push('第一条普通问答',preempt=True,ttl=5)
            self.assertTrue(first.wait(1),'普通preempt后第一句未播')
            bus.push('第二条普通问答',preempt=True,ttl=5)
            self.assertTrue(second.wait(1),'普通preempt后下一句未播')
            self.assertEqual(len(played),2)
            self.assertEqual(bus.stats().get('prepare_failed',0),0)
        finally:bus.detach_sink()


class PrepareDiagnosticsTests(unittest.TestCase):
    def test_prepare_failure_is_bounded_observable_and_never_counted_played(self):
        error=threading.Event()
        class Sink:
            def prepare(self,payload,cancel):error.set();raise RuntimeError('x'*4000)
            def speak(self,payload,cancel):raise AssertionError('must not play')
            def stop(self):pass
        bus=OutputBus();events=bus.subscribe();bus.attach_sink(Sink())
        try:
            bus.push('普通回答',preempt=True,ttl=5)
            self.assertTrue(error.wait(.5))
            deadline=time.monotonic()+1
            found=None
            while time.monotonic()<deadline:
                item=events.get(timeout=1)
                if item.get('type')=='output_error':found=item;break
            self.assertIsNotNone(found)
            self.assertFalse(found['played'])
            self.assertLessEqual(len(found['message']),512)
            state=bus.stats()
            self.assertEqual(state['prepare_failed'],1)
            self.assertEqual(state.get('sink_played',0),0)
            self.assertEqual(state['last_prepare_error']['stage'],'prepare')
        finally:bus.detach_sink()


if __name__=='__main__':unittest.main()
