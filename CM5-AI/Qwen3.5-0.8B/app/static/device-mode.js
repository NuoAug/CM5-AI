/* 远程测试台只控制服务端设备；状态未知或掉线时不退回电脑外设。 */
(() => {
  'use strict';
  const ui = window.CM5DeviceUI = { remote: true, status: null, live: false, pinned: false };
  let epoch = 0;
  let timer = null;
  let inflight = null;
  const notice = document.createElement('p');
  notice.className = 'hint';
  notice.id = 'device-source-status';
  notice.textContent = '正在确认外设来源…';
  const sourceSlot = $('device-source-slot');
  if (sourceSlot && sourceSlot.appendChild) sourceSlot.appendChild(notice);
  else $('sample-image').parentElement.before(notice);

  async function api(path, body) {
    const options = { cache: 'no-store' };
    if (body !== undefined) {
      options.method = 'POST';
      options.headers = { 'Content-Type': 'application/json' };
      options.body = JSON.stringify(body);
    }
    const response = await fetch(path, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || ('HTTP ' + response.status));
    return data;
  }
  function render(status) {
    if (typeof status.remote_device_mode !== 'boolean') throw new Error('服务端未提供外设模式，请刷新或升级服务');
    ui.status = status;
    ui.remote = !!status.remote_device_mode;
    if (!ui.remote) { notice.textContent = '开发模式：使用本机浏览器外设'; return; }
    const voice = status.voice || {};
    const camera = status.camera || {};
    notice.textContent = '摄像头：树莓派' + (camera.running ? '（采集中）' : '（未采集）') +
      ' · 麦克风：树莓派' + (voice.capture ? '（收音中）' : '（未收音）') +
      ' · 扬声器：树莓派' + (status.sink ? '' : '（不可用）') +
      ' · 图像模式：' + status.image_source;
    $('btn-cam').textContent = ui.live ? '关闭树莓派预览' : '打开树莓派摄像头';
    $('btn-shot').disabled = !camera.running;
    $('btn-voice').textContent = voice.running ? '停止树莓派收音' : '开启树莓派语音唤醒';
    voiceState(voice.error || (voice.running ? (voice.paused ? '树莓派播报中 · 暂停收音' : '树莓派监听中 · 说「小冰小冰」') : '树莓派收音已停止'), !!voice.running && !voice.error);
    $('audio').style.display = 'none';
    $('btn-tts').textContent = '在树莓派播报';
  }

  let confirmedVolume = null;
  let pendingVolume = null;
  let volumeBusy = false;
  let volumeTimer = null;
  function showVolume(value) {
    confirmedVolume = value;
    $('device-volume').value = value.volume_percent;
    $('device-volume').disabled = false;
    $('device-volume-value').textContent = value.volume_percent + '%' + (value.muted ? '（静音）' : '');
    $('device-mute').textContent = value.muted ? '取消静音' : '静音';
    $('device-mute').disabled = !value.mute_supported;
  }
  async function flushVolume() {
    clearTimeout(volumeTimer);
    if (volumeBusy || !pendingVolume) return;
    volumeBusy = true;
    $('device-mute').disabled = true;
    try {
      while (pendingVolume) {
        const change = pendingVolume;
        pendingVolume = null;
        const actual = await api('/api/device/volume', change);
        confirmedVolume = actual;
        if (!pendingVolume) showVolume(actual);
      }
    } catch (error) {
      pendingVolume = null;
      if (confirmedVolume) showVolume(confirmedVolume);
      toast('树莓派音量调节失败：' + error.message);
    } finally {
      volumeBusy = false;
      if (confirmedVolume) $('device-mute').disabled = !confirmedVolume.mute_supported;
    }
  }
  function queueVolume(change) {
    pendingVolume = Object.assign(pendingVolume || {}, change);
    clearTimeout(volumeTimer);
    volumeTimer = setTimeout(flushVolume, 180);
  }
  $('device-volume').oninput = () => {
    const value = Number($('device-volume').value);
    $('device-volume-value').textContent = value + '%';
    queueVolume({ volume_percent: value });
  };
  $('device-volume').onchange = flushVolume;
  $('device-mute').onclick = () => {
    if (!confirmedVolume) return;
    queueVolume({ muted: !confirmedVolume.muted });
    return flushVolume();
  };

  const ready = api('/api/device/status').then(status => { render(status); return status; });
  ready.catch(error => { notice.textContent = '设备状态不可用：' + error.message + '；不会启用电脑外设'; });
  async function remote() { await ready; return ui.remote; }

  const originalCloseCamera = closeCamera;
  closeCamera = function () {
    epoch += 1;
    ui.live = false;
    ui.pinned = false;
    clearTimeout(timer);
    if (inflight) inflight.abort();
    originalCloseCamera();
    if (ui.remote) {
      $('btn-cam').textContent = '打开树莓派摄像头';
      setCameraPlaceholder('树莓派预览已关闭', '点击「打开树莓派摄像头」恢复实时画面');
    }
  };

  async function loadFrame(path, generation, preview, label = '') {
    const controller = new AbortController();
    if (preview) inflight = controller;
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      const response = await fetch(path, { cache: 'no-store', signal: controller.signal });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || '树莓派取图失败');
      }
      const blob = await response.blob();
      if (generation !== epoch) return false;
      const at = response.headers.get('X-Captured-At') || '';
      if (preview) {
        await loadCameraPreviewFromBlob(blob, () => generation === epoch, '树莓派实时预览 · ' + at);
      } else {
        await loadImageFromBlob(blob, () => generation === epoch);
        if (generation !== epoch) return false;
        currentFrame = { id: response.headers.get('X-Frame-Id') || '', at, source: 'device_camera' };
        $('sample-image').value = '';
        setImageSource((label || '树莓派拍照（静态）') + ' · ' + at);
      }
      return true;
    } finally {
      clearTimeout(timeout);
      if (inflight === controller) inflight = null;
    }
  }
  async function preview(generation) {
    if (!ui.live || generation !== epoch) return;
    try {
      if (!document.hidden) await loadFrame('/api/camera/preview', generation, true);
    } catch (error) {
      if (generation === epoch && error.name !== 'AbortError') {
        setCameraPlaceholder('树莓派预览中断', error.message);
      }
    } finally {
      if (ui.live && generation === epoch) timer = setTimeout(() => preview(generation), 1000);
    }
  }
  async function openDeviceCamera() {
    if (ui.status.image_source !== 'standalone-device-test') {
      throw new Error('当前图像来源不支持设备摄像头，请启用独立树莓派相机测试');
    }
    closeCamera();
    setCameraPlaceholder('正在连接树莓派摄像头…', '首帧到达前请稍候');
    render(await api('/api/device/camera', { action: 'start' }));
    ui.live = true;
    $('btn-cam').textContent = '关闭树莓派预览';
    $('btn-shot').disabled = false;
    $('sample-image').value = '';
    preview(epoch);
  }
  const browserCameraClick = $('btn-cam').onclick;
  $('btn-cam').onclick = async () => {
    try {
      if (!await remote()) return browserCameraClick();
      if (ui.live) closeCamera();
      else await openDeviceCamera();
    } catch (error) {
      if (ui.remote) setCameraPlaceholder('树莓派摄像头不可用', error.message);
      toast(error.message);
    }
  };
  const browserShot = $('btn-shot').onclick;
  $('btn-shot').onclick = async () => {
    try {
      if (!await remote()) return browserShot();
      closeCamera();
      if (await loadFrame('/api/camera/still', epoch, false)) ui.pinned = true;
      render(await api('/api/device/status'));
    } catch (error) { toast(error.message); }
  };
  const browserCapture = captureFrame;
  captureFrame = async function () {
    try {
      if (!await remote()) return browserCapture();
      if (ui.live) {
        if (!await loadFrame('/api/camera/still', epoch, false, '自动抽帧（树莓派摄像头）')) return null;
        return { full: state.imageFull, small: state.imageVlm };
      }
      if (ui.pinned) return { full: state.imageFull, small: state.imageVlm };
      return browserCapture(); // 明确选择的上传/示例静态图，不采集电脑摄像头。
    } catch (error) { toast(error.message); return null; }
  };
  $('btn-voice').onclick = async () => {
    try {
      if (!await remote()) return vOn ? stopVoice() : startVoice();
      const status = await api('/api/device/status');
      render(await api('/api/device/voice', { action: status.voice.running ? 'stop' : 'start' }));
    } catch (error) { toast(error.message); }
  };
  async function deviceSpeak(text) {
    if (!text || !text.trim()) return;
    const result = await api('/api/device/speak', { text });
    $('meta-tts').textContent = result.queued ? '已提交至树莓派扬声器' : '未入队';
    return result;
  }
  const browserTts = playTts;
  playTts = async function (text) {
    try { return await remote() ? await deviceSpeak(text) : await browserTts(text); }
    catch (error) { toast(error.message); }
  };
  const browserSpeakAnswer = speakAnswer;
  speakAnswer = function (text) { if (!ui.remote) return browserSpeakAnswer(text); };
  const browserSynth = synthTts;
  synthTts = async function (...args) {
    if (await remote()) throw new Error('远程模式禁止在电脑播放，请使用树莓派播报接口');
    return browserSynth(...args);
  };
  ui.showSpeech = function (event) {
    $('out-eyes').classList.remove('empty');
    $('out-eyes').textContent = (event.meta && event.meta.answer) || event.text;
    $('meta-eyes').textContent = '树莓派输出 · ' + (event.source || '') + ' · ' + (event.utterance_id || '');
  };
  setInterval(async () => {
    if (document.hidden) return;
    try { render(await api('/api/device/status')); }
    catch (error) { notice.textContent = '树莓派连接中断：' + error.message + '；不会改用电脑外设'; }
  }, 5000);
  ready.then(async () => {
    if (ui.remote) {
      $('device-volume-controls').style.display = 'flex';
      try { showVolume(await api('/api/device/volume')); }
      catch (error) { $('device-volume-value').textContent = '不可用'; $('device-volume').disabled = true; }
    }
    if (ui.remote && ui.status.image_source === 'standalone-device-test') {
      try { await openDeviceCamera(); } catch (error) { setCameraPlaceholder('树莓派摄像头不可用', error.message); }
    }
  }).catch(() => {});
})();
