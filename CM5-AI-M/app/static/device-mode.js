/* 远程测试台只控制服务端设备；状态未知或掉线时不退回电脑外设。 */
(() => {
  'use strict';
  const ui = window.CM5DeviceUI = { remote: true, status: null, live: false, pinned: false, connected: false };
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
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    const options = { cache: 'no-store', signal: controller.signal };
    if (body !== undefined) {
      options.method = 'POST';
      options.headers = { 'Content-Type': 'application/json' };
      options.body = JSON.stringify(body);
    }
    try {
      const response = await fetch(path, options);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || ('HTTP ' + response.status));
      return data;
    } finally { clearTimeout(timeout); }
  }
  function render(status) {
    if (typeof status.remote_device_mode !== 'boolean') throw new Error('服务端未提供外设模式，请刷新或升级服务');
    if (!status.remote_device_mode || status.voice_source !== 'device' || status.output_sink !== 'alsa') {
      throw new Error('实机服务外设配置错误：必须使用树莓派device输入与ALSA输出');
    }
    ui.status = status; ui.connected = true; ui.remote = true;
    const voice = status.voice || {};
    const camera = status.camera || {};
    notice.textContent = '摄像头：树莓派' + (camera.running ? '（采集中）' : '（未采集）') +
      ' · 麦克风：树莓派' + (voice.capture ? '（收音中）' : '（未收音）') +
      ' · 扬声器：树莓派' + (status.sink ? '' : '（不可用）') +
      ' · 图像模式：' + status.image_source;
    $('btn-cam').textContent = ui.live ? '关闭树莓派预览' : '打开树莓派摄像头';
    $('btn-shot').disabled = !camera.running;
    $('btn-voice').textContent = voice.running ? '停止树莓派收音' : '开启树莓派语音唤醒';
    voiceState(voice.error || (voice.running ? (voice.output_active ? (voice.aec_reference_verified ? '持续采集 · 播报中，可测试插话' : '持续采集 · 播报期间识别受保护') : '树莓派监听中 · 说「小冰小冰」') : '树莓派收音已停止'), !!voice.running && !voice.error);
    const details = $('device-voice-details');
    details.textContent = (voice.aec_reference_verified ? 'AEC参考路径已声明，真人插话仍需本轮验证。' : 'AEC参考尚未验证；播报时持续采集，但屏蔽识别以免回声误触发。') + (voice.last_partial ? ' 临时转写：' + voice.last_partial : '');
    $('audio').style.display = 'none';
    $('btn-tts').textContent = '在树莓派播报';
  }

  let confirmedVolume = null;
  let pendingVolume = null;
  let volumeBusy = false;
  let volumeTimer = null;
  let volumeGeneration = 0;
  function showVolume(value) {
    if (!value || !Number.isInteger(value.volume_percent) || value.volume_percent < 0 || value.volume_percent > 100) throw new Error('树莓派返回的音量无效');
    confirmedVolume = value;
    $('device-volume').value = value.volume_percent;
    $('device-volume').disabled = !ui.connected;
    $('device-volume-value').textContent = value.volume_percent + '%' + (value.muted ? '（静音）' : value.partially_muted ? '（部分声道静音）' : '');
    $('device-mute').textContent = value.muted ? '取消静音' : '静音';
    $('device-mute').disabled = !ui.connected || !value.mute_supported;
    $('device-volume-status').textContent = '树莓派已读回 · ' + (value.card || '') + ' / ' + (value.control || '') + (value.mute_supported ? '' : ' · 无静音开关，可将音量调至0%');
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
      $('device-volume-status').textContent = '设置失败：' + error.message + '；滑块仅显示上次确认值，请点“读取音量”核实。';
      toast('树莓派音量调节失败：' + error.message);
    } finally {
      volumeBusy = false;
      if (confirmedVolume) $('device-mute').disabled = !ui.connected || !confirmedVolume.mute_supported;
    }
  }
  function queueVolume(change) {
    volumeGeneration += 1;
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

  function disconnected(error) {
    ui.connected = false;
    notice.textContent = '树莓派状态未确认：' + error.message + '；不会启用Windows外设';
    $('device-volume').disabled = true; $('device-mute').disabled = true;
    $('device-volume-status').textContent = '连接中断，当前音量未确认';
    voiceState('树莓派连接中断；设备是否仍收音需恢复连接后确认', false);
  }
  async function readVolume() {
    if (volumeBusy || pendingVolume) return;
    const generation = volumeGeneration;
    try {
      await remote(); const value = await api('/api/device/volume');
      if (generation === volumeGeneration && !volumeBusy && !pendingVolume) showVolume(value);
    }
    catch (error) {
      $('device-volume').disabled = true; $('device-mute').disabled = true;
      $('device-volume-value').textContent = '未确认';
      $('device-volume-status').textContent = '音量读取失败：' + error.message + '；可点“读取音量”重试';
    }
  }
  $('device-volume-refresh').onclick = readVolume;
  $('device-volume-test').onclick = async () => {
    try { await remote(); await deviceSpeak('这是树莓派扬声器音量测试，一二三四五。'); }
    catch (error) { toast('试听失败：' + error.message); }
  };
  const ready = api('/api/device/status').then(status => { render(status); return status; });
  ready.catch(disconnected);
  async function remote() {
    if (!ui.connected) {
      try { render(await api('/api/device/status')); }
      catch (error) { disconnected(error); throw error; }
    }
    return true;
  }

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
  $('btn-cam').onclick = async () => {
    try {
      await remote();
      if (ui.live) closeCamera();
      else await openDeviceCamera();
    } catch (error) {
      if (ui.remote) setCameraPlaceholder('树莓派摄像头不可用', error.message);
      toast(error.message);
    }
  };
  $('btn-shot').onclick = async () => {
    try {
      await remote();
      closeCamera();
      if (await loadFrame('/api/camera/still', epoch, false)) ui.pinned = true;
      render(await api('/api/device/status'));
    } catch (error) { toast(error.message); }
  };
  const browserCapture = captureFrame;
  captureFrame = async function () {
    try {
      await remote();
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
      await remote();
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
  playTts = async function (text) {
    try { await remote(); return await deviceSpeak(text); }
    catch (error) { toast(error.message); }
  };
  speakAnswer = function () {}; // 统一树莓派输出总线负责播报，网页不重复播放。
  synthTts = async function (...args) {
    throw new Error('实机版本禁止在Windows播放，请使用树莓派播报接口');
  };
  let displayedUtterance = '', displayedSequence = 0, displayedText = '';
  ui.showSpeech = function (event) {
    const uid = event.utterance_id || event.id || '';
    if (uid !== displayedUtterance) { displayedUtterance = uid; displayedSequence = 0; displayedText = ''; }
    if (event.sequence && event.sequence <= displayedSequence) return;
    if (event.sequence) displayedSequence = event.sequence;
    displayedText = (event.meta && event.meta.answer) || (displayedText + (event.text || ''));
    displayedText = displayedText.slice(-8000);
    $('out-eyes').classList.remove('empty');
    $('out-eyes').textContent = displayedText;
    $('meta-eyes').textContent = '树莓派输出已提交 · ' + (event.source || '') + ' · ' + uid + '（是否实际出声请人工确认）';
  };
  let statusBusy = false;
  setInterval(async () => {
    if (document.hidden || statusBusy) return;
    statusBusy = true;
    try { render(await api('/api/device/status')); }
    catch (error) { disconnected(error); }
    finally { statusBusy = false; }
  }, 5000);
  setInterval(() => { if (!document.hidden) readVolume(); }, 10000);
  ready.then(async () => {
    if (ui.remote) {
      $('device-volume-controls').style.display = 'flex';
      await readVolume();
    }
    if (ui.remote && ui.status.image_source === 'standalone-device-test') {
      try { await openDeviceCamera(); } catch (error) { setCameraPlaceholder('树莓派摄像头不可用', error.message); }
    }
  }).catch(() => {});
})();
