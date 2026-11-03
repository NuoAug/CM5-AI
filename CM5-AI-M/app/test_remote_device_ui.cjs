// 无浏览器硬件的行为回归：远程模式和掉线时不能调用本机外设。
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const code = fs.readFileSync(path.join(__dirname, 'static/device-mode.js'), 'utf8');

async function scenario(mode) {
  const calls = [];
  let local = 0;
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      id, style: {}, className: '', disabled: false, value: '',
      classList: { remove() {} }, parentElement: { before() {}, title: '' },
      onclick() { local++; },
    });
    return elements.get(id);
  }
  const status = { remote_device_mode: true, image_source: 'external',
    voice_source: 'device', output_sink: 'alsa', sink: true,
    camera: {running: false}, voice: {running: true, capture: true} };
  if (mode === 'legacy') delete status.remote_device_mode;
  if (mode === 'browser') status.remote_device_mode = false;
  const context = {
    window: {}, document: { hidden: false, createElement: () => element('notice') },
    $: element, setTimeout: () => 1, clearTimeout() {}, setInterval() {},
    AbortController, console,
    fetch: async (url, options) => {
      calls.push({url, body: options.body && JSON.parse(options.body)});
      if (mode === 'offline' || (mode === 'recover' && calls.length === 1)) throw new Error('offline');
      return {ok: true, json: async () => url.endsWith('/volume') ? {volume_percent:options.body ? 61 : 62,muted:false,mute_supported:true} : url.endsWith('/speak') ? {queued:true} : status};
    },
    toast() {}, voiceState() {},
    closeCamera() {}, captureFrame: async () => null,
    setCameraPlaceholder() {}, loadImageFromBlob: async () => {},
    loadCameraPreviewFromBlob: async () => {},
    playTts: async () => {local++;}, synthTts: async () => {local++;},
    speakAnswer: () => {local++;}, startVoice: () => {local++;}, stopVoice: () => {local++;},
    vOn: false,
  };
  vm.createContext(context);
  vm.runInContext(code, context);
  for (let i=0; i<40; i++) await Promise.resolve();
  await element('btn-voice').onclick();
  await element('btn-cam').onclick();
  await context.playTts('test');
  context.speakAnswer('late answer');
  await context.synthTts('late synth').catch(() => {});
  assert.equal(local, 0, mode + ': Pi edition must never use browser hardware');
  if (mode === 'remote' || mode === 'recover') {
    assert.ok(calls.some(c => c.url === '/api/device/voice' && c.body.action === 'stop'));
    assert.equal(calls.filter(c => c.url === '/api/device/speak').length, 1);
    assert.equal(calls.filter(c => c.url === '/api/device/camera').length, 0, 'unsupported image source must not start device camera');
  }
  if (mode === 'browser') assert.equal(calls.filter(c => c.url === '/api/device/speak').length, 0, 'bad remote config must fail closed');
  if (mode === 'remote') {
    const ui = context.window.CM5DeviceUI;
    ui.showSpeech({utterance_id:'u',sequence:1,text:'第一段。'});
    ui.showSpeech({utterance_id:'u',sequence:2,text:'第二段。'});
    ui.showSpeech({utterance_id:'u',sequence:2,text:'重复段。'});
    assert.equal(element('out-eyes').textContent, '第一段。第二段。');
    ui.showSpeech({utterance_id:'new',sequence:1,text:'新回答。'});
    assert.equal(element('out-eyes').textContent, '新回答。');
    element('device-volume').value = '64';
    element('device-volume').oninput();
    await element('device-volume').onchange();
    assert.equal(element('device-volume-value').textContent, '61%', 'show hardware readback, not requested value');
    await element('device-volume-test').onclick();
    assert.ok(calls.some(c => c.url === '/api/device/speak' && c.body.text.includes('音量测试')));
    assert.ok(element('device-voice-details').textContent.includes('屏蔽识别'));
  }
}
(async () => {
  for (const mode of ['remote','offline','legacy','browser','recover']) await scenario(mode);
  console.log('PASS: Pi-only peripherals, bad-config/offline fail-closed, reconnect, volume readback and segmented answer');
})();
