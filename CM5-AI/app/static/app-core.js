// 统一播放器与交互代次：页面与自动化测试共用，避免实现和测试分叉。
const player = { gen: 0, queue: [], running: false, audio: null, resolver: null };
let interactionGen = 0;
let activeCtrl = null;

function stopPlayer() {
  player.gen += 1;
  player.queue = [];
  if (window.speechSynthesis) {
    try { window.speechSynthesis.cancel(); } catch (e) { /* ignore */ }
  }
  if (player.audio) {
    const a = player.audio;
    a.onended = null;
    a.onerror = null;
    try { a.pause(); } catch (e) { /* ignore */ }
    a.src = '';
  }
  if (player.resolver) {
    const r = player.resolver;
    player.resolver = null;
    r();
  }
  player.running = false;
}

function splitSentences(text, maxLen = 90) {
  const raw = (text || '').split(/([。！？；\n])/);
  const pieces = [];
  let buf = '';
  for (let i = 0; i < raw.length; i += 2) {
    const seg = raw[i].trim();
    const punct = raw[i + 1] || '';
    if (!seg) continue;
    const piece = seg + (punct && punct !== '\n' ? punct : '。');
    if ((buf + piece).length <= maxLen) buf += piece;
    else {
      if (buf) pieces.push(buf);
      buf = piece;
    }
  }
  if (buf) pieces.push(buf);
  const out = [];
  for (let piece of pieces) {
    while (piece.length > maxLen) {
      const win = piece.slice(0, maxLen);
      const cut = Math.max(win.lastIndexOf('，'), win.lastIndexOf('、'));
      const take = cut >= Math.floor(maxLen / 3) ? cut + 1 : maxLen;
      out.push(piece.slice(0, take));
      piece = piece.slice(take);
    }
    if (piece) out.push(piece);
  }
  return out;
}

function playUrl(url) {
  return new Promise((resolve) => {
    const a = $('audio');
    player.audio = a;
    player.resolver = resolve;
    a.src = url;
    const done = () => {
      a.onended = null;
      a.onerror = null;
      if (player.resolver === resolve) player.resolver = null;
      resolve();
    };
    a.onended = done;
    a.onerror = done;
    a.play().catch(done);
  });
}

function pickZhVoice() {
  const synth = window.speechSynthesis;
  const voices = synth ? synth.getVoices() || [] : [];
  return voices.find((v) => /^zh|cmn|chinese/i.test(v.lang || '') || /chinese|huihui|yaoyao|kangkang/i.test(v.name || '')) || null;
}

function systemSpeak(text) {
  return new Promise((resolve) => {
    const synth = window.speechSynthesis;
    if (!synth) { resolve(false); return; }
    const u = new SpeechSynthesisUtterance(text);
    const voice = pickZhVoice();
    if (voice) u.voice = voice;
    u.lang = voice ? voice.lang : 'zh-CN';
    u.rate = 1.0;
    let done = false;
    let guard = null;
    const finish = (ok) => {
      if (done) return;
      done = true;
      if (guard) clearTimeout(guard);
      if (player.resolver === cancel) player.resolver = null;
      resolve(ok);
    };
    const cancel = () => finish(false);
    // 没有音频输出设备时 onend/onerror 可能都不触发，必须兜底，否则播放器会永久占用
    guard = setTimeout(() => {
      try { synth.cancel(); } catch (e) { /* ignore */ }
      finish(false);
    }, Math.max(6000, text.length * 400 + 4000));
    player.resolver = cancel;
    u.onend = () => finish(true);
    u.onerror = () => finish(false);
    synth.speak(u);
  });
}

async function speakSentence(text) {
  if (window.speechSynthesis && window.useSystemVoice !== false) {
    const ok = await systemSpeak(text);
    if (ok) return;
  }
  let url = null;
  try { url = (await synthTts(text)).url; } catch (e) { return; }
  if (!url) return;
  await playUrl(url);
  URL.revokeObjectURL(url);
}

async function pumpPlayer(gen) {
  if (player.running) return;
  player.running = true;
  try {
    while (player.queue.length && gen === player.gen) {
      const sentence = player.queue.shift();
      await speakSentence(sentence);
      if (gen !== player.gen) break;
    }
  } finally {
    player.running = false;
  }
}

async function speakAll(text) {
  if (!text) return;
  stopPlayer();
  const gen = player.gen;
  splitSentences(text).forEach((s) => player.queue.push(s));
  await pumpPlayer(gen);
}

function beginInteraction() {
  interactionGen += 1;
  stopPlayer();
  if (activeCtrl) {
    activeCtrl.abort();
    activeCtrl = null;
  }
  activeCtrl = new AbortController();
  return { gen: interactionGen, signal: activeCtrl.signal };
}

function cancelInteraction() {
  interactionGen += 1;
  stopPlayer();
  if (activeCtrl) {
    activeCtrl.abort();
    activeCtrl = null;
  }
}
