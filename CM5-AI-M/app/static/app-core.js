// 统一播放器与交互代次：页面与自动化测试共用，避免实现和测试分叉。
const player = { gen: 0, queue: [], running: false, audio: null, resolver: null, speed: null };
let interactionGen = 0;
let activeCtrl = null;

function newFrameId(prefix = 'f') {
  return prefix + '-' + Date.now().toString(36) + '-' + Math.random().toString(16).slice(2, 8);
}

function stopPlayer() {
  player.gen += 1;
  player.queue = [];
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

async function synthesizeSentence(text, speed) {
  try {
    return await synthTts(text, 'piper', speed);
  } catch (e) {
    return null;
  }
}

async function pumpPlayer(gen) {
  if (player.running) return;
  player.running = true;
  let next = null;
  try {
    while (player.queue.length && gen === player.gen) {
      const sentence = player.queue.shift();
      // 当前句播放时，提前合成下一句，避免每句都等 Piper 进程启动
      const current = next || synthesizeSentence(sentence, player.speed);
      next = null;
      if (player.queue.length) next = synthesizeSentence(player.queue[0], player.speed);
      const t = await current;
      if (!t || !t.url) continue;
      if (gen !== player.gen) {
        URL.revokeObjectURL(t.url);
        break;
      }
      await playUrl(t.url);
      URL.revokeObjectURL(t.url);
    }
  } finally {
    if (next) {
      next.then((t) => {
        if (t && t.url) URL.revokeObjectURL(t.url);
      }).catch(() => {});
    }
    player.running = false;
  }
}

function chunkSentences(text, maxLen = 180) {
  const parts = splitSentences(text);
  const chunks = [];
  let buf = '';
  for (const part of parts) {
    if (buf && (buf + part).length > maxLen) {
      chunks.push(buf);
      buf = '';
    }
    buf += part;
  }
  if (buf) chunks.push(buf);
  return chunks;
}

async function speakAll(text, opts = {}) {
  if (!text) return;
  streamPlayer.stop();
  stopPlayer();
  player.speed = Number(opts.speed) > 0 ? Number(opts.speed) : null;
  const gen = player.gen;
  document.dispatchEvent(new CustomEvent('cm5-tts-start', { detail: { text, speed: player.speed } }));
  chunkSentences(text, 60).forEach((s) => player.queue.push(s));
  try {
    await pumpPlayer(gen);
  } finally {
    player.speed = null;
    document.dispatchEvent(new CustomEvent('cm5-tts-end'));
  }
}

function beginInteraction() {
  interactionGen += 1;
  stopPlayer();
  streamPlayer.stop();
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
  streamPlayer.stop();
  if (activeCtrl) {
    activeCtrl.abort();
    activeCtrl = null;
  }
}

// 连续 AudioContext 播放器：同一回答的语义片段追加调度，不逐段切换 audio.src。
const streamPlayer = {
  ctx: null,
  gen: 0,
  playhead: 0,
  sources: new Set(),
  queue: [],
  seen: new Set(),
  activeUtterance: null,
  started: false,
  running: false,
  ctrl: null,
  underruns: 0,
  failures: 0,

  ensureContext() {
    if (!this.ctx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return null;
      this.ctx = new Ctx({ latencyHint: 'interactive' });
    }
    if (this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
    return this.ctx;
  },

  reset() {
    this.gen += 1;
    if (this.ctrl) {
      try { this.ctrl.abort(); } catch (e) { /* ignore */ }
      this.ctrl = null;
    }
    for (const src of this.sources) {
      try { src.stop(); } catch (e) { /* ignore */ }
    }
    this.sources.clear();
    this.queue = [];
    this.activeUtterance = null;
    this.playhead = 0;
    this.spokenText = '';
    this.running = false;
    if (this.started) {
      this.started = false;
      document.dispatchEvent(new CustomEvent('cm5-tts-end'));
    }
  },

  stop() {
    this.reset();
  },

  enqueue(job) {
    if (!job || !job.text) return;
    const uid = job.utteranceId || '';
    if (job.segmentId && this.seen.has(`${uid}|${job.segmentId}`)) return;
    if (job.segmentId) {
      this.seen.add(`${uid}|${job.segmentId}`);
      if (this.seen.size > 256) this.seen.clear();
    }
    if (!this.activeUtterance || uid !== this.activeUtterance) {
      this.reset();
      this.activeUtterance = uid;
    }
    if (this.queue.length >= 32) {
      this.failures += 1;
      this._trace(job, 'tts_queue_drop', { chars: (job.text || '').length });
      return;
    }
    this.queue.push(job);
    this._drain();
  },

  async _drain() {
    if (this.running) return;
    this.running = true;
    const gen = this.gen;
    const ctx0 = this.ensureContext();
    if (ctx0 && ctx0.state === 'suspended') {
      try { await ctx0.resume(); } catch (e) { /* ignore */ }
    }
    try {
      while (this.queue.length && gen === this.gen) {
        const job = this.queue.shift();
        if (job.deadlineMs && Date.now() - (job.receivedAt || Date.now()) > job.deadlineMs) continue;
        const ctrl = new AbortController();
        this.ctrl = ctrl;
        let t = null;
        for (let attempt = 0; attempt < 2 && !t; attempt += 1) {
          try {
            t = await synthTts(job.text, 'piper', job.speed, ctrl.signal, job.traceId || '');
          } catch (e) {
            if (e && e.name === 'AbortError') break;
            if (attempt === 1) {
              this._trace(job, 'tts_segment_failed', { chars: (job.text || '').length });
            }
          }
        }
        if (!t || !t.url) {
          if (!t && gen === this.gen) this.failures += 1;
          continue;
        }
        if (gen !== this.gen) {
          URL.revokeObjectURL(t.url);
          break;
        }
        let ab = null;
        try {
          ab = await fetch(t.url).then((r) => r.arrayBuffer());
        } catch (e) {
          ab = null;
        } finally {
          URL.revokeObjectURL(t.url);
        }
        if (!ab) { this.failures += 1; continue; }
        const ctx = this.ensureContext();
        if (!ctx) { this.failures += 1; continue; }
        let buf = null;
        try {
          buf = await ctx.decodeAudioData(ab.slice(0));
        } catch (e) {
          this.failures += 1;
          continue;
        }
        if (gen !== this.gen) break;
        this._schedule(buf, job, gen);
      }
    } finally {
      this.running = false;
      if (gen === this.gen) this._maybeEnd(gen);
    }
  },

  _schedule(buf, job, gen) {
    const ctx = this.ctx;
    const now = ctx.currentTime;
    const at = Math.max(now + 0.03, this.playhead || 0);
    if (this.started && at > now + 0.12) {
      this.underruns += 1;
      this._trace(job, 'underrun', { gap_ms: Math.round((at - now) * 1000) });
    }
    if (!this.started) {
      this.started = true;
      document.dispatchEvent(new CustomEvent('cm5-tts-start', {
        detail: { text: job.answer || '', speed: job.speed || null, utterance_id: job.utteranceId || '' },
      }));
    }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.onended = () => { this.sources.delete(src); };
    this.sources.add(src);
    src.start(at);
    this.playhead = at + buf.duration;
    this.spokenText = ((this.spokenText || '') + ' ' + (job.text || '')).slice(-600);
    document.dispatchEvent(new CustomEvent('cm5-tts-text', { detail: { text: this.spokenText } }));
    this._trace(job, 'play_start', {
      play_at_ms: Date.now() + Math.max(0, Math.round((at - now) * 1000)),
      audio_s: Math.round(buf.duration * 1000) / 1000,
      sequence: job.sequence || 0,
    });
  },

  _maybeEnd(gen) {
    if (!this.ctx || !this.started) return;
    const wait = Math.max(0, (this.playhead - this.ctx.currentTime) * 1000);
    setTimeout(() => {
      if (gen !== this.gen) return;
      if (this.queue.length || this.running) return;
      if (this.ctx.currentTime + 0.02 < this.playhead) {
        this._maybeEnd(gen);
        return;
      }
      if (this.started) {
        this.started = false;
        this.activeUtterance = null;
        document.dispatchEvent(new CustomEvent('cm5-tts-end'));
      }
    }, wait + 40);
  },

  _trace(job, stage, extra) {
    try {
      fetch('/api/trace/event', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          trace_id: job.traceId || job.utteranceId || '',
          utterance_id: job.utteranceId || '',
          segment_id: job.segmentId || '',
          stage,
          at_ms: Date.now(),
          ...extra,
        }),
      }).catch(() => {});
    } catch (e) { /* ignore */ }
  },
};
