// 摄像头识别的小后端：静态页 + 把浏览器拍的图转发给本地 llama-server。
// 浏览器 → 本进程（同源，无 CORS 问题）→ http://127.0.0.1:<PORT>/v1/chat/completions
import http from 'node:http'
import { exec } from 'node:child_process'
import { readFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import net from 'node:net'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = dirname(HERE)
const KEY_FILE = join(ROOT, 'api-key.txt')

const PORT = Number(process.env.VL_PORT || 8091)        // 模型服务端口
const LISTEN = Number(process.env.CAM_PORT || 8093)     // 这个页面自己的端口
const UPSTREAM = `http://127.0.0.1:${PORT}`
const ALIAS = 'qwen3-vl-2b-q4km'

function apiKey() {
  try { if (existsSync(KEY_FILE)) return readFileSync(KEY_FILE, 'utf8').trim() } catch { }
  return ''
}

function json(res, code, obj) {
  const body = Buffer.from(JSON.stringify(obj), 'utf8')
  res.writeHead(code, { 'content-type': 'application/json; charset=utf-8', 'content-length': body.length, 'cache-control': 'no-store' })
  res.end(body)
}

function readBody(req, limit = 32 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    const chunks = []
    let size = 0
    req.on('data', (c) => {
      size += c.length
      if (size > limit) { reject(new Error('图片太大')); req.destroy(); return }
      chunks.push(c)
    })
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', reject)
  })
}

const PAGES = {
  '/': 'camera.html',
  '/index.html': 'camera.html',
  '/camera.html': 'camera.html',
  '/blind.html': 'blind.html', '/phone.html': 'phone.html'
}
const pageCache = new Map()
function pageFor(name) {
  if (!pageCache.has(name)) pageCache.set(name, readFileSync(join(HERE, name)))
  return pageCache.get(name)
}
const START_PAGE = process.env.CAM_PAGE || 'camera.html'

// ---------- 实时画面：常驻拉流，缓存最新一帧 ----------
const JPEG_SOI = Buffer.from([0xff, 0xd8])
const JPEG_EOI = Buffer.from([0xff, 0xd9])
const streams = new Map()

function sleep(ms) { return new Promise(r => setTimeout(r, ms)) }

function normTarget(raw, user, pass) {
  let target = String(raw || '')
  let basic = ''
  try {
    const u = new URL(target)
    if (u.username) {
      basic = Buffer.from(`${decodeURIComponent(u.username)}:${decodeURIComponent(u.password)}`).toString('base64')
      u.username = ''; u.password = ''
      target = u.toString()
    }
  } catch { }
  if (!basic && user) basic = Buffer.from(`${user}:${pass}`).toString('base64')
  return { target, basic }
}

function makeHeaders(basic) {
  const h = { 'user-agent': 'Mozilla/5.0 (camera-proxy)', accept: 'multipart/x-mixed-replace,image/*,video/*,*/*' }
  if (basic) h.authorization = 'Basic ' + basic
  return h
}

function startStream(target, headers, key) {
  let s = streams.get(key)
  if (s) return s
  s = { latest: null, at: 0, err: '', count: 0, lastUse: Date.now(), abort: new AbortController() }
  streams.set(key, s)
  ;(async () => {
    while (!s.abort.signal.aborted) {
      try {
        const r = await fetch(target, { headers, signal: s.abort.signal, redirect: 'follow' })
        const ct = String(r.headers.get('content-type') || '')
        if (!r.ok) {
          if (r.status === 401) s.err = '手机要求登录（401）'
          else s.err = '手机返回 HTTP ' + r.status
          try { await r.body?.cancel() } catch { }
          await sleep(1500); continue
        }
        if (!ct.includes('multipart') && !ct.includes('video')) {
          // 静态快照类：每 500ms 取一张
          const b = Buffer.from(await r.arrayBuffer())
          if (b.slice(0, 2).toString('hex') === 'ffd8') { s.latest = b; s.at = Date.now(); s.count++; s.err = '' }
          else s.err = '这个地址返回的不是画面'
          await sleep(500); continue
        }
        // MJPEG：持续读流，切出每一帧
        const rd = r.body.getReader()
        let buf = Buffer.alloc(0)
        for (;;) {
          const { value, done } = await rd.read()
          if (done) break
          buf = Buffer.concat([buf, Buffer.from(value)])
          for (;;) {
            const a = buf.indexOf(JPEG_SOI)
            if (a < 0) { buf = Buffer.alloc(0); break }
            const b2 = buf.indexOf(JPEG_EOI, a + 2)
            if (b2 < 0) { buf = buf.subarray(a); break }
            s.latest = buf.subarray(a, b2 + 2); s.at = Date.now(); s.count++; s.err = ''
            if (Date.now() - s.lastUse > 30000) { s.abort.abort(); streams.delete(key); break }   // 空闲 30 秒自动回收
            buf = buf.subarray(b2 + 2)
          }
          if (buf.length > 4 * 1024 * 1024) buf = Buffer.alloc(0)
        }
      } catch (e) {
        if (s.abort.signal.aborted) break
        s.err = String(e.message || e)
        await sleep(1200)
      }
    }
  })()
  return s
}

// 从 MJPEG 流里切出第一张完整的 JPEG（SOI 0xFFD8 … EOI 0xFFD9）
async function firstJpeg(r, limitMs) {
  if (!r.body) return null
  const reader = r.body.getReader()
  const SOI = Buffer.from([0xff, 0xd8])
  const EOI = Buffer.from([0xff, 0xd9])
  let buf = Buffer.alloc(0)
  const t0 = Date.now()
  try {
    while (Date.now() - t0 < limitMs) {
      const { value, done } = await reader.read()
      if (done) break
      buf = buf.length ? Buffer.concat([buf, Buffer.from(value)]) : Buffer.from(value)
      const s = buf.indexOf(SOI)
      if (s >= 0) {
        const e = buf.indexOf(EOI, s + 2)
        if (e > s) return buf.subarray(s, e + 2)
      }
      if (buf.length > 16 * 1024 * 1024) break
    }
  } catch { }
  finally { try { await reader.cancel() } catch { } }
  return null
}
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://127.0.0.1')

  if (req.method === 'GET' && PAGES[url.pathname]) {
    const page = pageFor(PAGES[url.pathname])
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'content-length': page.length, 'cache-control': 'no-store' })
    return res.end(page)
  }

  // 诊断：分层告诉用户卡在哪一步（后端 / 手机端口 / 认证 / 取帧）
  if (req.method === 'GET' && url.pathname === '/diag') {
    const target0 = String(url.searchParams.get('url') || '')
    const user0 = String(url.searchParams.get('user') || '')
    const pass0 = String(url.searchParams.get('pass') || '')
    const out = { ok: false, target: target0, steps: [] }
    let target = target0
    let basic = ''
    try {
      const u = new URL(target0)
      if (u.username) {
        basic = Buffer.from(`${decodeURIComponent(u.username)}:${decodeURIComponent(u.password)}`).toString('base64')
        u.username = ''; u.password = ''
        target = u.toString()
      }
    } catch { out.steps.push({ name: '地址格式', ok: false, detail: '不是合法 URL：' + target0 }); return json(res, 200, out) }
    if (!basic && user0) basic = Buffer.from(`${user0}:${pass0}`).toString('base64')
    out.steps.push({ name: '模型服务', ok: true, detail: '本页后端在跑，模型地址 ' + UPSTREAM })

    // 1) TCP 能不能连上手机端口
    const u2 = new URL(target)
    const port = Number(u2.port || (u2.protocol === 'https:' ? 443 : 80))
    const t0 = Date.now()
    const tcpOk = await new Promise(resolve => {
      const s = net.connect({ host: u2.hostname, port })
      const done = v => { try { s.destroy() } catch { } resolve(v) }
      s.setTimeout(3000)
      s.once('connect', () => done(true))
      s.once('timeout', () => done(false))
      s.once('error', e => { out.steps.push({ name: '手机端口', ok: false, detail: e.code || e.message }); done(false) })
    })
    if (tcpOk) out.steps.push({ name: '手机端口', ok: true, detail: `${u2.hostname}:${port} 通（${Date.now() - t0} ms）` })
    else { out.steps.push({ name: '手机端口', ok: false, detail: `${u2.hostname}:${port} 连不上 —— 手机锁屏了？App 停了？IP 变了？` }); return json(res, 200, out) }

    // 2) 带认证请求一次，看状态码
    try {
      const headers = { 'user-agent': 'Mozilla/5.0 (camera-proxy)', accept: 'image/*,multipart/x-mixed-replace,video/*,*/*' }
      if (basic) headers.authorization = 'Basic ' + basic
      const r = await fetch(target, { headers, signal: AbortSignal.timeout(8000) })
      const ct = r.headers.get('content-type') || ''
      if (r.status === 401) { out.steps.push({ name: '认证', ok: false, detail: '手机要求登录（401）：用户名密码不对。这类 App 常见 admin / admin' }); try { await r.body?.cancel() } catch { } return json(res, 200, out) }
      if (!r.ok) { out.steps.push({ name: '认证', ok: false, detail: '手机返回 HTTP ' + r.status }); try { await r.body?.cancel() } catch { } return json(res, 200, out) }
      out.steps.push({ name: '认证', ok: true, detail: '通过（HTTP 200）' })
      // 3) 取一帧（只填了首页地址时，自动试常见画面路径）
      const t1 = Date.now()
      let bytes = 0
      let r2 = r
      let ct2 = ct
      let usedPath = ''
      if (ct.includes('text/html')) {
        try { await r.body?.cancel() } catch { }
        const b0 = new URL(target)
        const cands = ['/video', '/videostream.cgi', '/shot.jpg', '/photo.jpg', '/snapshot.jpg', '/image.jpg']
        let hit = null
        for (const c of cands) {
          const u3 = new URL(b0); u3.pathname = c; u3.search = ''
          try {
            const rr = await fetch(u3.toString(), { headers, signal: AbortSignal.timeout(6000) })
            const c3 = String(rr.headers.get('content-type') || '')
            if (rr.ok && (c3.startsWith('image/') || c3.includes('multipart') || c3.includes('video'))) { hit = { r: rr, ct: c3, path: c }; break }
            try { await rr.body?.cancel() } catch { }
          } catch { }
        }
        if (!hit) { out.steps.push({ name: '取帧', ok: false, detail: '这个地址是网页，不是画面。末尾加 /video 或 /shot.jpg 再试' }); return json(res, 200, out) }
        r2 = hit.r; ct2 = hit.ct; usedPath = hit.path
      }
      if (ct2.startsWith('image/')) {
        const b = Buffer.from(await r2.arrayBuffer()); bytes = b.length
      } else {
        const jp = await firstJpeg(r2, 8000); bytes = jp ? jp.length : 0
      }
      if (bytes > 0) { out.ok = true; out.steps.push({ name: '取帧', ok: true, detail: (usedPath ? '自动用了 ' + usedPath + ' · ' : '') + `${ct2 || 'jpeg'} · ${bytes} B · ${Date.now() - t1} ms` }) }
      else out.steps.push({ name: '取帧', ok: false, detail: '连上了但没取到 JPEG（' + (ct2 || '无 content-type') + '）' })
    } catch (e) {
      out.steps.push({ name: '请求手机', ok: false, detail: String(e.message || e) })
    }
    return json(res, 200, out)
  }
  // 调试：看常驻拉流的状态
  if (req.method === 'GET' && url.pathname === '/hub') {
    const out = []
    for (const [k, v] of streams) out.push({ key: k.replace(/\?[^/]*$/, ''), latest: v.latest ? v.latest.length : 0, at: v.at, count: v.count, err: v.err, ageMs: v.at ? Date.now() - v.at : -1 })
    return json(res, 200, { ok: true, streams: out })
  }

  // 手机实时画面：从常驻拉流的缓存帧重新推 multipart（手机只允许一条 MJPEG 连接，所以不能自己再开一条）
  if (req.method === 'GET' && url.pathname === '/stream') {
    const { target, basic } = normTarget(url.searchParams.get('url'), String(url.searchParams.get('user') || ''), String(url.searchParams.get('pass') || ''))
    if (!/^https?:\/\//i.test(target)) return json(res, 400, { ok: false, error: '地址要以 http:// 或 https:// 开头' })
    const B = '--BoundaryString'
    res.writeHead(200, { 'content-type': 'multipart/x-mixed-replace; boundary=' + B, 'cache-control': 'no-store', 'x-accel-buffering': 'no' })
    try { res.flushHeaders() } catch { }
    let closed = false
    req.on('close', () => { closed = true })
    let lastAt = 0
    let sent = 0
    while (!closed) {
      const s = startStream(target, makeHeaders(basic), target + '|' + (basic ? '1' : '0'))
      s.lastUse = Date.now()
      if (s.latest && s.at !== lastAt && Date.now() - s.at < 5000) {
        lastAt = s.at
        const head = Buffer.from(B + '\r\nContent-Type: image/jpeg\r\nContent-Length: ' + s.latest.length + '\r\n\r\n')
        try { res.write(head); res.write(s.latest); res.write('\r\n') } catch { break }
        sent++
      } else if (s.err && sent === 0) {
        // 还没出过画面且上游报错：直接说清楚
        try { res.writeHead(200, { 'content-type': 'text/plain; charset=utf-8' }); res.end('取不到手机画面：' + s.err) } catch { }
        return
      }
      await sleep(40)
    }
    try { res.end() } catch { }
    return
  }

  // 手机摄像头：把手机 App 的画面代理回来（同源，页面不碰 CORS）
  // 支持三种：单张 JPEG、MJPEG 实时流（从流里切一帧）、Basic 认证（?user=&pass=）
  if (req.method === 'GET' && url.pathname === '/frame') {
    let target = String(url.searchParams.get('url') || '')
    const user = String(url.searchParams.get('user') || '')
    const pass = String(url.searchParams.get('pass') || '')
    if (!/^https?:\/\//i.test(target)) return json(res, 400, { ok: false, error: '地址要以 http:// 或 https:// 开头' })

    // 允许把账号密码直接写在地址里：http://admin:admin@10.0.0.2:8081/video
    let basic = ''
    try {
      const u = new URL(target)
      if (u.username) {
        basic = Buffer.from(`${decodeURIComponent(u.username)}:${decodeURIComponent(u.password)}`).toString('base64')
        u.username = ''; u.password = ''
        target = u.toString()
      }
    } catch { }
    if (!basic && user) basic = Buffer.from(`${user}:${pass}`).toString('base64')

    // 有常驻流时等它的最新帧（手机 App 只允许一条 MJPEG 连接，不能自己再连一次）
    {
      const key = target + '|' + (basic ? '1' : '0')
      const live = streams.get(key)
      if (live) {
        live.lastUse = Date.now()
        const deadline = Date.now() + 5000
        while (Date.now() < deadline) {
          if (live.latest && Date.now() - live.at < 2500) {
            res.writeHead(200, { 'content-type': 'image/jpeg', 'content-length': live.latest.length, 'cache-control': 'no-store', 'x-frame-cache': '1' })
            return res.end(live.latest)
          }
          await sleep(40)
        }
        return json(res, 502, { ok: false, error: '手机那边还没出画面' + (live.err ? '（' + live.err + '）' : '') })
      }
    }

    const headers = { 'user-agent': 'Mozilla/5.0 (camera-proxy)', accept: 'image/*,multipart/x-mixed-replace,video/*,*/*' }
    if (basic) headers.authorization = 'Basic ' + basic

    try {
      let r = await fetch(target, { headers, signal: AbortSignal.timeout(12000), redirect: 'follow' })
      if (r.status === 401) return json(res, 401, { ok: false, error: '手机要求登录：填上用户名和密码（这类 App 常见默认是 admin / admin）' })
      if (!r.ok) return json(res, 502, { ok: false, error: `手机那边返回 ${r.status}` })

      let ctype = String(r.headers.get('content-type') || '')

      // 只填了 App 首页（http://ip:port 或 …/）时，自动试几个常见的画面路径
      if (ctype.includes('text/html')) {
        try { await r.body?.cancel() } catch { }
        let base = null
        try { base = new URL(target) } catch { }
        let found = null
        if (base && (base.pathname === '/' || base.pathname === '')) {
          const cands = ['/video', '/videostream.cgi', '/shot.jpg', '/photo.jpg', '/snapshot.jpg', '/image.jpg']
          for (const c of cands) {
            const u2 = new URL(base); u2.pathname = c; u2.search = ''
            try {
              const rr = await fetch(u2.toString(), { headers, signal: AbortSignal.timeout(6000), redirect: 'follow' })
              const ct2 = String(rr.headers.get('content-type') || '')
              if (rr.ok && (ct2.startsWith('image/') || ct2.includes('multipart') || ct2.includes('video'))) { found = { r: rr, ct: ct2, path: c }; break }
              try { await rr.body?.cancel() } catch { }
            } catch { }
          }
        }
        if (!found) {
          return json(res, 502, { ok: false, error: '这个地址返回的是网页，不是画面。在末尾加上 /video（实时流）或 /shot.jpg（单张快照）再试' })
        }
        r = found.r; ctype = found.ct
        res.setHeader('x-frame-path', found.path)
      }
      let jpeg = null
      if (ctype.includes('multipart/x-mixed-replace') || ctype.includes('video')) {
        jpeg = await firstJpeg(r, 10000)          // MJPEG：从流里切出第一帧完整 JPEG
        if (!jpeg) return json(res, 502, { ok: false, error: '从手机的视频流里没切出画面，试试把地址换成 /shot.jpg 这类快照地址' })
      } else {
        const buf = Buffer.from(await r.arrayBuffer())
        if (buf.length > 8 * 1024 * 1024) return json(res, 502, { ok: false, error: '这一帧太大' })
        if (buf.slice(0, 2).toString('hex') !== 'ffd8') return json(res, 502, { ok: false, error: '这个地址返回的不是 JPEG' })
        jpeg = buf
      }
      res.writeHead(200, { 'content-type': 'image/jpeg', 'content-length': jpeg.length, 'cache-control': 'no-store' })
      return res.end(jpeg)
    } catch (e) {
      return json(res, 502, { ok: false, error: '连不上手机：' + e.message })
    }
  }
  if (req.method === 'GET' && url.pathname === '/status') {
    try {
      const r = await fetch(`${UPSTREAM}/health`, { signal: AbortSignal.timeout(1500) })
      const j = await r.json().catch(() => ({}))
      return json(res, 200, { ok: String(j.status || '').toLowerCase() === 'ok', upstream: UPSTREAM })
    } catch {
      return json(res, 200, { ok: false, upstream: UPSTREAM, error: '模型服务没在跑' })
    }
  }

  if (req.method === 'POST' && url.pathname === '/ask') {
    let payload
    try { payload = JSON.parse(await readBody(req)) } catch (e) { return json(res, 400, { ok: false, error: '请求体不合法：' + e.message }) }
    const image = String(payload.image || '')
    if (!image.startsWith('data:image/')) return json(res, 400, { ok: false, error: '没有图片数据' })
    const prompt = String(payload.prompt || '用中文简洁描述这张图里有什么。')
    const temperature = Number.isFinite(payload.temperature) ? Math.max(0, Math.min(2, payload.temperature)) : 0.2
    const maxTokens = Number.isFinite(payload.max_tokens) ? Math.max(16, Math.min(2048, payload.max_tokens)) : 512
    const key = apiKey()
    try {
      const r = await fetch(`${UPSTREAM}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', ...(key ? { authorization: `Bearer ${key}` } : {}) },
        body: JSON.stringify({
          model: ALIAS,
          max_tokens: maxTokens,
          temperature,
          messages: [{
            role: 'user',
            content: [
              { type: 'text', text: prompt },
              { type: 'image_url', image_url: { url: image } }
            ]
          }]
        }),
        signal: AbortSignal.timeout(180000)
      })
      const j = await r.json()
      if (!r.ok) return json(res, 200, { ok: false, error: `上游 ${r.status}：${JSON.stringify(j).slice(0, 300)}` })
      const text = j.choices?.[0]?.message?.content || ''
      try {
        const dir = join(ROOT, 'shots')
        mkdirSync(dir, { recursive: true })
        const name = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19) + '.jpg'
        writeFileSync(join(dir, name), Buffer.from(image.split(',')[1], 'base64'))
      } catch { /* 存图失败不影响识别 */ }
      return json(res, 200, { ok: true, text, tokens: j.usage?.completion_tokens ?? 0 })
    } catch (e) {
      const refused = /ECONNREFUSED|fetch failed|terminated/i.test(String(e.message))
      return json(res, 200, {
        ok: false,
        error: refused
          ? `模型服务没在跑（${UPSTREAM}）。先双击包里的「一键部署.bat」或「启动.bat」，等它说「服务已就绪」再来。`
          : `请求失败：${e.message}`
      })
    }
  }

  json(res, 404, { ok: false, error: 'not found' })
})

server.on('error', (e) => {
  if (e && e.code === 'EADDRINUSE') {
    console.error(`端口 ${LISTEN} 已经被占用了 —— 多半是已经开着一个识别页面。`)
    console.error(`直接在浏览器打开 http://127.0.0.1:${LISTEN}/ 就行。`)
  } else {
    console.error('启动失败：' + (e && e.message ? e.message : String(e)))
  }
  process.exitCode = 1
})

server.listen(LISTEN, '127.0.0.1', () => {
  const url = `http://127.0.0.1:${LISTEN}/${START_PAGE === 'camera.html' ? '' : START_PAGE}`
  console.log(`${START_PAGE === 'blind.html' ? '盲人导航识别' : '摄像头识别'}已启动：${url}`)
  console.log(`  模型服务：${UPSTREAM}   模型名：${ALIAS}`)
  console.log('  关掉这个窗口即停止。')
  if (!process.env.NO_BROWSER) exec(`start "" "${url}"`, () => { })
})
