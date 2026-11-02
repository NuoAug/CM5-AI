/* v1.0.0 服务端引导的远程测试页：只读状态，不在浏览器采集或发送引导帧。 */
(() => {
  'use strict';
  const tabs = document.querySelector('.right .tabs');
  const card = document.querySelector('.right > .card');
  if (!tabs || !card || document.getElementById('tab-guide')) return;

  const tab = document.createElement('button');
  tab.type = 'button';
  tab.className = 'tab';
  tab.dataset.tab = 'guide';
  tab.id = 'tab-guide';
  tab.textContent = '无盲道引导';
  tabs.appendChild(tab);
  const panel = document.createElement('div');
  panel.id = 'panel-guide';
  panel.className = 'panel';
  panel.innerHTML = '<h2>无盲道引导（试验）</h2>' +
    '<p class="hint">使用树莓派摄像头持续判断，提示从树莓派扬声器播报。切换页签或关闭网页不会停止设备引导，结束时请点“停止引导”。</p>' +
    '<div class="row"><button class="primary" id="btn-guide-start" disabled>开始树莓派引导</button>' +
    '<button id="btn-guide-stop" disabled>停止引导</button><button id="btn-guide-refresh">刷新状态</button></div>' +
    '<div class="result empty" id="out-guide" role="status" aria-live="polite" style="font-size:24px">正在查询引导状态…</div>' +
    '<div class="meta" id="guide-model"></div><div class="meta" id="meta-guide"></div>' +
    '<div class="meta" id="guide-timing"></div><div class="meta" id="guide-errors"></div>' +
    '<p class="hint" style="margin-top:14px">单目图像实验；宽度为像素，未标定真实通行空间。不用于过马路或上下楼梯。</p>';
  card.appendChild(panel);
  const el = id => document.getElementById(id);
  const startButton = el('btn-guide-start');
  const stopButton = el('btn-guide-stop');
  const result = el('out-guide');
  const labels = {straight:'直行',left:'向左一点',right:'向右一点',stop:'停下',unavailable:'暂不可用'};
  const reasons = {
    clear:'通道候选连续',obstacle:'前方有物体',narrow:'通道偏窄',road:'道路',stairs:'台阶',person:'行人',
    blocked:'近处受阻',ahead:'前方无法确认',uncertain:'判断不确定',no_seed:'脚下地面不可靠',
    low_quality:'画面质量不足',tilted:'相机姿态异常',stale:'画面过期',model_stuck:'模型无响应',
    no_frame:'等待新画面',stopped:'已停止',manual:'手动停止',depth_obstacle:'近处障碍',
    model_error:'模型错误',error:'处理错误',hold:'停止保持中',recovering:'恢复确认中'
  };
  const state = {status:null, timer:null, inFlight:null, operation:false, revision:0};
  const number = (value, suffix='') => typeof value === 'number' && Number.isFinite(value) ? value + suffix : '--';
  const reason = value => reasons[value] || value || '--';
  const active = () => tab.classList.contains('active') && !document.hidden;

  async function request(path, body) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      const options = {cache:'no-store', signal:controller.signal};
      if (body !== undefined) {
        options.method = 'POST';
        options.headers = {'Content-Type':'application/json'};
        options.body = JSON.stringify(body);
      }
      const response = await fetch(path, options);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'HTTP ' + response.status);
      return data;
    } finally { clearTimeout(timeout); }
  }
  function buttons() {
    startButton.disabled = state.operation || !state.status || !!state.status.running || !(state.status.model || {}).ready;
    stopButton.disabled = state.operation || !state.status || !state.status.running;
  }
  function render(status) {
    state.status = status;
    const model = status.model || {};
    const decision = status.decision || {};
    const stats = status.stats || {};
    const metrics = status.metrics || {};
    const unknown = !status.running || !decision.state;
    result.classList.toggle('empty', !status.running);
    result.style.color = status.running && decision.state === 'stop' ? 'var(--red)' : (unknown || decision.state === 'unavailable' ? 'var(--amber)' : 'var(--text)');
    result.textContent = status.running
      ? (labels[decision.state] || decision.state || '等待新画面') + ' · ' + reason(decision.reason)
      : (model.ready ? '引导已停止' : '引导模型不可用');
    el('guide-model').textContent = '模型 ' + (model.engine || '--') + ' · ' + (model.ready ? (model.loaded ? '已加载' : '就绪，按需加载') : '不可用') +
      ' · 输入 ' + (model.input || '--') + ' · 线程 ' + number(model.threads);
    el('meta-guide').textContent = '设备引导 ' + (status.running ? '运行中' : '未运行') + ' · 模式 ' + (status.mode || '--') +
      ' · 帧来源 ' + (status.frame_source || '--') + ' · 帧 ' + (status.frame_id || '--') +
      ' · 帧龄 ' + number(status.frame_age_ms, ' ms') + ' · 处理 ' + number(stats.processed) + ' 帧 · 丢弃 ' + number(stats.dropped) + ' 帧';
    el('guide-timing').textContent = '推理 ' + number(stats.last_inference_ms, ' ms') + ' · P50 ' + number(stats.inference_p50_ms, ' ms') +
      ' · P95 ' + number(stats.inference_p95_ms, ' ms') + ' · 过期帧 ' + number(stats.stale) + ' · 错误 ' + number(stats.errors) +
      ' · 宽度单位 ' + (metrics.width_unit || 'pixel') + ' · ' + (metrics.calibrated === true ? '已标定' : '未标定');
    const issues = [model.error, stats.last_error, status.stuck ? '看门狗：模型无响应' : null];
    const quality = status.quality || {};
    if (quality.ok === false) issues.push('画面质量：' + ((quality.reasons || []).join('、') || '不足'));
    if (status.running && status.mode !== 'device') issues.push('当前不是设备送帧模式；请停止后从本页重新开始');
    el('guide-errors').textContent = issues.filter(Boolean).join(' · ');
    startButton.title = !model.ready ? model.error || '缺少引导模型，请检查设备部署' : '';
    buttons();
  }
  function failure(error) {
    result.classList.remove('empty');
    result.style.color = 'var(--amber)';
    result.textContent = '引导状态未知 · ' + (error.name === 'AbortError' ? '请求超时' : error.message);
    el('guide-errors').textContent = '未显示旧方向；连接恢复前请确认设备实际状态。';
  }
  function schedule() {
    clearTimeout(state.timer);
    state.timer = null;
    if (active() && !state.operation) state.timer = setTimeout(refresh, 1000);
  }
  function refresh() {
    if (state.inFlight) return state.inFlight;
    const revision = state.revision;
    state.inFlight = request('/api/guide/status')
      .then(status => { if (revision === state.revision) render(status); return status; })
      .catch(error => { if (revision === state.revision) failure(error); return null; })
      .finally(() => { state.inFlight = null; schedule(); });
    return state.inFlight;
  }
  async function command(action) {
    if (state.operation) return;
    if (action === 'start' && (!state.status || !(state.status.model || {}).ready)) return;
    state.operation = true;
    state.revision += 1;
    clearTimeout(state.timer);
    buttons();
    try {
      const status = await request('/api/guide/' + action, action === 'start' ? {source:'device'} : {reason:'remote_testbench'});
      state.revision += 1;
      render(status);
    } catch (error) {
      failure(error);
      state.revision += 1;
    } finally {
      state.operation = false;
      buttons();
      schedule();
    }
  }
  tab.onclick = () => {
    document.querySelectorAll('.tab').forEach(item => item.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(item => item.classList.remove('active'));
    tab.classList.add('active');
    panel.classList.add('active');
    refresh();
  };
  startButton.onclick = () => command('start');
  stopButton.onclick = () => command('stop');
  el('btn-guide-refresh').onclick = refresh;
  tabs.addEventListener('click', schedule);
  document.addEventListener('visibilitychange', () => { if (active()) refresh(); else schedule(); });
  window.CM5GuideUI = {state, refresh, start:() => command('start'), stop:() => command('stop')};
  refresh();
})();
