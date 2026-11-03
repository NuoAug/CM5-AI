/* 实机测试批次与人工评分。声卡/相机任务仍由设备端和现有交互入口执行。 */
(() => {
  'use strict';
  const ui = window.CM5TestSession = { session: null, busy: false, connected: false };
  let polling = false;
  const ratings = ['test-recognition', 'test-image', 'test-fluency', 'test-stability', 'test-latency'];
  async function api(path, body) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 12000);
    const options = {cache: 'no-store', signal: controller.signal};
    if (body !== undefined) {
      options.method = 'POST'; options.headers = {'Content-Type': 'application/json'};
      options.body = JSON.stringify(body);
    }
    try {
      const response = await fetch(path, options);
      const data = await response.json();
      if (!response.ok || data.ok === false) throw new Error(data.error || data.message || ('HTTP ' + response.status));
      return data;
    } finally { clearTimeout(timeout); }
  }
  function buttons() {
    const active = !!(ui.session && ui.session.active);
    $('test-session-start').disabled = ui.busy || !ui.connected || active;
    $('test-session-stop').disabled = ui.busy || !ui.connected || !active;
    $('test-case-start').disabled = ui.busy || !ui.connected || !active;
    $('test-review-save').disabled = ui.busy || !ui.connected || !active || !ui.session.current_case_id;
    $('test-session-export').disabled = ui.busy || !ui.session;
    $('test-session-refresh').disabled = ui.busy;
  }
  function safe(value, max=1000) { return typeof value === 'string' ? value.slice(0,max) : ''; }
  function render(data) {
    ui.connected = true;
    ui.session = data.session || null;
    const session = ui.session;
    if (!session) {
      $('test-session-status').textContent = '尚未开始测试批次；开始后自动记录树莓派交互与资源样本。';
      $('test-session-summary').textContent = '尚无测试数据';
      buttons(); return;
    }
    const counts = session.counts || {};
    $('test-session-status').textContent = (session.active ? '记录中' : '批次已结束') + ' · ' +
      (session.label || '') + ' · ' + session.test_session_id + ' · 当前题：' + (session.current_case_id || '尚未选择');
    const limits = [];
    if (session.logging_limited) limits.push('日志已达到上限');
    if (session.logging_error) limits.push('日志写入错误：' + safe(String(session.logging_error),240));
    if (session.frame_logging_limited) limits.push('原图证据缓存已达到上限');
    if (limits.length) $('test-session-status').textContent += ' · 记录受限：' + limits.join('；') + '。请结束并导出本批次，核实缺失证据后再开始新批次。';
    if (!$('test-case-id').value && session.current_case_id) $('test-case-id').value = session.current_case_id;
    const lines = ['题目 ' + (counts.cases || 0) + ' · 交互 ' + (counts.interactions || 0) +
      ' · 人工评价 ' + (counts.reviews || 0) + ' · 资源采样 ' + (counts.resource_samples || 0)];
    const interactions = data.recent_interactions || [];
    const latest = interactions.length ? interactions[interactions.length-1] : null;
    if (latest) {
      const request = latest.request || {};
      const response = latest.response || {};
      const transcript = latest.final_transcript || latest.transcript || request.final_transcript || latest.text;
      const answer = latest.answer || response.answer;
      if (transcript) lines.push('最近识别／提问：' + safe(transcript));
      if (answer) lines.push('最近回答：' + safe(answer));
    }
    if (data.summary) lines.push('统计（仅软件计时与已评题；不代表物理首声／字错率）：\n' + JSON.stringify(data.summary,null,2));
    $('test-session-summary').classList.remove('empty');
    $('test-session-summary').textContent = lines.join('\n');
    buttons();
  }
  function failed(error) {
    ui.connected = false;
    $('test-session-status').textContent = '测试记录状态未确认：' + error.message + '。恢复连接后点“刷新记录状态”；不要重复提交同一评价。';
    buttons();
  }
  async function refresh() {
    if (polling || ui.busy) return;
    polling = true;
    try { render(await api('/api/test/session/status')); }
    catch (error) { failed(error); }
    finally { polling = false; }
  }
  async function action(path, body, success) {
    if (ui.busy) return;
    ui.busy = true; buttons();
    try {
      const result = await api(path, body);
      if (Object.prototype.hasOwnProperty.call(result, 'session')) render(result);
      else render(await api('/api/test/session/status'));
      if (success) success(result);
    } catch (error) { failed(error); toast(error.message); }
    finally { ui.busy = false; buttons(); }
  }
  $('test-session-start').onclick = () => action('/api/test/session/start', {
    label: $('test-session-label').value.trim() || 'MiniCPM 实机测试 ' + new Date().toLocaleString(),
    notes: 'Windows仅查看与控制，外设位于树莓派。'
  });
  $('test-session-stop').onclick = () => action('/api/test/session/stop', {notes: $('test-review-notes').value.trim()});
  $('test-case-start').onclick = () => {
    const id = $('test-case-id').value.trim();
    if (!id) return toast('请先填写题号，例如 A01');
    return action('/api/test/case', {case_id:id, category:$('test-case-category').value,
      prompt:$('test-case-prompt').value.trim(), expected:$('test-case-expected').value.trim(), notes:''}, () => {
      for (const id of ratings) $(id).value = '';
      $('test-outcome').value = 'untested'; $('test-review-notes').value = '';
      $('test-observed-answer').value = ''; $('test-spoken-reference').value = '';
      toast('本题已开始；现在使用树莓派语音、摄像头或网页提问');
    });
  };
  function booleanValue(id) { return $(id).value === '' ? null : $(id).value === 'true'; }
  function score(id) { return $(id).value === '' ? null : Number($(id).value); }
  $('test-review-save').onclick = () => {
    if (!ui.session || !ui.session.current_case_id) return toast('请先开始本题');
    return action('/api/test/review', {case_id:ui.session.current_case_id,
      recognition_correct:booleanValue('test-recognition'), image_correct:booleanValue('test-image'),
      fluency:score('test-fluency'), stability:score('test-stability'), latency_rating:score('test-latency'),
      outcome:$('test-outcome').value, spoken_reference:$('test-spoken-reference').value.trim(),
      observed_answer:$('test-observed-answer').value.trim(), notes:$('test-review-notes').value.trim()
    }, () => toast('本题评价已保存；下一题请更换题号并点“开始本题”'));
  };
  $('test-session-export').onclick = () => {
    if (!ui.session) return;
    const link = document.createElement('a');
    link.href = '/api/test/session/export?test_session_id=' + encodeURIComponent(ui.session.test_session_id);
    link.download = ''; document.body.appendChild(link); link.click(); link.remove();
  };
  $('test-session-refresh').onclick = refresh;
  refresh();
  setInterval(() => { if (!document.hidden) refresh(); }, 5000);
})();
