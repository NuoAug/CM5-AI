// 不启浏览器设备；验证测试标记、人工评分与服务端记录的边界。
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const code = fs.readFileSync(path.join(__dirname,'static/test-session.js'),'utf8');
const elements = new Map(); const calls = []; const downloads = []; const intervals = [];
function element(id) {
  if (!elements.has(id)) elements.set(id,{value:'',disabled:false,textContent:'',classList:{remove(){}},
    click(){downloads.push(this.href);},remove(){}});
  return elements.get(id);
}
let active = null, offline = false;
const context = {window:{},document:{hidden:false,body:{appendChild(){}},createElement:()=>element('link')},
  $:element,AbortController,console,toast(){},setTimeout:()=>1,clearTimeout(){},setInterval:f=>intervals.push(f),
  fetch:async(url,options)=>{
    const body = options.body && JSON.parse(options.body); calls.push({url,body});
    if (offline) throw new Error('network failed');
    if (url.endsWith('/start')) active={test_session_id:'batch/one',label:body.label,active:true,current_case_id:null,counts:{cases:0,reviews:0}};
    if (url.endsWith('/case')) {active.current_case_id=body.case_id;active.counts.cases++;}
    if (url.endsWith('/review')) active.counts.reviews++;
    if (url.endsWith('/stop')) active.active=false;
    return {ok:true,json:async()=>({ok:true,session:active,summary:{reviewed_cases:active ? active.counts.reviews : 0}})};
  }};
vm.createContext(context);vm.runInContext(code,context);
async function settle(){for(let i=0;i<30;i++)await Promise.resolve();}
(async()=>{
  await settle();
  assert.equal(element('test-session-start').disabled,false);
  assert.equal(element('test-review-save').disabled,true);
  element('test-session-label').value='安静测试';await element('test-session-start').onclick();
  element('test-case-id').value='A01';element('test-case-category').value='speech';
  element('test-case-prompt').value='小冰，今天心情不好';element('test-case-expected').value='保留原句';
  await element('test-case-start').onclick();
  assert.equal(active.current_case_id,'A01');
  assert.equal(element('test-review-save').disabled,false);
  element('test-recognition').value='false';element('test-image').value='';element('test-fluency').value='3';
  element('test-stability').value='';element('test-latency').value='';element('test-outcome').value='fail';
  element('test-spoken-reference').value='实际说的是另一句';element('test-review-notes').value='识别错字';
  await element('test-review-save').onclick();
  const review=calls.find(c=>c.url.endsWith('/review')).body;
  assert.equal(review.recognition_correct,false);assert.equal(review.image_correct,null);assert.equal(review.fluency,3);
  assert.equal(review.stability,null);assert.equal(review.spoken_reference,'实际说的是另一句');
  assert.equal(review.case_id,'A01');assert.equal(review.outcome,'fail');
  assert.ok(calls.every(c=>!['/api/ask','/api/voice','/api/talk'].includes(c.url)),'record controls must not submit model tasks');
  await element('test-session-stop').onclick();
  assert.equal(element('test-case-start').disabled,true);assert.equal(element('test-session-export').disabled,false);
  element('test-session-export').onclick();assert.equal(downloads[0],'/api/test/session/export?test_session_id=batch%2Fone');
  active.logging_limited=true;active.frame_logging_limited=true;
  await element('test-session-refresh').onclick();
  assert.ok(element('test-session-status').textContent.includes('记录受限'));
  assert.ok(element('test-session-status').textContent.includes('原图证据缓存'));
  offline=true;await element('test-session-refresh').onclick();
  assert.ok(element('test-session-status').textContent.includes('未确认'));
  assert.equal(element('test-session-start').disabled,true);
  offline=false;await element('test-session-refresh').onclick();assert.equal(element('test-session-start').disabled,false);
  console.log('PASS: test batch/case/review/export, unknown values, no auto inference and reconnect');
})().catch(error=>{console.error(error);process.exitCode=1;});
