const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.resolve(__dirname, 'static/guide-mode.js'), 'utf8');
function fixture(ready=true) {
  const nodes = new Map(), calls=[], timers=new Map(); let nextTimer=0;
  const classes = initial => { const set=new Set((initial||'').split(' ').filter(Boolean)); return { add:x=>set.add(x), remove:x=>set.delete(x), contains:x=>set.has(x), toggle:(x,v)=>v?set.add(x):set.delete(x)}; };
  function node(id='') {
    const result={id, dataset:{}, style:{}, disabled:false, textContent:'', _className:'', classList:classes(), listeners:{},
      appendChild(child){if(child.id)nodes.set(child.id,child);}, addEventListener(type,cb){this.listeners[type]=cb;}};
    Object.defineProperty(result,'className',{get(){return this._className;},set(v){this._className=v;this.classList=classes(v);}});
    Object.defineProperty(result,'innerHTML',{set(html){for(const match of html.matchAll(/id="([^"]+)"/g))nodes.set(match[1],node(match[1]));}});
    if(id)nodes.set(id,result); return result;
  }
  const tabs=node('tabs'), card=node('card'), eyes=node('tab-eyes'), eyesPanel=node('panel-eyes');
  eyes.className='tab active'; eyesPanel.className='panel active';
  const document={hidden:false,listeners:{},createElement:()=>node(),getElementById:id=>nodes.get(id),
    querySelector:selector=>selector==='.right .tabs'?tabs:selector==='.right > .card'?card:null,
    querySelectorAll:selector=>[...nodes.values()].filter(n=>n.classList.contains(selector.slice(1))),
    addEventListener(type,cb){this.listeners[type]=cb;}};
  let status={running:false,mode:'device',model:{ready,engine:'segformer-b0-ade'},decision:{state:'unavailable',reason:'stopped'},stats:{processed:0}};
  let failing=false, held=null, release=null;
  const context={window:{},document,AbortController,console,setTimeout:(fn,delay)=>{const id=++nextTimer;timers.set(id,{fn,delay});return id;},clearTimeout:id=>timers.delete(id),
    fetch:async(url,options)=>{calls.push({url,options});if(failing)throw new Error('offline');if(held){await new Promise(r=>{release=r;});}
      if(options.method==='POST')status={...status,running:url.endsWith('/start'),decision:{state:url.endsWith('/start')?'straight':'unavailable',reason:url.endsWith('/start')?'clear':'stopped'}};
      return{ok:true,json:async()=>status};}};
  vm.runInNewContext(source,context);
  return{nodes,calls,timers,document,tabs,eyes,eyesPanel,ui:context.window.CM5GuideUI,setFailure:v=>failing=v,setHold:v=>held=v,release:()=>release&&release()};
}
const settle=async()=>{for(let i=0;i<20;i++)await Promise.resolve();};
(async()=>{
  const f=fixture();await settle();
  assert.equal(f.nodes.get('btn-guide-start').disabled,false);
  f.nodes.get('tab-guide').onclick();await settle();
  assert.equal([...f.timers.values()].filter(x=>x.delay===1000).length,1);
  await f.ui.start();
  const start=f.calls.find(c=>c.url==='/api/guide/start');
  assert.deepEqual(JSON.parse(start.options.body),{source:'device'});
  assert.match(f.nodes.get('out-guide').textContent,/直行/);
  assert.equal(f.nodes.get('btn-guide-stop').disabled,false);
  assert.equal(f.nodes.get('btn-guide-start').disabled,true);
  f.document.hidden=true;f.document.listeners.visibilitychange();
  assert.equal([...f.timers.values()].filter(x=>x.delay===1000).length,0);
  assert.equal(f.calls.filter(c=>c.url==='/api/guide/stop').length,0);
  f.document.hidden=false;f.nodes.get('tab-guide').classList.remove('active');f.eyes.classList.add('active');f.tabs.listeners.click();
  assert.equal(f.calls.filter(c=>c.url==='/api/guide/stop').length,0);
  await f.ui.stop();assert.equal(f.calls.filter(c=>c.url==='/api/guide/stop').length,1);
  assert.match(f.nodes.get('out-guide').textContent,/已停止/);
  assert.equal(f.calls.some(c=>c.url==='/api/guide/frame'||c.url==='/api/device/camera'),false);
  const m=fixture(false);await settle();assert.equal(m.nodes.get('btn-guide-start').disabled,true);await m.ui.start();
  assert.equal(m.calls.some(c=>c.options.method==='POST'),false);assert.match(m.nodes.get('out-guide').textContent,/模型不可用/);
  f.setFailure(true);await f.ui.refresh();assert.match(f.nodes.get('out-guide').textContent,/状态未知/);assert.doesNotMatch(f.nodes.get('out-guide').textContent,/直行/);
  const h=fixture();await settle();h.setHold(true);const count=h.calls.length;const p=h.ui.refresh();const q=h.ui.refresh();
  assert.equal(h.calls.length,count+1);assert.equal(p,q);h.release();await p;
  console.log('PASS: device-only start/stop, bounded polling, hidden/tab switch keeps device guide, missing model blocks start, failed status clears old direction, single in-flight request');
})().catch(error=>{console.error(error);process.exitCode=1;});
