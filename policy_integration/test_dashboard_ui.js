/* Execute real dashboard script with a small DOM harness. Not a layout test. */
'use strict';
const vm=require('vm'),fs=require('fs'),assert=require('assert');
const html=fs.readFileSync(process.argv[2]||'policy_integration/dashboard.html','utf8');
const code=html.match(/<script>([\s\S]*?)<\/script>/)[1];
function lab(width=736){
  class Element{
    constructor(value=''){this.value=value;this.textContent='';this.dataset={};this.handlers={};this.attrs={};}
    addEventListener(k,f){this.handlers[k]=f;}
    setAttribute(k,v){this.attrs[k]=v;}
    replaceChildren(...children){this.children=children;}
    getBoundingClientRect(){return {width};}
    get selectedOptions(){return [{textContent:this.value}];}
    click(){this.handlers.click();}
    change(v){this.value=v;this.handlers.change?.();}
  }
  const elements={},get=k=>elements[k]||(elements[k]=new Element());
  for(const [k,v]of Object.entries({policy:'fixed',sensing:'camera',fault:'none',rate:'4'}))get('[data-field="'+k+'"]').value=v;
  const moves=['forward','left','stop','right','back'].map(move=>{const b=get('[data-move="'+move+'"]');b.dataset.move=move;b.textContent=move;return b;});
  const root={querySelector:get,querySelectorAll:()=>moves};get('svg').parentElement=new Element();
  const queue=new Map();let id=0,wall=0;
  const document={getElementById:()=>root,createElement:()=>new Element(),createElementNS:()=>new Element(),addEventListener:()=>{},hidden:false};
  const context={document,console,ResizeObserver:class{observe(){}},requestAnimationFrame:f=>{queue.set(++id,f);return id;},cancelAnimationFrame:i=>queue.delete(i),matchMedia:()=>({matches:false})};
  vm.runInNewContext(code,context);
  function animate(frames){for(let i=0;i<frames&&queue.size;i++){const jobs=[...queue.values()];queue.clear();wall+=50;jobs.forEach(f=>f(wall));}}
  return {get,animate,queue};
}
function value(l,key){return l.get('[data-value="'+key+'"]').textContent;}
function action(l,key){l.get('[data-action="'+key+'"]').click();}
function field(l,key,v){l.get('[data-field="'+key+'"]').change(v);}
function position(l){return value(l,'position').split(' / ').map(Number);}
for(const [move,axis,sign]of [['forward',0,1],['back',0,-1],['left',1,1],['right',1,-1]]){
  const l=lab();const initial=position(l);l.get('[data-move="'+move+'"]').click();l.animate(200);
  assert((position(l)[axis]-initial[axis])*sign>.1,move+' moves in correct body direction');
  assert.equal(l.queue.size,0,'manual pulse settles and pauses');
}
{
 const l=lab();assert.equal(l.queue.size,0,'no autoplay');action(l,'run');l.animate(20);action(l,'pause');
 const before=value(l,'position');l.animate(20);assert.equal(value(l,'position'),before,'pause freezes physics');
 action(l,'run');l.animate(2000);assert(value(l,'status').includes('sensors_timed_out')||value(l,'status').includes('persistent_pose_stale'),'two-tag route aborts on pose loss');
 assert.equal(l.queue.size,0);action(l,'run');assert.equal(l.queue.size,0,'terminal run cannot resume');
 action(l,'reset');assert.equal(position(l)[0],0,'reset restores start');
}
for(const p of ['fixed','look']){
 const l=lab();field(l,'policy',p);field(l,'sensing','oracle');action(l,'run');l.animate(2200);
 assert(value(l,'status').includes('complete'),p+' completes under ideal sensing');
}
for(const f of ['delay','hidden','link']){
 const l=lab();field(l,'sensing','oracle');field(l,'fault',f);action(l,'run');l.animate(1000);
 assert(value(l,'state').includes('abort'),f+' aborts');assert.equal(value(l,'rc'),'0 / 0 / 0 / 0');
}
{
 const l=lab();field(l,'sensing','oracle');action(l,'run');l.animate(15);action(l,'kick');l.animate(8);
 assert(l.get('[data-value="events"]').children.some(e=>/brake|aim/.test(e.textContent)),'sharp turn triggers brake or aim');
 action(l,'reset');l.get('[data-move="forward"]').click();l.animate(3);l.get('[data-move="stop"]').click();l.animate(100);assert.equal(l.queue.size,0);
}
for(const w of [320,736,1024]){const l=lab(w);assert(l.get('svg').attrs.viewBox.startsWith('0 0 '+w+' '),'responsive SVG at '+w);}
console.log('Dashboard interactions passed: manual directions, stop, pause/resume, terminal latch, reset, both policies, faults, sharp turn, SVG widths. Layout not browser-verified.');
