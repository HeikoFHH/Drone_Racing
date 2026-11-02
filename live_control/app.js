'use strict';
const token=document.querySelector('meta[name="bridge-token"]').content;
const session=crypto.randomUUID(),$=id=>document.getElementById(id);
let sequence=0,held=null,busy=false,urgent=null,lastHeartbeat=0,latest=null,lastMessage='';
const directionButtons=[...document.querySelectorAll('[data-direction]')];
function clearHeld(){held=null;directionButtons.forEach(b=>b.setAttribute('aria-pressed','false'));}
function stop(){clearHeld();urgent={action:'stop'};pump();}
async function send(body){
  const response=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json','X-Bridge-Token':token},
    body:JSON.stringify({...body,session,seq:sequence++}),signal:AbortSignal.timeout(700)});
  const result=await response.json();if(!response.ok)throw new Error(result.error||'Control rejected');
}
async function pump(){
  if(busy)return;
  let body=null;
  const focused=!document.hidden&&document.hasFocus();
  if(urgent){body=urgent;urgent=null;}
  else if(!focused)return;
  else if(performance.now()-lastHeartbeat>200){body={action:'heartbeat'};lastHeartbeat=performance.now();}
  else if(held)body={action:'move',direction:held};
  if(!body)return;
  busy=true;
  try{await send(body);$('error').textContent='';}
  catch(error){clearHeld();$('error').textContent=error.message;}
  finally{busy=false;}
}
function press(button){
  if(button.disabled)return;clearHeld();held=button.dataset.direction;
  button.setAttribute('aria-pressed','true');pump();
}
for(const button of directionButtons){
  button.setAttribute('aria-pressed','false');
  button.addEventListener('pointerdown',event=>{if(event.button!==0)return;button.setPointerCapture(event.pointerId);press(button);event.preventDefault();});
  for(const type of ['pointerup','pointercancel','lostpointercapture'])button.addEventListener(type,stop);
  button.addEventListener('keydown',event=>{if([' ','Enter'].includes(event.key)){event.preventDefault();if(!event.repeat)press(button);}});
  button.addEventListener('keyup',event=>{if([' ','Enter'].includes(event.key)){event.preventDefault();stop();}});
  button.addEventListener('blur',()=>{if(held)stop();});
}
$('stop').addEventListener('click',stop);
$('takeoff').addEventListener('click',()=>{clearHeld();urgent={action:'takeoff'};pump();});
$('land').addEventListener('click',()=>{clearHeld();urgent={action:'land'};pump();});
$('run-policy').addEventListener('click',()=>{clearHeld();urgent={action:'run_policy',policy:$('policy').value};pump();});
$('stop-policy').addEventListener('click',()=>{clearHeld();urgent={action:'stop_policy'};pump();});
window.addEventListener('blur',stop);
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
window.addEventListener('pagehide',()=>{
  clearHeld();fetch('/api/control',{method:'POST',keepalive:true,headers:{'Content-Type':'application/json','X-Bridge-Token':token},
    body:JSON.stringify({action:'stop',session,seq:sequence++})}).catch(()=>{});
});
function age(value){return value===null||value===undefined?'—':value.toFixed(2)+' s';}
async function status(){
  try{
    const response=await fetch('/api/status',{signal:AbortSignal.timeout(700)});if(!response.ok)throw new Error('Bridge unavailable');
    latest=await response.json();const real=latest.backend==='real';
    $('backend').textContent=real?'REAL TELLO CONNECTION':'DEMO · NO DRONE CONNECTED';$('backend').classList.toggle('real',real);
    $('flight-mode').textContent=real?(latest.flight_enabled?'Flight enabled · operator takeoff required':'Camera-only · flight disabled'):'Simulated controls and camera';
    $('stage').textContent=latest.stage.replaceAll('_',' ').toUpperCase();$('rc').textContent='RC '+latest.rc.join(' / ');
    $('battery').textContent=latest.telemetry.bat===undefined?'—':latest.telemetry.bat.toFixed(0)+'%';
    $('height').textContent=latest.telemetry.h===undefined?'—':(latest.telemetry.h/100).toFixed(2)+' m';
    $('state-age').textContent=age(latest.telemetry_age_s);$('frame-age').textContent=age(latest.video_age_s);
    $('video-state').textContent=latest.video_fresh?(real?'Live · decoded frames':'Synthetic frames'):'No fresh video';
    $('camera-empty').hidden=latest.video_fresh;
    const ready=latest.flight_enabled&&latest.link_fresh&&latest.video_fresh&&!latest.land_requested;
    $('takeoff').disabled=!(ready&&latest.stage==='ready'&&latest.telemetry.bat>=25);
    $('land').disabled=!latest.flight_enabled||!['takeoff_queued','taking_off','airborne','landing_requested','landing_acknowledged'].includes(latest.stage);
    $('run-policy').disabled=!(ready&&latest.stage==='airborne'&&latest.policy_available&&!latest.policy_used);
    $('policy').disabled=latest.policy_active;
    $('policy-state').textContent=latest.policy_active?'RUNNING · '+latest.policy_name+' · '+JSON.stringify(latest.policy_info):latest.policy_available?(latest.policy_used?'Stopped · land and restart for another policy run':'Ready for policy start after takeoff'):'Mapped policy setup required · manual control available';
    directionButtons.forEach(b=>{b.disabled=!(ready&&latest.stage==='airborne');});
    if(lastMessage!==latest.message){$('status').textContent=latest.message;lastMessage=latest.message;}
  }catch(error){
    clearHeld();directionButtons.forEach(b=>b.disabled=true);$('takeoff').disabled=true;$('run-policy').disabled=true;
    $('camera-empty').hidden=false;$('camera-empty').textContent='Bridge unavailable · do not rely on the last image';
    $('status').textContent='Local bridge unavailable; movement renewal stopped';
  }
}
async function frame(){
  // One image request at a time; a delayed frame never queues extra requests.
  const image=$('camera');
  await new Promise(resolve=>{
    const timeout=setTimeout(()=>{image.hidden=true;resolve();},1200);
    image.onload=()=>{clearTimeout(timeout);image.hidden=false;resolve();};
    image.onerror=()=>{clearTimeout(timeout);image.hidden=true;resolve();};
    image.src='/camera?t='+Date.now();
  });
}
async function loop(task,delay){await task();setTimeout(()=>loop(task,delay),delay);}
setInterval(pump,75);loop(status,200);loop(frame,200);pump();
