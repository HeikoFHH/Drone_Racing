"""Self-contained interactive replay (one HTML file, no external assets) of recorded rollouts."""

from __future__ import annotations

import json
from pathlib import Path

PHASES = ["IDLE", "AT_START", "OUTBOUND", "TURNED", "FINISHED"]


def pack_run(label: str, ro, i: int) -> dict:
    tr = ro.tracks[i]
    modes = sorted(set(m.split(":")[0] for m in tr["mode"]))
    r3 = lambda a: [round(float(v), 3) for v in a]
    lap = ro.lap_time_s[i]
    return {
        "label": label, "controller": ro.controller, "scenario": ro.scenario, "seed": int(ro.seeds[i]),
        "outcome": ro.outcome[i], "reason": ro.land_reason[i], "lap": lap, "lap_est": ro.lap_time_est_s[i],
        "takeover": list(ro.takeover[i]) if ro.takeover[i] else None, "sensing": ro.meta["sensing"],
        "dt": float(ro.dt), "modes": modes,
        "t": r3(tr["t"]), "x": r3(tr["x"]), "y": r3(tr["y"]), "z": r3(tr["z"]), "yaw": r3(tr["yaw"]),
        "rc": [[int(v) for v in row] for row in tr["rc"]],
        "phase": [PHASES.index(p) for p in tr["phase"]], "mode": [modes.index(m.split(":")[0]) for m in tr["mode"]],
        "ex": r3(tr["est_x"]), "ey": r3(tr["est_y"]), "lost": [int(v) for v in tr["lost"]],
        "seen": [int(a) + 2 * int(b) for a, b in tr["seen"]],
    }


def write_replay(runs: list[dict], geom, path: Path, title: str = "Tello race replay") -> None:
    course = {"a": list(geom.pole_a_xy), "b": list(geom.pole_b_xy), "gate": geom.gate_radius_m, "pole_r": 0.25,
              "geofence": {"x": [-1.0, 5.0], "y": [-1.0, 1.5], "z": [0.4, 2.0]}}
    html = TEMPLATE.replace("__DATA__", json.dumps({"course": course, "runs": runs}, separators=(",", ":"))).replace("__TITLE__", title)
    Path(path).write_text(html, encoding="utf-8")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--mut:#6b7385;--line:#d9dde5;--grid:#eceff4;--a:#2f6db3;--g:#2a9d4b;--r:#d1453b;--o:#e8a317;--p:#8e44ad;--pole:#222;--fov:rgba(47,109,179,.12)}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#12151b;--card:#1b2029;--ink:#e6e9ef;--mut:#98a1b3;--line:#2c3340;--grid:#232a35;--pole:#cfd5e1;--fov:rgba(90,150,230,.16)}}
:root[data-theme=dark]{--bg:#12151b;--card:#1b2029;--ink:#e6e9ef;--mut:#98a1b3;--line:#2c3340;--grid:#232a35;--pole:#cfd5e1;--fov:rgba(90,150,230,.16)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}h1{font-size:18px;margin:0 0 12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;margin-bottom:12px}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}select,button,input[type=range]{font:inherit;color:var(--ink)}
select,button{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 10px;cursor:pointer}
button:hover{border-color:var(--a)}input[type=range]{flex:1;min-width:160px}
canvas{width:100%;display:block;border-radius:8px;background:var(--card)}
.grid{display:grid;grid-template-columns:1fr;gap:12px}@media(min-width:900px){.grid{grid-template-columns:1.6fr 1fr}}
.stat{display:flex;gap:16px;flex-wrap:wrap;color:var(--mut)}.stat b{color:var(--ink)}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;border:1px solid var(--line)}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--mut)}.legend i{display:inline-block;width:14px;height:4px;border-radius:2px;margin-right:4px;vertical-align:middle}
</style></head><body><main>
<h1>__TITLE__</h1>
<div class="card"><div class="row">
<label>Run <select id="run"></select></label>
<button id="play">Pause</button>
<label>Speed <select id="speed"><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select>x</label>
<input id="scrub" type="range" min="0" max="1" value="0" step="1" aria-label="time">
<span id="clock" class="stat"></span></div>
<div id="info" class="stat" style="margin-top:8px"></div></div>
<div class="grid"><div class="card"><canvas id="top" width="900" height="420" aria-label="top-down view"></canvas>
<canvas id="side" width="900" height="150" aria-label="side view" style="margin-top:8px"></canvas>
<div class="legend" id="legend" style="margin-top:8px"></div></div>
<div class="card"><canvas id="sticks" width="360" height="250" aria-label="rc sticks"></canvas>
<div id="state" style="margin-top:8px"></div></div></div>
<script>
const D=__DATA__;const C=D.course;const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const PH=["idle","at start","outbound","turned","finished"];
const PHC=()=>[css("--mut"),css("--a"),css("--g"),css("--r"),css("--p")];
const MODEC={backup:"#e67e22",landing:"#c0392b",hold:"#f1c40f"};
let run=D.runs[0],k=0,playing=true,last=performance.now();
const sel=$("run");D.runs.forEach((r,i)=>{const o=document.createElement("option");o.value=i;o.textContent=r.label;sel.appendChild(o)});
function setRun(i){run=D.runs[i];k=0;$("scrub").max=run.t.length-1;$("scrub").value=0;
 const lap=run.lap!=null?run.lap.toFixed(1)+" s":"-";
 $("info").innerHTML=`<span><b>${run.controller}</b></span><span>scenario <b>${run.scenario}</b></span><span>seed <b>${run.seed}</b></span><span>sensing <b>${run.sensing}</b></span>`+
 `<span>outcome <b>${run.outcome}</b>${run.reason?" ("+run.reason+")":""}</span><span>lap <b>${lap}</b></span>`+(run.takeover?`<span>Maude backup at <b>${run.takeover[0].toFixed(1)} s</b> (${run.takeover[1]})</span>`:"");
 $("legend").innerHTML=PH.slice(1).map((n,j)=>`<span><i style="background:${PHC()[j+1]}"></i>${n}</span>`).join("")+Object.entries(MODEC).filter(([m])=>run.modes.includes(m)).map(([m,c])=>`<span><i style="background:${c}"></i>${m}</span>`).join("")+
 (run.sensing==="realistic"?`<span><i style="background:${css("--ink")};height:2px"></i>estimate (ghost)</span><span><i style="background:rgba(90,150,230,.4)"></i>camera field of view</span>`:"");}
sel.onchange=()=>setRun(+sel.value);$("play").onclick=()=>{playing=!playing;$("play").textContent=playing?"Pause":"Play";last=performance.now()};
$("scrub").oninput=e=>{k=+e.target.value;playing=false;$("play").textContent="Play"};
function fit(cv){const r=cv.getBoundingClientRect(),d=window.devicePixelRatio||1;const w=Math.round(r.width*d),h=Math.round(cv.height/cv.width*r.width*d);if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h}return d}
function draw(){
 const cv=$("top"),g=cv.getContext("2d");fit(cv);const W=cv.width,H=cv.height;g.clearRect(0,0,W,H);
 const x0=-1.4,x1=Math.max(C.b[0],4)+1.4,y0=-1.7,y1=1.7;const s=Math.min(W/(x1-x0),H/(y1-y0));const ox=(W-s*(x1-x0))/2,oy=(H-s*(y1-y0))/2;
 const X=x=>ox+(x-x0)*s,Y=y=>oy+(y1-y)*s;
 g.strokeStyle=css("--grid");g.lineWidth=1;for(let x=Math.ceil(x0);x<=x1;x++){g.beginPath();g.moveTo(X(x),Y(y0));g.lineTo(X(x),Y(y1));g.stroke()}for(let y=Math.ceil(y0);y<=y1;y++){g.beginPath();g.moveTo(X(x0),Y(y));g.lineTo(X(x1),Y(y));g.stroke()}
 g.fillStyle=css("--mut");g.font=`${11*(window.devicePixelRatio||1)}px system-ui`;g.fillText("X: A to B (m)",X(x1)-80*(window.devicePixelRatio||1),Y(y0)-6);
 const gf=C.geofence;g.setLineDash([4,4]);g.strokeStyle=css("--line");g.strokeRect(X(gf.x[0]),Y(gf.y[1]),(gf.x[1]-gf.x[0])*s,(gf.y[1]-gf.y[0])*s);g.setLineDash([]);
 [[C.a,"A",1],[C.b,"B",-1]].forEach(([p,n,f])=>{g.setLineDash([5,5]);g.strokeStyle=css("--line");g.beginPath();g.arc(X(p[0]),Y(p[1]),C.gate*s,0,7);g.stroke();g.setLineDash([]);
  g.fillStyle=css("--pole");g.beginPath();g.arc(X(p[0]),Y(p[1]),C.pole_r*s,0,7);g.fill();g.strokeStyle="#e8a317";g.lineWidth=4;g.beginPath();g.moveTo(X(p[0]+f*.25),Y(p[1]));g.lineTo(X(p[0]+f*.5),Y(p[1]));g.stroke();g.lineWidth=1;
  g.fillStyle=css("--mut");g.fillText("Pole "+n,X(p[0])-20,Y(p[1])+(C.gate+.15)*s)});
 const ph=PHC();g.lineWidth=2.2;g.lineCap="round";
 for(let i=1;i<=k;i++){const m=run.modes[run.mode[i]];g.strokeStyle=MODEC[m]||ph[run.phase[i]];g.beginPath();g.moveTo(X(run.x[i-1]),Y(run.y[i-1]));g.lineTo(X(run.x[i]),Y(run.y[i]));g.stroke()}
 const x=run.x[k],y=run.y[k],yw=run.yaw[k];
 if(run.sensing==="realistic"){const hf=27.7*Math.PI/180;g.fillStyle=css("--fov");g.beginPath();g.moveTo(X(x),Y(y));g.arc(X(x),Y(y),3.2*s,-(yw+hf),-(yw-hf));g.closePath();g.fill();
  g.strokeStyle=css("--ink");g.globalAlpha=.55;g.beginPath();g.arc(X(run.ex[k]),Y(run.ey[k]),7,0,7);g.stroke();g.globalAlpha=1;
  const tg=[C.a,C.b];tg.forEach((p,j)=>{if(run.seen[k]&(1<<j)){g.strokeStyle="#e8a317";g.lineWidth=1.5;g.setLineDash([3,3]);g.beginPath();g.moveTo(X(x),Y(y));g.lineTo(X(p[0]),Y(p[1]));g.stroke();g.setLineDash([]);g.lineWidth=2.2}})}
 g.save();g.translate(X(x),Y(y));g.rotate(-yw);g.fillStyle=run.lost[k]?"#c0392b":css("--ink");g.beginPath();g.moveTo(13,0);g.lineTo(-9,7);g.lineTo(-5,0);g.lineTo(-9,-7);g.closePath();g.fill();g.restore();
 // side view
 const sc=$("side"),h=sc.getContext("2d");fit(sc);const SW=sc.width,SH=sc.height;h.clearRect(0,0,SW,SH);
 const sx=(SW)/(x1-x0),zmax=3,sy=SH/zmax;const SX=x=>(x-x0)*sx,SZ=z=>SH-z*sy;
 h.fillStyle="rgba(42,157,75,.08)";h.fillRect(0,SZ(2.5),SW,SZ(.4)-SZ(2.5));h.strokeStyle=css("--line");h.beginPath();h.moveTo(0,SZ(0));h.lineTo(SW,SZ(0));h.stroke();
 [C.a,C.b].forEach(p=>{h.fillStyle=css("--pole");h.fillRect(SX(p[0])-3,SZ(1.2),6,SZ(0)-SZ(1.2));h.fillStyle="#e8a317";h.fillRect(SX(p[0])-3,SZ(1.1),6,6)});
 h.strokeStyle=css("--a");h.lineWidth=2;h.beginPath();for(let i=0;i<=k;i++){const px=SX(run.x[i]),pz=SZ(run.z[i]);i?h.lineTo(px,pz):h.moveTo(px,pz)}h.stroke();
 h.fillStyle=css("--ink");h.beginPath();h.arc(SX(x),SZ(run.z[k]),5,0,7);h.fill();h.fillStyle=css("--mut");h.fillText("height (m), 0 to 3",6,12);
 // sticks
 const st=$("sticks"),q=st.getContext("2d");fit(st);const QW=st.width,QH=st.height;q.clearRect(0,0,QW,QH);
 const names=["right","forward","up","yaw"],cols=["#d98200","#2a9d4b","#888","#d1453b"];const bw=QW/4;
 names.forEach((n,i)=>{const v=run.rc[k][i]/100,cx=bw*i+bw/2,mid=QH/2-8,half=QH/2-30;q.strokeStyle=css("--line");q.strokeRect(cx-14,mid-half,28,2*half);q.beginPath();q.moveTo(cx-18,mid);q.lineTo(cx+18,mid);q.stroke();
  q.fillStyle=cols[i];q.fillRect(cx-12,v>=0?mid-v*half:mid,24,Math.abs(v)*half);q.fillStyle=css("--mut");q.textAlign="center";q.fillText(n,cx,QH-8);q.fillText(run.rc[k][i],cx,14);q.textAlign="left"});
 $("state").innerHTML=`<div class="stat"><span>phase <b>${PH[run.phase[k]]}</b></span><span>mode <b>${run.modes[run.mode[k]]}</b></span><span>height <b>${run.z[k].toFixed(2)} m</b></span></div>`+
  (run.sensing==="realistic"?`<div class="stat"><span class="badge" style="color:${run.lost[k]?"#c0392b":"var(--g)"}">${run.lost[k]?"tag lost: dead reckoning":"tracking"}</span><span>tag A <b>${run.seen[k]&1?"seen":"-"}</span><span>tag B <b>${run.seen[k]&2?"seen":"-"}</b></span></div>`:"");
 $("clock").innerHTML=`<span><b>${run.t[k].toFixed(1)}</b> / ${run.t[run.t.length-1].toFixed(1)} s</span>`;$("scrub").value=k}
function loop(now){const sp=+$("speed").value;if(playing){const adv=(now-last)/1000*sp/run.dt;if(adv>=1){k=Math.min(run.t.length-1,k+Math.floor(adv));last=now;if(k>=run.t.length-1){playing=false;$("play").textContent="Replay";$("play").onclick=()=>{k=0;playing=true;last=performance.now();$("play").textContent="Pause";$("play").onclick=()=>{playing=!playing;$("play").textContent=playing?"Pause":"Play";last=performance.now()}}}}}else last=now;draw();requestAnimationFrame(loop)}
setRun(0);requestAnimationFrame(loop);addEventListener("resize",draw);
</script></main></body></html>
"""
