"""Turn/visibility diagnostics using the real policy and a synthetic planar plant.

No physical flights or rendered tag detections. Point visibility ignores tag
facing, full corners, blur and distortion, so a visibility pass is optimistic.
Oracle pose mode isolates control logic; it does NOT model a real sensor system.
"""

from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import math
import random
import hashlib
import re
import subprocess
import sys
import types

from .policy import BaselinePolicy, Decision, Observation, PolicyConfig, ZERO


@dataclass(frozen=True)
class Camera:
    width: int = 960
    height: int = 720
    fx: float = 914.5379930571795
    fy: float = 914.4056412418461
    cx: float = 470.599937031912
    cy: float = 360.9839617497804

    def visible(self, p, yaw, point):
        dx, dy, dz = (v - q for v, q in zip(point, p))
        co, si = math.cos(yaw), math.sin(yaw)
        forward, right = co * dx + si * dy, si * dx - co * dy
        if forward <= 0:
            return False
        u, v = self.cx + self.fx * right / forward, self.cy - self.fy * dz / forward
        return 0 <= u <= self.width and 0 <= v <= self.height


def geometry(camera=Camera(), lateral_offset=0.5, gate_radius=0.8):
    if not 0 < lateral_offset < gate_radius:
        raise ValueError("offset must be positive and smaller than gate radius")
    exit_x = 4 - camera.fx * lateral_offset / (camera.width-camera.cx)
    enter_x = 4 - math.sqrt(gate_radius**2-lateral_offset**2)
    return {"camera_source": "example config/camera/default.yaml; replace per drone",
            "horizontal_fov_deg": round(math.degrees(math.atan(camera.cx/camera.fx)
                                                   +math.atan((camera.width-camera.cx)/camera.fx)),3),
            "tag_center_exits_frame_at_x_m": round(exit_x,3),
            "gate_entry_x_m": round(enter_x,3),
            "unobserved_gap_before_gate_m": round(enter_x-exit_x,3)}


@dataclass(frozen=True)
class TurnScenario:
    id: str
    camera_limited: bool = False
    delay_s: float = 0.0
    yaw_kick_deg: float = 0.0
    kick_at_s: float = 5.0
    velocity_kick_mps: float = 0.0
    response_tau_s: float = 0.18
    command_delay_s: float = 0.0
    hide_from_s: float | None = None
    hide_duration_s: float = 0.0
    wind_y_mps: float = 0.0
    horizon_s: float = 90.0


def candidate_config():
    # A review candidate only: orient toward B outbound, toward A on return.
    # It still needs proven localization coverage through the turn.
    return PolicyConfig(heading_targets_w=((4,0,1.2),(4,0,1.2),(0,0,1.2)),
                        arrival_speed_mps=0.05)


def run(scenario, seed=0, candidate=False, policy_type=BaselinePolicy):
    cfg = candidate_config() if candidate else PolicyConfig()
    policy = policy_type(cfg)
    rng = random.Random(seed)
    dt=0.05
    p=[rng.uniform(-0.1,0.1),0.5+rng.uniform(-0.04,0.04),1.0]
    v=[0.0,0.0,0.0]
    yaw=0.0
    camera=Camera()
    tags=((0,0,1.2),(4,0,1.2))
    frames=deque();commands=deque();last_fix=None;applied_rc=ZERO
    modes=Counter();max_gap=0.0;last_seen=0.0;clearance=float('inf')
    kicked=False;recovered_at=None;peak_error=0.0;truth_b=False;left_a=False;truth_finished=False
    decision=Decision(ZERO,"disabled","awaiting_initial_fix",0)
    trace=[]
    true_violation=None
    for tick in range(int(scenario.horizon_s/dt)+1):
        now=round(tick*dt,8)
        if not kicked and now>=scenario.kick_at_s:
            yaw += math.radians(scenario.yaw_kick_deg)
            v[0] += scenario.velocity_kick_mps
            kicked=True
        hidden=scenario.hide_from_s is not None and scenario.hide_from_s <= now < scenario.hide_from_s+scenario.hide_duration_s
        visible=(not scenario.camera_limited or any(camera.visible(p,yaw,tag) for tag in tags)) and not hidden
        if visible:
            measured=tuple(value+rng.gauss(0,0.003) for value in p)
            frames.append((now+scenario.delay_s,now,measured,tuple(v),yaw))
            last_seen=now
        max_gap=max(max_gap,now-last_seen)
        while frames and frames[0][0] <= now+1e-8:
            _,captured,position,velocity,heading=frames.popleft()
            last_fix=(captured,position,velocity,heading)
        if last_fix is not None:
            captured,position,velocity,heading=last_fix
            obs=Observation(captured,now,position,velocity,heading,0.03,0.05,80.0,True,True,True)
            decision=policy.step(obs,now)
        else:
            decision=Decision(ZERO,"disabled","awaiting_initial_fix",0)
        modes[decision.mode]+=1
        distance_a=math.hypot(p[0],p[1]);distance_b=math.hypot(p[0]-4,p[1])
        clearance=min(clearance,distance_a,distance_b)
        if distance_a>0.8: left_a=True
        if left_a and distance_b<=0.8: truth_b=True
        if truth_b and distance_a<=0.8: truth_finished=True
        if min(distance_a,distance_b)<cfg.pole_clearance_m:
            true_violation="truth_pole_clearance"
        if not cfg.inside(p):
            true_violation="truth_geofence"
        # For heading-kick recovery metrics use the commanded heading at this tick.
        target=0.0
        if cfg.heading_targets_w is not None:
            point=cfg.heading_targets_w[policy.waypoint_index]
            target=math.atan2(point[1]-p[1],point[0]-p[0])
        error=abs(math.atan2(math.sin(target-yaw),math.cos(target-yaw)))
        if kicked and scenario.yaw_kick_deg:
            peak_error=max(peak_error,error)
            if recovered_at is None and error<=cfg.max_yaw_error_rad and math.hypot(*v[:2])<=cfg.turn_entry_speed_mps:
                recovered_at=now
        if tick%4==0 or decision.request_land or true_violation:
            trace.append({"t":now,"p":[round(x,3) for x in p],"yaw_deg":round(math.degrees(yaw),2),
                          "mode":decision.mode,"rc":decision.rc,"waypoint":policy.waypoint_index,"visible":visible})
        if decision.request_land or true_violation:
            break
        commands.append((now+scenario.command_delay_s,decision.rc))
        while commands and commands[0][0]<=now+1e-8:
            _,applied_rc=commands.popleft()
        right,forward,up,cw=applied_rc
        f=forward/100*cfg.horizontal_mps_at_100;r=right/100*cfg.horizontal_mps_at_100
        desired=(math.cos(yaw)*f+math.sin(yaw)*r,
                 math.sin(yaw)*f-math.cos(yaw)*r+scenario.wind_y_mps,
                 up/100*cfg.vertical_mps_at_100)
        alpha=1-math.exp(-dt/scenario.response_tau_s)
        for axis in range(3):
            v[axis]+=(desired[axis]-v[axis])*alpha
            p[axis]+=v[axis]*dt
        yaw-=cw/100*cfg.yaw_rps_at_100*dt
    complete=decision.mode=="complete"
    outcome="truth_violation" if true_violation else "complete" if complete else "abort" if decision.mode=="abort" else "timeout"
    return {"scenario":asdict(scenario),"seed":seed,"controller":"look_at_candidate" if candidate else "fixed_heading_baseline",
            "outcome":outcome,"reason":true_violation or decision.reason,"elapsed_s":now,
            "truth_reached_b":truth_b,"truth_lap_finished":truth_finished,
            "min_true_pole_distance_m":round(clearance,3),"max_visibility_gap_s":round(max_gap,3),
            "max_heading_error_after_kick_deg":round(math.degrees(peak_error),2),
            "heading_recovery_s":None if recovered_at is None else round(recovered_at-scenario.kick_at_s,3),
            "mode_ticks":dict(modes),"landing_simulated":False,"trace":trace}


SCENARIOS=(
    TurnScenario("oracle_nominal"),
    *(TurnScenario("oracle_yaw_"+str(angle),yaw_kick_deg=angle) for angle in (-90,-60,-30,30,60,90,180)),
    TurnScenario("oracle_yaw_velocity",yaw_kick_deg=60,velocity_kick_mps=0.3),
    TurnScenario("oracle_yaw_delay",yaw_kick_deg=60,delay_s=0.15),
    TurnScenario("oracle_yaw_command_delay",yaw_kick_deg=60,command_delay_s=0.15),
    TurnScenario("oracle_sluggish_turn",yaw_kick_deg=60,response_tau_s=0.6),
    TurnScenario("two_tag_visibility",camera_limited=True),
    TurnScenario("two_tag_yaw",camera_limited=True,yaw_kick_deg=60),
    TurnScenario("two_tag_yaw_delay",camera_limited=True,yaw_kick_deg=60,delay_s=0.15),
    TurnScenario("oracle_hidden_turn",yaw_kick_deg=60,hide_from_s=5,hide_duration_s=1),
)


def load_reference(ref):
    """Load the previously reviewed policy from an explicit local Git commit."""
    if re.fullmatch(r"[0-9a-fA-F]{7,40}",ref) is None:
        raise ValueError("reference must be a hexadecimal Git commit ID")
    code=subprocess.check_output(["git","show",ref+":policy_integration/policy.py"],text=True)
    name="_reviewed_policy_"+ref
    module=types.ModuleType(name);sys.modules[name]=module
    exec(compile(code,"git:"+ref+":policy.py","exec"),module.__dict__)
    return module.BaselinePolicy,hashlib.sha256(code.encode()).hexdigest()


def suite(seeds=10, reference_ref=None):
    cells=[]
    reference,reference_hash=(load_reference(reference_ref) if reference_ref else (None,None))
    for scenario in SCENARIOS:
        for candidate in (False,True):
            runs=[run(scenario,seed,candidate) for seed in range(seeds)]
            for result in runs: result.pop("trace")
            cells.append({"id":scenario.id,"controller":"look_at_candidate" if candidate else "fixed_heading_baseline",
                          "outcomes":dict(Counter(r["outcome"] for r in runs)),"runs":runs})
        if reference is not None:
            runs=[run(scenario,seed,False,reference) for seed in range(seeds)]
            for result in runs:
                result.pop("trace")
                result["controller"]="previous_reviewed_policy"
            cells.append({"id":scenario.id,"controller":"previous_reviewed_policy",
                          "outcomes":dict(Counter(r["outcome"] for r in runs)),"runs":runs})
    return {"model":"synthetic_planar_velocity_plant_not_rigid_body_or_hardware",
            "assumptions":["oracle means ideal pose coverage, not implemented sensing","camera uses point-tag projection only",
                           "no tag-face/full-corner/blur/distortion/roll/pitch model","landing requests stop traces; no landing is simulated"],
            "geometry":geometry(),"seeds":seeds,"reference_ref":reference_ref,
            "reference_policy_sha256":reference_hash,
            "current_policy_sha256":hashlib.sha256(Path(__file__).with_name("policy.py").read_bytes()).hexdigest(),
            "cells":cells}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds",type=int,default=10)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--run-log",type=Path)
    parser.add_argument("--reference-ref")
    args=parser.parse_args(argv)
    if not 1<=args.seeds<=1000: parser.error("seeds must be between 1 and 1000")
    report=suite(args.seeds,args.reference_ref)
    if args.run_log:
        args.run_log.parent.mkdir(parents=True,exist_ok=True)
        with args.run_log.open('w') as stream:
            for cell in report["cells"]:
                runs=cell.pop("runs")
                for result in runs: stream.write(json.dumps(result,separators=(',',':'))+'\n')
                cell["reasons"]=dict(Counter(r["reason"] for r in runs))
                cell["truth_finished"]=sum(r["truth_lap_finished"] for r in runs)
                cell["truth_reached_b"]=sum(r["truth_reached_b"] for r in runs)
                cell["min_true_pole_distance_m"]=min(r["min_true_pole_distance_m"] for r in runs)
                cell["failed_seeds"]=[r["seed"] for r in runs if r["outcome"]!="complete"]
        report["run_log"]=str(args.run_log)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({"geometry":report["geometry"],"cells":[{k:c[k] for k in ("id","controller","outcomes")} for c in report["cells"]]},indent=2))


if __name__=="__main__": main()
