"""Adapt the repository's policies and tag sensing; never write to the drone."""
from dataclasses import replace
import json
import math
from pathlib import Path
import threading

from policy_integration.adapter import PositionerAdapter,matvec
from policy_integration.policy import ActionSample,BaselinePolicy,Observation,PolicyConfig


class DemoPose:
    error=None
    def __init__(self,transport):self.transport=transport
    def sample(self,now,telemetry,telemetry_at,airborne):
        t=self.transport
        return Observation(now,telemetry_at,tuple(t.p),tuple(t.v),0,.03,.05,
                           telemetry.get('bat',0),True,airborne,True)
    def close(self):pass


class MappedPose:
    """Use FilteredPositioner, retaining each accepted fix's original pose/time.

    The operator must supply the measured transform, camera file and video-delay
    upper bound. This is deliberately not the sim's default frame fixture.
    """
    def __init__(self,transport,config_path):
        from core.positional_kalman_filter import FilteredPositioner
        config=json.loads(Path(config_path).read_text())
        if config.get('verified_for_this_drone_and_course') is not True:
            raise ValueError('Policy setup must be verified for this drone and course')
        self.adapter=PositionerAdapter(config['R_world_map'],config['t_world_map'])
        self.delay=config['video_delay_upper_s']
        if type(self.delay) not in (float,int) or not math.isfinite(self.delay) or not 0<=self.delay<=1:
            raise ValueError('Invalid measured video delay upper bound')
        root=Path(config_path).resolve().parent
        def file(key):
            path=Path(config[key]);return path if path.is_absolute() else root/path
        self.map_path=file('map_path');self.course_path=file('course_path')
        self.positioner=FilteredPositioner(config['drone'],map_path=self.map_path,
                                         camera_path=file('camera_path'),tags_path=file('tags_path'))
        self.transport=transport;self.lock=threading.Lock();self.latest=None;self.error='Waiting for an accepted mapped tag fix'
        self.stop=threading.Event();self.thread=threading.Thread(target=self._watch,daemon=True,name='mapped-tag-positioner');self.thread.start()
    def _watch(self):
        seen=None
        while not self.stop.wait(.01):
            packet=self.transport.frame_packet
            if packet is None or packet[0]==seen:continue
            decoded_at,frame=packet;seen=decoded_at
            try:
                captured=decoded_at-self.delay
                estimate=self.positioner.locate(frame,captured)
                if not estimate['accepted']:
                    self.error='No accepted mapped tag in latest frame';continue
                raw=estimate['raw'];position=estimate['xyz'];velocity=estimate['vel']
                # Rotate diagonal map covariance into the course frame without
                # replacing missing uncertainty by a made-up small constant.
                def std(variance):
                    if variance is None or len(variance)!=3 or any(not math.isfinite(v) or v<0 for v in variance):
                        raise ValueError('Missing/invalid estimator covariance')
                    return max(math.sqrt(sum(r*r*v for r,v in zip(row,variance))) for row in self.adapter.R)
                pos_std,vel_std=std(estimate['position_variance']),std(estimate['velocity_variance'])
                result=dict(raw,xyz=list(position),raw_positioner=raw)
                with self.lock:self.latest=(captured,result,tuple(velocity),pos_std,vel_std)
                self.error=None
            except Exception as exc:self.error=type(exc).__name__+': '+str(exc)[:180]
    def sample(self,now,telemetry,telemetry_at,airborne):
        with self.lock:fix=self.latest
        if fix is None:return None
        captured,result,velocity,pos_std,vel_std=fix
        return self.adapter.adapt(result,pose_time_s=captured,telemetry_time_s=telemetry_at,
            velocity_w=matvec(self.adapter.R,velocity),position_std_m=pos_std,velocity_std_mps=vel_std,
            battery_pct=telemetry.get('bat',0),link_ok=now-telemetry_at<=.5,airborne=airborne,enabled=True)
    def map_fix(self):
        with self.lock:return self.latest
    def close(self):self.stop.set();self.thread.join(timeout=1.)


class PolicySession:
    """An existing policy instance plus its input adapter, not a new controller."""
    def __init__(self,name,source,transport):
        self.name=name;self.source=source;self.transport=transport;self.last_info={}
        self.cfg=PolicyConfig();self.guard=None
        if name=='baseline':self.policy=BaselinePolicy(self.cfg)
        elif name=='race':
            from race.race_policy import Obs,PolicyCfg,RacePolicy
            from race.lap_timer import CourseConfig,LapTimer,load_course,pole_poses_from_map
            self.obs_type=Obs
            if transport.real:
                course=load_course(source.course_path);poles=pole_poses_from_map(source.map_path)
                self.adapter=source.adapter
            else:
                course=CourseConfig('demo',1,2,.8,.4,2.)
                poles={1:(0,0,0),2:(4,0,0)}
                self.adapter=PositionerAdapter(((1,0,0),(0,0,-1),(0,1,0)),(0,0,1))
            self.timer=LapTimer(course,poles)
            self.policy=RacePolicy(poles,course.pole_a_tag,course.pole_b_tag,
                PolicyCfg(cruise_mps=.16,max_rc=20,max_up_rc=20,rel_height_m=0.))
            targets=tuple(self.adapter.point_to_world([float(v) for v in p]) for p in (self.policy.gate_a,self.policy.gate_b,self.policy.gate_a))
            world_poles=tuple(self.adapter.point_to_world([float(v) for v in p]) for p in (self.policy.pole_a,self.policy.pole_b))
            separation=tuple(b-a for a,b in zip(*world_poles))
            if separation[0]<=0 or abs(separation[1])>.05 or abs(separation[2])>.05:
                raise ValueError('Map transform must align course +X from A to B with level poles')
            self.cfg=replace(self.cfg,waypoints_w=targets,poles_xy=tuple(p[:2] for p in world_poles),
                             heading_targets_w=(world_poles[1],)*3)
            self.guard=BaselinePolicy(self.cfg,mode='learned',expected_model_id='repo-race-policy')
            self.last_fix=None;self.settled=None;self.last_now=None
        else:raise ValueError('Unknown policy')
    def ready(self,now,telemetry,telemetry_at):
        obs=self.source.sample(now,telemetry,telemetry_at,True)
        if obs is None:return False,'No accepted mapped position'
        # A disposable existing-policy instance validates the entire observation
        # envelope without consuming the actual run's state or first dwell tick.
        probe=BaselinePolicy(self.cfg).step(obs,now)
        if probe.mode in ('abort','hold'):return False,probe.reason
        return True,'Ready'
    def step(self,now,telemetry,telemetry_at):
        obs=self.source.sample(now,telemetry,telemetry_at,True)
        if obs is None:raise ValueError('Mapped position unavailable')
        if self.name=='baseline':
            decision=self.policy.step(obs,now);self.last_info=decision.to_dict();return decision
        from policy_integration.policy import Decision
        if self.transport.real:
            fix=self.source.map_fix();captured,filtered,velocity,_,_=fix
            pos=filtered['xyz'];fresh=captured!=self.last_fix
            raw_for_policy=filtered['raw_positioner'] if fresh else None;self.last_fix=captured
        else:
            import numpy as np
            pos=self.adapter.point_to_map(obs.position_w);velocity=matvec(tuple(zip(*self.adapter.R)),obs.velocity_w)
            forward=np.array([1.,0.,0.]);right=np.array([0.,-1.,0.]);down=np.array([0.,0.,-1.])
            R=np.asarray(self.adapter.R).T @ np.column_stack((right,down,forward))
            raw_for_policy={'xyz':list(pos),'R':R.tolist(),'tags':[1,2]}
        status=self.timer.update(now,tuple(pos));phase=status['phase']
        if phase=='FINISHED':
            self.last_info={'mode':'complete','phase':phase,'lap_time_s':self.timer.lap_time_s}
            return Decision(ZERO,'complete','race_timer_finished',2,True)
        if phase=='AT_START' and not self.policy.armed:
            self.settled=now if self.settled is None else self.settled
            if now-self.settled>=1.5:self.policy.armed=True
        elif not self.policy.armed:self.settled=None
        race_obs=self.obs_type(now,list(pos),list(velocity),raw_for_policy,now-obs.pose_time_s>.25,
                               raw_for_policy.get('tags',[]) if raw_for_policy else [])
        dt=.05 if self.last_now is None else now-self.last_now;self.last_now=now
        rc,info=self.policy.step(race_obs,phase,dt)
        if info.get('mode')=='land':return Decision(ZERO,'abort','race_policy_requested_land',self.guard.waypoint_index,True)
        # Guard uses Maude's unchanged packet validation, fresh-pose rules,
        # horizontal cap/slew, heading/braking, clearance and abort latch.
        index=2 if phase=='TURNED' else 1 if phase=='OUTBOUND' or self.policy.armed else 0
        self.guard.waypoint_index=index
        decision=self.guard.step(obs,now,ActionSample(tuple(v/100 for v in rc),now,obs.pose_time_s,'repo-race-policy',index))
        self.last_info={**info,'guard_mode':decision.mode,'guard_reason':decision.reason,'lap_time_s':self.timer.lap_time_s}
        return decision


ZERO=(0,0,0,0)
