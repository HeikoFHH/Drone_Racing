/* Browser preview of the CLASSIC policy. No SDK/network/model inference.
 * Config is exported from Python; parity tests compare sequential decisions.
 */
(function (scope) {
  'use strict';
  const clamp = (v, n) => Math.max(-n, Math.min(n, v));
  const norm = v => Math.hypot(...v);
  const wrap = v => Math.atan2(Math.sin(v), Math.cos(v));
  const round = v => { const lo = Math.floor(v), f = v - lo;
    return (f === .5 ? (lo % 2 === 0 ? lo : lo + 1) : Math.round(v)) || 0; };
  class ClassicPreview {
    constructor(config) {
      this.c = config; this.waypoint_index = 0; this.lastNow = null;
      this.lastPose = null; this.lastTelemetry = null; this.accepted = null;
      this.prev = [0,0,0,0]; this.dwell = 0; this.arrival = null;
      this.fault = null; this.completed = false; this.stale = null;
      this.started = null; this.braking = false;
    }
    zero(mode, reason, land=false, latch=false) {
      this.prev = [0,0,0,0]; this.dwell=0; this.arrival=null; this.braking=false;
      if (latch) this.fault=reason;
      return {rc:[0,0,0,0],mode,reason,waypoint_index:this.waypoint_index,request_land:land};
    }
    inside(p) { return [this.c.x_bounds,this.c.y_bounds,this.c.z_bounds].every((b,i)=>b[0]<=p[i]&&p[i]<=b[1]); }
    clear(a,b) {
      const dx=b[0]-a[0],dy=b[1]-a[1],den=dx*dx+dy*dy;
      return this.c.poles_xy.every(([x,y])=>{
        const u=den===0?0:Math.max(0,Math.min(1,((x-a[0])*dx+(y-a[1])*dy)/den));
        return Math.hypot(a[0]+u*dx-x,a[1]+u*dy-y)>=this.c.pole_clearance_m;
      });
    }
    step(o,now) {
      const c=this.c, abort=r=>this.zero('abort',r,true,true);
      if(this.fault) return this.zero('abort',this.fault,true);
      const scalars=[now,o.pose_time_s,o.telemetry_time_s,o.yaw_rad,o.position_std_m,o.velocity_std_mps,o.battery_pct];
      if(!Array.isArray(o.position_w)||!Array.isArray(o.velocity_w)||o.position_w.length!==3||o.velocity_w.length!==3||
         ![...scalars,...o.position_w,...o.velocity_w].every(v=>typeof v==='number'&&Number.isFinite(v))||
         ![o.link_ok,o.airborne,o.enabled].every(v=>typeof v==='boolean')||o.battery_pct<0||o.battery_pct>100||
         o.position_std_m<0||o.velocity_std_mps<0||o.frame!=='course_flu_v1'||o.schema_version!==1) return abort('invalid_observation');
      if(o.airborne) {
        if(this.started===null)this.started=now;
        if(now<this.started)return abort('control_clock_or_deadline');
        if(now-this.started>=c.max_session_s)return abort('session_timeout');
      }
      if(!o.enabled){this.lastNow=null;this.stale=null;return this.zero('disabled','operator_not_enabled');}
      if(!o.airborne){this.lastNow=null;this.stale=null;return this.zero('ground','takeoff_owned_by_runner');}
      const dt=this.lastNow===null?.05:now-this.lastNow;
      if(dt<=0||dt>c.max_control_gap_s)return abort('control_clock_or_deadline');
      this.lastNow=now;
      if(!o.link_ok)return abort('link_lost');
      if(o.battery_pct<c.min_battery_pct)return abort('low_battery');
      const pa=now-o.pose_time_s,ta=now-o.telemetry_time_s;
      if(pa<0||ta<0)return abort('future_timestamp');
      if((this.lastPose!==null&&o.pose_time_s<this.lastPose)||(this.lastTelemetry!==null&&o.telemetry_time_s<this.lastTelemetry))return abort('out_of_order_timestamp');
      this.lastPose=o.pose_time_s;this.lastTelemetry=o.telemetry_time_s;
      if(ta>c.max_telemetry_age_s||pa>c.land_after_pose_age_s)return abort('sensors_timed_out');
      if(pa>c.max_pose_age_s){
        if(this.stale===null)this.stale=now;
        if(now-this.stale>=c.max_stale_hold_s)return abort('persistent_pose_stale');
        return this.zero('hold','pose_stale');
      }
      this.stale=null;
      if(o.position_std_m>c.max_position_std_m||o.velocity_std_mps>c.max_velocity_std_mps)return abort('estimator_uncertain');
      if(!this.inside(o.position_w))return abort('geofence');
      if(norm(o.velocity_w)>c.max_measured_speed_mps)return abort('excess_speed');
      if(this.accepted){
        const [t,p,std]=this.accepted,elapsed=o.pose_time_s-t,movement=norm(o.position_w.map((v,i)=>v-p[i]));
        if(elapsed===0&&movement>1e-6)return abort('pose_changed_without_new_frame');
        if(elapsed>0&&movement>c.max_measured_speed_mps*elapsed+3*(o.position_std_m+std))return abort('implausible_pose_jump');
      }
      this.accepted=[o.pose_time_s,[...o.position_w],o.position_std_m];
      if(this.completed)return this.zero('complete','route_complete_timer_must_confirm',true);
      const goal=c.waypoints_w[this.waypoint_index];
      if(!this.clear(o.position_w,goal))return abort('path_near_pole');
      const err=goal.map((v,i)=>v-o.position_w[i]);let target=0;
      if(c.heading_targets_w){
        const point=c.heading_targets_w[this.waypoint_index],dx=point[0]-o.position_w[0],dy=point[1]-o.position_w[1];
        if(Math.hypot(dx,dy)<c.pole_clearance_m)return abort('heading_landmark_too_close');
        target=Math.atan2(dy,dx);
      }
      const ye=wrap(target-o.yaw_rad),aiming=Math.abs(ye)>c.max_yaw_error_rad;
      const braking=aiming&&Math.hypot(...o.velocity_w.slice(0,2))>c.turn_entry_speed_mps;
      if(!aiming&&norm(err)<=c.waypoint_tolerance_m&&norm(o.velocity_w)<=c.arrival_speed_mps){
        if(this.arrival!==null&&o.pose_time_s>this.arrival)this.dwell+=Math.min(o.pose_time_s-this.arrival,dt,.1);
        this.arrival=o.pose_time_s;
        if(this.dwell>=c.arrival_dwell_s){
          this.dwell=0;
          if(this.waypoint_index===c.waypoints_w.length-1){this.completed=true;return this.zero('complete','route_complete_timer_must_confirm',true);}
          this.waypoint_index++;return this.zero('waypoint','advance');
        }
      }else{this.dwell=0;this.arrival=null;}
      const co=Math.cos(o.yaw_rad),si=Math.sin(o.yaw_rad),desired=err.map((v,i)=>c.kp*v-c.kd*o.velocity_w[i]);
      let raw=[100*(si*desired[0]-co*desired[1])/c.horizontal_mps_at_100,
        100*(co*desired[0]+si*desired[1])/c.horizontal_mps_at_100,
        100*desired[2]/c.vertical_mps_at_100,-100*c.yaw_kp*ye/c.yaw_rps_at_100];
      if(braking){
        if(!this.braking)this.prev=[0,0,this.prev[2],this.prev[3]];
        const vx=-c.kd*o.velocity_w[0],vy=-c.kd*o.velocity_w[1];
        raw=[100*(si*vx-co*vy)/c.horizontal_mps_at_100,100*(co*vx+si*vy)/c.horizontal_mps_at_100,raw[2],0];
      }
      const h=Math.hypot(raw[0],raw[1]);if(h>c.rc_limit){raw[0]*=c.rc_limit/h;raw[1]*=c.rc_limit/h;}
      const limited=raw.map(v=>clamp(v,c.rc_limit)),delta=limited.map((v,i)=>v-this.prev[i]);
      const largest=Math.max(Math.abs(delta[0]),Math.abs(delta[1])),fraction=largest?Math.min(1,c.rc_slew_per_s*dt/largest):1;
      const slew=[this.prev[0]+fraction*delta[0],this.prev[1]+fraction*delta[1],this.prev[2]+clamp(delta[2],c.rc_slew_per_s*dt),this.prev[3]+clamp(delta[3],c.rc_slew_per_s*dt)];
      if(aiming&&!braking){slew[0]=0;slew[1]=0;}
      this.prev=slew;this.braking=braking;
      return {rc:slew.map(round),mode:braking?'brake':aiming?'aim':'track',reason:'classic_pd',waypoint_index:this.waypoint_index,request_land:false};
    }
  }
  scope.DronePreview={ClassicPreview,wrap};
})(typeof module!=='undefined'?module.exports:globalThis);
