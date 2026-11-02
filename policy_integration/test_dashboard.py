"""Compare the browser CLASSIC preview with the real Python policy, not RL."""
import json
from dataclasses import asdict, replace
from pathlib import Path
import shutil
import subprocess
import unittest

from .policy import BaselinePolicy, Observation, PolicyConfig
from .turn_sim import run, SCENARIOS, candidate_config


@unittest.skipUnless(shutil.which('node'), 'Node required for browser policy parity')
class DashboardParity(unittest.TestCase):
    def test_interactive_controls(self):
        from tempfile import TemporaryDirectory
        from .build_dashboard import build
        with TemporaryDirectory() as folder:
            page=Path(folder)/'preview.html'
            page.write_text(build())
            result=subprocess.run(['node','policy_integration/test_dashboard_ui.js',str(page)],
                                  text=True,capture_output=True,cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode,0,result.stderr)

    def compare(self, batches):
        program = """
const fs=require('fs'),assert=require('assert');
const {DronePreview}=require('./policy_integration/dashboard_core.js');
const batches=JSON.parse(fs.readFileSync(0,'utf8'));
let count=0;
for(const b of batches){
 const p=new DronePreview.ClassicPreview(b.config);
 for(const f of b.frames){
  assert.deepStrictEqual(p.step(f.obs,f.now),f.expected,`frame ${count}`);count++;
 }
}
console.log(count+' matching sequential decisions');
"""
        result=subprocess.run(['node','-e',program],input=json.dumps(batches),text=True,
                              capture_output=True,cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertGreater(int(result.stdout.split()[0]),0)

    def test_scenario_sequences_match_python(self):
        batches=[]
        for candidate in (False,True):
            for scenario in SCENARIOS:
                frames=[]
                class Recorder(BaselinePolicy):
                    def step(self,obs,now_s,candidate_action=None):
                        decision=super().step(obs,now_s,candidate_action)
                        frames.append({'obs':asdict(obs),'now':now_s,'expected':asdict(decision)})
                        return decision
                run(scenario,seed=3,candidate=candidate,policy_type=Recorder)
                batches.append({'config':asdict(candidate_config() if candidate else PolicyConfig()),'frames':frames})
        self.compare(batches)

    def test_guard_sequences_match_python(self):
        base=Observation(0,0,(0,.5,1),(0,0,0),0,.03,.05,80,True,True,True)
        changes=[{'link_ok':False},{'battery_pct':20},{'position_std_m':.3},
                 {'position_w':(8,.5,1)},{'velocity_w':(2,0,0)},
                 {'frame':'wrong'},{'schema_version':2},{'enabled':False},
                 {'airborne':False},{'position_w':(0,0,1)},
                 {'pose_time_s':.1},{'telemetry_time_s':.1}]
        batches=[]
        for change in changes:
            p=BaselinePolicy();frames=[]
            for now,obs in [(0,replace(base,**change)),(.05,replace(base,pose_time_s=.05,telemetry_time_s=.05))]:
                frames.append({'now':now,'obs':asdict(obs),'expected':asdict(p.step(obs,now))})
            batches.append({'config':asdict(p.cfg),'frames':frames})
        self.compare(batches)


if __name__=='__main__':unittest.main()
