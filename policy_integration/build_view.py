"""Render a data-backed simulation fragment for in-conversation playback."""

import argparse
import json
from pathlib import Path
from .stress_sim import SCENARIOS, run


def build(report_path, output):
    report = json.loads(report_path.read_text())
    notes = {
        "baseline": "Reference run. Approximate motion model; not proof of physical flight.",
        "mild-wind": "A small synthetic lateral velocity disturbance is added to the plant.",
        "strong-wind": "This PD baseline stalls off target under the stronger constant disturbance.",
        "delay-150": "The policy receives positions captured 150 ms earlier; the square shows that delayed estimate.",
        "delay-300": "Continuous 300 ms delay exceeds the policy's pose-age limit and triggers an abort.",
        "pose-loss": "Video freezes at 5 s. Zero RC and landing request are policy decisions; landing is not simulated.",
        "link-loss": "Link loss at 5 s latches abort. The trace stops at the landing request, not after a physical landing.",
        "deadband": "Commands below 12 are ignored by this synthetic actuator; showing a failing seed.",
    }
    data=[]
    for summary, scenario in zip(report['scenarios'], SCENARIOS):
        if summary['id'] != scenario.id:
            raise ValueError('scenario order mismatch')
        seed=0
        if scenario.id=='deadband':
            seed=next((r['seed'] for r in summary['runs'] if r['outcome']!='complete'), 0)
        result=run(scenario,seed,trace=True)
        rows=result['trace']
        # Keep all mode transitions plus 5 Hz samples. Actions in between remain
        # in raw output; playback is a downsampled view of actual Python runs.
        keep=[row for i,row in enumerate(rows) if i%4==0 or i==len(rows)-1 or (i and rows[i-1][11:]!=row[11:])]
        data.append({'id':scenario.id,'label':scenario.label,'counts':summary['counts'],'seed_count':report['seeds_per_scenario'],
                     'seed':seed,'elapsed_s':result['elapsed_s'],'outcome':result['outcome'],
                     'reason':result['reason'],'note':notes[scenario.id],'trace':keep})
    source=Path(__file__).with_name('trace_view.html').read_text()
    fragment=source.replace('__TRACE_DATA__',json.dumps(data,separators=(',',':')).replace('</','<\\/'))
    output.write_text(fragment)
    return len(fragment.encode())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    print(f'{args.output}: {build(args.report,args.output)} bytes')


if __name__=='__main__':
    main()
