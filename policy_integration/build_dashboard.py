"""Build a self-contained simulation preview; no network or flight connection."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .policy import PolicyConfig
from .turn_sim import candidate_config


def build():
    folder=Path(__file__).parent
    html=(folder/'dashboard_fragment.html').read_text()
    html=html.replace('/*__CORE__*/',(folder/'dashboard_core.js').read_text())
    return html.replace('/*__CONFIGS__*/',json.dumps({'fixed':asdict(PolicyConfig()),'look':asdict(candidate_config())}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fragment',type=Path)
    parser.add_argument('--standalone',type=Path)
    args=parser.parse_args()
    fragment=build()
    if args.fragment:args.fragment.write_text(fragment)
    if args.standalone:
        theme='''<style>
html{color-scheme:light dark;--background:light-dark(#fff,#181818);--foreground:light-dark(#202020,#eee);--border:light-dark(#ddd,#444);--muted-foreground:light-dark(#555,#bbb);--viz-series-1:light-dark(#176c9a,#75c8ef);--viz-series-2:light-dark(#a84c14,#efa36b);font:14px system-ui;background:var(--background);color:var(--foreground)}
body{max-width:960px;margin:24px auto;padding:0 16px}button,select{font:inherit;max-width:100%;padding:8px;margin:3px}button{cursor:pointer}.viz-row,.viz-controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.form-label{display:flex;flex-direction:column;gap:4px}.text-small{font-size:12px}.viz-badge{font-weight:500}.tabular-nums{font-variant-numeric:tabular-nums}.btn-primary{font-weight:500}
</style>'''
        args.standalone.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Drone control lab — simulation</title>'+theme+'</head><body>'+fragment+'</body></html>\n')
    if not args.fragment and not args.standalone:parser.error('select --fragment or --standalone')


if __name__=='__main__':main()
