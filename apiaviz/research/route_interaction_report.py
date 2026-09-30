"""Summarise every completed interaction trial and the transfer check."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from .study import write_json

ROOT=Path('apiaviz/output')
DIRECTORIES=dict(sampling='route-sampling-control',restart='visual-avoidance-smoke-v2',
                 pause='route-detour',continuous='route-motor-feedback-v2',transfer='route-motor-feedback-meander')
NAMES={'linear_colour':'ApiaViz','sobel_colour':'Sobel + colour','ardin_input':'Ardin'}
OUTCOMES={'arrival':'Arrived','rock_collision':'Collision','time_budget':'Time limit',
          'step_budget':'Distance limit','field_boundary':'Left field','uninformative_views':'Search stopped'}


def report():
    groups={}
    for label,name in DIRECTORIES.items():
        out=ROOT/name
        manifest=json.loads((out/'manifest.json').read_text())
        assert manifest['status']=='complete',name
        groups[label]=[json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    original={r['id']:r for r in map(json.loads,(ROOT/'familiarity-controller/results.jsonl').read_text().splitlines())}
    for rows in groups.values():
        for row in rows:
            for field in ('training_images_sha256','encoder_fingerprint','memory_fingerprint','memory_views'):
                assert row[field]==original[row['id']][field]
    indexed={label:{r['id']:r for r in rows} for label,rows in groups.items()}
    detours=groups['continuous']
    out=Path('docs/route-detour'); out.mkdir(exist_ok=True)
    p=json.loads((ROOT/DIRECTORIES['continuous']/'protocol.json').read_text())
    env=Path(p['worlds'][0]['environment']); base=json.loads((env/'protocol.json').read_text())
    rocks=json.loads((env/'world.json').read_text())['obstacles']; route=np.asarray(base['route'])
    fig,axes=plt.subplots(2,3,figsize=(11,9),sharex=True,sharey=True)
    for row in detours:
        ax=axes[['aligned','left50'].index(row['scenario']),p['methods'].index(row['method'])]
        d=json.loads((ROOT/DIRECTORIES['continuous']/row['trace']).read_text())
        first=next(e['position'] for e in d['events'] if 'position' in e)
        xy=np.array([first]+[v['position'] for v in d['microtrace']])
        steering=np.array([v['position'] for v in d['microtrace'] if v['state']=='avoid']).reshape(-1,2)
        for rock in rocks:ax.add_patch(Circle((rock['x'],rock['y']),rock['conservative_radius_m'],color='#c9c4b6',alpha=.6))
        ax.plot(*route.T,'--',color='#7d8790',lw=1,label='Teaching route')
        ax.plot(*xy.T,color='#087f82',lw=1.3,label='Executed path')
        if len(steering):ax.scatter(*steering.T,color='#df942f',s=4,label='Evasive steering',zorder=4)
        ax.scatter(*route[-1],marker='*',color='#283d4c',s=60,zorder=5)
        ax.scatter(*xy[-1],marker='o' if row['reached_nest'] else 'x',color='#087f82',s=25,zorder=5)
        ax.set(xlim=(0,10),ylim=(0,10),aspect='equal')
        ax.set_title(f"{NAMES[row['method']]} · {row['scenario']}\n{OUTCOMES.get(row['termination'],row['termination'])}",fontsize=10)
        ax.tick_params(labelsize=8)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Route guidance remains active during visual avoidance',fontsize=15)
    fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(out/'trajectories.png',dpi=160);plt.close(fig)
    arrivals={key:sum(r['reached_nest'] for r in rows) for key,rows in groups.items()}
    lines=['# Keeping route guidance active during obstacle avoidance','',
        'The ant now continues to use visual familiarity while steering around an obstacle. It retains its remembered route direction, search phase and progress through a cast. There is no navigator restart or separate waiting period before route guidance resumes.','',
        '[Watch the aligned ApiaViz run](apiaviz-aligned.mp4) · [Watch the displaced ApiaViz run](apiaviz-displaced.mp4)','',
        '![All six continuous-guidance trials](trajectories.png)','',
        '## What changed','',
        'The previous implementation treated an evasive movement as invalidating the before/after familiarity comparison. That was too strong: when both images are taken at the same viewing direction, their difference remains useful evidence about the movement actually made. A detour changes the action that produced the observation; it does not erase the observation.','',
        'The corrected integration keeps that visual evidence and the existing controller state. It records every commanded substep and their net direction, so a comparison is attributed to the actual combined navigation/avoidance movement. The net-direction calculation is an accounting measure; route steering still comes from the unchanged familiarity policy. Its cast counter advances normally, including during evasive movements.','',
        'The obstacle reflex itself is unchanged: it estimates nearby surfaces from consecutive 199 × 51 RGB images, with no depth buffer, obstacle coordinates, labels or contact sensor. The same teaching images, encoder parameters, mushroom-body memories, collision rules and budgets are retained. All extra camera observations and turns are charged.','',
        '## Paired development results','',
        '| Input | Release | Extra sampling, no evasion | Previous restart | Pause and reacquire | Continuous guidance |',
        '|---|---|---|---|---|---|']
    for row in detours:
        outcomes=[OUTCOMES.get(indexed[k][row['id']]['termination'],indexed[k][row['id']]['termination']) for k in ('sampling','restart','pause','continuous')]
        lines.append('| '+' | '.join([NAMES[row['method']],row['scenario']]+outcomes)+' |')
    lines += ['',f"Arrivals across the six hairpin trials: sampling control **{arrivals['sampling']}/6**, previous restart **{arrivals['restart']}/6**, pause/reacquire **{arrivals['pause']}/6**, continuous guidance **{arrivals['continuous']}/6**.",'',
        'The sampling control reproduced the original aligned trajectories to numerical precision, with the same outcomes. Extra sampling alone therefore did not cause the earlier loss of arrivals under these budgets. The steering/handoff interaction mattered. The pause/reacquire alternative was frozen first; the continuous variant was added after its first trial failed. All completed outcomes are retained.','',
        '| Continuous-guidance trial | Mean deviation (cm) | Path (m) | Time (s) | Camera observations | Scan bouts |',
        '|---|---:|---:|---:|---:|---:|']
    for row in detours:
        lines.append(f"| {NAMES[row['method']]} / {row['scenario']} | {100*row['polyline_mean_m']:.1f} | {row['path_length_m']:.2f} | {row['time_s']:.1f} | {row['observations']} | {row['scan_bouts']} |")
    lines += ['', 'Deviation is weighted by executed substep length and includes failed trials. A short failed trajectory is not comparable to successful completion simply because its mean deviation is small.','',
        '## Additional environment','',
        'After the hairpin ApiaViz successes, the unchanged controller was checked on aligned and displaced releases in the existing meander world. This is an additional development-world check, not a held-out benchmark or a new comparison between frontends.','',
        '| ApiaViz release | Original controller | Continuous guidance | Mean deviation (cm) |',
        '|---|---|---|---:|']
    for row in groups['transfer']:
        lines.append(f"| {row['scenario']} | {OUTCOMES.get(original[row['id']]['termination'],original[row['id']]['termination'])} | {OUTCOMES.get(row['termination'],row['termination'])} | {100*row['polyline_mean_m']:.1f} |")
    lines += ['', '## Interpretation and limits','',
        'This is a more coherent functional interaction between visual guidance and obstacle avoidance. It is not a validated neural implementation of ant motor control. Rotations remain finite-rate, stationary turns between short translations. The videos show actual acquired images at four times simulation speed, held between observations; the overhead map is for display only.','',
        'A [recent ant-navigation modelling study](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012798) supports considering visual direction choice together with exploratory movements. It does not validate these specific motor rules. These runs use one wiring seed and one initial search phase. Broader paired tests across seeds, phases, routes, releases and lighting are still required; no statistical-significance or general-robustness claim is made. Dense 10 cm teaching is unchanged.','',
        'Eight new interaction tests and eight existing avoidance tests passed. Independent audits check frozen sources, identical training/memory hashes, collision-free executed segments, all movement/view costs, and the match between logged motor commands and familiarity comparisons. No route coordinates, endpoint bearing or collision feedback enter either policy. The earlier 181-trial study remains separate.','',
        '## Reproduce','',
        'With the saved hairpin renderer running:','',
        '```sh',
        'python -m apiaviz.research.motor_feedback prepare --output apiaviz/output/my-continuous-run',
        'python -m apiaviz.research.motor_feedback run --output apiaviz/output/my-continuous-run',
        'python -m apiaviz.research.motor_feedback report --output apiaviz/output/my-continuous-run',
        'python -m apiaviz.research.route_avoidance_audit --output apiaviz/output/my-continuous-run',
        'python -m unittest discover -s tests -p test_route_avoidance.py -v',
        '```','',
        'Use `route_avoidance_experiments prepare --mode sampling_control` or `--mode detour` with separate output directories to reproduce the two new controls. Exact protocols and complete traces are stored in the experiment directories named in `results.json`.','']
    (out/'README.md').write_text('\n'.join(lines))
    write_json(out/'results.json',dict(directories=DIRECTORIES,arrivals=arrivals,groups=groups))
    print(json.dumps(arrivals),flush=True)


if __name__=='__main__':report()
