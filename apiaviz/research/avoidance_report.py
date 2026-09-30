"""Report all completed avoidance development outcomes, including regressions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from .study import write_json


def rows(path):
    return [json.loads(s) for s in (path/'results.jsonl').read_text().splitlines()]


def report():
    out=Path('docs/visual-avoidance')
    main=Path('apiaviz/output/visual-avoidance-smoke-v2')
    baseline=Path('apiaviz/output/familiarity-controller')
    old={r['id']:r for r in rows(baseline)}
    p=json.loads((main/'protocol.json').read_text())
    env=Path(p['worlds'][0]['environment'])
    base=json.loads((env/'protocol.json').read_text())
    route=np.asarray(base['route'])
    rocks=json.loads((env/'world.json').read_text())['obstacles']
    fig,axes=plt.subplots(2,3,figsize=(11,9),sharex=True,sharey=True)
    names={'linear_colour':'ApiaViz','sobel_colour':'Sobel + colour','ardin_input':'Ardin'}
    main_rows=rows(main)
    lines=['## Completed route results','',
        'All comparisons below use the same 91 teaching images, frozen encoder parameters and mushroom-body memories. These are development results from one scene, one wiring seed and one initial controller phase. They do not establish statistical significance.','',
        '| Input | Release | Original outcome | Avoidance outcome | Walked (m) | Returned to route | Views |',
        '|---|---|---|---|---:|---|---:|']
    for r in main_rows:
        b=old[r['id']]
        for k in ('encoder_fingerprint','memory_fingerprint','training_images_sha256'):
            assert r[k] == b[k]
        lines.append(f"| {names[r['method']]} | {r['scenario']} | {b['termination']} | {r['termination']} | {r['path_length_m']:.2f} | {('yes' if r['recovered'] else 'no') if r['recovery_applicable'] else '—'} | {r['observations']} |")
        ax=axes[['aligned','left50'].index(r['scenario']),p['methods'].index(r['method'])]
        ax.set_aspect('equal')
        for rock in rocks:
            ax.add_patch(Circle((rock['x'],rock['y']),rock['conservative_radius_m'],color='#c6c3b9',alpha=.6))
        ax.plot(*route.T,'--',color='#6b7278',lw=1,label='Teaching route')
        angle=np.deg2rad(base['headings'][0])
        start=route[0]+(.5 if r['scenario']=='left50' else 0)*np.array([-np.sin(angle),np.cos(angle)])
        old_trace=json.loads((baseline/b['trace']).read_text())['trajectory']
        oldpath=np.array([start]+[t['position'] for t in old_trace])
        ax.plot(*oldpath.T,color='#d48a2b',lw=1.3,label='Original controller')
        detail=json.loads((main/r['trace']).read_text())
        newpath=np.array([start]+[t['position'] for t in detail['microtrace']])
        ax.plot(*newpath.T,color='#137c80',lw=1.1,label='With visual avoidance')
        ax.scatter(*start,color='#343c46',s=16,zorder=4)
        ax.scatter(*newpath[-1],marker='x',color='#137c80',s=30,zorder=4)
        ax.scatter(*route[-1],marker='*',color='#343c46',s=55,zorder=4)
        ax.set_xlim(0,10); ax.set_ylim(0,10)
        ax.set_title(f"{names[r['method']]} · {r['scenario']}\n{r['termination'].replace('_',' ')}",fontsize=10)
        ax.tick_params(labelsize=8)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Avoiding rocks does not guarantee arrival',fontsize=16)
    fig.tight_layout(rect=[0,.04,1,.96])
    fig.savefig(out/'route-results.png',dpi=160)
    plt.close(fig)
    lines += ['', '![All six route trajectories](route-results.png)', '',
        'The reflex can remove an immediate collision without solving homing. It also increases visual sampling and turning costs, and its detours change the views encountered by the route controller. A return to the route means three successive macro-step endpoints within 10 cm; it does not imply a later arrival.','',
        'Two additional development checks are retained rather than discarded:','',
        '| Variant | Input/release | Outcome | Walked (m) | Final nest distance (m) |',
        '|---|---|---|---:|---:|']
    all_variants=[]
    for name,path in [('Preserve motor state (rejected)',Path('apiaviz/output/visual-avoidance-continuity')),
                      ('8 cm lookahead',Path('apiaviz/output/visual-avoidance-near'))]:
        for r in rows(path):
            lines.append(f"| {name} | {names[r['method']]} / {r['scenario']} | {r['termination']} | {r['path_length_m']:.2f} | {r['final_nest_distance_m']:.2f} |")
            all_variants.append(dict(variant=name,**r))
    lines += ['', 'The motor-state variant was stopped after exposing a search-state problem; its incomplete protocol is marked rejected. The shorter-lookahead protocol was limited to two ApiaViz trials before running it. Neither should be pooled with the six reference trials or the earlier 181-trial study.','']
    text=(out/'README.md').read_text().split('## Completed route results')[0]
    (out/'README.md').write_text(text+'\n'+'\n'.join(lines))
    write_json(out/'results.json',dict(reference=main_rows,variants=all_variants,
        original_arrivals=sum(old[r['id']]['reached_nest'] for r in main_rows),
        avoidance_arrivals=sum(r['reached_nest'] for r in main_rows),
        reference_rock_collisions=sum(r['termination']=='rock_collision' for r in main_rows)))
    print(f'Reported {len(main_rows)} reference and {len(all_variants)} additional completed trials')


if __name__ == '__main__':
    report()
