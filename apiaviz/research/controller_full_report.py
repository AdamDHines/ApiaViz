"""Paired analysis of the expanded, fixed-controller development matrix."""
import argparse
from collections import Counter, defaultdict
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from .study import file_hash, write_json

METHODS = ['linear_colour','sobel_colour','ardin_input']
NAMES = dict(linear_colour='ApiaViz',sobel_colour='Sobel + colour',ardin_input='Ardin-style')
POLICIES = ['exhaustive','active','familiarity']
LABELS = dict(exhaustive='Exhaustive scan',active='Previous active',familiarity='New controller')
COLORS = dict(exhaustive='#637b93',active='#cf934d',familiarity='#168e88')
DISPLACED = {'left50','right50','kick_left50','kick_right50'}


def world_interval(values):
    """World-level bootstrap; exact enumeration for this three-world study.

    Percentiles use the empirical inverse CDF, without interpolating between
    unattainable values. Three worlds support descriptive intervals only.
    """
    values=np.asarray(values,dtype=float)
    if not len(values):return dict(mean=None,low=None,high=None,worlds=0)
    if len(values)>6:raise ValueError('Exact enumeration is for small world counts')
    samples=np.array([np.mean(values[list(indices)]) for indices in itertools.product(range(len(values)),repeat=len(values))])
    low,high=np.quantile(samples,[.025,.975],method='inverted_cdf')
    return dict(mean=float(values.mean()),low=float(low),high=float(high),worlds=len(values))


def paired_effect(rows, metric, left, right):
    """Pair before aggregation; average phases, then cases within each world.

    left/right are (controller, method-or-None). For controller comparisons,
    method remains part of each matched case. For encoder comparisons, it does
    not. Missing pairs are excluded explicitly and their count is returned.
    """
    if (left[1] is None)!=(right[1] is None):raise ValueError('Incompatible comparison')
    cells=[defaultdict(list),defaultdict(list)]
    for i,(controller,method) in enumerate((left,right)):
        for r in rows:
            if r['controller']!=controller or (method is not None and r['method']!=method):continue
            key=(r['world'],r['seed'],r['scenario'])+((r['method'],) if method is None else ())
            if r[metric] is not None:cells[i][key].append(float(r[metric]))
    keys=set(cells[0])&set(cells[1])
    world=defaultdict(list)
    for key in sorted(keys):
        world[key[0]].append(float(np.mean(cells[0][key])-np.mean(cells[1][key])))
    means={name:float(np.mean(values)) for name,values in sorted(world.items())}
    return dict(metric=metric,left=left,right=right,paired_cases=len(keys),
        missing_pairs=len(set(cells[0])^set(cells[1])),by_world=means,**world_interval(list(means.values())))


def invalid_blocks(rows):
    """Exclude the complete paired block, not only the policy that was kicked."""
    invalid={'displacement_into_rock','displacement_outside_field','invalid_release_rock','invalid_release_bounds'}
    return {(r['world'],r['seed'],r['scenario']) for r in rows if r['termination'] in invalid}


def retention(trace, scenario, kick_before_step=36, consecutive=3):
    if scenario not in DISPLACED:return dict(return_found=False,lost_after_return=False)
    first=kick_before_step if scenario.startswith('kick_') else 1
    after=[t for t in trace if t['step']>=first]
    streak=0;returned=None
    for i,t in enumerate(after):
        streak=streak+1 if t['polyline_m']<=.1 else 0
        if streak>=consecutive:
            returned=i;break
    if returned is None:return dict(return_found=False,lost_after_return=False)
    streak=0;lost=False
    for t in after[returned+1:]:
        streak=streak+1 if t['polyline_m']>.2 else 0
        if streak>=consecutive:lost=True;break
    return dict(return_found=True,lost_after_return=lost)


def enrich(out,p,rows):
    expected={(w['name'],seed,m,s['name'],c['name'],c['phase'])
        for w in p['worlds'] for seed in p['seeds'] for m in p['methods']
        for s in p['scenarios'] for c in p['controllers']}
    actual={(r['world'],r['seed'],r['method'],r['scenario'],r['controller'],r['phase']) for r in rows}
    assert actual==expected and len(rows)==len(expected)
    enriched=[]
    for row in rows:
        saved=json.loads((out/row['trace']).read_text())
        trace=saved['trajectory'];distance=[t['polyline_m'] for t in trace]
        kept=retention(trace,row['scenario'],p['evaluation']['kick_before_step'])
        assert kept['return_found']==row['recovered'], row['id']
        actions={t['step'] for t in trace}
        states=Counter(d.get('state',d['state_before']) for d in saved['decisions'] if d['step'] in actions)
        enriched.append(dict(row,**kept,
            deviation95_m=float(np.quantile(distance,.95)) if distance else None,
            observations_per_m=row['observations']/row['path_length_m'] if row['path_length_m'] else None,
            returned_then_arrived=bool(row['recovered'] and row['reached_nest']),
            followed_moves=states['follow'],cast_moves=states['cast'],
            phase_value=row['phase'] if row['controller']=='familiarity' else None))
    return enriched


def groups(rows):
    result=[]
    for m in METHODS:
        for c in POLICIES:
            for s in sorted({r['scenario'] for r in rows}):
                selected=[r for r in rows if (r['method'],r['controller'],r['scenario'])==(m,c,s)]
                if not selected:continue
                errors=[r['polyline_mean_m'] for r in selected if r['polyline_mean_m'] is not None]
                result.append(dict(method=m,controller=c,scenario=s,n=len(selected),
                    arrivals=sum(r['reached_nest'] for r in selected),returns=sum(r['recovered'] for r in selected),
                    lost_after_return=sum(r['lost_after_return'] for r in selected),
                    returned_then_arrived=sum(r['returned_then_arrived'] for r in selected),
                    zero_move_trials=sum(r['steps']==0 for r in selected),
                    mean_deviation_m=float(np.mean(errors)) if errors else None,
                    deviation_n=len(errors),median_observations=float(np.median([r['observations'] for r in selected])),
                    median_time_s=float(np.median([r['time_s'] for r in selected])),
                    terminations=dict(Counter(r['termination'] for r in selected))))
    return result


def comparisons(rows):
    results=[]
    for population,subset in [('all',rows),('displaced',[r for r in rows if r['scenario'] in DISPLACED]),
                               ('aligned',[r for r in rows if r['scenario']=='aligned'])]:
        metrics=['reached_nest']+(['recovered'] if population=='displaced' else [])
        for metric in metrics:
            for baseline in ('active','exhaustive'):
                results.append(dict(population=population,comparison='controller',
                    **paired_effect(subset,metric,('familiarity',None),(baseline,None))))
            for method in ('sobel_colour','ardin_input'):
                results.append(dict(population=population,comparison='preprocessing',
                    **paired_effect(subset,metric,('familiarity','linear_colour'),('familiarity',method))))
    return results


def figures(docs,p,rows,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scenarios=[s['name'] for s in p['scenarios']]
    titles=['Aligned','Left 50 cm','Right 50 cm','Heading +40°','Heading −40°','Kick left','Kick right']
    fig,axes=plt.subplots(3,1,figsize=(12,8),layout='constrained')
    for ax,method in zip(axes,METHODS):
        values=np.zeros((3,len(scenarios)));counts=[]
        for i,c in enumerate(POLICIES):
            cc=[]
            for j,s in enumerate(scenarios):
                g=next(g for g in summary if (g['method'],g['controller'],g['scenario'])==(method,c,s))
                values[i,j]=g['arrivals']/g['n'];cc.append(f"{g['arrivals']}/{g['n']}")
            counts.append(cc)
        ax.imshow(values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        for i in range(3):
            for j in range(len(scenarios)):
                ax.text(j,i,counts[i][j],ha='center',va='center',color='white' if values[i,j]>.6 else '#20313c')
        ax.set(xticks=range(len(scenarios)),xticklabels=titles,yticks=range(3),
               yticklabels=[LABELS[c] for c in POLICIES],title=NAMES[method])
    fig.suptitle('Endpoint arrivals across three worlds and three wiring seeds\nNew controller includes both initial search phases; trials share environments',fontsize=12)
    fig.savefig(docs/'arrivals.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained')
    for ax,method in zip(axes,METHODS):
        for c in POLICIES:
            selected=[r for r in rows if r['method']==method and r['controller']==c and r['scenario'] in DISPLACED]
            times=np.arange(0,361,2)
            recovered=[r['recovery_time_s'] for r in selected if r['recovered']]
            ax.step(times,[sum(t<=now for t in recovered)/len(selected) for now in times],
                    where='post',label=LABELS[c],color=COLORS[c])
        ax.set(title=NAMES[method],xlabel='Seconds since release or displacement',ylabel='Fraction returning to route',ylim=(0,1))
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle('Sustained return within 10 cm · intended displacement trials\nCollisions, missed kicks and other failures remain in the denominator',fontsize=11)
    fig.savefig(docs/'recovery.png',dpi=170);plt.close(fig)

    outcomes=['arrival','rock_collision','field_boundary','uninformative_views','time_budget','step_budget','displacement_into_rock','displacement_outside_field']
    colors=['#168e88','#b85d50','#dcaa50','#7184b2','#9f80a5','#a5a7a7','#6b4440','#776340']
    extra=sorted({r['termination'] for r in rows}-set(outcomes))
    outcomes+=extra;colors+=['#454545']*len(extra)
    fig,axes=plt.subplots(1,3,figsize=(12,5),layout='constrained')
    for ax,method in zip(axes,METHODS):
        bottom=np.zeros(3)
        for outcome,color in zip(outcomes,colors):
            fractions=[]
            for c in POLICIES:
                rs=[r for r in rows if r['method']==method and r['controller']==c]
                fractions.append(sum(r['termination']==outcome for r in rs)/len(rs))
            ax.bar(range(3),fractions,bottom=bottom,label=outcome.replace('_',' '),color=color)
            bottom+=fractions
        ax.set(title=NAMES[method],xticks=range(3),xticklabels=['Full scan','Old active','New'],ylim=(0,1),ylabel='Fraction of all trials')
    axes[-1].legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
    fig.suptitle('How trials end · all predeclared scenarios, no failure exclusions')
    fig.savefig(docs/'terminations.png',dpi=170);plt.close(fig)


def run(out,docs):
    from .controller_report import audit
    p=json.loads((out/'protocol.json').read_text())
    rows=[json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    rows=enrich(out,p,rows)
    docs.mkdir(parents=True,exist_ok=True)
    print('Auditing trajectories, budgets, memories and sampled queries',flush=True)
    checked=audit(out)
    imports=json.loads((out/'imports.json').read_text())
    for trial in imports['trials']:
        row=next(r for r in rows if r['id']==trial['id'])
        assert file_hash(out/row['trace'])==trial['trace_sha256']
    checked['imported_trials_exact']=len(imports['trials'])
    write_json(docs/'audit.json',checked)
    write_json(docs/'protocol.json',p)
    with (docs/'trials.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    blocked=invalid_blocks(rows)
    valid=[r for r in rows if (r['world'],r['seed'],r['scenario']) not in blocked]
    summary=groups(rows)
    result=dict(trials=len(rows),groups=summary,comparisons=comparisons(rows),
        sensitivity=dict(excluded_blocks=sorted(blocked),remaining_trials=len(valid),
                         groups=groups(valid),comparisons=comparisons(valid)),
        phases=[dict(method=m,phase=phase,n=len(rs),arrivals=sum(r['reached_nest'] for r in rs),returns=sum(r['recovered'] for r in rs))
                for m in METHODS for phase in (-1,1)
                for rs in [[r for r in rows if r['method']==m and r['controller']=='familiarity' and r['phase']==phase]]],
        phase_comparisons={str(phase):comparisons([r for r in rows if r['controller']!='familiarity' or r['phase']==phase]) for phase in (-1,1)},
        report_source_sha256=file_hash(Path(__file__)),
        inference='World-level descriptive intervals only. Three worlds do not support a strong population-level significance claim; seeds and phases are technical repeats.')
    write_json(docs/'summary.json',result)
    figures(docs,p,rows,summary)
    lines=['# Expanded controller experiments','',
        'The controller, teaching set and visual encoders were held fixed throughout this 756-trial matrix. '
        'The three saved worlds were tested with three wiring seeds, all three front ends, seven scenarios, '
        'and four controller conditions. The new controller includes two initial search phases. '
        'Seventy-two verified smoke trials were imported; 684 additional trials were run.','',
        '## Endpoint arrival and route recovery','',
        '| Front end | Controller | Arrivals, all cases | Returns after displacement | Lost again after return |',
        '|---|---|---:|---:|---:|']
    for m in METHODS:
        for c in POLICIES:
            allrows=[r for r in rows if r['method']==m and r['controller']==c]
            rs=[r for r in allrows if r['scenario'] in DISPLACED]
            nreturn=sum(r['recovered'] for r in rs)
            lines.append(f"| {NAMES[m]} | {LABELS[c]} | {sum(r['reached_nest'] for r in allrows)}/{len(allrows)} | {nreturn}/{len(rs)} | {sum(r['lost_after_return'] for r in rs)}/{nreturn} |")
    lines+=['','A return requires three consecutive positions within 10 cm of the route. Renewed loss requires '
        'three later consecutive positions more than 20 cm away. Endpoint arrival is a separate outcome. '
        'Trials that fail before a scheduled kick remain failures in the intended-displacement analysis.','',
        '![Arrival rates](arrivals.png)','','![Recovery over time](recovery.png)','','![Trial endings](terminations.png)','',
        '## Paired comparisons','',
        'Both new-controller phases are averaged within each world, seed, method and scenario before comparison. '
        'Case differences are then averaged within worlds. The intervals below enumerate all 27 bootstrap '
        'resamples of the three worlds; they describe variation among these worlds and do not establish '
        'population-level statistical significance. More wiring seeds do not increase the number of independent environments.','',
        '| Population | Outcome | New minus baseline | Difference | Descriptive 95% interval |',
        '|---|---|---|---:|---:|']
    for comparison in result['comparisons']:
        if comparison['comparison']!='controller':continue
        lines.append(f"| {comparison['population']} | {comparison['metric']} | {LABELS[comparison['right'][0]]} | {comparison['mean']*100:+.1f} pp | [{comparison['low']*100:+.1f}, {comparison['high']*100:+.1f}] pp |")
    lines+=['','The [complete summary](summary.json) also includes paired front-end comparisons, world-level '
        'effects, initial-phase sensitivity and a paired sensitivity analysis of invalid imposed displacements. '
        f"That analysis excludes {len(blocked)} world/seed/scenario blocks across every method and controller, "
        f"leaving {len(valid)} trials. Such exclusions depend on the trajectories and are not the primary result. "
        'Ordinary collisions, including collisions on the first attempted movement, remain failures.','',
        '## Reproducibility','',
        f"The [audit](audit.json) checked {checked['observations']:,} observations and "
        f"{checked['matched_gaze_comparisons']:,} matched-gaze movement comparisons. "
        f"All {checked['original_baselines_exact']} overlapping original-baseline trials reproduced exactly; "
        f"all {checked['imported_trials_exact']} imported smoke traces retained their hashes. "
        f"Memories were rebuilt and {checked['independently_reencoded_scores']:,} selected scores were re-encoded exactly.",
        '', 'The [trial table](trials.csv) includes deviation, time, observations, phase, termination and retention. '
        'Deviation is undefined for zero-movement trials; low deviation or low sensing cost after an early failure '
        'must not be interpreted as successful efficient navigation.','',
        '```sh',
        'MPLCONFIGDIR=/tmp/apiaviz-mpl .pixi/envs/default/bin/python -m apiaviz.research.controller_sweep prepare --output apiaviz/output/controller-full-reproduction',
        '# Start the three saved Blender workers with --resume, as documented in ../familiarity-controller/README.md.',
        'MPLCONFIGDIR=/tmp/apiaviz-mpl .pixi/envs/default/bin/python -m apiaviz.research.controller_sweep run --output apiaviz/output/controller-full-reproduction',
        'MPLCONFIGDIR=/tmp/apiaviz-mpl .pixi/envs/default/bin/python -m apiaviz.research.controller_full_report --output apiaviz/output/controller-full-reproduction --docs docs/controller-full-reproduction',
        '```','',
        'This expands controller testing only. Teaching spacing, retinal resolution, lighting and scene geometry '
        'were not varied, and the original question about training density remains a separate experiment.']
    (docs/'README.md').write_text('\n'.join(lines)+'\n')
    print('Report complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('apiaviz/output/controller-full'))
    parser.add_argument('--docs',type=Path,default=Path('docs/controller-full'))
    args=parser.parse_args();run(args.output,args.docs)
