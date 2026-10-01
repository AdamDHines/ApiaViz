"""Presentation figures from the immutable compact graded-UV result table."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from apiaviz.research import uv_trials as base
from apiaviz.research.spectral_input import file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/apiaviz-graded-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((a.report/'results.json').read_text())['summary']
    cases=[(False,'pairwise','Fixed, pairwise'),(False,'combined','Fixed, combined'),(True,'pairwise','Adaptive, pairwise'),(True,'combined','Adaptive, combined')]
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True,layout='constrained')
    for ax,gain in zip(axes,[.25,1.,4.]):
        for adaptive,mode,label in cases:
            rs=[next(r for r in summary if r['world']=='all' and r['kind']=='all' and r['adaptive']==adaptive and r['mode']==mode and r['readout']==read and r['gain']==gain) for read in ['graded','spikes']]
            ax.plot([0,1],[r['heading_error_mean'] for r in rs],'o-',label=label)
        ax.set(title=f'{gain:g} × radiance',xticks=[0,1],xticklabels=['Graded currents','Spike identities'],ylim=(0,20));ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean heading error (degrees)');axes[-1].legend(fontsize=8)
    fig.suptitle('Equal stream weights in both readouts • 84 positions × 3 seeds × 2 scan directions')
    fig.savefig(a.report/'factorial-comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for adaptive,mode,label in cases:
        rs=[next(r for r in summary if r['world']=='all' and r['kind']=='all' and r['adaptive']==adaptive and r['mode']==mode and r['readout']=='spikes' and r['gain']==gain) for gain in [.25,1.,4.]]
        for ax,field in zip(axes,['illumination_heading_change_mean','scan_direction_change_mean']):
            ax.plot([.25,1.,4.],[r[field] for r in rs],'o-',label=label)
            ax.set_xscale('log',base=4);ax.set_xticks([.25,1.,4.],['0.25','1','4']);ax.set_xlabel('Radiance multiplier')
    axes[0].set_ylabel('Change from normal lighting (degrees)');axes[1].set_ylabel('Left/right scan disagreement (degrees)')
    axes[1].legend(fontsize=8);fig.suptitle('Heading stability • equal-stream spike readout')
    fig.savefig(a.report/'scan-stability.png',dpi=160);plt.close(fig)
    base.atomic_json(a.report/'presentation.json',dict(script_sha256=file_sha(Path(__file__)),
        results_sha256=file_sha(a.report/'results.json'),
        figures={n:file_sha(a.report/n) for n in ['factorial-comparison.png','scan-stability.png']}))


if __name__=='__main__':main()
