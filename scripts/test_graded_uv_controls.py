"""Frozen graded adaptation × opponency × readout experiment on verified images."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from apiaviz.research import uv_trials as base
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.graded_uv import AdaptationConfig, GradedUVEncoder, initialize, advance, response_sequence, opponent_maps
from apiaviz.research.study import encode, fingerprint
from apiaviz.research.spectral_input import file_sha
from apiaviz.research.mechanisms import polyline_distance
from visual_response_controls import select_poses, winner

SEEDS=[19,31,43]
GAINS=[.25,1.,4.]
DT=.05+5/180


def tensor_sha(x):return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()


def pooled(model,response,mode):
    chunks=[model.pooled(response[i:i+16],mode) for i in range(0,len(response),16)]
    return [torch.cat([c[j] for c in chunks]) for j in range(3)]


def summarize(matrix,angles,pose,route,anchor):
    scores,stations=matrix.max(1)
    i=winner(scores,angles,anchor);heading=float(angles[i])
    pos=np.asarray(pose['position']);v=np.array([np.cos(np.deg2rad(heading)),np.sin(np.deg2rad(heading))])
    before,after=polyline_distance([pos,pos+.1*v],route)
    # Forward progress is a geometric diagnostic only, not policy input.
    tangent=np.array([np.cos(np.deg2rad(pose['tangent'])),np.sin(np.deg2rad(pose['tangent']))])
    near=int(np.argmin(abs((angles-pose['tangent']+180)%360-180)))
    return dict(heading=heading,heading_error=abs((heading-pose['tangent']+180)%360-180),
        matched_station=int(stations[i]),score=float(scores[i]),
        correct_station_score=float(matrix[near,pose['station']]),
        forward_projection_m=float(.1*v@tangent),route_distance_change_m=float(after-before),
        scores=scores.tolist(),stations=stations.tolist())


def stimulus_check(out):
    initial=torch.ones(1,3,8,8,dtype=torch.float64);rows=[]
    for gain_tau in [.1,1.,10.]:
        for integration_tau in [.01,.02,.04]:
            config=AdaptationConfig(gain_tau_s=gain_tau,integration_tau_s=integration_tau)
            for gain in GAINS:
                state=initialize(initial);sequence=[]
                for i in range(400):
                    y,state=advance(initial*gain,state,.01,config)
                    sequence.append([float(y.mean()),*opponent_maps(y,'combined').mean((0,2,3)).tolist()])
                rows.append(dict(config=asdict(config),gain=gain,dt=.01,response=sequence))
    # Explicitly check the declared steady-state invariance, not fitted physiology.
    pattern=initial.clone();pattern[:,:,:,4:]*=2
    reference,_=advance(pattern,initialize(pattern),1.)
    errors=[]
    for gain in GAINS:
        value,_=advance(pattern*gain,initialize(pattern*gain),1.)
        errors.append(float((reference-value).abs().max()))
    assert max(errors)<1e-12
    colour=[]
    for receptor in range(3):
        state=initialize(initial)
        x=initial.clone();x[:,receptor]*=2
        values=[]
        for i in range(400):
            y,state=advance(x,state,.01)
            values.append(opponent_maps(y,'combined').mean((0,2,3)).tolist())
        colour.append(dict(stimulated_receptor=['uv','blue','green'][receptor],response=values))
    base.atomic_json(out/'stimuli.json',dict(step_responses=rows,colour_steps=colour,
        steady_achromatic_scale_errors=errors,limitation='Receptor-excitation stimuli, not monochromatic radiance or measured voltage traces. Equations are tested, not fitted to physiology.'))


def job(source_string,out_string,w):
    torch.set_num_threads(2)
    source,out=Path(source_string),Path(out_string)
    parent=json.loads((source/'protocol.json').read_text());study=Path(parent['study'])
    p=json.loads((study/'protocol.json').read_text());env=study/'worlds'/w['name']
    camera=DualCamera(source/w['name'],json.loads((env/'render.json').read_text()))
    bank=torch.load(env/'teaching.pt',weights_only=True)
    models={seed:GradedUVEncoder(seed) for seed in SEEDS}
    references={seed:base.make_model(p,'apiaviz_uv',seed) for seed in SEEDS}
    frozen={str(s):fingerprint(m) for s,m in models.items()}
    memory={};memory_hashes={};adaptation_records=[]
    for adaptive in [False,True]:
        responses,state=response_sequence(bank['uv'][:,None],bank['uv'][:1],1.,adaptive=adaptive)
        responses=responses[:,0]
        adaptation_records.append(dict(stage='teaching',adaptive=adaptive,
            response_sha256=tensor_sha(responses),final_background=state.background.flatten().tolist()))
        for mode in ['pairwise','combined']:
            features=pooled(models[19],responses,mode)
            for seed,m in models.items():
                for readout,c in m.readouts(features).items():
                    key=(adaptive,mode,seed,readout);memory[key]=c
                    memory_hashes[str(key)]=tensor_sha(c)
    for seed,m in references.items():
        memory[('current',seed)]=F.normalize((encode(m,bank['uv'])>0).float(),dim=1)
        memory_hashes[str(('current',seed))]=tensor_sha(memory['current',seed])
    rows=[];poses=select_poses(w);anchor=float(w['headings'][0]);route=np.asarray(w['route'])
    for pose in tqdm(poses,desc=w['name']+' graded probes'):
        initial=camera.scan(pose['position'],[anchor],uv=True)
        for direction in [-1,1]:
            angles=anchor+direction*np.arange(0.,360.,5.)
            raw=camera.scan(pose['position'],angles,uv=True)
            for gain in GAINS:
                for adaptive in [False,True]:
                    response,state=response_sequence(raw[:,None]*gain,initial,DT,adaptive=adaptive)
                    response=response[:,0]
                    adaptation_records.append(dict(stage='query',station=pose['station'],kind=pose['kind'],direction=direction,
                        gain=gain,adaptive=adaptive,response_sha256=tensor_sha(response),
                        initial_background=initial.mean((2,3)).flatten().tolist(),
                        final_background=state.background.flatten().tolist()))
                    for mode in ['pairwise','combined']:
                        features=pooled(models[19],response,mode)
                        for seed,m in models.items():
                            for readout,query in m.readouts(features).items():
                                matrix=query@memory[adaptive,mode,seed,readout].T
                                rows.append(dict(world=w['name'],**pose,seed=seed,adaptive=adaptive,mode=mode,readout=readout,
                                    direction=direction,gain=gain,**summarize(matrix,angles,pose,route,anchor)))
                # Original production model is a separate reference, not a factorial cell.
                if direction==1:
                    for seed,m in references.items():
                        c=F.normalize((encode(m,raw*gain)>0).float(),dim=1)
                        rows.append(dict(world=w['name'],**pose,seed=seed,adaptive=False,mode='current',readout='production_spikes',
                            direction=direction,gain=gain,**summarize(c@memory['current',seed].T,angles,pose,route,anchor)))
    assert frozen=={str(s):fingerprint(m) for s,m in models.items()}
    for key,c in memory.items():assert tensor_sha(c)==memory_hashes[str(key)]
    base.atomic_json(out/f'{w["name"]}.json',dict(records=rows,adaptation=adaptation_records,
        raw_records=camera.references,teaching_sha256=file_sha(env/'teaching.pt'),memory_sha256=memory_hashes,encoder_fingerprints=frozen))
    return w['name']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'apiaviz/output/visual-response-controls-v1')
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    parent=json.loads((a.source/'protocol.json').read_text());study=Path(parent['study'])
    p=json.loads((study/'protocol.json').read_text());base.verify_sources(p)
    protocol=dict(schema='graded-uv-controls-v1',source=str(a.source.resolve()),source_protocol_sha256=file_sha(a.source/'protocol.json'),
        source_sha256=base.sources(),adaptation=asdict(AdaptationConfig()),seeds=SEEDS,gains=GAINS,
        factorial=dict(adaptive=[False,True],opponency=['pairwise','combined'],readout=['graded','spikes','production_spikes'],uv_stream=['on','off']),
        primary_readout_comparison='Graded currents versus binary spike identities; each stream normalized to equal norm in BOTH. Production global binary normalization is a separate control.',
        receptor_model='Fast exponential irradiance state Z; slow exponential retinal-mean background A independently per UV/B/G receptor class; response Z/(Z+max(A,1e-4)). Broad-field adaptation approximation; not a receptor-local voltage model.',
        opponent_model='pairwise: U-B and U-G; combined: U-(B+G)/2 and B-G. Each axis split into positive/negative planes. Coefficients fixed, not fitted physiology.',
        common_frontend='All factorial cells use angular spatial filters including UV; stable DC subtraction; same 4000/4000/2000 wiring. This differs from historical production, retained as a separate reference.',
        teaching='Chronological stored route images, held 1s per 10cm segment; initial receptor state equilibrated to first image. Full history retained. Snapshot/hold approximation to traversing the route.',
        recall=dict(initial='Explicit equilibrium at same-pose initial-world-heading image under original lighting. No route tangent supplied.',
            lighting='Multiply linear UV/B/G radiances by common gain at start of scan. Spectrum, sun, shadows, geometry unchanged.',
            angles='72 successive 5deg headings, starting at initial-world heading, in both directions.',
            seconds_per_view=DT,scan_duration_s=72*DT,
            approximation='Hold each acquired view through nominal 50ms observation plus 5deg/180deg-per-second turn allowance; no interpolated images during turns. This is a controlled exposure schedule, not executed locomotion.'),
        physiology_sources=['https://doi.org/10.1371/journal.pone.0025989','https://doi.org/10.1371/journal.pone.0310282'],
        parameter_status='20ms integration and 1s gain are engineering hypotheses. Published impulse half-widths are NOT equated to these time constants. Stimulus checks include 10/20/40ms and .1/1/10s, without selecting a winner.',
        limitations=['Offline direction/retrieval test, not a navigation-success result.','No precision retracing requirement; route-distance change is diagnostic only.',
            'Three inspected development worlds. Positions, seeds, and scan directions are correlated.',
            'Broad-field receptor-class adaptation is not empirically fitted local photoreceptor physiology.',
            'No added UV to Sobel or Ardin; these are within-ApiaViz controls, not a new model-ranking study.'])
    base.atomic_json(out/'protocol.json',protocol)
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in protocol['source_sha256']:archive.write(ROOT/name,name)
    stimulus_check(out)
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(job,str(a.source.resolve()),str(out),w) for w in p['worlds']]
        for future in futures:print('Completed',future.result(),flush=True)
    base.atomic_json(out/'complete.json',dict(protocol_sha256=file_sha(out/'protocol.json'),
        results={w['name']:file_sha(out/f'{w["name"]}.json') for w in p['worlds']},stimuli_sha256=file_sha(out/'stimuli.json')))


if __name__=='__main__':main()
