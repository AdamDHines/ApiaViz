"""Paired grassland smoke test with the existing spiking navigation pipeline.

The Blender worker supplies RGB panoramas; this adapter samples the legacy ray
grid. All methods share acquisition, frozen wiring, spikes, memory and steering.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image
import torch

from apiaviz.mbant.config import NavigationConfig
from apiaviz.mbant.io_utils import load_world_data
from .frontend_refinements import RefinementEncoder, DEFINITIONS
from .mechanisms import code_statistics, polyline_distance
from .navigation import Scorer, evaluate_route
from .openloop import acquisition_views
from .spike_overlap import SpikeOverlapMemory
from .study import encode, fingerprint, file_hash, write_json

DEFAULT = Path('apiaviz/output/grassland-smoke')
METHODS = ['linear_colour','sobel_colour','ardin_input']


def make_route():
    """Smooth 7.8 m meander, sampled at .1 m, quantised as in prepare_route."""
    t=np.linspace(0,1,10001)
    dense=np.column_stack([6.3-1.1*t+.60*np.sin(2*np.pi*t)*np.sin(np.pi*t),8.45-7.4*t])
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(dense,axis=0),axis=1))]
    samples=np.arange(0,arc[-1],.1)
    reference=np.column_stack([np.interp(samples,arc,dense[:,axis]) for axis in (0,1)])
    d=np.diff(reference,axis=0)
    raw=np.degrees(np.arctan2(d[:,1],d[:,0]))
    headings=np.where(np.floor(raw).astype(int)%2==0,np.floor(raw),np.ceil(raw))
    positions=np.vstack([reference[0],reference[0]+np.cumsum(.1*np.column_stack([np.cos(np.radians(headings)),np.sin(np.radians(headings))]),axis=0)])
    return positions,headings


def prepare(out):
    if (out/'protocol.json').exists():
        raise FileExistsError(f'{out} already has a protocol; use a new directory')
    out.mkdir(parents=True,exist_ok=True)
    (out/'queue').mkdir()
    (out/'panoramas').mkdir()
    world_file=Path('apiaviz/mbant/data/antview/world5000_gray.mat')
    old=load_world_data(str(world_file))
    bounds=[float(old['X'].min()),float(old['X'].max()),float(old['Y'].min()),float(old['Y'].max())]
    positions,headings=make_route()
    protocol=dict(role='Smoke test on one generated route, three paired wiring seeds; no tuning or significance claim.',
                  geometry_seed=20260930, methods=METHODS, wiring_seeds=[19,31,43],
                  world_bounds_m=bounds, original_world_sha256=file_hash(world_file),
                  route=positions.tolist(),headings=headings.tolist(),route_length_m=len(headings)*.1,
                  route_generation='Fixed analytic meander; 10 cm steps, headings rounded to even degrees before any results.',
                  acquisition=dict(viewpoints=9,width=.2,lookahead=0.),max_steps=200,
                  renderer=dict(engine='Cycles',samples=32,seed=20260929,denoising=False,
                                panorama_size=[720,152],latitude_bounds_deg=[-15.5,60.5],
                                eye_height_m=.01,view_transform='AgX',exposure=.3),
                  sensory=dict(shape=[3,18,74],horizontal_rays_deg=[-148,148],vertical_rays_deg=[60,-15],
                               sampling='Bilinear samples of fixed-world 360-degree panorama at the legacy endpoint-inclusive ray grid; no heading-specific rendering noise.',
                               frontend_channels='G and B; red unused by all methods'),
                  nav=asdict(NavigationConfig(step_size=.1,scan_range=120,scan_step=10,dis_threshold=.2)),
                  definitions={m:DEFINITIONS[m] for m in METHODS})
    write_json(out/'protocol.json',protocol)
    print(json.dumps(dict(output=str(out),world_bounds_m=bounds,route_length_m=len(headings)*.1,teaching_views=len(headings)*9)),flush=True)


def position_key(position):
    return hashlib.sha256(np.asarray(position,dtype='<f8').tobytes()).hexdigest()[:24]


def sample_panorama(panorama, headings, shape=(18,74)):
    """Blender camera faces +X, image left +Y; legacy columns increase azimuth."""
    h,w=panorama.shape[:2]
    if len(shape)!=2 or any(int(n)!=n or n<2 for n in shape):
        raise ValueError('shape must contain integer height and width of at least two')
    angles=np.asarray(headings)[:,None]+np.linspace(-148,148,shape[1])[None]
    x=((180-angles)%360)/360*w-.5
    y=(60.5-np.linspace(60,-15,shape[0]))/76*h-.5
    x0=np.floor(x).astype(int); y0=np.floor(y).astype(int)
    dx=x-x0; dy=y-y0
    a=panorama[y0[None,:,None],(x0%w)[:,None,:]]
    b=panorama[y0[None,:,None],((x0+1)%w)[:,None,:]]
    c=panorama[(y0+1)[None,:,None],(x0%w)[:,None,:]]
    d=panorama[(y0+1)[None,:,None],((x0+1)%w)[:,None,:]]
    top=a*(1-dx[:,None,:,None])+b*dx[:,None,:,None]
    bottom=c*(1-dx[:,None,:,None])+d*dx[:,None,:,None]
    image=top*(1-dy[None,:,None,None])+bottom*dy[None,:,None,None]
    return torch.from_numpy(np.ascontiguousarray(image.transpose(0,3,1,2)/255,dtype=np.float32))


class GrasslandWorld:
    def __init__(self,out):
        self.out=out
        self.protocol=json.loads((out/'protocol.json').read_text())
        self.nav=NavigationConfig(**self.protocol['nav'])
        self.requests=0

    def obtain(self,positions):
        unique={position_key(p):np.asarray(p).tolist() for p in positions}
        missing=[dict(key=k,position=p) for k,p in unique.items() if not (self.out/'panoramas'/f'{k}.png').exists()]
        if missing:
            self.requests+=1
            name=f'{time.time_ns()}-{self.requests:06d}'
            request=self.out/'queue'/f'{name}.request.json'
            temp=request.with_suffix('.tmp')
            temp.write_text(json.dumps(dict(views=missing)))
            temp.replace(request)
            response=request.with_name(f'{name}.response.json')
            start=time.monotonic()
            while not response.exists():
                if time.monotonic()-start > max(180,30*len(missing)):
                    raise TimeoutError(f'Blender worker did not finish {request}')
                time.sleep(.025)
            result=json.loads(response.read_text())
            if not result['ok']:
                raise RuntimeError(result['error'])
        return unique

    def render(self,positions,headings):
        self.obtain(positions)
        images=[]
        for p,h in zip(positions,headings):
            with Image.open(self.out/'panoramas'/f'{position_key(p)}.png') as im:
                panorama=np.asarray(im.convert('RGB'))
            images.append(sample_panorama(panorama,[h]))
        return torch.cat(images)

    def scan(self,position,headings):
        self.obtain([position])
        with Image.open(self.out/'panoramas'/f'{position_key(position)}.png') as im:
            panorama=np.asarray(im.convert('RGB'))
        return sample_panorama(panorama,headings)


def validate_calibration(environment):
    with Image.open(environment/'calibration.png') as im:
        panorama=np.asarray(im.convert('RGB'))
    # Both central columns straddle forward by 2.03 degrees; both must see the
    # expected physical cardinal sphere. The marker centres are at elevation 0.
    views=sample_panorama(panorama,[0,90,180,270]).numpy()
    centre=views[:,:,13:15,36:38].mean((2,3))
    assert centre[0,0] > centre[0,1]+.2 and centre[0,0] > centre[0,2]+.2
    assert centre[1,1] > centre[1,0]+.2 and centre[1,1] > centre[1,2]+.2
    assert centre[2,2] > centre[2,0]+.2 and centre[2,2] > centre[2,1]+.2
    assert min(centre[3,:2]) > centre[3,2]+.2
    assert torch.equal(sample_panorama(panorama,[0]),sample_panorama(panorama,[360]))
    return dict(cardinal_directions_passed=True,periodicity_passed=True,centre_rgb=centre.tolist())


def run(out,environment=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    environment=environment or out
    if not (out/'protocol.json').exists():
        out.mkdir(parents=True,exist_ok=True)
        (out/'protocol.json').write_bytes((environment/'protocol.json').read_bytes())
    protocol=json.loads((out/'protocol.json').read_text())
    if (out/'results.jsonl').exists():
        raise FileExistsError('Use a fresh run directory to avoid mixing results')
    world=GrasslandWorld(environment)
    assert protocol==world.protocol, 'Run protocol must match its environment'
    calibration=validate_calibration(environment)
    positions,headings=np.array(protocol['route']),np.array(protocol['headings'])
    teach_pos,teach_head=acquisition_views(positions,headings,**protocol['acquisition'])
    write_json(out/'acquisition.json',dict(positions=teach_pos.tolist(),headings=teach_head.tolist()))
    sources=[p for p in sorted(Path('apiaviz').rglob('*.py')) if 'output' not in p.parts and 'data' not in p.parts]
    with zipfile.ZipFile(out/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sources: archive.write(p,str(p))
    manifest=dict(status='rendering_training',protocol_sha256=file_hash(out/'protocol.json'),
                  source_sha256={str(p):file_hash(p) for p in sources},encoders={},
                  environment=str(environment.resolve()),calibration=calibration,
                  scene_sha256=file_hash(environment/'grassland.blend'))
    write_json(out/'manifest.json',manifest)
    batches=[]
    for i in range(0,len(teach_head),45):
        batches.append(world.render(teach_pos[i:i+45],teach_head[i:i+45]))
        print(f'Teaching images {min(i+45,len(teach_head))}/{len(teach_head)}',flush=True)
    images=torch.cat(batches)
    assert images.shape==(len(teach_head),3,18,74) and torch.isfinite(images).all()
    torch.save(dict(training=images),out/'training.pt')
    image_hash=hashlib.sha256(images.numpy().tobytes()).hexdigest()
    manifest.update(status='navigation',training_images_sha256=image_hash,training_views=len(images))
    write_json(out/'manifest.json',manifest)
    for seed in protocol['wiring_seeds']:
        model=RefinementEncoder(seed=seed).eval()
        frozen=fingerprint(model)
        torch.save(dict(state_dict=model.state_dict()),out/f'encoder-{seed}.pt')
        manifest['encoders'][str(seed)]=dict(fingerprint=frozen,encoder=asdict(model.config),circuit=asdict(model.circuit_config))
        for method in protocol['methods']:
            started=time.monotonic()
            model.method=method
            codes=encode(model,images)
            memory=SpikeOverlapMemory(codes)
            frozen_memory=fingerprint(memory)
            scorer=Scorer(world,model,memory,encode,'clean',0.,protocol['geometry_seed'])
            print(f'Navigation seed={seed} method={method}',flush=True)
            result=evaluate_route(positions,headings,scorer,'free',max_steps=protocol['max_steps'])
            trace=result.pop('trace')
            errors=polyline_distance([r['position'] for r in trace],positions)
            xmin,xmax,ymin,ymax=protocol['world_bounds_m']
            outside=sum(not (xmin<=r['position'][0]<=xmax and ymin<=r['position'][1]<=ymax) for r in trace)
            assert fingerprint(model)==frozen and fingerprint(memory)==frozen_memory
            name=f'seed-{seed}-{method}.json'
            write_json(out/name,dict(trajectory=trace,decisions=scorer.decisions))
            row=dict(seed=seed,preprocessing=method,trace=name,training_images_sha256=image_hash,
                     memory_fingerprint=frozen_memory,encoder_fingerprint=frozen,memory_views=len(codes),
                     **result,polyline_mean_m=float(errors.mean()),steps_outside_landmark_field=outside,
                     **code_statistics(codes),elapsed_s=time.monotonic()-started)
            with (out/'results.jsonl').open('a') as handle:
                handle.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(row),flush=True)
            write_json(out/'manifest.json',manifest)
    manifest['status']='complete'
    write_json(out/'manifest.json',manifest)
    print('COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--environment',type=Path,help='Reuse an existing saved scene and panorama cache for a new run')
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.output)
    else: run(args.output,args.environment)


if __name__=='__main__':
    main()
