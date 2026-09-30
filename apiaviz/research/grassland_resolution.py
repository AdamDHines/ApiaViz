"""Pre-specified resolution sweep in the saved grassland environment.

27 paired input-resolution trials and 12 angular-filter controls. No change to
teaching poses, projection wiring, common pooling, circuit, memory or controller.
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

from .angular_frontend import AngularEncoder
from .frontend_refinements import RefinementEncoder
from .grassland_smoke import DEFAULT as ENVIRONMENT, GrasslandWorld, position_key, sample_panorama
from .mechanisms import code_statistics, polyline_distance
from .navigation import Scorer, evaluate_route
from .spike_overlap import SpikeOverlapMemory
from .study import encode, fingerprint, file_hash, write_json

DEFAULT=Path('apiaviz/output/grassland-resolution')
SHAPES=[(18,74),(39,149),(51,199)]
METHODS=['linear_colour','sobel_colour','ardin_input']


def conditions():
    return [dict(shape=list(shape),mode=mode,methods=METHODS if mode=='pixels' else METHODS[:2])
            for mode in ('pixels','angles') for shape in SHAPES if mode=='pixels' or shape!=SHAPES[0]]


class ResolutionWorld(GrasslandWorld):
    def __init__(self,out,shape,cache_only=False):
        super().__init__(out)
        self.shape=tuple(shape)
        self.cache_only=cache_only

    def obtain(self,positions):
        if self.cache_only:
            for p in positions:
                if not (self.out/'panoramas'/f'{position_key(p)}.png').exists():
                    raise AssertionError(f'Cache miss at {p}')
        else:
            super().obtain(positions)

    def render(self,positions,headings):
        self.obtain(positions)
        result=[]
        for p,h in zip(positions,headings):
            with Image.open(self.out/'panoramas'/f'{position_key(p)}.png') as image:
                panorama=np.asarray(image.convert('RGB'))
            result.append(sample_panorama(panorama,[h],self.shape))
        return torch.cat(result)

    def scan(self,position,headings):
        self.obtain([position])
        with Image.open(self.out/'panoramas'/f'{position_key(position)}.png') as image:
            panorama=np.asarray(image.convert('RGB'))
        return sample_panorama(panorama,headings,self.shape)


def prepare(out,environment):
    if (out/'protocol.json').exists():
        raise FileExistsError('Protocol exists; use a fresh directory')
    out.mkdir(parents=True,exist_ok=True)
    base=json.loads((environment/'protocol.json').read_text())
    protocol=dict(role='One-route resolution sensitivity study; fixed before new outcomes. No significance/generalisation claim.',
                  environment=str(environment.resolve()),base_protocol_sha256=file_hash(environment/'protocol.json'),
                  scene_sha256=file_hash(environment/'grassland.blend'),
                  acquisition_sha256=file_hash(environment/'acquisition.json'),
                  wiring_seeds=base['wiring_seeds'],conditions=conditions(),trials=39,
                  primary_endpoint='Whole-trajectory nearest-taught-point mean deviation, including failures, equally averaged over seeds.',
                  secondary_endpoints=['completion','continuous-polyline mean deviation','maximum deviation','code sparsity'],
                  unchanged='World, sun, teaching positions/headings, route, 8x64 feature pooling, projection checkpoint, 8000 cells, LIF/inhibition settings, spike-overlap memory, 13-heading scan, 10 cm steps, 20 cm stopping radius, 200-step limit.',
                  pixels='Original frontend code and pixel kernels at each input shape; finer angular input AND smaller angular filter footprints.',
                  angles='ApiaViz/Sobel discrete spatial tap offsets and weights fixed to the original visual degrees. Fractional offsets use bilinear interpolation; hex row phase anchored to reference angular rows. Same reflected boundaries, gains, nonlinearities and common pooling. Approximate numerical control, not a calibrated bee retina.',
                  ardin='Original 10x36 downsampling, L2 normalization and default native-resolution CLAHE retained. No separately scaled Ardin baseline and no redundant angular-control runs.',
                  optics='Same 0.5-degree cached renderer panoramas and bilinear ray samples. No new retinal optical-blur model; isolate input sampling/filter scale from optics.',
                  stopping='Complete all 39 trials with no performance-based retuning. Low-resolution runs must reproduce the prior smoke exactly.')
    write_json(out/'protocol.json',protocol)
    print(json.dumps(protocol,indent=2),flush=True)


def load_model(environment,seed,method,mode):
    cls=RefinementEncoder if mode=='pixels' else AngularEncoder
    model=cls(method=method,seed=seed).eval()
    model.load_state_dict(torch.load(environment/f'encoder-{seed}.pt',weights_only=True,map_location='cpu')['state_dict'])
    return model


def run(out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol=json.loads((out/'protocol.json').read_text())
    environment=Path(protocol['environment'])
    assert file_hash(environment/'protocol.json')==protocol['base_protocol_sha256']
    assert file_hash(environment/'grassland.blend')==protocol['scene_sha256']
    assert file_hash(environment/'acquisition.json')==protocol['acquisition_sha256']
    if (out/'results.jsonl').exists(): raise FileExistsError('Refusing to mix runs')
    base=json.loads((environment/'protocol.json').read_text())
    acquisition=json.loads((environment/'acquisition.json').read_text())
    previous={(r['seed'],r['preprocessing']):r for r in map(json.loads,(environment/'results.jsonl').read_text().splitlines())}
    positions,headings=np.asarray(base['route']),np.asarray(base['headings'])
    sources=[p for p in sorted(Path('apiaviz').rglob('*.py')) if 'output' not in p.parts and 'data' not in p.parts]
    with zipfile.ZipFile(out/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sources: archive.write(p,str(p))
    manifest=dict(status='running',protocol_sha256=file_hash(out/'protocol.json'),
                  source_sha256={str(p):file_hash(p) for p in sources},training={},encoders={})
    for seed in protocol['wiring_seeds']:
        model=load_model(environment,seed,'linear_colour','pixels')
        manifest['encoders'][str(seed)]=dict(fingerprint=fingerprint(model),checkpoint_sha256=file_hash(environment/f'encoder-{seed}.pt'),encoder=asdict(model.config),circuit=asdict(model.circuit_config))
    write_json(out/'manifest.json',manifest)
    for condition in protocol['conditions']:
        h,w=condition['shape'];mode=condition['mode'];resolution=f'{w}x{h}'
        world=ResolutionWorld(environment,(h,w))
        training=out/f'training-{resolution}.pt'
        if training.exists():
            images=torch.load(training,weights_only=True)['training']
        else:
            images=world.render(acquisition['positions'],acquisition['headings'])
            torch.save(dict(training=images),training)
            if (h,w)==SHAPES[0]:
                assert torch.equal(images,torch.load(environment/'training.pt',weights_only=True)['training'])
        image_hash=hashlib.sha256(images.numpy().tobytes()).hexdigest()
        manifest['training'][resolution]=dict(images_sha256=image_hash,file_sha256=file_hash(training),shape=list(images.shape))
        write_json(out/'manifest.json',manifest)
        for seed in protocol['wiring_seeds']:
            for method in condition['methods']:
                started=time.monotonic()
                model=load_model(environment,seed,method,mode)
                frozen=fingerprint(model)
                assert frozen==manifest['encoders'][str(seed)]['fingerprint']
                codes=encode(model,images)
                memory=SpikeOverlapMemory(codes)
                memory_hash=fingerprint(memory)
                print(f'Navigation {resolution} {mode} seed={seed} {method}',flush=True)
                scorer=Scorer(world,model,memory,encode,'clean',0.,base['geometry_seed'])
                result=evaluate_route(positions,headings,scorer,'free',max_steps=base['max_steps'])
                trace=result.pop('trace')
                errors=polyline_distance([r['position'] for r in trace],positions)
                assert fingerprint(model)==frozen and fingerprint(memory)==memory_hash
                if (h,w)==SHAPES[0]:
                    old=previous[seed,method]
                    assert memory_hash==old['memory_fingerprint']
                    old_trace=json.loads((environment/old['trace']).read_text())
                    assert trace==old_trace['trajectory'] and scorer.decisions==old_trace['decisions']
                name=f'{resolution}-{mode}-seed-{seed}-{method}.json'
                write_json(out/name,dict(trajectory=trace,decisions=scorer.decisions))
                row=dict(resolution=resolution,shape=[h,w],mode=mode,seed=seed,preprocessing=method,
                         trace=name,training_images_sha256=image_hash,encoder_fingerprint=frozen,
                         memory_fingerprint=memory_hash,memory_views=len(images),**result,
                         polyline_mean_m=float(errors.mean()),**code_statistics(codes),elapsed_s=time.monotonic()-started)
                with (out/'results.jsonl').open('a') as handle:
                    handle.write(json.dumps(row,allow_nan=False)+'\n')
                print(json.dumps({k:row[k] for k in ('resolution','mode','seed','preprocessing','reached_nest','route_deviation_mean_m','polyline_mean_m','elapsed_s')}),flush=True)
    manifest['status']='complete'
    write_json(out/'manifest.json',manifest)
    print('COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--environment',type=Path,default=ENVIRONMENT)
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.output,args.environment)
    else: run(args.output)


if __name__=='__main__': main()
