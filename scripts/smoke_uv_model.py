"""Bounded ApiaViz UV encoding/memory checks; no navigation or baseline trials.

Consumes two independently sampled calibrated render batches of the same poses.
The UV-only fixtures are diagnostic stimuli, not measured scene radiances.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from apiaviz.research.spectral_input import file_sha
from apiaviz.research.uv_input import load_views
from apiaviz.research.uv_encoder import ApiaVizUVEncoder,UVEncoderConfig
from apiaviz.research.spike_overlap import SpikeOverlapMemory
from apiaviz.research.study import fingerprint,write_json


def batch_views(directory,calibration,offsets):
    protocol=json.loads((directory/'protocol.json').read_text())
    completed=json.loads((directory/'complete.json').read_text())
    assert completed['protocol_sha256']==file_sha(directory/'protocol.json')
    assert completed['frames']==len(protocol['poses'])
    views=[];sources={}
    for i,pose in enumerate(protocol['poses']):
        path=directory/f'{i:05d}.json'
        images,record=load_views(path,np.array(offsets)+pose['heading'],calibration_sha256=calibration)
        assert record['pose']==pose and record['scene_sha256']==protocol['scene_sha256']
        assert record['seed']==protocol['seed']
        views.append(images)
        sources[str(path)]=dict(sidecar_sha256=file_sha(path),arrays=record['arrays'])
    return torch.cat(views),protocol,sources


def fixture_views():
    """Four patterns distinguished only by UV; GB values are exactly shared."""
    y,x=torch.meshgrid(torch.linspace(0,1,51),torch.linspace(0,1,199),indexing='ij')
    images=torch.empty(4,3,51,199)
    images[:,1]=.25; images[:,2]=.2
    for i,centre in enumerate((.18,.39,.61,.82)):
        images[i,0]=.06+.35*torch.exp(-((x-centre)/.08).square()-((y-.5)/.25).square())
    probes=images.clone(); probes[:,0]=torch.roll(images[:,0],1,-1)
    return images,probes


def overlaps(a,b):
    return F.normalize((a>0).float(),dim=1)@F.normalize((b>0).float(),dim=1).T


def run(args):
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    offsets=[-60.,-30.,0.,30.,60.]
    calibration=file_sha(ROOT/'docs/uv-calibration/calibration.json')
    training,first,source_a=batch_views(args.frames.resolve(),calibration,[0.])
    probes,second,source_b=batch_views(args.repeat.resolve(),calibration,offsets)
    assert first['seed']!=second['seed'],'Require independent Monte Carlo seeds'
    for key in first.keys()-{'seed','source_sha256'}:
        assert first[key]==second[key],f'Repeated rendering differs beyond seed: {key}'
    protocol=dict(schema='apiaviz-uv-smoke-v1',role='Implementation smoke; not a navigation comparison',
        seeds=[19,23,31],encoder=asdict(UVEncoderConfig()),offsets_deg=offsets,
        training_frames=str(args.frames.resolve()),probe_frames=str(args.repeat.resolve()),
        scene_sha256=first['scene_sha256'],calibration_sha256=calibration,
        shape=[51,199],controls=['UV pathway enabled','UV pathway disabled with identical allocated populations',
            'UV-only synthetic patterns','UV horizontal-roll counterfactual','common exposure factors 0.5 and 2'],
        limits='Three correlated poses in one simple scene; no Sobel/Ardin changes, UV polarization, closed-loop navigation or performance tuning')
    sources=[f for f in (ROOT/'apiaviz').rglob('*.py') if 'output' not in f.relative_to(ROOT).parts and 'data' not in f.relative_to(ROOT).parts]
    sources.append(Path(__file__).resolve())
    manifest=dict(status='running',sources={str(f.relative_to(ROOT)):file_sha(f) for f in sources},
        inputs={**source_a,**source_b},torch=str(torch.__version__),numpy=np.__version__)
    write_json(out/'protocol.json',protocol); write_json(out/'manifest.json',manifest)
    with zipfile.ZipFile(out/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sources: archive.write(path,str(path.relative_to(ROOT)))
    torch.save(dict(training=training,probes=probes),out/'retinal-inputs.pt')
    fixtures,fixture_queries=fixture_views()
    rows=[]; all_scores=[]
    for seed in protocol['seeds']:
        model=ApiaVizUVEncoder(UVEncoderConfig(seed=seed))
        frozen=fingerprint(model)
        torch.save(dict(state_dict=model.state_dict(),metadata=model.metadata()),out/f'encoder-{seed}.pt')
        started=time.perf_counter()
        diag=model.diagnostics(training); train=diag['codes']; query=model(probes)
        elapsed=time.perf_counter()-started
        assert torch.isfinite(train).all() and torch.isfinite(query).all()
        memory=SpikeOverlapMemory(train); memory_hash=fingerprint(memory)
        scores=memory(query).reshape(len(training),len(offsets))
        off_train=model(training,uv_enabled=False); off_query=model(probes,uv_enabled=False)
        off_memory=SpikeOverlapMemory(off_train)
        off_scores=off_memory(off_query).reshape_as(scores)
        torch.testing.assert_close(off_train[:,:8000],train[:,:8000],atol=0,rtol=0)
        assert not torch.count_nonzero(off_train[:,8000:])
        torch.testing.assert_close(memory(train),torch.full((len(train),),-1.),atol=1e-6,rtol=0)
        fixture_codes=model(fixtures); fixture_probe=model(fixture_queries)
        fixture_similarity=overlaps(fixture_probe,fixture_codes)
        fixture_off=overlaps(model(fixture_queries,uv_enabled=False),model(fixtures,uv_enabled=False))
        # All visible-only fixture codes must tie; don't call argmax's first
        # index a meaningful prediction. Full UV decoding should break the tie.
        torch.testing.assert_close(fixture_off,fixture_off[:,:1].expand_as(fixture_off),atol=0,rtol=0)
        foil=fixture_similarity.clone(); foil.fill_diagonal_(-float('inf'))
        rolled=training.clone(); rolled[:,0]=torch.roll(rolled[:,0],49,-1)
        rolled_codes=model(rolled)
        torch.testing.assert_close(rolled_codes[:,:8000],train[:,:8000],atol=0,rtol=0)
        uv_overlap=overlaps(train[:,8000:],rolled_codes[:,8000:]).diag()
        exposure={str(scale):(-memory(model(training*scale))).tolist() for scale in (.5,2.)}
        torch.save(dict(state_dict=memory.state_dict(),teaching_codes=train),out/f'memory-{seed}.pt')
        checkpoint=torch.load(out/f'encoder-{seed}.pt',weights_only=True,map_location='cpu')
        restored=ApiaVizUVEncoder(UVEncoderConfig(**checkpoint['metadata']['encoder']))
        restored.load_state_dict(checkpoint['state_dict'])
        torch.testing.assert_close(restored(training),train,atol=0,rtol=0)
        assert fingerprint(model)==frozen and fingerprint(memory)==memory_hash
        def retrieval(s):
            off_indices=[0,1,3,4]
            return dict(best_heading_deg=[offsets[i] for i in s.argmin(1).tolist()],
                correct_heading_margin=(s[:,off_indices].min(1).values-s[:,2]).tolist(),
                familiarity=(-s).tolist())
        rows.append(dict(seed=seed,encoder_fingerprint=frozen,memory_fingerprint=memory_hash,
            stream_activity={name:float(s['counts'].mean()) for name,s in zip(model.stream_names,diag['streams'])},
            encoding_s=elapsed,encoded_views=len(training)+len(probes),
            full_uv=retrieval(scores),uv_disabled=retrieval(off_scores),
            fixture_correct=int((fixture_similarity.argmax(1)==torch.arange(4)).sum()),fixture_n=4,
            fixture_margins=(fixture_similarity.diag()-foil.max(1).values).tolist(),
            visible_only_fixture='All four memories tie exactly; the cues exist only in UV',
            uv_rolled_overlap=uv_overlap.tolist(),exposure_recall=exposure))
        all_scores.append((scores.numpy(),off_scores.numpy()))
        print(json.dumps(rows[-1],allow_nan=False),flush=True)
    summary=dict(passed=True,protocol=protocol,results=rows,
        checks=['Verified linear frame/calibration hashes and angular sampling',
            'UV changes leave visible stream codes exactly unchanged',
            'UV ablation suppresses only the third stream',
            'Finite spike codes and exact teaching-view retrieval',
            'Checkpoint reload reproduces codes exactly',
            'Encoder and learned memory remain frozen during recall'])
    write_json(out/'results.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,5.6),layout='constrained')
    vmax=float(torch.quantile(training,.99))
    for i,name in enumerate(('UV','Blue','Green')):
        axes[0,i].imshow(training[0,i],cmap='gray',vmin=0,vmax=vmax,aspect='auto')
        axes[0,i].set_title(name+' · linear receptor input'); axes[0,i].set_axis_off()
    maps=model.backbone(training[:1])['uv'][0]
    for ax,index,name in ((axes[1,0],0,'UV ON spatial contrast'),(axes[1,1],3,'Rectified UV − blue')):
        ax.imshow(maps[index],cmap='magma',aspect='auto'); ax.set_title(name); ax.set_axis_off()
    values=np.asarray(all_scores)
    for i,name in enumerate(('UV enabled','UV pathway disabled')):
        axes[1,2].plot(offsets,-values[:,i].mean((0,1)),marker='o',label=name)
    axes[1,2].set(xlabel='Heading offset (degrees)',ylabel='Mean memory familiarity',title='Independent render seed recall')
    axes[1,2].legend(fontsize=8)
    fig.suptitle('ApiaViz UV smoke · three poses, three wiring seeds · no navigation claim')
    fig.savefig(out/'diagnostic.png',dpi=160); plt.close(fig)
    manifest.update(status='complete',results_sha256=file_sha(out/'results.json'))
    write_json(out/'manifest.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames',type=Path,required=True)
    parser.add_argument('--repeat',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
