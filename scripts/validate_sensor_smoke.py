"""Compare the tiny Blender fixture's collision/UV exports and linear renders."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from apiaviz.research.spectral_input import load_frame


def validate(output):
    collision=json.loads((output/'collision.json').read_text())
    geometry=json.loads((output/'geometry/geometry.json').read_text())
    assert collision['scene_sha256']==geometry['scene_sha256']
    item=next(m for m in geometry['meshes'] if m['material']=='Limestone')
    mesh=np.load(output/'geometry'/item['file'],allow_pickle=False)
    triangles=mesh['vertices'][mesh['faces'],:2]
    projected=np.asarray(collision['rocks'][0]['triangles_xy_m'])
    # Spectral export stores float32; evaluator retains float64 transformed points.
    np.testing.assert_allclose(triangles,projected,atol=1e-7,rtol=0)
    frames=[]
    for i in range(len(geometry['poses'])):
        a,_,_=load_frame(output/'uv-linear'/f'{i:05d}.json')
        b,stokes,_=load_frame(output/'uv-stokes'/f'{i:05d}.json')
        assert stokes is not None and np.any(stokes[1:]<0) and np.any(stokes[1:]>0)
        np.testing.assert_allclose(a,b,rtol=2e-5,atol=1e-6)
        frames.append(dict(frame=i,shape=list(a.shape),
            intensity_difference=float(abs(a-b).max()),
            max_degree=float((np.linalg.norm(stokes[1:],axis=0)/np.maximum(b,1e-15)).max())))
    record=dict(passed=True,scene_sha256=geometry['scene_sha256'],
        scene='synthetic one-rock integration fixture; not historical study geometry',
        rock_triangles=len(triangles),geometry_max_abs_error_m=float(abs(triangles-projected).max()),frames=frames)
    (output/'validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    validate(parser.parse_args().output)
