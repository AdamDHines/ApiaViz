"""Small, bounded Blender camera server for recorded synthetic navigation demos."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'apiaviz/research'))
sys.path.insert(0,str(ROOT/'scripts'))
from terrain_concepts import Mesh, blade, material, rock, setup_light_camera
from export_collision_geometry import export


def build(out,p):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rng=random.Random(p['geometry_seed'])
    soil=material('Grassland earth','A49475','62563E',scale=35,bump=.0003)
    stone=material('Limestone','B8B3A1','54534A',scale=95,bump=.0006)
    grass=material('Grass 0','637C3D','29452F',scale=35,bump=.00004,leaf=True)
    bpy.ops.mesh.primitive_plane_add(size=40)
    bpy.context.object.name='Walkable terrain'
    bpy.context.object.data.materials.append(soil)
    obstacles=[]
    for i,spec in enumerate(p['rocks']):
        x,y=spec['position']
        rock(f'Limestone {i:03d}',(x,y,spec['scale'][2]*.4),spec['scale'],stone,rng)
        obstacles.append(dict(x=x,y=y,conservative_radius_m=spec['legacy_radius_m']))
    foliage=Mesh('Grass landmarks',[grass])
    for _ in range(85):
        x=rng.uniform(-.8,2.3); y=rng.choice([-1,1])*rng.uniform(.6,1.3)
        for _ in range(7):
            blade(foliage,(x+rng.uniform(-.02,.02),y+rng.uniform(-.02,.02),0),rng.uniform(0,math.tau),
                  rng.uniform(.1,.28),.008,rng.uniform(.02,.05),0)
    foliage.finish()
    setup_light_camera('grassland')
    scene=bpy.context.scene
    scene.cycles.device='CPU'
    scene.render.threads_mode='FIXED'; scene.render.threads=4
    scene.cycles.samples=p['renderer']['samples']
    scene.cycles.use_denoising=False
    scene.cycles.seed=p['renderer']['seed']; scene.cycles.use_animated_seed=False
    scene.render.use_persistent_data=True
    scene.render.resolution_x,scene.render.resolution_y=p['renderer']['panorama_size']
    cam=scene.camera.data; cam.type='PANO'; cam.panorama_type='EQUIRECTANGULAR'
    cam.longitude_min,cam.longitude_max=-math.pi,math.pi
    cam.latitude_min,cam.latitude_max=map(math.radians,(-15.5,60.5))
    scene.camera.rotation_euler=Vector((1,0,0)).to_track_quat('-Z','Y').to_euler()
    bpy.ops.wm.save_as_mainfile(filepath=str((out/'grassland.blend').resolve()))
    scene_hash=hashlib.sha256((out/'grassland.blend').read_bytes()).hexdigest()
    (out/'world.json').write_text(json.dumps(dict(scene_sha256=scene_hash,obstacles=obstacles,
        role='New synthetic video demonstration; not historical grassland',renderer=p['renderer']),indent=2))
    export(out/'grassland.blend',out/'collision.json')


def main(out):
    p=json.loads((out/'protocol.json').read_text())
    build(out,p)
    (out/'ready.json').write_text('{"ready":true}')
    deadline=time.monotonic()+1200
    while not (out/'stop').exists() and time.monotonic()<deadline:
        jobs=sorted((out/'queue').glob('*.request.json'))
        if not jobs:
            time.sleep(.025); continue
        for job in jobs:
            try:
                request=json.loads(job.read_text())
                for view in request['views']:
                    target=out/'panoramas'/f"{view['key']}.png"
                    if target.exists(): continue
                    x,y=view['position']; bpy.context.scene.camera.location=(x,y,.01)
                    bpy.context.scene.render.filepath=str(target.resolve())
                    bpy.ops.render.render(write_still=True)
                result=dict(ok=True)
            except Exception:
                result=dict(ok=False,error=traceback.format_exc())
            target=job.with_name(job.name.replace('.request.json','.response.json'))
            tmp=target.with_suffix('.tmp'); tmp.write_text(json.dumps(result)); tmp.replace(target)
            job.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    main(p.parse_args(sys.argv[sys.argv.index('--')+1:]).output.resolve())
