"""Build fresh flat-ground procedural trial scenes; no renderer service started."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'apiaviz/research'))
from terrain_concepts import Mesh,blade,material,rock,setup_light_camera


def build(out):
    p=json.loads((out/'protocol.json').read_text())
    if (out/'grassland.blend').exists(): raise FileExistsError('Scene already exists')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rng=random.Random(p['geometry_seed'])
    route=[Vector(v) for v in p['route']]
    def distance(x,y):
        v=Vector((x,y))
        return min((v-(a+(b-a)*max(0.,min(1.,(v-a).dot(b-a)/(b-a).length_squared)))).length
            for a,b in zip(route[:-1],route[1:]))
    xmin,xmax,ymin,ymax=p['world_bounds_m']
    def location(clearance):
        for _ in range(10000):
            x,y=rng.uniform(xmin+.2,xmax-.2),rng.uniform(ymin+.2,ymax-.2)
            if distance(x,y)>clearance: return x,y
        raise RuntimeError('Cannot place landmark outside teaching corridor')
    soil=material('Grassland earth','A49475','776A51',bump=0)
    stone=material('Limestone','AAA796','827F73',bump=0)
    grass=[material(f'Grass {i}',c,bump=0) for i,c in enumerate(('798653','AFA779','536347','C2AE7B'))]
    bpy.ops.mesh.primitive_plane_add(size=80)
    bpy.context.object.name='Walkable terrain'; bpy.context.object.data.materials.append(soil)
    obstacles=[]
    for i in range(p['rock_count']):
        radius=rng.uniform(.08,.30)
        x,y=location(.22+1.7*radius)
        rock(f'Limestone {i:03d}',(x,y,radius*.35),(radius*1.2,radius,radius*.8),stone,rng,detail=2)
        obstacles.append(dict(x=x,y=y,conservative_radius_m=1.7*radius))
    foliage=Mesh('Grass landmarks',grass)
    for _ in range(p['grass_count']):
        x,y=location(.25)
        for _ in range(8):
            blade(foliage,(x+rng.uniform(-.025,.025),y+rng.uniform(-.025,.025),0),
                rng.uniform(0,math.tau),rng.uniform(.08,.3),rng.uniform(.004,.012),
                rng.uniform(.015,.08),rng.randrange(4))
    foliage.finish()
    setup_light_camera('grassland')
    bpy.context.scene.camera.location=(*p['route'][0],.01)
    bpy.context.scene.camera.rotation_euler=Vector((1,0,0)).to_track_quat('-Z','Y').to_euler()
    bpy.ops.wm.save_as_mainfile(filepath=str((out/'grassland.blend').resolve()),compress=True)
    (out/'world.json').write_text(json.dumps(dict(scene_sha256=hashlib.sha256((out/'grassland.blend').read_bytes()).hexdigest(),
        geometry_seed=p['geometry_seed'],obstacles=obstacles,blender=bpy.app.version_string,
        terrain='Flat plane z=0; camera z=0.01m. Foliage is visual, rocks are solid.',
        scope='New seeded development scene, not a reproduction of cancelled experiments'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    build(p.parse_args(sys.argv[sys.argv.index('--')+1:]).output.resolve())
