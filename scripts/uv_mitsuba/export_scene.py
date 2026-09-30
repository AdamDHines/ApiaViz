"""Export evaluated Blender geometry without modifying the saved experiment scene."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument('--environment', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
bpy.ops.wm.open_mainfile(filepath=str((args.environment/'grassland.blend').resolve()))
args.output.mkdir(parents=True, exist_ok=False)
groups = {}
deps = bpy.context.evaluated_depsgraph_get()
for obj in bpy.context.scene.objects:
    if obj.type != 'MESH' or obj.hide_render:
        continue
    for modifier in obj.modifiers:
        if modifier.show_viewport != modifier.show_render or (modifier.type == 'SUBSURF' and modifier.levels != modifier.render_levels):
            raise ValueError(f'{obj.name}: viewport and render modifiers differ; bake a separate export scene first')
    evaluated = obj.evaluated_get(deps)
    mesh = evaluated.to_mesh()
    mesh.calc_loop_triangles()
    vertices = np.empty(len(mesh.vertices)*3, dtype=np.float32)
    mesh.vertices.foreach_get('co', vertices)
    vertices = vertices.reshape(-1, 3)
    mat = np.array(evaluated.matrix_world)
    vertices = vertices@mat[:3,:3].T+mat[:3,3]
    triangles = np.array([t.vertices[:] for t in mesh.loop_triangles], dtype=np.int32)
    materials = np.array([t.material_index for t in mesh.loop_triangles])
    for idx in np.unique(materials):
        name = mesh.materials[int(idx)].name if len(mesh.materials) else 'Unknown'
        entries = groups.setdefault(name, [])
        selected = triangles[materials == idx]
        used, inverse = np.unique(selected, return_inverse=True)
        entries.append((vertices[used], inverse.reshape(-1,3)))
    evaluated.to_mesh_clear()
assets = []
for i, (name, entries) in enumerate(sorted(groups.items())):
    vertices, faces = [], []
    offset = 0
    for v, f in entries:
        vertices.append(v)
        faces.append(f+offset)
        offset += len(v)
    filename = f'mesh-{i:02d}.npz'
    vv, ff = np.concatenate(vertices).astype(np.float32), np.concatenate(faces).astype(np.uint32)
    np.savez_compressed(args.output/filename, vertices=vv, faces=ff)
    assets.append(dict(material=name, file=filename, vertices=len(vv), triangles=len(ff),
                      sha256=hashlib.sha256((args.output/filename).read_bytes()).hexdigest()))
p = json.loads((args.environment/'protocol.json').read_text())
ground = bpy.data.objects['Walkable terrain']
poses = []
for idx in (0, len(p['route'])//2, len(p['route'])-1):
    x,y = p['route'][idx]
    hit, point, _, _ = ground.ray_cast(Vector((x,y,10)), Vector((0,0,-1)))
    assert hit
    poses.append(dict(station=idx, position=[x,y,float(point.z)+.01], heading=p['headings'][min(idx,len(p['headings'])-1)]))
source = args.environment/'grassland.blend'
metadata = dict(scene=str(source.resolve()), scene_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                blender=bpy.app.version_string, meshes=assets, poses=poses,
                note='Evaluated triangle geometry grouped by material. Procedural RGB shaders are replaced with explicitly documented spectral materials.')
(args.output/'geometry.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(dict(materials=len(assets), triangles=sum(a['triangles'] for a in assets), poses=poses)))
