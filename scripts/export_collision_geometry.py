"""Run with Blender: export evaluated rock projections without editing a scene."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apiaviz.research.collision_geometry import SCHEMA


def export(scene, output):
    scene, output = Path(scene), Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh geometry output; do not replace frozen assets')
    bpy.ops.wm.open_mainfile(filepath=str(scene.resolve()))
    deps = bpy.context.evaluated_depsgraph_get()
    rocks = []
    for obj in sorted(bpy.context.scene.objects, key=lambda o:o.name):
        if obj.type != 'MESH' or obj.hide_render or not obj.name.startswith('Limestone '):
            continue
        for modifier in obj.modifiers:
            if modifier.show_viewport != modifier.show_render or (modifier.type == 'SUBSURF' and modifier.levels != modifier.render_levels):
                raise ValueError(f'{obj.name}: viewport and render modifiers differ; bake a separate export scene first')
        evaluated = obj.evaluated_get(deps)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            vertices = np.array([v.co[:] for v in mesh.vertices], dtype=float)
            matrix = np.array(evaluated.matrix_world)
            vertices = vertices@matrix[:3, :3].T+matrix[:3, 3]
            faces = np.array([t.vertices[:] for t in mesh.loop_triangles])
            rocks.append(dict(name=obj.name, triangles_xy_m=vertices[faces, :2].tolist()))
        finally:
            evaluated.to_mesh_clear()
    if not rocks:
        raise ValueError('No visible Limestone rock objects found')
    record = dict(schema=SCHEMA, scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
        blender=bpy.app.version_string, rocks=rocks,
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        projection='Union of world-space evaluated triangles projected into XY; includes buried vertices and overhangs; no climbing')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(record, f, allow_nan=False)
    print(json.dumps(dict(output=str(output), rocks=len(rocks), scene_sha256=record['scene_sha256'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    export(args.scene, args.output)
