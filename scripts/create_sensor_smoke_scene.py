"""Blender-only small synthetic fixture for export/render integration checks.

This is not a research environment or a navigation trial. No image is rendered.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy


def main(output):
    output.mkdir(parents=True,exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    def material(name): return bpy.data.materials.new(name)
    bpy.ops.mesh.primitive_plane_add(size=10)
    ground=bpy.context.object
    ground.name='Walkable terrain'
    ground.data.materials.append(material('Grassland earth'))
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,radius=.2,location=(.5,0,.06))
    rock=bpy.context.object
    rock.name='Limestone 000'
    rock.scale=(1.2,.7,.8)
    rock.rotation_euler=(.1,.2,.35)
    rock.data.materials.append(material('Limestone'))
    modifier=rock.modifiers.new('Evaluated subdivision','SUBSURF')
    modifier.levels=modifier.render_levels=1
    bpy.ops.wm.save_as_mainfile(filepath=str((output/'grassland.blend').resolve()))
    protocol=dict(route=[[0,0],[0,.1],[0,.2]],headings=[0,0,0],world_bounds_m=[-2,2,-2,2])
    (output/'protocol.json').write_text(json.dumps(protocol)+'\n')
    (output/'world.json').write_text(json.dumps(dict(scene_sha256=hashlib.sha256((output/'grassland.blend').read_bytes()).hexdigest(),
        obstacles=[dict(x=.5,y=0,conservative_radius_m=.3)],role='synthetic integration fixture'))+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args(sys.argv[sys.argv.index('--')+1:]).output)
