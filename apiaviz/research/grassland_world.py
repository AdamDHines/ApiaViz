"""Blender worker for the grassland navigation smoke test.

Run with Blender, after ``python -m apiaviz.research.grassland_smoke prepare``.
File requests keep Blender's scene resident while the unchanged Python navigation
controller asks for new positions. Every heading at a position uses one panorama.
"""
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from terrain_concepts import (Mesh, blade, height, leaf, material, rock,
                              setup_light_camera, tube)


def build(protocol, output):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rng = random.Random(protocol['geometry_seed'])
    xmin, xmax, ymin, ymax = protocol['world_bounds_m']
    route = [Vector(p) for p in protocol['route']]

    def distance(x, y):
        p = Vector((x, y))
        return min((p - (a + (b-a)*max(0., min(1., (p-a).dot(b-a)/(b-a).length_squared)))).length
                   for a, b in zip(route[:-1], route[1:]))

    def position(clearance):
        for _ in range(10000):
            x, y = rng.uniform(xmin+.45, xmax-.45), rng.uniform(ymin+.45, ymax-.45)
            if distance(x, y) > clearance:
                return x, y
        raise RuntimeError('Could not place an object outside the teaching corridor')

    ground = material('Grassland earth', 'A49475', '776A51', scale=4, bump=.00065)
    stone = material('Limestone', 'AAA796', '827F73', scale=17, bump=.0005)
    grass = [material(f'Grass {i}', c, scale=25, roughness=.62, bump=.00004, leaf=True)
             for i, c in enumerate(('798653', 'AFA779', '536347', 'C2AE7B'))]
    stem_mat = material('Scrub stems', '718052', '536046', scale=30, bump=.00003)
    bark = material('Weathered twigs', '7D715F', '655C4B', scale=8, bump=.0015, bark=True)
    litter_mat = [material(f'Litter {i}', c, bump=.00005)
                  for i, c in enumerate(('A88953', '765C3D', 'B29265', '615A39'))]
    terrain = Mesh('Walkable terrain', [ground])
    n = 220
    vertices = [(xmin+(xmax-xmin)*i/n, ymin+(ymax-ymin)*j/n,
                 height(xmin+(xmax-xmin)*i/n, ymin+(ymax-ymin)*j/n))
                for j in range(n+1) for i in range(n+1)]
    faces = [(j*(n+1)+i, j*(n+1)+i+1, (j+1)*(n+1)+i+1, (j+1)*(n+1)+i)
             for j in range(n) for i in range(n)]
    terrain.add(vertices, faces)
    terrain.finish()
    bpy.ops.mesh.primitive_plane_add(size=200, location=(5, 5, -.065))
    bpy.context.object.name = 'Distant ground beyond landmark field'
    bpy.context.object.data.materials.append(ground)
    foliage, stems = Mesh('Grass and scrub leaves', grass), Mesh('Scrub stems', [stem_mat])
    litter, wood = Mesh('Leaf litter', litter_mat), Mesh('Twigs', [bark])
    obstacles = []
    for i in range(390):
        x, y = position(.36)
        for _ in range(rng.randint(9, 21)):
            bx, by = x+rng.gauss(0, .025), y+rng.gauss(0, .025)
            blade(foliage, (bx, by, height(bx, by)), rng.uniform(0, math.tau),
                  rng.uniform(.08, .24), rng.uniform(.004, .011),
                  rng.uniform(.015, .10), rng.randrange(4))
    for i in range(110):
        r = rng.uniform(.018, .08) if i < 96 else rng.uniform(.14, .34)
        x, y = position(.25+1.5*r)
        rock(f'Limestone {i:03d}', (x, y, height(x,y)+r*.28),
             (r*1.2, r, r*.7), stone, rng)
        obstacles.append(dict(x=x, y=y, conservative_radius_m=1.5*r))
    for i in range(20):
        size = rng.uniform(.30, .75)
        x, y = position(.25+size*.8)
        base = Vector((x, y, height(x,y)))
        branches = []
        for j in range(7):
            angle = j*math.tau/7+rng.uniform(-.3,.3)
            end = base+Vector((math.cos(angle)*size*.6, math.sin(angle)*size*.6, rng.uniform(.55,1)*size))
            tube(stems, [base, base.lerp(end,.6), end], [.009*size,.004*size,.0015*size], sides=6)
            branches.append(end)
        for _ in range(180):
            angle = rng.uniform(0,math.tau)
            p = base.lerp(rng.choice(branches),rng.uniform(.15,1))
            direction = Vector((math.cos(angle),math.sin(angle),rng.uniform(-.25,.45))).normalized()
            tip = p+direction*rng.uniform(.018,.065)*size
            tube(stems, [p,tip], [.0013*size,.0004*size], sides=5)
            length = rng.uniform(.05,.14)*max(.65,size)
            leaf(foliage,tip,direction,length,length*.3,rng.choice([0,2]))
    for _ in range(350):
        x, y = rng.uniform(xmin,xmax), rng.uniform(ymin,ymax)
        angle, length = rng.uniform(0,math.tau),rng.uniform(.018,.085)
        leaf(litter,(x,y,height(x,y)+.0018),(math.cos(angle),math.sin(angle),.02),
             length,length*rng.uniform(.22,.6),rng.randrange(4),curl=.07)
    for _ in range(100):
        x,y = position(.3)
        angle,length = rng.uniform(0,math.tau),rng.uniform(.04,.20)
        p = Vector((x,y,height(x,y)+.003))
        delta=Vector((math.cos(angle),math.sin(angle),.04))*length
        tube(wood,[p,p+delta*.5+Vector((0,0,.004)),p+delta],[.0024,.002,.0008],sides=6)
    for mesh in (foliage, stems, litter, wood):
        mesh.finish()
    setup_light_camera('grassland')
    scene = bpy.context.scene
    scene.render.use_persistent_data = True
    scene.cycles.samples = protocol['renderer']['samples']
    scene.cycles.seed = protocol['renderer']['seed']
    scene.cycles.use_animated_seed = False
    scene.cycles.use_denoising = False
    scene.render.resolution_x, scene.render.resolution_y = protocol['renderer']['panorama_size']
    camera = scene.camera.data
    camera.type = 'PANO'
    camera.panorama_type = 'EQUIRECTANGULAR'
    camera.longitude_min, camera.longitude_max = -math.pi, math.pi
    camera.latitude_min, camera.latitude_max = map(math.radians, (-15.5,60.5))
    # Conventional photograph: image left is positive azimuth from forward +X.
    scene.camera.rotation_euler = Vector((1,0,0)).to_track_quat('-Z','Y').to_euler()
    set_position(protocol['route'][0])
    bpy.ops.wm.save_as_mainfile(filepath=str(output/'grassland.blend'),compress=True)
    metadata = dict(bounds_m=protocol['world_bounds_m'], geometry_seed=protocol['geometry_seed'],
                    mesh_faces=sum(len(o.data.polygons) for o in scene.objects if o.type=='MESH'),
                    objects=len(scene.objects), blender=bpy.app.version_string,
                    obstacles=obstacles, renderer=protocol['renderer'],
                    placement='Random positions; reserve an unobstructed teaching corridor. No route marking or ground-colour change.',
                    sun=dict(azimuth_deg=35, elevation_deg=38),
                    scene_sha256=hashlib.sha256((output/'grassland.blend').read_bytes()).hexdigest())
    (output/'world.json').write_text(json.dumps(metadata,indent=2)+'\n')


def ground_z(x,y):
    ground = bpy.data.objects['Walkable terrain']
    hit, point, _, _ = ground.ray_cast(Vector((x,y,10)),Vector((0,0,-1)))
    return point.z if hit else -.065


def set_position(position):
    x,y=position
    bpy.context.scene.camera.location=(x,y,ground_z(x,y)+.01)


def render_position(position, path):
    set_position(position)
    scene=bpy.context.scene
    scene.render.filepath=str(path)
    bpy.ops.render.render(write_still=True)


def calibrate(output):
    """Render physical cardinal markers to check the azimuth/elevation adapter."""
    scene=bpy.context.scene
    objects=[o for o in scene.objects if o.type=='MESH']
    for obj in objects: obj.hide_render=True
    markers=[]
    for name,position,colour in [('east',(2,0,.01),(1,0,0,1)),
                                  ('north',(0,2,.01),(0,1,0,1)),
                                  ('west',(-2,0,.01),(0,0,1,1)),
                                  ('south',(0,-2,.01),(1,1,0,1))]:
        mat=bpy.data.materials.new('Calibration '+name)
        mat.use_nodes=True
        nodes=mat.node_tree.nodes
        nodes.clear()
        emitter=nodes.new('ShaderNodeEmission')
        emitter.inputs['Color'].default_value=colour
        output_node=nodes.new('ShaderNodeOutputMaterial')
        mat.node_tree.links.new(emitter.outputs[0],output_node.inputs['Surface'])
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=12,radius=.2,location=position)
        obj=bpy.context.object
        obj.data.materials.append(mat)
        markers.append(obj)
    scene.camera.location=(0,0,.01)
    scene.render.filepath=str(output/'calibration.png')
    bpy.ops.render.render(write_still=True)
    for obj in markers: bpy.data.objects.remove(obj,do_unlink=True)
    for obj in objects: obj.hide_render=False


def previews(protocol,output):
    """Viewing exports only; do not overwrite the scene or sensory cache."""
    output.mkdir(parents=True,exist_ok=True)
    scene=bpy.context.scene
    scene.cycles.samples=64
    scene.cycles.use_denoising=True
    scene.render.resolution_x,scene.render.resolution_y=1500,844
    camera=scene.camera.data
    camera.type='PERSP'
    camera.angle=math.radians(100)
    heading=math.radians(protocol['headings'][0])
    scene.camera.rotation_euler=Vector((math.cos(heading),math.sin(heading),.05)).to_track_quat('-Z','Y').to_euler()
    render_position(protocol['route'][0],output/'grassland-view.png')
    xmin,xmax,ymin,ymax=protocol['world_bounds_m']
    scene.camera.location=((xmin+xmax)/2,(ymin+ymax)/2,15)
    scene.camera.rotation_euler=(0,0,0)
    camera.type='ORTHO'
    camera.ortho_scale=xmax-xmin
    scene.render.resolution_x=1400
    scene.render.resolution_y=round(1400*(ymax-ymin)/(xmax-xmin))
    scene.render.filepath=str(output/'world-overhead.png')
    bpy.ops.render.render(write_still=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--build-only',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Load the saved scene and continue filling its panorama cache')
    parser.add_argument('--preview-dir',type=Path,help='Export perspective/overhead views and exit; requires --resume')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    out=args.output.resolve()
    protocol=json.loads((out/'protocol.json').read_text())
    if args.preview_dir and not args.resume:
        raise ValueError('Viewing exports require --resume to use the saved evaluated scene')
    if args.resume:
        metadata=json.loads((out/'world.json').read_text())
        assert metadata['scene_sha256']==hashlib.sha256((out/'grassland.blend').read_bytes()).hexdigest()
        assert metadata['bounds_m']==protocol['world_bounds_m']
        assert metadata['geometry_seed']==protocol['geometry_seed'] and metadata['renderer']==protocol['renderer']
        bpy.ops.wm.open_mainfile(filepath=str(out/'grassland.blend'))
        preferences=bpy.context.preferences.addons['cycles'].preferences
        try:
            preferences.compute_device_type='METAL'
            preferences.refresh_devices()
            for device in preferences.devices: device.use=device.type=='METAL'
            bpy.context.scene.cycles.device='GPU'
        except (TypeError,RuntimeError):
            bpy.context.scene.cycles.device='CPU'
        if not args.preview_dir and (out/'stop').exists(): (out/'stop').unlink()
    else:
        if (out/'world.json').exists():
            raise FileExistsError('Saved world exists: use --resume or a new output directory')
        build(protocol,out)
        calibrate(out)
        render_position(protocol['route'][0],out/'start-panorama.png')
    if args.preview_dir:
        previews(protocol,args.preview_dir.resolve())
        return
    (out/'ready.json').write_text(json.dumps(dict(ready=True)))
    if args.build_only:
        return
    queue=out/'queue'
    queue.mkdir(exist_ok=True)
    while not (out/'stop').exists():
        jobs=sorted(queue.glob('*.request.json'))
        if not jobs:
            time.sleep(.025)
            continue
        for job in jobs:
            request=json.loads(job.read_text())
            try:
                for view in request['views']:
                    target=out/'panoramas'/f"{view['key']}.png"
                    if not target.exists():
                        render_position(view['position'],target)
                response=dict(ok=True)
            except Exception:
                response=dict(ok=False,error=traceback.format_exc())
            target=job.with_name(job.name.replace('.request.json','.response.json'))
            temp=target.with_suffix('.tmp')
            temp.write_text(json.dumps(response))
            temp.replace(target)
            job.unlink()


if __name__=='__main__':
    main()
