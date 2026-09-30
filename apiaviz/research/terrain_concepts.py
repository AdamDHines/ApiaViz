"""Procedural ant-height terrain previews. Run with Blender's Python.

These are visual prototypes, not a replacement for any evaluated environment.
All geometry is seeded; the sun, camera and materials are recorded explicitly.
No downloaded assets or generated background pictures are used.

blender --background --factory-startup --python apiaviz/research/terrain_concepts.py -- --scene grassland
"""
import argparse
import json
import math
from pathlib import Path
import random
import sys

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260929
SUN_AZIMUTH = 35.
SUN_ELEVATION = 38.
EYE_HEIGHT = .01

CONCEPTS = {
    "grassland": dict(number=1, title="Dry grassland", ground=("A49475", "776A51"),
                      grass=("798653", "AFA779", "536347", "C2AE7B"),
                      rocks=("AAA796", "827F73"), description="Rounded limestone, olive and straw tussocks, scattered low scrub."),
    "woodland": dict(number=2, title="Open woodland", ground=("796148", "4C4536"),
                     grass=("63734D", "8F8956", "435D43", "A19162"),
                     rocks=("837967", "625F53"), description="Tree trunks, bark, curled leaf litter and dappled light on warm soil."),
    "dune": dict(number=3, title="Coastal dune", ground=("D4C29C", "AD9975"),
                 grass=("9AAB7C", "B8B288", "6E885F", "CDC093"),
                 rocks=("BFB39A", "999688"), description="Pale sand, fine dune grasses, smooth pebbles and weathered driftwood."),
    "garden": dict(number=4, title="Garden edge", ground=("897758", "534E3E"),
                   grass=("5C7B42", "7C9657", "3E603D", "9B9C65"),
                   rocks=("8C9183", "6B736A"), description="Broad green leaves, small white and yellow flowers, soil and rounded stones."),
}


def colour(hex_colour):
    rgb = [int(hex_colour[i:i+2], 16)/255 for i in (0, 2, 4)]
    return tuple(v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in rgb) + (1.,)


def material(name, a, b=None, scale=8., roughness=.85, bump=.0008, leaf=False, bark=False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Roughness"].default_value = roughness
    coords = nodes.new("ShaderNodeNewGeometry")
    texture = nodes.new("ShaderNodeTexNoise")
    texture.inputs["Scale"].default_value = scale
    texture.inputs["Detail"].default_value = 4
    vector = coords.outputs["Position"]
    if bark:
        stretch = nodes.new("ShaderNodeVectorMath")
        stretch.operation = "MULTIPLY"
        stretch.inputs[1].default_value = (4., 4., .3)
        links.new(vector, stretch.inputs[0])
        vector = stretch.outputs[0]
    links.new(vector, texture.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = .15
    ramp.color_ramp.elements[0].color = colour(b or a)
    ramp.color_ramp.elements[1].position = .85
    ramp.color_ramp.elements[1].color = colour(a)
    links.new(texture.outputs["Fac"], ramp.inputs[0])
    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    if bump:
        fine = nodes.new("ShaderNodeTexNoise")
        fine.inputs["Scale"].default_value = 900 if not bark else 42
        fine.inputs["Detail"].default_value = 2
        links.new(vector, fine.inputs["Vector"])
        normal = nodes.new("ShaderNodeBump")
        normal.inputs["Strength"].default_value = .28 if not bark else .5
        normal.inputs["Distance"].default_value = bump
        links.new(fine.outputs["Fac"], normal.inputs["Height"])
        links.new(normal.outputs["Normal"], shader.inputs["Normal"])
        if name.startswith("Ground"):
            grain_colour = nodes.new("ShaderNodeMixRGB")
            grain_colour.blend_type = "MULTIPLY"
            grain_colour.inputs[0].default_value = .65
            links.new(ramp.outputs["Color"], grain_colour.inputs[1])
            links.new(fine.outputs["Fac"], grain_colour.inputs[2])
            links.new(grain_colour.outputs[0], shader.inputs["Base Color"])
    if leaf:
        translucent = nodes.new("ShaderNodeBsdfTranslucent")
        links.new(ramp.outputs["Color"], translucent.inputs["Color"])
        mix = nodes.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = .13
        links.new(shader.outputs[0], mix.inputs[1])
        links.new(translucent.outputs[0], mix.inputs[2])
        links.new(mix.outputs[0], output.inputs["Surface"])
    else:
        links.new(shader.outputs[0], output.inputs["Surface"])
    mat.diffuse_color = colour(a)
    return mat


class Mesh:
    def __init__(self, name, materials):
        self.name, self.materials = name, materials
        self.vertices, self.faces, self.indices = [], [], []

    def add(self, vertices, faces, material_index=0):
        offset = len(self.vertices)
        self.vertices.extend(tuple(p) for p in vertices)
        self.faces.extend(tuple(offset+i for i in f) for f in faces)
        self.indices.extend([material_index]*len(faces))

    def finish(self, smooth=True):
        mesh = bpy.data.meshes.new(self.name)
        mesh.from_pydata(self.vertices, [], self.faces)
        mesh.update()
        obj = bpy.data.objects.new(self.name, mesh)
        bpy.context.collection.objects.link(obj)
        for mat in self.materials:
            mesh.materials.append(mat)
        for polygon, index in zip(mesh.polygons, self.indices):
            polygon.material_index = index
            polygon.use_smooth = smooth
        return obj


def height(x, y, dune=False):
    # Millimetre-scale irregularity near the camera, broader relief further away.
    fine = .0025*math.sin(x*5.7+y*1.1)*math.sin(y*3.2-.3*x)
    broad = (.027 if dune else .012)*math.sin(x*.75+.7)*math.sin(y*.63)
    return fine + broad


def tube(mesh, points, radii, material_index=0, sides=8):
    points = [Vector(p) for p in points]
    vertices = []
    for i, (p, radius) in enumerate(zip(points, radii)):
        direction = (points[min(i+1, len(points)-1)] - points[max(0, i-1)]).normalized()
        right = direction.cross(Vector((0, 0, 1)))
        if right.length < .01:
            right = direction.cross(Vector((0, 1, 0)))
        right.normalize()
        up = direction.cross(right).normalized()
        vertices.extend(p + radius*(math.cos(a)*right+math.sin(a)*up)
                        for a in [j*2*math.pi/sides for j in range(sides)])
    faces = []
    for i in range(len(points)-1):
        for j in range(sides):
            k = (j+1) % sides
            faces.append((i*sides+j, i*sides+k, (i+1)*sides+k, (i+1)*sides+j))
    faces += [tuple(reversed(range(sides))), tuple((len(points)-1)*sides+j for j in range(sides))]
    mesh.add(vertices, faces, material_index)


def blade(mesh, origin, angle, length, width, lean, material_index):
    origin = Vector(origin)
    along = Vector((math.cos(angle), math.sin(angle), 0))
    side = Vector((-along.y, along.x, 0))
    vertices = []
    for i in range(7):
        t = i/6
        mid = origin + along*lean*t*t + Vector((0, 0, length*(t-.2*t*t)))
        half = width*(1-t)**.65/2
        # A folded cross-section gives a real surface for directional shading.
        vertices.extend([mid-side*half, mid+Vector((0, 0, width*.16*(1-t))), mid+side*half])
    faces = []
    for i in range(6):
        for j in (0, 1):
            faces.append((3*i+j, 3*i+j+1, 3*(i+1)+j+1, 3*(i+1)+j))
    mesh.add(vertices, faces, material_index)


def leaf(mesh, origin, direction, length, width, material_index, curl=.15):
    origin, forward = Vector(origin), Vector(direction).normalized()
    right = forward.cross(Vector((0, 0, 1)))
    if right.length < .01:
        right = Vector((1, 0, 0))
    right.normalize()
    up = right.cross(forward).normalized()
    vertices = []
    for i in range(7):
        t = i/6
        mid = origin + forward*length*t + up*(length*curl*math.sin(math.pi*t))
        half = width/2*math.sin(math.pi*t)**.8
        vertices.extend([mid-right*half, mid+up*(half*.18), mid+right*half])
    faces = [(3*i+j, 3*i+j+1, 3*(i+1)+j+1, 3*(i+1)+j) for i in range(6) for j in (0, 1)]
    mesh.add(vertices, faces, material_index)


def rock(name, position, scale, mat, rng, smoothness=.07, detail=3):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=detail, radius=1, location=position)
    obj = bpy.context.object
    obj.name = name
    phase = rng.uniform(0, 6.28)
    for v in obj.data.vertices:
        x, y, z = v.co
        factor = 1 + smoothness*(math.sin(x*4.2+y*2+phase) + .5*math.cos(z*5.1-y*2.2))
        v.co *= factor
    obj.scale = scale
    obj.rotation_euler = [rng.uniform(-.15, .15), rng.uniform(-.15, .15), rng.uniform(0, math.tau)]
    obj.data.materials.append(mat)
    for face in obj.data.polygons:
        face.use_smooth = True
    return obj


def build(kind):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    cfg = CONCEPTS[kind]
    rng = random.Random(SEED+cfg["number"])
    dune = kind == "dune"
    ground = material("Ground / layered earth and fine grain", *cfg["ground"], scale=4, bump=.00065)
    stone = material("Stone / mineral variation", *cfg["rocks"], scale=17, roughness=.83, bump=.0005)
    bark = material("Bark / weathered wood", "A79B81" if dune else "7D715F", "655C4B", scale=8, bump=.0015, bark=True)
    greens = [material(f"Leaf / {i}", c, scale=25, roughness=.62, bump=.00004, leaf=True) for i, c in enumerate(cfg["grass"])]
    litter_mats = [material(f"Leaf litter / {i}", c, roughness=.94, bump=.00005)
                   for i, c in enumerate(("A88953", "765C3D", "B29265", "615A39"))]
    stems_mat = material("Plant stems", "718052", "536046", scale=30, bump=.00003)
    foliage, stems, litter = Mesh("Curved grass and leaves", greens), Mesh("Plant stems", [stems_mat]), Mesh("Curled leaf litter", litter_mats)
    woody = Mesh("Branches and driftwood", [bark])

    # A fine, mildly undulating ground mesh above a distant matching ground plane.
    terrain = Mesh("Uneven ground", [ground])
    n, span = 180, 28.
    vertices = []
    for j in range(n+1):
        y = -9+span*j/n
        for i in range(n+1):
            x = -span/2+span*i/n
            vertices.append((x, y, height(x, y, dune)))
    faces = [(j*(n+1)+i, j*(n+1)+i+1, (j+1)*(n+1)+i+1, (j+1)*(n+1)+i)
             for j in range(n) for i in range(n)]
    terrain.add(vertices, faces)
    terrain.finish()
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -.065))
    bpy.context.object.name = "Distant ground"
    bpy.context.object.data.materials.append(ground)

    # Irregular bare patches rather than an explicit route painted into the world.
    def place(radius_lo=.25, radius_hi=9.):
        angle = rng.uniform(0, math.tau)
        radius = rng.uniform(radius_lo, radius_hi)
        return radius*math.cos(angle), radius*math.sin(angle)

    count = {"grassland": 400, "woodland": 160, "dune": 190, "garden": 270}[kind]
    for _ in range(count):
        x, y = place(.24, 10.)
        if y > 0 and abs(x-.22*math.sin(y*1.2)) < .28 and rng.random() < .78:
            continue
        for _ in range(rng.randint(9, 21)):
            bx, by = x+rng.gauss(0, .035), y+rng.gauss(0, .035)
            h = rng.uniform(.08, .24) * (1.3 if dune else 1.)
            blade(foliage, (bx, by, height(bx, by, dune)), rng.uniform(0, math.tau), h,
                  rng.uniform(.004, .011), rng.uniform(.015, .13), rng.randrange(4))

    # Rocks are smoothly deformed volumes, not isolated flat triangles.
    for i in range(95):
        x, y = place(.28, 7.)
        radius = rng.uniform(.015, .085)
        rock(f"Pebble {i:03d}", (x, y, height(x, y, dune)+radius*.28),
             (radius, radius*rng.uniform(.65, 1.35), radius*rng.uniform(.45, .85)), stone, rng)
    for i in range(240):
        a = rng.uniform(0, math.tau)
        distance = math.exp(rng.uniform(math.log(.022), math.log(.48)))
        x, y = distance*math.cos(a), distance*math.sin(a)
        radius = rng.uniform(.0005, .0025)
        rock(f"Ground grain {i:03d}", (x, y, height(x, y, dune)+radius*.28),
             (radius, radius*rng.uniform(.6, 1.3), radius*.6), stone, rng, detail=2)
    landmarks = [(-.55, .9, .20), (.85, 2.1, .32), (-1.35, 3.8, .40)]
    if kind == "woodland": landmarks = [(-.6, 1.1, .12), (1.1, 3.1, .21)]
    if kind == "garden": landmarks = [(-.38, .85, .14), (.55, 1.4, .17), (-1.1, 3.3, .25)]
    if dune: landmarks = [(-.65, 1.5, .11), (.85, 2.6, .15)]
    for i, (x, y, r) in enumerate(landmarks):
        rock(f"Landmark stone {i}", (x, y, height(x, y, dune)+r*.28), (r*1.2, r, r*.7), stone, rng)

    if kind in ("woodland", "grassland", "garden"):
        for i in range(600 if kind == "woodland" else 130):
            x, y = place(.06, 7.)
            a = rng.uniform(0, math.tau)
            length = rng.uniform(.018, .085)
            leaf(litter, (x, y, height(x, y, dune)+.0018), (math.cos(a), math.sin(a), rng.uniform(-.03, .08)),
                 length, length*rng.uniform(.22, .6), rng.randrange(4), curl=rng.uniform(.02, .14))
        for i in range(65):
            x, y = place(.15, 7.)
            a = rng.uniform(0, math.tau)
            length = rng.uniform(.04, .20)
            p = Vector((x, y, height(x, y, dune)+.003))
            delta = Vector((math.cos(a), math.sin(a), .04))*length
            tube(woody, [p, p+delta*.5+Vector((0, 0, .004)), p+delta], [.0024, .002, .0008], sides=6)

    if kind in ("woodland", "dune"):
        for i, (x, y, angle, length, radius) in enumerate(((-.8, 1.9, -.2, 1.0, .047), (1.0, 3.7, 2.2, 1.4, .075))):
            points = []
            for j in range(7):
                t = j/6
                px = x+math.cos(angle)*length*t
                py = y+math.sin(angle)*length*t
                points.append((px, py, height(px, py, dune)+radius*.7 + .04*math.sin(t*math.pi)))
            tube(woody, points, [radius*(1-.55*j/6) for j in range(7)], sides=14)
            start = Vector(points[3])
            tube(woody, [start, start+Vector((.13, .15, .10)), start+Vector((.23, .24, .12))],
                 [radius*.38, radius*.20, .005], sides=9)

    def shrub(x, y, size, leaf_count=90):
        base = Vector((x, y, height(x, y, dune)))
        branches = []
        for j in range(7):
            a = j*math.tau/7+rng.uniform(-.3, .3)
            end = base + Vector((math.cos(a)*size*.6, math.sin(a)*size*.6, rng.uniform(.55, 1.)*size))
            tube(stems, [base, base+(end-base)*.6, end], [.009*size, .004*size, .0015*size], sides=6)
            branches.append(end)
        for j in range(leaf_count):
            a = rng.uniform(0, math.tau)
            stem_point = base.lerp(rng.choice(branches), rng.uniform(.15, 1.))
            direction = Vector((math.cos(a), math.sin(a), rng.uniform(-.25, .45))).normalized()
            p = stem_point + direction*rng.uniform(.018, .065)*size
            tube(stems, [stem_point, p], [.0013*size, .0004*size], sides=5)
            length = rng.uniform(.05, .14)*max(.65, size)
            leaf(foliage, p, direction, length,
                 length*(.55 if kind == "garden" else .3), rng.choice([0, 2]), curl=.11)

    if kind in ("grassland", "dune"):
        for x, y, size in ((-2.5, 4, .7), (2.8, 5, .9), (-1.5, 8, 1.0), (1.8, 10, 1.1), (-4, 1, .65), (4, -1, .8)):
            shrub(x, y, size*.6 if dune else size, 400)

    if kind == "woodland":
        for i, (x, y, radius, h) in enumerate(((-2.1, 3.7, .17, 3.5), (2.9, 5.4, .23, 4.1),
                                               (-4.8, 7.2, .27, 4.7), (4.4, 10., .18, 4.0), (1., 11., .13, 3.7))):
            base = Vector((x, y, height(x, y)))
            tube(woody, [base, base+Vector((.06, 0, h*.4)), base+Vector((-.08, .04, h*.75)), base+Vector((.10, 0, h))],
                 [radius*1.15, radius*.78, radius*.5, radius*.22], sides=16)
            for j in range(5):
                a = rng.uniform(0, math.tau)
                start = base+Vector((0, 0, h*rng.uniform(.42, .76)))
                end = start+Vector((math.cos(a)*1.1, math.sin(a)*1.1, .65))
                tube(woody, [start, (start+end)/2+Vector((0, 0, .1)), end], [radius*.32, radius*.15, .012], sides=10)
                for k in range(14):
                    twig_start = start.lerp(end, rng.uniform(.5, .95))
                    tip = end+Vector((rng.gauss(0, .48), rng.gauss(0, .48), rng.gauss(0, .23)))
                    tube(woody, [twig_start, tip], [.007, .0012], sides=6)
                    for l in range(12):
                        p = twig_start.lerp(tip, rng.uniform(.25, 1.))
                        theta = rng.uniform(0, math.tau)
                        leaf(foliage, p, (math.cos(theta), math.sin(theta), rng.uniform(-1.5, .2)),
                             rng.uniform(.10, .22), rng.uniform(.02, .055), rng.choice([0, 2]), curl=.06)
        for x, y in ((-1.1, 2.7), (1.2, 5.), (-3.4, 1.5)):
            shrub(x, y, .45, 110)

    if kind == "garden":
        cream = material("Flower petals / ivory", "F0EAD2", "DDD8BC", roughness=.7, bump=0, leaf=True)
        yellow = material("Flower centre / ochre", "D1AE43", "A98629", roughness=.9, bump=.0002)
        flowers = Mesh("Small white flowers", [cream])
        for x, y, size in ((-.45, .62, .35), (.53, .95, .4), (-.85, 1.7, .5), (.9, 2.5, .7), (-2.1, 4, 1.1), (2.4, 4.7, 1.4)):
            shrub(x, y, size, 95)
        flower_positions = [(.10, .38, .105), (-.18, .62, .17), (-.26, .55, .10), (.40, .80, .16), (-.55, 1.3, .14)]
        flower_positions += [(rng.choice([-1, 1])*rng.uniform(.35, 1.65), rng.uniform(.4, 3.9), rng.uniform(.075, .23)) for _ in range(28)]
        for i, (x, y, h) in enumerate(flower_positions):
            z = height(x, y)
            centre = Vector((x, y, z+h))
            tube(stems, [(x, y, z), (x+.01, y, z+h*.6), centre], [.0015, .001, .0007], sides=6)
            u, v = Vector((1, 0, 0)), Vector((0, .55, .835)).normalized()
            normal = u.cross(v).normalized()
            for j in range(9):
                a = j*math.tau/9
                leaf(flowers, centre, math.cos(a)*u+math.sin(a)*v-.12*normal, .014, .006, 0, curl=.14)
            flower_centre = rock(f"Flower centre {i}", centre, (.005, .005, .0028), yellow, rng, smoothness=.015)
            flower_centre.rotation_euler = normal.to_track_quat("Z", "Y").to_euler()
        flowers.finish()

    foliage.finish()
    stems.finish()
    litter.finish()
    woody.finish()
    setup_light_camera(kind)
    return cfg


def setup_light_camera(kind):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 5
    scene.cycles.diffuse_bounces = 3
    scene.cycles.transparent_max_bounces = 6
    scene.cycles.seed = SEED
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = .3
    preferences = bpy.context.preferences.addons["cycles"].preferences
    try:
        preferences.compute_device_type = "METAL"
        preferences.refresh_devices()
        for device in preferences.devices:
            device.use = device.type == "METAL"
        scene.cycles.device = "GPU"
    except (TypeError, RuntimeError):
        scene.cycles.device = "CPU"

    world = bpy.data.worlds.new("Daylight sky")
    world.use_nodes = True
    scene.world = world
    sky = world.node_tree.nodes.new("ShaderNodeTexSky")
    available = [e.identifier for e in sky.bl_rna.properties["sky_type"].enum_items]
    sky.sky_type = "HOSEK_WILKIE"
    az, el = math.radians(SUN_AZIMUTH), math.radians(SUN_ELEVATION)
    direction = Vector((math.cos(az)*math.cos(el), math.sin(az)*math.cos(el), math.sin(el)))
    sky.sun_direction = direction
    sky.turbidity = 2.5
    sky.ground_albedo = .25
    for name, value in (("sun_elevation", math.radians(SUN_ELEVATION)),
                        ("sun_rotation", math.radians(SUN_AZIMUTH)), ("sun_disc", False),
                        ("altitude", .03), ("air_density", 1.), ("dust_density", 1.)):
        if hasattr(sky, name):
            setattr(sky, name, value)
    background = world.node_tree.nodes.get("Background")
    background.inputs["Strength"].default_value = 3.
    world.node_tree.links.new(sky.outputs["Color"], background.inputs["Color"])
    light = bpy.data.lights.new("Sun / explicit azimuth and elevation", "SUN")
    light.energy = 4.0
    light.angle = math.radians(.7)
    light.color = (1., .95, .86)
    sun = bpy.data.objects.new("Sun", light)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
    camera = bpy.data.cameras.new("Ant-eye camera")
    obj = bpy.data.objects.new("Ant-eye camera", camera)
    bpy.context.collection.objects.link(obj)
    obj.location = (0, 0, height(0, 0, kind == "dune")+EYE_HEIGHT)
    obj.rotation_euler = Vector((0, 1, .05)).to_track_quat("-Z", "Y").to_euler()
    camera.type = "PERSP"
    camera.sensor_fit = "HORIZONTAL"
    camera.angle = math.radians(100)
    camera.clip_start = .0005
    camera.clip_end = 200
    camera.dof.use_dof = False
    scene.camera = obj


def render(kind, output, draft=False, panorama=False):
    cfg = build(kind)
    scene = bpy.context.scene
    stem = f"{cfg['number']:02d}-{kind}"
    output.mkdir(parents=True, exist_ok=True)
    scene.render.resolution_x = 1000 if draft else 1800
    scene.render.resolution_y = 625 if draft else 1125
    scene.cycles.samples = 20 if draft else 64
    scene.render.filepath = str(output / f"{stem}{'-draft' if draft else ''}.png")
    print(f"RENDER {kind}: perspective", flush=True)
    bpy.ops.render.render(write_still=True)
    files = [Path(scene.render.filepath).name]
    if panorama and not draft:
        camera = scene.camera.data
        camera.type = "PANO"
        camera.panorama_type = "EQUIRECTANGULAR"
        camera.longitude_min = math.radians(-148)
        camera.longitude_max = math.radians(148)
        camera.latitude_min = math.radians(-15)
        camera.latitude_max = math.radians(60)
        scene.camera.rotation_euler = Vector((0, 1, 0)).to_track_quat("-Z", "Y").to_euler()
        scene.render.resolution_x = 1776
        scene.render.resolution_y = 450
        scene.render.filepath = str(output / f"{stem}-panorama.png")
        print(f"RENDER {kind}: navigation-FOV panorama", flush=True)
        bpy.ops.render.render(write_still=True)
        files.append(Path(scene.render.filepath).name)
        camera.type = "PERSP"
        scene.camera.rotation_euler = Vector((0, 1, .05)).to_track_quat("-Z", "Y").to_euler()
        scene.render.resolution_x, scene.render.resolution_y = 1800, 1125
    if not draft:
        scene.render.filepath = str(output / f"{stem}.png")
        bpy.ops.wm.save_as_mainfile(filepath=str(output / f"{stem}.blend"), compress=True)
    metadata = dict(concept=kind, title=cfg["title"], description=cfg["description"],
                    geometry_seed=SEED+cfg["number"], render_seed=SEED,
                    sun=dict(azimuth_deg=SUN_AZIMUTH, elevation_deg=SUN_ELEVATION,
                             convention="azimuth counterclockwise from world +X; +Z up", energy=4., angle_deg=.7),
                    camera=dict(eye_height_above_ground_m=EYE_HEIGHT, position=list(scene.camera.location),
                                perspective_hfov_deg=100, perspective_pitch_deg=math.degrees(math.atan(.05)),
                                depth_of_field=False, heading="world +Y"),
                    panorama=dict(hfov_deg=296, elevation_min_deg=-15, elevation_max_deg=60),
                    render=dict(engine="Cycles", device=scene.cycles.device, samples=scene.cycles.samples,
                                view_transform="AgX", exposure=.3, denoising=True, blender=bpy.app.version_string),
                    sky=dict(model="HOSEK_WILKIE", strength=3., turbidity=2.5, ground_albedo=.25),
                    objects=len(scene.objects), mesh_faces=sum(len(o.data.polygons) for o in scene.objects if o.type=="MESH"),
                    images=files, draft=draft,
                    scope="Visual terrain prototypes only. Not used for training, evaluation or model input. All assets procedural; no external photographs or generated images.")
    (output / f"{stem}{'-draft' if draft else ''}.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"COMPLETE {kind}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=[*CONCEPTS, "all"], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/terrain-concepts")
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--panorama", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    for kind in CONCEPTS if args.scene == "all" else [args.scene]:
        render(kind, args.output.resolve(), args.draft, args.panorama)


if __name__ == "__main__":
    main()
