"""Make audited, explanatory videos of saved linear-colour navigation trials.

The displayed high-resolution panorama is for the viewer only. All decisions,
feature maps and spike patterns are recomputed at the original 18 x 74 input
resolution and checked against the saved trial. Playback timing is illustrative.
"""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .fast_render import accelerate_world, SparseWorldRenderer
from .frontend_refinements import RefinementEncoder, refinement_maps
from .navigation import World, choose_heading
from .spike_overlap import SpikeOverlapMemory
from .study import encode, fingerprint, file_hash, write_json

ROOT = Path(__file__).resolve().parents[2]
SIZE = (1920, 1080)
BG = (8, 20, 27)
CARD = (17, 36, 45)
EDGE = (36, 60, 69)
WHITE = (238, 246, 245)
MUTED = (163, 186, 193)
TEAL = (90, 230, 195)
GOLD = (255, 193, 112)
BLUE = (122, 172, 255)
PURPLE = (196, 155, 255)


@lru_cache(None)
def font(size, bold=False):
    candidates = [("/System/Library/Fonts/Avenir Next.ttc", 2 if bold else 7),
                  ("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0),
                  ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0)]
    for path, index in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size, index=index)
    return ImageFont.load_default(size=size)


def text(draw, xy, value, size=22, fill=WHITE, bold=False, anchor=None):
    draw.text(xy, str(value), font=font(size, bold), fill=fill, anchor=anchor)


def rgb_image(array):
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def tinted(values, scale, colour, signed=False):
    v = np.asarray(values, dtype=np.float32) / max(float(scale), 1e-6)
    base = np.array(BG, dtype=np.float32)
    strength = np.sqrt(np.clip(np.abs(v), 0, 1))[..., None]
    tint = np.broadcast_to(np.array(colour), (*v.shape, 3)).copy()
    if signed:
        tint[v < 0] = BLUE
    return rgb_image(base + strength * (tint - base))


def case(ant, seed):
    found = []
    for name in json.loads((ROOT / "docs/frontend-factorial/runs.json").read_text())["runs"]:
        run = ROOT / name
        manifest = json.loads((run / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        for row in map(json.loads, (run / "results.jsonl").read_text().splitlines()):
            if (row["ant"], row["seed"], row["preprocessing"]) == (ant, seed, "linear_colour"):
                found.append((run, manifest, row))
    assert len(found) == 1, (ant, seed, len(found))
    return found[0]


@torch.no_grad()
def prepare(ant, seed, world, cache_root):
    run, manifest, row = case(ant, seed)
    trace_file = run / row["trace"]
    saved = json.loads(trace_file.read_text())
    source = ROOT / manifest["stimulus_sources"][str(ant)]["path"]
    bank_file = source / f"ant-{ant}-images.pt"
    checkpoint = ROOT / manifest["encoders"][str(seed)]["source"]
    assert file_hash(bank_file) == manifest["stimulus_sources"][str(ant)]["images_sha256"]
    assert file_hash(checkpoint) == manifest["encoders"][str(seed)]["sha256"]
    provenance = dict(source_run=str(run.relative_to(ROOT)), trace_sha256=file_hash(trace_file),
                      bank_sha256=file_hash(bank_file), checkpoint_sha256=file_hash(checkpoint),
                      encoder_source_sha256=file_hash(Path(__file__).with_name("frontend_refinements.py")),
                      renderer_source_sha256=file_hash(Path(__file__).with_name("fast_render.py")),
                      video_source_sha256=file_hash(Path(__file__)))
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = cache_root / f"ant-{ant:02d}-seed-{seed}.npz"
    meta_file = cache.with_suffix(".json")
    if cache.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        # Video styling can change without recalculating audited model outputs.
        compare = [k for k in provenance if k != "video_source_sha256"]
        if all(meta["provenance"][k] == provenance[k] for k in compare):
            print(f"Ant {ant}: using audited sensor cache", flush=True)
            with np.load(cache) as z:
                data = {k: z[k] for k in z.files}
            return row, saved, data, dict(meta, provenance=provenance)

    model = RefinementEncoder("linear_colour", seed=seed)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu")["state_dict"])
    assert fingerprint(model) == manifest["encoders"][str(seed)]["fingerprint"]
    training = torch.load(bank_file, weights_only=True, map_location="cpu")["training"]
    memory = SpikeOverlapMemory(encode(model, training))
    assert fingerprint(memory) == row["memory_fingerprint"]
    raw, sampled, maps, spikes, winners = [], [], [], [], []
    max_error = 0.
    route, headings = world.route(ant, 2)
    offsets = np.arange(60., -61., -10.)
    previous_heading = float(headings[0])
    for i, (decision, step) in enumerate(zip(saved["decisions"], saved["trajectory"])):
        images = world.scan(decision["position"], decision["headings"])
        codes = encode(model, images)
        scores = memory(codes).numpy()
        reference = np.asarray(decision["scores"])
        np.testing.assert_array_equal(scores, reference)
        max_error = max(max_error, float(np.max(np.abs(scores-reference))))
        winner, silent = choose_heading(scores, offsets)
        np.testing.assert_allclose(decision["headings"], previous_heading + offsets, atol=1e-9, rtol=0)
        assert float(decision["headings"][winner]) == step["heading"]
        assert silent == step["no_evidence"]
        position = np.asarray(decision["position"]) + .1 * np.array([
            np.cos(np.radians(step["heading"])), np.sin(np.radians(step["heading"]))])
        np.testing.assert_allclose(position, step["position"], atol=1e-9, rtol=0)
        previous_heading = step["heading"]
        feature = refinement_maps(images, "linear_colour", model.features.backbone)
        sampled_gb = (model.features.backbone.spatial_sampler(images[:, -2:] * 2 - 1) + 1) / 2
        raw.append((images.permute(0, 2, 3, 1).numpy()*255).round().astype(np.uint8))
        sampled.append((sampled_gb.permute(0, 2, 3, 1).numpy()*255).round().astype(np.uint8))
        maps.append(torch.cat(feature, 1).numpy().astype(np.float16))
        spikes.append(np.packbits((codes.numpy() > 0), axis=1))
        winners.append(winner)
        if (i+1) % 40 == 0:
            print(f"Ant {ant}: verified {i+1}/{len(saved['decisions'])} complete scans", flush=True)
    data = dict(raw=np.stack(raw), sampled=np.stack(sampled), maps=np.stack(maps),
                spikes=np.stack(spikes), winners=np.asarray(winners))
    # Fixed within each clip, only for visualisation; never fed back to the model.
    scales = [float(np.quantile(np.abs(data["maps"][:, :, c].astype(np.float32)), .995)) for c in range(5)]
    meta = dict(provenance=provenance, ant=ant, seed=seed, model="linear_colour",
                verified_scans=len(winners), verified_heading_scores=len(winners)*13,
                score_max_absolute_error=max_error, feature_display_scales=scales,
                memory_fingerprint=fingerprint(memory), n_teaching_views=len(training))
    np.savez_compressed(cache, **data)
    write_json(meta_file, meta)
    return row, saved, data, meta


class Dashboard:
    def __init__(self, ant, seed, row, saved, data, meta, world):
        self.ant, self.seed, self.row = ant, seed, row
        self.saved, self.data, self.meta, self.world = saved, data, meta, world
        self.route, self.initial_headings = world.route(ant, 2)
        self.steps = saved["trajectory"]
        self.positions = np.array([saved["decisions"][0]["position"]] + [s["position"] for s in self.steps])
        self.bits = np.unpackbits(data["spikes"], axis=2, count=8000)
        self.base = self.make_base()
        raw_world = {k: getattr(world.renderer, k) for k in ("X", "Y", "Z", "colp")}
        self.display_renderer = SparseWorldRenderer(raw_world, device="cpu", hfov=360., resolution=.5,
                                                    color=True, triangle_color=world.renderer.triangle_color)
        self.panoramas = {}

    def make_base(self):
        canvas = Image.new("RGB", SIZE, BG)
        d = ImageDraw.Draw(canvas)
        text(d, (36, 20), "APIA VIZ   /   NAVIGATION", 18, TEAL, True)
        text(d, (36, 49), "Following a remembered route", 38, WHITE, True)
        text(d, (38, 98), "Linear colour  ·  Original form filters  ·  Spiking route memory", 19, MUTED)
        text(d, (1884, 27), f"ANT {self.ant:02d}  /  ROUTE 2", 30, WHITE, True, "ra")
        text(d, (1884, 73), f"Wiring seed {self.seed}  ·  Recorded evaluation", 19, MUTED, anchor="ra")
        for box in ((36, 138, 1166, 500), (36, 520, 1166, 880),
                    (36, 900, 1166, 1036), (1186, 138, 1884, 680), (1186, 700, 1884, 1036)):
            d.rounded_rectangle(box, radius=19, fill=CARD, outline=EDGE, width=2)
        text(d, (58, 153), "01   THE ANT'S VIEW", 22, WHITE, True)
        text(d, (58, 468), "296° panorama  ·  Detailed render for viewing; the model input is shown below", 17, MUTED)
        text(d, (58, 535), "02   FROM AN IMAGE TO VISUAL FEATURES", 22, WHITE, True)
        text(d, (65, 579), "Actual input · 74 × 18 RGB", 19, WHITE, True)
        text(d, (65, 697), "Neighbour averaging · G + B", 19, WHITE, True)
        text(d, (420, 579), "Bright contrast", 18, TEAL, True)
        text(d, (598, 579), "Dark contrast", 18, BLUE, True)
        text(d, (420, 697), "Local contrast", 19, WHITE, True)
        text(d, (808, 579), "More green than blue", 19, TEAL, True)
        text(d, (808, 697), "More blue than green", 19, PURPLE, True)
        text(d, (65, 829), "Red is not used", 17, MUTED)
        text(d, (420, 829), "Original form pathway", 17, MUTED)
        text(d, (808, 829), "Linear colour opponency", 17, MUTED)
        text(d, (58, 855), "Five maps  /  each pooled to 64 × 8  /  fixed connections  /  spiking cells", 16, MUTED)
        text(d, (58, 912), "03   THE SPARSE CODE", 21, WHITE, True)
        text(d, (1143, 916), "Each dot is a cell; lit cells fired", 17, MUTED, anchor="ra")
        text(d, (1208, 153), "04   ROUTE PROGRESS", 22, WHITE, True)
        text(d, (1208, 188), "Map for the viewer; not an input to steering", 17, MUTED)
        text(d, (1208, 716), "05   LOOK, COMPARE, STEP", 22, WHITE, True)
        text(d, (1208, 753), "Match to route memory · higher is more familiar", 17, MUTED)
        d.line((397, 583, 397, 845), fill=EDGE, width=1)
        d.line((785, 583, 785, 845), fill=EDGE, width=1)
        self.make_map(canvas)
        return canvas

    def make_map(self, canvas):
        # Full trial bounds are a viewer camera choice, never an agent input.
        points = np.concatenate([self.route, self.positions])
        lo, hi = points.min(0)-.65, points.max(0)+.65
        self.map_rect = (1210, 220, 1860, 570)
        x0, y0, x1, y1 = self.map_rect
        self.map_scale = min((x1-x0)/(hi[0]-lo[0]), (y1-y0)/(hi[1]-lo[1]))
        self.map_centre = (lo+hi)/2
        self.map_pixel_centre = np.array([(x0+x1)/2, (y0+y1)/2])
        tile = Image.new("RGB", (x1-x0, y1-y0), BG)
        d = ImageDraw.Draw(tile)
        for x in np.arange(math.floor(lo[0]), math.ceil(hi[0])+1):
            p = self.xy([x, self.map_centre[1]]) - [x0, y0]
            d.line((p[0], 0, p[0], y1-y0), fill=(17, 37, 45))
        for y in np.arange(math.floor(lo[1]), math.ceil(hi[1])+1):
            p = self.xy([self.map_centre[0], y]) - [x0, y0]
            d.line((0, p[1], x1-x0, p[1]), fill=(17, 37, 45))
        r = self.world.renderer
        for X, Y, colour in zip(r.X.numpy(), r.Y.numpy(), r.triangle_color.numpy()):
            polygon = [tuple(self.xy(p)-[x0, y0]) for p in zip(X, Y)]
            muted = tuple((.17*colour + .83*np.array(BG)).astype(int))
            d.polygon(polygon, fill=muted)
        for i in range(len(self.route)-1):
            if i % 3 != 2:
                d.line([tuple(self.xy(p)-[x0, y0]) for p in self.route[i:i+2]], fill=MUTED, width=3)
        canvas.paste(tile, (x0, y0))
        d = ImageDraw.Draw(canvas)
        for point, label, colour in ((self.route[0], "START", GOLD), (self.route[-1], "NEST", TEAL)):
            x, y = self.xy(point)
            radius = max(7, .2*self.map_scale) if label == "NEST" else 5
            d.ellipse((x-radius, y-radius, x+radius, y+radius), outline=colour, width=2)
            text(d, (x+13, y-10), label, 16, colour, True)
        d.line((1222, 552, 1222+self.map_scale, 552), fill=MUTED, width=3)
        text(d, (1222, 530), "1 m", 15, MUTED)
        d.line((1210, 593, 1248, 593), fill=MUTED, width=3)
        text(d, (1260, 580), "Taught route", 17, MUTED)
        d.line((1440, 593, 1478, 593), fill=TEAL, width=4)
        text(d, (1490, 580), "Actual path", 17, MUTED)
        d.line((1656, 593, 1694, 593), fill=GOLD, width=3)
        text(d, (1706, 580), "Looking", 17, MUTED)

    def xy(self, point):
        return self.map_pixel_centre + (np.asarray(point)-self.map_centre)*self.map_scale*np.array([1, -1])

    @torch.no_grad()
    def panorama(self, position):
        key = tuple(float(x) for x in position)
        if key not in self.panoramas:
            r = self.display_renderer.render_single(*key, self.world.ic.eye_height, 0.)
            self.panoramas[key] = r.permute(1, 2, 0).numpy().astype(np.uint8)
            if len(self.panoramas) > 16:
                self.panoramas.pop(next(iter(self.panoramas)))
        return self.panoramas[key]

    def view(self, position, heading):
        panorama = self.panorama(position)
        angles = (np.linspace(-148, 148, 1076) + heading + 180) % 360
        x = angles / 360 * (panorama.shape[1]-1)
        y = np.linspace(0, panorama.shape[0]-1, 252)
        xx, yy = np.meshgrid(x, y)
        return Image.fromarray(cv2.remap(panorama, xx.astype(np.float32), yy.astype(np.float32),
                                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP))

    def sensor_tiles(self, index, look):
        im = Image.new("RGBA", (1090, 216), (0, 0, 0, 0))
        raw = rgb_image(self.data["raw"][index, look])
        im.paste(raw.resize((312, 76), Image.Resampling.NEAREST), (0, 0))
        gb = self.data["sampled"][index, look]
        sampled = np.zeros((*gb.shape[:2], 3), dtype=np.uint8)
        sampled[..., 1:] = gb
        im.paste(rgb_image(sampled).resize((312, 76), Image.Resampling.NEAREST), (0, 118))
        maps = self.data["maps"][index, look]
        scale = self.meta["feature_display_scales"]
        for c, x, y, w, h, tint in ((0, 355, 0, 154, 76, TEAL),
                                   (1, 533, 0, 154, 76, BLUE),
                                   (2, 355, 118, 332, 76, TEAL),
                                   (3, 743, 0, 332, 76, TEAL),
                                   (4, 743, 118, 332, 76, PURPLE)):
            panel = tinted(maps[c], scale[c], tint, signed=(c == 2))
            # ON/OFF images retain their native aspect ratio, centred in the slot.
            if c < 2:
                panel = panel.resize((154, 38), Image.Resampling.NEAREST)
                im.paste(panel, (x, y+19))
            else:
                im.paste(panel.resize((w, h), Image.Resampling.NEAREST), (x, y))
        return im

    def ant_icon(self, heading, moving, phase):
        icon = Image.new("RGBA", (84, 84))
        d = ImageDraw.Draw(icon)
        gait = 3*math.sin(phase*2*math.pi) if moving else 0
        for sign in (-1, 1):
            for j in range(3):
                x = 32+j*6
                shift = gait*((-1)**j)*sign
                d.line((x, 42+sign*4, x-6+shift, 42+sign*14,
                        x-11+shift, 42+sign*20), fill=WHITE, width=2)
        d.ellipse((15, 33, 36, 51), fill=WHITE, outline=BG, width=2)
        d.ellipse((33, 36, 45, 48), fill=WHITE, outline=BG, width=2)
        d.ellipse((45, 35, 59, 49), fill=GOLD, outline=BG, width=2)
        d.line((55, 38, 65, 30, 69, 32), fill=GOLD, width=2)
        d.line((55, 46, 65, 54, 69, 52), fill=GOLD, width=2)
        return icon.rotate(float(heading), resample=Image.Resampling.BICUBIC)

    def frame(self, index, look, phase="scan", move=0.):
        canvas = self.base.copy()
        d = ImageDraw.Draw(canvas)
        decision, step = self.saved["decisions"][index], self.steps[index]
        winner = int(self.data["winners"][index])
        position = self.positions[index]*(1-move) + self.positions[index+1]*move
        initial_heading = float(self.initial_headings[0] if index == 0 else self.steps[index-1]["heading"])
        view_heading = float(decision["headings"][look])
        body_heading = step["heading"] if phase == "walk" else initial_heading
        # The first-person display moves continuously; model panels hold the last
        # evaluated view during a step, since the model senses only at scan poses.
        canvas.paste(self.view(position, view_heading), (63, 199))
        d = ImageDraw.Draw(canvas)
        d.line((601, 418, 601, 443), fill=WHITE, width=2)
        d.line((590, 431, 612, 431), fill=WHITE, width=2)
        labels = dict(scan=f"SCANNING  {look+1:02d} / 13", choose="CHOOSING A HEADING", walk="MOVING 10 cm")
        colour = GOLD if phase == "scan" else TEAL
        text(d, (1138, 157), labels[phase], 18, colour, True, "ra")
        tiles = self.sensor_tiles(index, look)
        canvas.paste(tiles, (65, 608), tiles)
        d = ImageDraw.Draw(canvas)
        bits = self.bits[index, look]
        for j, (tint, name, x) in enumerate(((TEAL, "Form", 65), (PURPLE, "Colour", 614))):
            active = bits[j*4000:(j+1)*4000]
            text(d, (x, 945), f"{name}   {active.sum():,} / 4,000 cells", 17, tint)
            img = np.empty((20, 200, 3), dtype=np.uint8)
            img[:] = (31, 52, 61)
            img[active.reshape(20, 200) > 0] = tint
            canvas.paste(rgb_image(img).resize((520, 52), Image.Resampling.NEAREST), (x, 972))
        d = ImageDraw.Draw(canvas)
        trail = [tuple(self.xy(p)) for p in self.positions[:index+1]] + [tuple(self.xy(position))]
        if len(trail) > 1:
            d.line(trail, fill=(22, 87, 82), width=10, joint="curve")
            d.line(trail, fill=TEAL, width=4, joint="curve")
        px, py = self.xy(position)
        for angle in initial_heading + np.array([-60, 0, 60]):
            end = self.xy(position + .6*np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))]))
            d.line((px, py, *end), fill=(70, 86, 91), width=1)
        end = self.xy(position + .85*np.array([np.cos(np.radians(view_heading)), np.sin(np.radians(view_heading))]))
        d.line((px, py, *end), fill=GOLD, width=3)
        icon = self.ant_icon(body_heading, phase == "walk", move)
        canvas.paste(icon, (round(px)-42, round(py)-42), icon)
        d = ImageDraw.Draw(canvas)
        deviation = np.linalg.norm(self.route-position, axis=1).min()
        nest_distance = np.linalg.norm(self.route[-1]-position)
        for x, label, value in ((1210, "TO NEST", f"{nest_distance:.2f} m"),
                                (1438, "FROM ROUTE", f"{deviation*100:.1f} cm"),
                                (1665, "DISTANCE WALKED", f"{.1*(index+move):.1f} m")):
            text(d, (x, 620), label, 14, MUTED, True)
            text(d, (x, 640), value, 25, WHITE, True)
        scores = -np.asarray(decision["scores"])
        x0, y0, x1, y1 = 1247, 792, 1849, 940
        for value in (0., .5, 1.):
            y = y1-value*(y1-y0)
            d.line((x0, y, x1, y), fill=EDGE, width=1)
            text(d, (1234, y-9), f"{value:.1f}", 14, MUTED, anchor="ra")
        bar_width = (x1-x0)/13
        for j, score in enumerate(scores):
            x = x0+j*bar_width
            revealed = phase != "scan" or j <= look
            tint = GOLD if j == look and phase == "scan" else (TEAL if j == winner and phase != "scan" else (88, 118, 130))
            if revealed:
                d.rounded_rectangle((x+6, y1-max(1, float(score)*(y1-y0)), x+bar_width-6, y1), radius=3, fill=tint)
        text(d, (x0, 947), "LEFT +60°", 14, MUTED)
        text(d, ((x0+x1)/2, 947), "AHEAD", 14, MUTED, anchor="ma")
        text(d, (x1, 947), "RIGHT −60°", 14, MUTED, anchor="ra")
        offset = 60-10*(look if phase == "scan" else winner)
        if phase == "scan":
            message = f"Looking {abs(offset):.0f}° {'left' if offset>0 else 'right' if offset<0 else 'ahead'}  ·  match {scores[look]:.3f}"
        elif step["no_evidence"]:
            message = "No clear preference · continue straight"
        else:
            message = f"Selected: {abs(offset):.0f}° {'left' if offset>0 else 'right' if offset<0 else 'ahead'}  ·  match {scores[winner]:.3f}"
        text(d, (1208, 981), message, 20, colour, True)
        text(d, (1208, 1011), "Scan all 13 directions, then take one step", 15, MUTED)
        text(d, (38, 1052), f"STEP {index+1:03d} / {len(self.steps)}", 17, WHITE, True)
        d.rounded_rectangle((223, 1060, 882, 1065), radius=2, fill=EDGE)
        d.rounded_rectangle((223, 1060, 223+max(1,659*(index+move)/len(self.steps)), 1065), radius=2, fill=TEAL)
        footer = "Sensor panels hold the last scan while walking" if phase == "walk" else "Input, features and cells follow the highlighted scan direction"
        text(d, (1884, 1052), footer, 16, MUTED, anchor="ra")
        return canvas

    def title(self, canvas, outro=False):
        overlay = Image.new("RGBA", SIZE, (2, 10, 16, 170))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay)
        d = ImageDraw.Draw(canvas)
        d.rounded_rectangle((330, 310, 1590, 758), radius=28, fill=(*CARD, 255), outline=(*EDGE, 255), width=2)
        if outro:
            arrived = self.row["reached_nest"]
            text(d, (385, 356), "NEST REACHED" if arrived else "STEP LIMIT REACHED", 19, TEAL if arrived else GOLD, True)
            text(d, (385, 405), f"Ant {self.ant:02d} {'returns home' if arrived else 'does not reach the nest'}", 43, WHITE, True)
            text(d, (385, 483), f"{self.row['steps']} steps   ·   {self.row['path_length_m']:.1f} m walked   ·   {100*self.row['route_deviation_mean_m']:.1f} cm mean deviation", 25, WHITE)
            text(d, (385, 548), f"Final distance to nest: {self.row['final_nest_distance_m']:.2f} m", 24, MUTED)
            text(d, (385, 611), "Same teaching procedure, memory and steering as the evaluation.", 22, MUTED)
            text(d, (385, 662), f"Linear colour + original form   /   Wiring seed {self.seed}", 20, TEAL)
        else:
            text(d, (385, 354), f"APIA VIZ   /   ANT {self.ant:02d}", 20, TEAL, True)
            text(d, (385, 401), "How the ant finds its way", 44, WHITE, True)
            for y, n, label in ((482, "1", "Look around: compare 13 viewing directions."),
                                (532, "2", "Match each view to the route learned earlier."),
                                (582, "3", "Choose a direction and move 10 cm.")):
                text(d, (389, y), n, 25, TEAL, True)
                text(d, (435, y), label, 25, WHITE)
            text(d, (385, 665), "Replay: 1 step/s; first two scans slowed. Not a physical walking speed.", 20, MUTED)
            text(d, (385, 704), "The detailed panorama is for viewing. The model receives only 74 × 18 pixels.", 19, MUTED)
        return canvas.convert("RGB")


def write_video(dashboard, out, fps=30, preview_only=False):
    out.mkdir(parents=True, exist_ok=True)
    stem = f"ant-{dashboard.ant:02d}-linear-colour"
    # Mid-scan poster displays each major part of the system without an overlay.
    poster = dashboard.frame(0, 8)
    poster.save(out / f"{stem}.png")
    if preview_only:
        dashboard.title(poster).save(out / f"{stem}-intro.png")
        return
    target = out / f"{stem}.mp4"
    temporary = target.with_name(target.stem + ".partial.mp4")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "1920x1080", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "2",
               "-movflags", "+faststart", "-metadata", f"title=ApiaViz | Ant {dashboard.ant:02d} | Linear colour",
               str(temporary)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    frames = 0

    def emit(image, repeats=1):
        nonlocal frames
        pixels = image.tobytes()
        for _ in range(repeats):
            process.stdin.write(pixels)
            frames += 1

    try:
        emit(dashboard.title(poster), 6*fps)
        n = len(dashboard.steps)
        for i in range(n):
            slowdown = 3 if i < 2 else 1
            for j in range(13):
                emit(dashboard.frame(i, j, "scan"), slowdown)
            winner = int(dashboard.data["winners"][i])
            chosen = dashboard.frame(i, winner, "choose")
            emit(chosen, 7*slowdown)
            for j in range(10):
                image = dashboard.frame(i, winner, "walk", (j+1)/10)
                emit(image, slowdown)
            if (i+1) % 20 == 0:
                print(f"Ant {dashboard.ant}: composed {i+1}/{n} navigation steps", flush=True)
        emit(dashboard.title(image, outro=True), 5*fps)
    finally:
        process.stdin.close()
    code = process.wait()
    if code:
        raise RuntimeError(f"ffmpeg exited with {code}")
    temporary.replace(target)
    metadata = dict(dashboard.meta, result=dashboard.row, video=str(target.relative_to(ROOT)),
                    frames=frames, fps=fps, duration_s=frames/fps, width=1920, height=1080,
                    video_sha256=file_hash(target),
                    display=dict(high_resolution_panorama="720 x 150 at each display pose; cropped to 296 degrees for the viewer, never encoded",
                                 model_input="74 x 18 RGB; red discarded; all scan scores reproduce exactly",
                                 feature_maps="Before 64 x 8 pooling; square-root display contrast with fixed per-clip channel scales",
                                 spike_grid="8,000 actual binary cell outputs, displayed as two grids of 4,000; grid layout is not retinotopy",
                                 movement="Linear interpolation between recorded endpoints; last evaluated model inputs held during movement",
                                 timing="One complete decision per second; first two steps slowed threefold; six-second introduction and five-second result card",
                                 sound="Silent, with explanations embedded on screen"))
    write_json(out / f"{stem}.json", metadata)
    print(f"Saved {target} ({frames/fps:.1f} s)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ants", type=int, nargs="+", default=[4, 7, 13])
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--output", type=Path, default=ROOT / "docs/navigation-videos")
    p.add_argument("--cache", type=Path, default=ROOT / "apiaviz/output/navigation-video-cache")
    p.add_argument("--preview-only", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    world = accelerate_world(World(ROOT / "apiaviz/mbant/data/antview", seed=99))
    for ant in args.ants:
        row, saved, data, meta = prepare(ant, args.seed, world, args.cache)
        dashboard = Dashboard(ant, args.seed, row, saved, data, meta, world)
        write_video(dashboard, args.output.resolve(), preview_only=args.preview_only)


if __name__ == "__main__":
    main()
