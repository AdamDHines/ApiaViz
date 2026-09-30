"""Navigation protocols with a common sensory, memory and action budget."""
import math
import numpy as np
import torch

from apiaviz.mbant.config import ImageConfig, NavigationConfig
from apiaviz.mbant.io_utils import load_ant_data, load_world_data, prepare_route
from apiaviz.nav.torch_route import TorchWorldRenderer, route_corridor_bank, corridor_offsets
from apiaviz.nav.retino_kc import AntiHebbianMBON, MBONPopulation
from apiaviz.nav.torch_route import CosineRouteMemory
from apiaviz.eval import build_landmark_color
from .stimuli import perturb, sample_key


def make_memory(codes, kind, segments=16, graded=False):
    if kind == "cosine":
        return CosineRouteMemory(codes)
    if kind == "single":
        memory = AntiHebbianMBON(codes.shape[1], graded=graded).to(codes.device)
        memory.store(codes)
        return memory
    if kind == "population":
        return MBONPopulation(codes, segments, graded=graded).to(codes.device)
    raise ValueError(kind)


class World:
    def __init__(self, root, device="cpu", seed=99, landmark_fraction=.5, chroma=60.):
        self.device = torch.device(device)
        self.ic = ImageConfig(resolution=4., hfov=296.)
        self.nav = NavigationConfig(step_size=.1, scan_range=120., scan_step=10., dis_threshold=.2)
        self.ants = load_ant_data(str(root / "AntData.mat"))
        world = load_world_data(str(root / "world5000_gray.mat"))
        renderer = TorchWorldRenderer(world, device=self.device, hfov=self.ic.hfov,
                                      resolution=self.ic.resolution, chunk_size=512, color=True)
        colours = build_landmark_color(renderer.triangle_grey, landmark_fraction, chroma, seed, self.device)
        self.renderer = TorchWorldRenderer(world, device=self.device, hfov=self.ic.hfov,
                         resolution=self.ic.resolution, chunk_size=512, color=True, triangle_color=colours)
        self.cache = {}

    def jobs(self, ants, routes=()):
        return [(ant, route) for ant in ants if f"Ant{ant}" in self.ants
                for route in self.ants[f"Ant{ant}"]["available_routes"] if not routes or route in routes]

    def route(self, ant, route):
        data = self.ants[f"Ant{ant}"]["routes"][f"Route{route}"]
        positions, headings, _ = prepare_route(data, img_separation=self.ic.img_separation)
        return positions, headings

    @torch.no_grad()
    def render(self, positions, headings):
        # Chunk the renderer: nine-view corridor banks can otherwise exhaust memory.
        positions = torch.as_tensor(np.asarray(positions), device=self.device, dtype=torch.float32)
        headings = torch.as_tensor(np.asarray(headings), device=self.device, dtype=torch.float32)
        return torch.cat([self.renderer.render_batch(positions[i:i + 16], headings[i:i + 16],
                                                     eye_height=self.ic.eye_height) / 255.
                          for i in range(0, len(headings), 16)])

    def scan(self, position, headings):
        key = (tuple(np.asarray(position).tolist()), tuple(np.asarray(headings).tolist()))
        if key not in self.cache:
            if len(self.cache) >= 128:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = self.render(np.tile(position, (len(headings), 1)), headings)
        return self.cache[key]


class Scorer:
    def __init__(self, world, encoder, memory, encode, condition, severity, seed):
        self.world, self.encoder, self.memory, self.encode = world, encoder, memory, encode
        self.condition, self.severity, self.seed = condition, severity, seed
        self.perturbations = []
        self.decisions = []

    def __call__(self, pos, headings):
        images = self.world.scan(pos, headings)
        keys = [sample_key(pos, h) for h in headings]
        images, diagnostics = perturb(images, self.condition, self.severity, self.seed, keys)
        self.perturbations.append(diagnostics)
        codes = self.encode(self.encoder, images)
        values = self.memory(codes)
        # All readouts agree that a silent visual code provides no direction.
        values = torch.where(codes.abs().sum(1) > 0, values, float("inf"))
        scores = values.cpu().numpy()
        self.decisions.append({"position": np.asarray(pos).tolist(),
                               "headings": np.asarray(headings).tolist(),
                               "scores": [float(v) if np.isfinite(v) else None for v in scores],
                               "active_fraction": float((codes > 0).float().mean()),
                               "memory_context": getattr(self.memory, "cursor", None)})
        return scores

    def commit(self, winner):
        if hasattr(self.memory, "commit"):
            self.memory.commit(winner)


def choose_heading(scores, offsets):
    """No-evidence/tie policy: prefer the smallest turn, then the lower index."""
    finite = np.isfinite(scores)
    # Float32 cosine reductions can differ by ~1e-7 for equivalent sparse codes.
    tolerance = 1e-6
    if not finite.any() or np.ptp(scores[finite]) <= tolerance:
        return int(np.argmin(abs(offsets))), True
    best = np.flatnonzero(finite & np.isclose(scores, np.min(scores[finite]), rtol=0, atol=tolerance))
    return int(best[np.argmin(abs(offsets[best]))]), False


def evaluate_route(positions, headings, scorer, protocol, max_steps=200, offline_stride=10,
                   initial_position=None, initial_heading=None):
    nav = scorer.world.nav
    offsets = np.arange(nav.scan_range / 2, -nav.scan_range / 2 - 1e-6, -nav.scan_step)
    trace, deviations, corrections = [], [], 0
    pos, heading = positions[0].copy(), float(headings[0])
    if initial_position is not None:
        pos = np.asarray(initial_position, dtype=float).copy()
        if pos.shape != (2,) or not np.isfinite(pos).all():
            raise ValueError("initial_position must be a finite 2D position")
    if initial_heading is not None:
        heading = float(initial_heading)
        if not math.isfinite(heading):
            raise ValueError("initial_heading must be finite")
    reached = np.linalg.norm(pos - positions[-1]) <= nav.dis_threshold
    silent_scans = 0
    if protocol == "offline":
        # Lateral probes distinguish exact-view recognition from navigable gradients.
        errors, margins = [], []
        for idx in range(0, len(headings), offline_stride):
            for lateral in (-.2, 0., .2):
                radians = math.radians(float(headings[idx]))
                pos = positions[idx] + lateral * np.array([-math.sin(radians), math.cos(radians)])
                scores = scorer(pos, headings[idx] + offsets)
                winner, silent = choose_heading(scores, offsets)
                silent_scans += silent
                # Offline scans are centred on the correct heading. A flat/silent
                # response must not get a perfect score just by keeping that heading.
                errors.append(180. if silent else abs(float(offsets[winner])))
                finite = np.sort(scores[np.isfinite(scores)])
                margins.append(float(finite[1] - finite[0]) if len(finite) > 1 else 0.)
                trace.append({"route_index": idx, "lateral_m": lateral,
                              "scores": [float(x) if np.isfinite(x) else None for x in scores],
                              "selected_offset_deg": float(offsets[winner]), "no_evidence": silent})
        return {"heading_error_deg": float(np.mean(errors)), "familiarity_margin": float(np.mean(margins)),
                "silent_scans": silent_scans, "scans": len(trace), "trace": trace}
    if protocol not in ("reset", "free"):
        raise ValueError(protocol)
    for step in range(max_steps):
        if reached:
            break
        if hasattr(scorer, "steer"):
            turn, silent = scorer.steer(pos, heading)
        else:
            scores = scorer(pos, heading + offsets)
            winner, silent = choose_heading(scores, offsets)
            turn = float(offsets[winner])
            if not silent and hasattr(scorer, "commit"):
                scorer.commit(winner)
        silent_scans += silent
        heading += turn
        rad = math.radians(heading)
        pos = pos + nav.step_size * np.array([math.cos(rad), math.sin(rad)])
        distances = np.linalg.norm(positions - pos, axis=1)
        nearest, deviation = int(np.argmin(distances)), float(np.min(distances))
        deviations.append(deviation)
        reached = np.linalg.norm(pos - positions[-1]) <= nav.dis_threshold
        entry = {"step": step + 1, "position": pos.tolist(), "heading": heading,
                 "deviation_m": deviation, "no_evidence": silent, "reset_to": None}
        if protocol == "reset" and deviation > nav.dis_threshold and not reached:
            # Explicit ground-truth assistance; do not include teleportation in path length.
            reset_index = min(nearest + 1, len(headings) - 1)
            pos, heading = positions[reset_index].copy(), float(headings[reset_index])
            corrections += 1
            entry["reset_to"] = pos.tolist()
        trace.append(entry)
    return {"reached_nest": bool(reached), "final_nest_distance_m": float(np.linalg.norm(pos - positions[-1])),
            "route_deviation_mean_m": float(np.mean(deviations)) if deviations else 0.,
            "route_deviation_max_m": max(deviations, default=0.),
            "path_length_m": len(trace) * nav.step_size, "corrective_resets": corrections,
            "steps": len(trace), "silent_scans": silent_scans, "trace": trace}


def memory_bank(world, positions, headings, viewpoints, width):
    pos, head = route_corridor_bank(positions, headings, corridor_offsets(viewpoints, width))
    return world.render(pos, head), pos
