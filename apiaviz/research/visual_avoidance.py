"""Untrained local obstacle avoidance from consecutive RGB camera images.

No renderer, obstacle list, position, route or contact signal enters this module.
Images use the experiment's 296 x 75 degree angular grid (columns increase yaw).
Matched-gaze translation separates parallax from rotation. Metric scale comes
from commanded self-motion, not from a depth sensor. This is an engineering
optic-flow controller, not a proposed complete insect neural circuit.
"""
from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass(frozen=True)
class Settings:
    probe_m: float = .005
    stride_m: float = .02
    lookahead_m: float = .16
    clearance_m: float = .045
    retain_distance_m: float = .12
    min_flow_px: float = .25
    consistency_px: float = .8
    min_elevation_deg: float = 2.
    max_elevation_deg: float = 28.
    max_range_m: float = .65
    turn_step_deg: float = 10.
    max_turn_deg: float = 100.


def gray(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError('Expected finite H x W x 3 RGB image')
    return cv2.cvtColor(np.round(np.clip(image, 0, 1)*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def flow_points(before, after, distance, settings=Settings()):
    """Estimate local planar surface points from same-yaw forward translation.

    Horizontal spherical triangulation ignores vertical terrain motion. Pixels
    near the focus of expansion, sky, ground and inconsistent matches are
    rejected. Farneback is deterministic classical optical flow, with no weights
    or training. A column needs multiple supported pixels before contributing.
    """
    if distance <= 0:
        raise ValueError('Positive commanded translation required')
    a, b = gray(before), gray(after)
    if a.shape != b.shape:
        raise ValueError('Matched image shapes required')
    forward = cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 9, 5, 5, 1.1, 0)
    backward = cv2.calcOpticalFlowFarneback(b, a, None, .5, 3, 9, 5, 5, 1.1, 0)
    h, w = a.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    inverse = cv2.remap(backward, xx+forward[..., 0], yy+forward[..., 1],
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    error = np.linalg.norm(forward+inverse, axis=2)
    az = np.deg2rad(np.linspace(-148, 148, w))[None, :]
    el = np.linspace(60, -15, h)[:, None]
    da = forward[..., 0] * np.deg2rad(296/(w-1))
    # Exact planar two-ray triangulation; range measured at the second image.
    ranges = distance*np.sin(az) / np.where(abs(np.sin(da)) > 1e-8, np.sin(da), np.nan)
    texture = np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
    valid = ((abs(az) > np.deg2rad(8)) & (abs(az) < np.deg2rad(135)) &
             (el >= settings.min_elevation_deg) & (el <= settings.max_elevation_deg) &
             (abs(forward[..., 0]) >= settings.min_flow_px) & (error < settings.consistency_px) &
             (texture > 8) & (ranges > .008) & (ranges < settings.max_range_m) &
             (xx+forward[..., 0] >= 1) & (xx+forward[..., 0] < w-2))
    points = []
    for col in range(w):
        rows = np.flatnonzero(valid[:, col])
        if len(rows) < 3:
            continue
        r = float(np.quantile(ranges[rows, col], .3))
        theta = float(az[0, col] + np.median(da[rows, col]))
        points.append([r*np.cos(theta), r*np.sin(theta)])
    return np.asarray(points, dtype=float).reshape(-1, 2), dict(
        supported_pixels=int(valid.sum()), supported_columns=len(points),
        median_consistency_px=float(np.median(error)))


def wrap(angle):
    return (angle+180.) % 360. - 180.


class VisualAvoidance:
    """Short-lived egocentric visual surface memory and persistent passing side."""
    def __init__(self, settings=Settings(), phase=1):
        self.settings = settings
        self.phase = 1 if phase >= 0 else -1
        self.points = np.empty((0, 2))
        self.ages = np.empty(0)
        self.frame_heading = None
        self.side = 0
        self.clear_moves = 0
        self.decisions = []

    def moved(self, before, after, heading, distance):
        """Only RGB, own yaw and commanded displacement are accepted."""
        if self.frame_heading is not None:
            theta = np.deg2rad(wrap(self.frame_heading-heading))
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            self.points = self.points @ rot.T - [distance, 0.]
            self.ages += distance
            keep = self.ages <= self.settings.retain_distance_m
            self.points, self.ages = self.points[keep], self.ages[keep]
        new, diagnostic = flow_points(before, after, distance, self.settings)
        self.points = np.concatenate([self.points, new])
        self.ages = np.concatenate([self.ages, np.zeros(len(new))])
        self.frame_heading = heading
        return diagnostic

    def choose(self, desired_heading):
        s = self.settings
        if self.frame_heading is None:
            return desired_heading, s.probe_m, dict(state='initial_visual_probe', points=0)
        offsets = np.arange(-s.max_turn_deg, s.max_turn_deg+.1, s.turn_step_deg)
        candidates = desired_heading + offsets
        angle = np.deg2rad(candidates-self.frame_heading)
        directions = np.stack([np.cos(angle), np.sin(angle)], axis=1)
        along = self.points @ directions.T
        across = abs(self.points[:, 0, None]*directions[None, :, 1] -
                     self.points[:, 1, None]*directions[None, :, 0])
        # Inflate visual surfaces for finite body clearance and depth error.
        radius = s.clearance_m + .1*np.linalg.norm(self.points, axis=1, keepdims=True)
        blocked = (along > -.01) & (along < s.lookahead_m) & (across < radius)
        risk = np.sum(blocked*np.exp(-np.maximum(along, 0)/s.lookahead_m), axis=0)
        centre = int(np.argmin(abs(offsets)))
        clear = risk < .5
        if clear[centre]:
            target = centre
            self.clear_moves += 1
            if self.clear_moves >= 5:
                self.side = 0
        else:
            self.clear_moves = 0
            cost = abs(offsets) + 25*((offsets*self.side < 0) if self.side else 0)
            cost += .001*(offsets*self.phase < 0)
            available = np.flatnonzero(clear)
            target = int(available[np.argmin(cost[available])]) if len(available) else int(np.argmin(risk*100+cost))
            if self.side == 0 and offsets[target] != 0:
                self.side = int(np.sign(offsets[target]))
        # Continue sensing at 5 mm in uncertain/blocked directions.
        stride = s.stride_m if clear[target] else s.probe_m
        record = dict(state='avoid' if target != centre else 'clear',
                      desired_heading=float(desired_heading), heading=float(candidates[target]),
                      offset_deg=float(offsets[target]), side=self.side, points=len(self.points),
                      frontal_risk=float(risk[centre]), chosen_risk=float(risk[target]),
                      stride_m=stride)
        self.decisions.append(record)
        return float(candidates[target]), stride, record

    def configuration(self):
        return asdict(self.settings)
