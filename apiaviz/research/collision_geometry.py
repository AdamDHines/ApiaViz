"""Evaluator-only swept circular body against evaluated rock projections.

The union of projected triangles preserves concavities (unlike a convex hull).
The complete mesh is projected, including overhangs and buried vertices: this
is a planar no-climbing model, not a height-resolved animal contact model.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'rock-mesh-projection-v1'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def point_segment_distance(p, a, b):
    delta = b-a
    t = np.clip(np.sum((p-a)*delta, axis=-1) /
                np.maximum(np.sum(delta*delta, axis=-1), 1e-30), 0., 1.)
    return np.linalg.norm(p-(a+t[..., None]*delta), axis=-1)


def swept_hit(a, b, triangles, radius):
    """Closed segment capsule intersects any triangle, including degenerate edges."""
    lo, hi = np.minimum(a, b)-radius, np.maximum(a, b)+radius
    t = triangles[np.all(triangles.max(axis=1) >= lo-1e-12, axis=1) &
                  np.all(triangles.min(axis=1) <= hi+1e-12, axis=1)]
    if not len(t):
        return False
    start, end = t, np.roll(t, -1, axis=1)
    edges = end-start
    area = cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
    for p in (a, b):
        side = cross(edges, p-start)
        inside = (np.all(side >= -1e-12, axis=1) | np.all(side <= 1e-12, axis=1))
        if np.any(inside & (abs(area) > 1e-15)):
            return True
    # Proper segment crossings; collinear overlap is covered by distances below.
    s1, s2 = cross(b-a, start-a), cross(b-a, end-a)
    s3, s4 = cross(edges, a-start), cross(edges, b-start)
    if np.any((s1*s2 < 0) & (s3*s4 < 0)):
        return True
    distances = np.minimum.reduce([
        point_segment_distance(a, start, end), point_segment_distance(b, start, end),
        point_segment_distance(start, a, b), point_segment_distance(end, a, b)])
    return bool(np.any(distances <= radius+1e-12))


class RockGeometry:
    """Physics object. Never pass this object or its results to a visual policy."""
    def __init__(self, rocks, body_radius_m, contact_response='block', provenance=None):
        if not np.isfinite(body_radius_m) or body_radius_m < 0:
            raise ValueError('Body radius must be finite and nonnegative')
        if contact_response not in ('block', 'terminate'):
            raise ValueError('Unknown contact response')
        self.body_radius_m = float(body_radius_m)
        self.contact_response = contact_response
        self.provenance = provenance or {}
        self.rocks = []
        for name, triangles in rocks:
            t = np.asarray(triangles, dtype=float)
            if t.ndim != 3 or t.shape[1:] != (3, 2) or not len(t) or not np.isfinite(t).all():
                raise ValueError('Expected finite nonempty projected triangles (N, 3, 2)')
            self.rocks.append((name, t, t.min(axis=(0, 1)), t.max(axis=(0, 1))))

    def hits(self, a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a, b]).all():
            raise ValueError('Expected finite planar endpoints')
        radius = self.body_radius_m
        lo, hi = np.minimum(a, b)-radius, np.maximum(a, b)+radius
        return [name for name, triangles, rlo, rhi in self.rocks
                if np.all(rhi >= lo-1e-12) and np.all(rlo <= hi+1e-12)
                and swept_hit(a, b, triangles, radius)]

    def intersects(self, a, b):
        return bool(self.hits(a, b))

    @classmethod
    def load(cls, path, scene_sha256, body_radius_m, contact_response='block', expected_sha256=None):
        path = Path(path)
        digest = sha256(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError('Collision geometry hash mismatch')
        record = json.loads(path.read_text())
        if record['schema'] != SCHEMA or record['scene_sha256'] != scene_sha256:
            raise ValueError('Collision geometry schema or scene hash mismatch')
        if not record['rocks']:
            raise ValueError('Rock export is empty; refusing an unverified collision-free world')
        return cls([(r['name'], r['triangles_xy_m']) for r in record['rocks']], body_radius_m,
                   contact_response, dict(schema=SCHEMA, geometry_sha256=digest,
                                          scene_sha256=scene_sha256))


def protocol_geometry(protocol, world, metadata):
    """Historical protocols keep discs; corrected protocols must supply hashed meshes."""
    physics = protocol.get('physics')
    if physics is None:
        return metadata['obstacles']
    if physics['schema'] != SCHEMA:
        raise ValueError('Unknown physics schema')
    asset = physics['worlds'][world['name']]
    result = RockGeometry.load(asset['path'], metadata['scene_sha256'], physics['body_radius_m'],
                               physics['contact_response'], asset['sha256'])
    if len(result.rocks) != len(metadata['obstacles']):
        raise ValueError('Exported rock count differs from world metadata')
    return result
