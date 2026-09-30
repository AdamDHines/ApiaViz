"""Versioned evaluator safeguards. No placement/geometry feedback goes to policies."""
from dataclasses import dataclass, asdict

import numpy as np

from .active_navigation import segment_collision

SCHEMA = 'navigation-safety-v1'


@dataclass(frozen=True)
class EvaluationSafety:
    schema: str = SCHEMA
    placement_margin_m: float = 1e-5

    def __post_init__(self):
        if self.schema != SCHEMA:
            raise ValueError('Unknown evaluation safety schema')
        if not np.isfinite(self.placement_margin_m) or self.placement_margin_m <= 0:
            raise ValueError('Placement margin must be finite and positive')

    def configuration(self):
        return asdict(self)


def inside_body(position, bounds, obstacles):
    radius = getattr(obstacles, 'body_radius_m', 0.)
    x0,x1,y0,y1 = bounds
    x,y = position
    return bool(x0+radius <= x <= x1-radius and y0+radius <= y <= y1-radius)


def proposal_reason(a, b, bounds, obstacles):
    if not inside_body(b, bounds, obstacles):
        return 'field_boundary'
    if segment_collision(a,b,obstacles):
        return 'rock_contact'
    return None


def safe_displacement(origin, requested, bounds, obstacles, settings=EvaluationSafety()):
    """Largest collision-free prefix, backed off a numerical safety margin.

    Sweep the whole displacement, not merely its endpoint: a kick cannot tunnel
    through a thin rock. Never search sideways or choose a favorable location.
    Invalid anchors are setup errors, raised before trial sensing begins.
    """
    origin, requested = np.asarray(origin,dtype=float), np.asarray(requested,dtype=float)
    bounds=np.asarray(bounds,dtype=float)
    if origin.shape!=(2,) or requested.shape!=(2,) or bounds.shape!=(4,) or not np.isfinite(np.r_[origin,requested,bounds]).all():
        raise ValueError('Finite planar placement and field bounds required')
    if not bounds[0]<bounds[1] or not bounds[2]<bounds[3]:
        raise ValueError('Invalid field bounds')
    if proposal_reason(origin,origin,bounds,obstacles):
        raise ValueError('Invalid placement anchor: body intersects rock or field boundary')
    reason=proposal_reason(origin,origin+requested,bounds,obstacles)
    fraction=1.
    length=float(np.linalg.norm(requested))
    if reason:
        low,high=0.,1.
        for _ in range(48):
            mid=(low+high)/2
            if proposal_reason(origin,origin+mid*requested,bounds,obstacles): high=mid
            else: low=mid
        # Record the first limiting surface, which can differ from the endpoint
        # violation (e.g. a rock before a farther field boundary).
        reason=proposal_reason(origin,origin+high*requested,bounds,obstacles)
        fraction=max(0.,low-settings.placement_margin_m/max(length,1e-30))
    applied=fraction*requested
    final=origin+applied
    if proposal_reason(origin,final,bounds,obstacles):
        raise AssertionError('Placement safeguard failed')
    return final,dict(origin=origin.tolist(),requested_displacement=requested.tolist(),
        requested_position=(origin+requested).tolist(),position=final.tolist(),
        displacement=applied.tolist(),requested_distance_m=length,
        applied_distance_m=float(np.linalg.norm(applied)),fraction=float(fraction),
        adjusted=bool(reason),constraint=reason,schema=SCHEMA)
