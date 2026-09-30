"""Sparse rasterization of the existing painter's renderer.

Geometry, ordering and float32 barycentric arithmetic follow TorchWorldRenderer.
Only pixel candidates inside each triangle's bounding rectangle are evaluated.
This is a computational optimization, not a new sensory model. Validate against
the original renderer before using it for a new study.
"""
import math
import torch
from apiaviz.nav.torch_route import TorchWorldRenderer


class SparseWorldRenderer(TorchWorldRenderer):
    def render_single(self, x, y, z, heading_deg):
        dx, dy, dz = self.X - float(x), self.Y - float(y), self.Z.abs() - float(z)
        radius = torch.sqrt(dx.square() + dy.square() + dz.square()).clamp_min(1e-6)
        az = self._wrap_pi(torch.atan2(dy, dx) - math.radians(float(heading_deg)))
        el = torch.atan2(dz, torch.sqrt(dx.square() + dy.square()).clamp_min(1e-6))
        nonwrap = az.max(1).values - az.min(1).values < math.pi
        ap, ep, rp, cp = [az[nonwrap]], [el[nonwrap]], [radius[nonwrap]], [self.colp[nonwrap]]
        rgbp = [self.triangle_color[nonwrap]] if self.color and self.triangle_color is not None else None
        if bool((~nonwrap).any()):
            pos, neg = az[~nonwrap].clone(), az[~nonwrap].clone()
            pos[pos <= 0] += 2 * math.pi
            neg[neg > 0] -= 2 * math.pi
            ap.extend([pos, neg]); ep.extend([el[~nonwrap]] * 2)
            rp.extend([radius[~nonwrap]] * 2); cp.extend([self.colp[~nonwrap]] * 2)
            if rgbp is not None: rgbp.extend([self.triangle_color[~nonwrap]] * 2)
        a, e, r, c = torch.cat(ap), torch.cat(ep), torch.cat(rp), torch.cat(cp).mean(1)
        order = torch.argsort(r.mean(1), descending=True)
        a, e, c = a[order], e[order], c[order]
        c = torch.full_like(c, 255.) if self.render_mode == "binary_objects" else ((c-self.colp_min)/self.colp_range).clamp(0,1)*255
        rgb = torch.cat(rgbp)[order] if rgbp is not None else None
        # Ascending y index is reversed below to preserve original image rows.
        gx = self.grid_x[:self.width].contiguous()
        gy = self.grid_y.view(self.height, self.width)[:, 0].flip(0).contiguous()
        margin = 1e-4
        x0 = torch.searchsorted(gx, a.min(1).values-margin)
        x1 = torch.searchsorted(gx, a.max(1).values+margin, right=True)
        y0 = torch.searchsorted(gy, e.min(1).values-margin)
        y1 = torch.searchsorted(gy, e.max(1).values+margin, right=True)
        nx, ny = x1-x0, y1-y0
        counts = nx*ny
        ti = torch.repeat_interleave(torch.arange(len(a), device=self.device), counts)
        local = torch.arange(len(ti), device=self.device) - torch.repeat_interleave(counts.cumsum(0)-counts, counts)
        xi = x0[ti] + local % nx[ti]
        yi = self.height-1-(y0[ti]+local//nx[ti])
        pi = yi*self.width + xi
        aa, ee = a[ti], e[ti]
        ax, bx, cx = aa.unbind(1); ay, by, cy = ee.unbind(1)
        px, py = self.grid_x[pi], self.grid_y[pi]
        denom = (by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        valid = denom.abs() > 1e-8
        denom = torch.where(valid, denom, torch.ones_like(denom))
        w1 = ((by-cy)*(px-cx)+(cx-bx)*(py-cy))/denom
        w2 = ((cy-ay)*(px-cx)+(ax-cx)*(py-cy))/denom
        w3 = 1-w1-w2
        inside = valid & (w1>=0) & (w2>=0) & (w3>=0)
        winner = torch.full((self.grid_x.numel(),), -1, dtype=torch.long, device=self.device)
        winner.scatter_reduce_(0, pi[inside], ti[inside], reduce="amax", include_self=True)
        ground = self.grid_y <= math.atan2(-float(z), 10.5)
        if self.render_mode == "binary_objects":
            image = torch.zeros((3, len(winner)) if self.color else (len(winner),), device=self.device)
        elif self.color:
            image = torch.empty((3, len(winner)), device=self.device)
            for channel, (sky, soil) in enumerate(zip(self._SKY_RGB, self._GROUND_RGB)):
                image[channel].fill_(sky); image[channel, ground] = soil
        else:
            image = torch.full((len(winner),), 255., device=self.device); image[ground] = 183.
        mask = winner >= 0
        if self.color:
            image[:,mask] = rgb[winner[mask]].T if rgb is not None else c[winner[mask]][None]
        else: image[mask] = c[winner[mask]]
        return image.view(3,self.height,self.width) if self.color else image.view(self.height,self.width)


def accelerate_world(world):
    """Reuse the original renderer state; no world data or settings change."""
    fast = SparseWorldRenderer.__new__(SparseWorldRenderer)
    fast.__dict__.update(world.renderer.__dict__)
    world.renderer = fast
    return world
