"""Explicit camera-only recovery control for deterministic static acquisitions.

Identical textured views after commanded translation are evidence of visual
stagnation, not free space. No contact flag or measured displacement is accepted.
This narrow detector is not calibrated for moving scenes or camera noise.
"""
import numpy as np

from .visual_avoidance import VisualAvoidance, flow_points, gray, wrap


class VisualStallRecovery(VisualAvoidance):
    schema = 'textured-image-stall-recovery-v2'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stationary_commands = 0
        self.recovery_heading = None
        self.recovery_progress = 0.

    def moved(self, before, after, heading, distance):
        a, b = np.asarray(before), np.asarray(after)
        # Validate with the same camera contract as the base reflex. Quantized
        # grayscale tests texture; raw float RGB tests change without discarding
        # subpixel evidence. Ignore the sky for the texture requirement.
        g = gray(a)
        gray(b)
        if a.shape != b.shape or not np.isfinite(distance) or distance <= 0:
            raise ValueError('Matched images and positive commanded motion required')
        # A smooth sky/ground elevation gradient can be unchanged under real
        # translation. Require azimuthal structure within rows, not merely
        # contrast between the top and bottom of an image.
        texture = float(np.std(g[g.shape[0]//2:], axis=1).mean())
        difference = float(np.max(np.abs(a.astype(float)-b.astype(float))))
        stationary = texture >= 2. and difference <= 1e-7
        if stationary:
            # Rotate retained visual surfaces, but do not walk them through the
            # animal or age them away while images show no displacement.
            if self.frame_heading is not None:
                theta = np.deg2rad(wrap(self.frame_heading-heading))
                rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                self.points = self.points @ rot.T
            _, diagnostic = flow_points(before, after, distance, self.settings)
            self.frame_heading = heading
            self.stationary_commands += 1
            self.recovery_progress = 0.
            if self.stationary_commands >= 2:
                self.recovery_heading = wrap(heading+self.phase*90.)
                self.stationary_commands = 0
        else:
            diagnostic = super().moved(before, after, heading, distance)
            self.stationary_commands = 0
            if self.recovery_heading is not None:
                self.recovery_progress += distance
                if self.recovery_progress >= .06-1e-10:
                    self.recovery_heading = None
                    self.recovery_progress = 0.
        diagnostic.update(visual_stagnation=stationary, image_max_change=difference,
                          lower_view_azimuth_texture_std=texture, recovery_schema=self.schema)
        return diagnostic

    def choose(self, desired_heading):
        if self.recovery_heading is None:
            return super().choose(desired_heading)
        heading = self.recovery_heading
        record = dict(state='visual_stall_recovery', desired_heading=float(desired_heading),
                      heading=float(heading), offset_deg=float(wrap(heading-desired_heading)),
                      side=self.phase, points=len(self.points), visually_clear=False,
                      stride_m=self.settings.probe_m, recovery_schema=self.schema)
        self.decisions.append(record)
        return heading, self.settings.probe_m, record
