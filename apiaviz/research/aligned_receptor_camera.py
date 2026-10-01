"""Explicit acquisition-order control; never inject UV into visible baselines.

Interpolate linear band radiance first for both camera paths. The visible camera
then applies its fixed response; ApiaViz applies the corresponding response in
its encoder. Historical DualCamera behavior is preserved in its own class.
"""
from .dual_camera import DualCamera, visible_image
from .uv_input import sample_receptors


class AlignedReceptorCamera(DualCamera):
    schema = 'linear-retinal-sampling-before-response-v3'

    def scan(self, position, headings, *, uv=False):
        raw = self.frame(position)
        linear = sample_receptors(raw[:, :, :3] if uv else raw[:, :, 3:], headings,
            elevation=self.config['elevation_deg'], shape=tuple(self.config['retina_shape']))
        return linear if uv else visible_image(linear, self.config['visible_white'])
