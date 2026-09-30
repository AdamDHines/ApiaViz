import unittest
import torch
from apiaviz.research.frontend_refinements import RefinementEncoder, METHODS, refinement_maps
from apiaviz.research.paper_baselines import FrontendEncoder


class FrontendRefinements(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.images = torch.rand(3, 3, 18, 74, generator=torch.Generator().manual_seed(19))

    def test_reference_encoders_are_exact(self):
        for method in ("apiaviz", "sobel_colour", "ardin_input"):
            old, new = FrontendEncoder(method, code_dim=128), RefinementEncoder(method, code_dim=128)
            torch.testing.assert_close(old(self.images), new(self.images), atol=0, rtol=0)

    def test_interventions_preserve_unmodified_stream(self):
        base = RefinementEncoder(code_dim=128)
        original = base.currents(self.images)
        for method in ("unit_dc", "balanced_form", "oriented_form", "linear_colour"):
            base.method = method
            changed = base.currents(self.images)
            fixed = 0 if method == "linear_colour" else 1
            torch.testing.assert_close(changed[fixed], original[fixed], atol=0, rtol=0)

    def test_combination_preserves_each_single_variant_stream_exactly(self):
        model = RefinementEncoder("oriented_form", code_dim=128)
        form = model.currents(self.images)[0]
        form_spikes = model(self.images)[:, :64]
        model.method = "linear_colour"
        colour = model.currents(self.images)[1]
        colour_spikes = model(self.images)[:, 64:]
        model.method = "oriented_linear"
        combined = model.currents(self.images)
        spikes = model(self.images)
        torch.testing.assert_close(combined[0], form, atol=0, rtol=0)
        torch.testing.assert_close(combined[1], colour, atol=0, rtol=0)
        torch.testing.assert_close(spikes[:, :64], form_spikes, atol=0, rtol=0)
        torch.testing.assert_close(spikes[:, 64:], colour_spikes, atol=0, rtol=0)

    def test_linear_colour_is_invariant_to_common_additive_pattern(self):
        model = RefinementEncoder("linear_colour", code_dim=128)
        x = .2 + .4 * self.images
        pattern = .2 * torch.rand(3, 1, 18, 74, generator=torch.Generator().manual_seed(4))
        a = refinement_maps(x, model.method, model.features.backbone)[1]
        b = refinement_maps(x + pattern, model.method, model.features.backbone)[1]
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)

    def test_black_and_uniform_views_are_finite(self):
        for method in METHODS:
            model = RefinementEncoder(method, code_dim=128)
            for level in (0., .5, 1.):
                self.assertTrue(torch.isfinite(model(torch.full_like(self.images, level))).all())


if __name__ == "__main__":
    unittest.main()
