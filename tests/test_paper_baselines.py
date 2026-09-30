import unittest

import torch

from apiaviz.research.latency import LatencyEncoder, LatencyConfig
from apiaviz.research.paper_baselines import FrontendEncoder, METHODS, feature_maps
from apiaviz.research.study import fingerprint


class MatchedFrontends(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.images = torch.rand(3, 3, 18, 74, generator=torch.Generator().manual_seed(47))

    def test_apia_is_the_previously_frozen_model(self):
        circuit = LatencyConfig(current_gain=.1, time_bin_ms=1, inhibition_delay_ms=1)
        original = LatencyEncoder(code_dim=128, sparsity=.02825, circuit=circuit)
        compared = FrontendEncoder(code_dim=128)
        self.assertEqual(fingerprint(original), fingerprint(compared))
        torch.testing.assert_close(original(self.images), compared(self.images), rtol=0, atol=0)

    def test_only_preprocessing_changes_and_both_pools_are_usable(self):
        model = FrontendEncoder(code_dim=128)
        before = fingerprint(model)
        for method in METHODS:
            model.method = method
            form, auxiliary = feature_maps(self.images, method, model.features.backbone)
            self.assertEqual(form.shape[1], 3)
            self.assertEqual(auxiliary.shape[1], 2)
            for drive in model.currents(self.images):
                self.assertTrue(torch.isfinite(drive).all())
                self.assertGreater(float(drive.max()), 0.)
            self.assertEqual(fingerprint(model), before)
            self.assertEqual(model.features.legacy[0].active_units, 2)

    def test_achromatic_processing_discards_colour_but_opponency_retains_it(self):
        model = FrontendEncoder(code_dim=128)
        swapped = self.images[:, [0, 2, 1]]
        for method in ("gray", "ardin_input"):
            a = feature_maps(self.images, method, model.features.backbone)
            b = feature_maps(swapped, method, model.features.backbone)
            for x, y in zip(a, b): torch.testing.assert_close(x, y)
        _, a = feature_maps(self.images, "opponent", model.features.backbone)
        _, b = feature_maps(swapped, "opponent", model.features.backbone)
        torch.testing.assert_close(a.flip(1), b)


if __name__ == "__main__":
    unittest.main()
