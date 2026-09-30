import unittest
from types import SimpleNamespace

import numpy as np
import torch

from apiaviz.research.openloop import acquisition_views, restoring_probe
from apiaviz.research.lateralized import LateralizedMemory, LateralizedScorer
from apiaviz.research.sequential import SequentialMemory
from apiaviz.research.latency import LatencyEncoder, LatencyConfig, latency_race
from apiaviz.research.prototypes import PrototypeMemory
from apiaviz.research.spike_overlap import SpikeOverlapMemory
from apiaviz.research.study import fingerprint
from apiaviz.research.navigation import evaluate_route
from apiaviz.mbant.config import NavigationConfig


class SpikeOverlapTests(unittest.TestCase):
    def test_retrieval_ignores_amplitude_but_retains_spike_identity(self):
        patterns = torch.tensor([[2., 3., 0., 0.], [0., 0., 4., 1.]])
        memory = SpikeOverlapMemory(patterns)
        before = fingerprint(memory)
        queries = torch.tensor([[.1, 7., 0., 0.], [1., 0., 1., 0.], [0., 0., 0., 0.]])
        scores = memory(queries)
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])
        torch.testing.assert_close(scores, memory(queries * torch.tensor([9., .2, 7., .3])))
        other = SpikeOverlapMemory(patterns * torch.tensor([.1, 8., 2., 9.]))
        torch.testing.assert_close(scores, other(queries))
        self.assertEqual(before, fingerprint(memory))


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.column_stack([np.arange(11) * .1, np.zeros(11)])
        self.headings = np.zeros(10)

    def test_convergent_teaching_points_inward_without_changing_positions(self):
        pos, parallel = acquisition_views(self.positions, self.headings, 3, .2)
        other, convergent = acquisition_views(self.positions, self.headings, 3, .2, .5)
        np.testing.assert_array_equal(pos, other)
        np.testing.assert_array_equal(parallel, np.zeros(30))
        self.assertTrue(np.all(convergent[0::3] > 0))
        self.assertTrue(np.all(convergent[2::3] < 0))
        np.testing.assert_array_equal(convergent[1::3], np.zeros(10))
        self.assertAlmostEqual(convergent[0], np.degrees(np.arctan2(.2, .5)), places=5)

    def test_rotation_equivariance_of_acquisition(self):
        pos, head = acquisition_views(self.positions, self.headings, 3, .2, .5)
        rotated = self.positions[:, ::-1].copy()
        other, angle = acquisition_views(rotated, self.headings + 90, 3, .2, .5)
        np.testing.assert_allclose(other, np.column_stack([-pos[:, 1], pos[:, 0]]), atol=1e-7)
        np.testing.assert_allclose((angle - head + 180) % 360 - 180, 90, atol=1e-5)

    def test_local_convergence_preserves_curved_route_tangents(self):
        positions = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        headings = np.array([0., 90., 180.])
        _, angle = acquisition_views(positions, headings, 3, .2, .5, "tangent")
        error = (angle.reshape(-1, 3) - headings[:, None] + 180) % 360 - 180
        np.testing.assert_allclose(error[:, 1], 0., atol=1e-5)
        self.assertTrue(np.all(error[:, 0] > 0))
        self.assertTrue(np.all(error[:, 2] < 0))

    def test_restoring_probe_distinguishes_recognition_from_recovery(self):
        def tangent(pos, headings):
            return np.abs(headings)

        def inward(pos, headings):
            target = -20 if pos[1] > 0 else 20
            return np.abs(headings - target)

        flat = lambda pos, headings: np.zeros(len(headings))
        for scorer, expected in ((tangent, 0.), (inward, 1.), (flat, 0.)):
            result = restoring_probe(None, scorer, self.positions, self.headings)
            self.assertEqual(result["restoring_fraction"], expected)


class LateralizedTests(unittest.TestCase):
    def test_error_direction_silence_and_frozen_synapses(self):
        # Left-looking views must produce right turns, and conversely.
        codes = torch.eye(4)
        memory = LateralizedMemory(codes, [20., -20., 20., -20.], [0, 0, 1, 1])
        before = fingerprint(memory)
        result = memory(torch.cat([codes, torch.zeros(1, 4)]))
        self.assertTrue(torch.all(result["turn_signal"][[0, 2]] < 0))
        self.assertTrue(torch.all(result["turn_signal"][[1, 3]] > 0))
        self.assertEqual(float(result["turn_signal"][-1]), 0.)
        self.assertTrue(bool(result["no_evidence"][-1]))
        self.assertEqual(fingerprint(memory), before)
        self.assertEqual(list(memory.parameters()), [])

    def test_controller_observes_only_current_heading(self):
        class FakeWorld:
            def scan(self, pos, headings):
                self.headings = headings
                return torch.tensor([[1., 0.]])

        world = FakeWorld()
        memory = LateralizedMemory(torch.eye(2), [20., -20.])
        scorer = LateralizedScorer(world, None, memory, lambda model, images: images, 60.)
        proposed = np.arange(90., -31., -10.)  # centre heading = 30 degrees
        scores = scorer(np.zeros(2), proposed)
        np.testing.assert_array_equal(world.headings, [30.])
        self.assertLess(proposed[np.argmin(scores)], 30.)


class SequenceTests(unittest.TestCase):
    def test_context_changes_only_after_committed_visual_choice(self):
        memory = SequentialMemory(torch.eye(6), 6, ahead=1, behind=1)
        # A perfect match at a distant place is excluded by acquisition order.
        scores = memory(torch.eye(6))
        self.assertEqual(float(scores[5]), 1.)
        self.assertEqual(memory.cursor, 0)
        self.assertLess(float(scores[1]), float(scores[5]))
        memory.commit(1)
        self.assertEqual(memory.cursor, 1)
        scores = memory(torch.eye(6))
        self.assertLess(float(scores[2]), float(scores[5]))
        memory.commit(0)
        self.assertEqual(memory.cursor, 1)
        memory.reset()
        self.assertEqual(memory.cursor, 0)

    def test_continuous_policy_does_not_receive_route_geometry(self):
        class Policy:
            world = SimpleNamespace(nav=NavigationConfig(step_size=.1, dis_threshold=.2))

            def steer(self, position, heading):
                return 2.5, False

        route_a = np.array([[0., 0.], [1., 0.], [10., 0.]])
        route_b = np.array([[0., 0.], [-5., 5.], [10., 0.]])
        outputs = [evaluate_route(p, [0., 0.], Policy(), "free", 3,
                                  initial_position=[0., .2], initial_heading=30.)
                   for p in (route_a, route_b)]
        np.testing.assert_array_equal([r["position"] for r in outputs[0]["trace"]],
                                      [r["position"] for r in outputs[1]["trace"]])
        self.assertEqual(outputs[0]["trace"][0]["heading"], 32.5)
        self.assertEqual(outputs[0]["corrective_resets"], 0)


class LatencyTests(unittest.TestCase):
    def test_spike_times_solve_lif_and_silence_is_preserved(self):
        drive = torch.tensor([[0., .5, 1., 2.]])
        result = latency_race(drive, 2)
        times = result["spike_times_ms"]
        emitted = result["counts"].bool()
        voltage = (1 + drive) * (1 - torch.exp(-times / 10.))
        torch.testing.assert_close(voltage[emitted], torch.ones(2))
        torch.testing.assert_close(result["codes"], torch.tensor([[0., 0., 1., 2.]]))
        self.assertTrue(torch.isinf(times[0, 0]))
        self.assertEqual(int(latency_race(torch.zeros_like(drive), 2)["counts"].sum()), 0)

    def test_delayed_inhibition_allows_extra_spikes_and_ties_are_simultaneous(self):
        drive = torch.tensor([[.5, 1., 2., 3.]])
        ideal = latency_race(drive, 1)
        delayed = latency_race(drive, 1, LatencyConfig(inhibition_delay_ms=10.))
        self.assertGreater(float(delayed["counts"].sum()), float(ideal["counts"].sum()))
        tied = latency_race(torch.ones(1, 4), 1)
        self.assertEqual(int(tied["counts"].sum()), 4)
        binned = latency_race(drive, 2, LatencyConfig(time_bin_ms=1.))
        times = binned["spike_times_ms"]
        self.assertTrue(torch.equal(times[torch.isfinite(times)], times[torch.isfinite(times)].ceil()))
        self.assertTrue(torch.all(binned["codes"] <= drive + 1e-6))

    def test_functional_reference_preserves_historical_graded_code(self):
        encoder = LatencyEncoder(code_dim=128)
        images = torch.rand(2, 3, 18, 74, generator=torch.Generator().manual_seed(8))
        # This is an explicit conversion-fidelity contract, not a superiority test.
        torch.testing.assert_close(encoder(images), encoder.features(images), atol=2e-6, rtol=2e-5)

    def test_current_gain_spreads_latency_without_changing_ideal_information(self):
        drive = torch.tensor([[1., 2., 3., 4.]])
        standard = latency_race(drive, 2)
        slower = latency_race(drive, 2, LatencyConfig(current_gain=.1))
        emitted = slower["counts"].bool()
        self.assertTrue(torch.all(slower["spike_times_ms"][emitted] > standard["spike_times_ms"][emitted]))
        torch.testing.assert_close(slower["codes"], standard["codes"])
        voltage = (1 + .1 * drive) * (1 - torch.exp(-slower["spike_times_ms"] / 10.))
        torch.testing.assert_close(voltage[emitted], torch.ones(2))


class PrototypeTests(unittest.TestCase):
    def test_repeated_exposure_does_not_saturate_and_capacity_is_bounded(self):
        codes = torch.tensor([[1., .5, 0.], [1., .25, 0.], [0., 0., 1.], [0., 0., 1.]])
        memory = PrototypeMemory(codes, 2)
        repeated = PrototypeMemory(codes.repeat_interleave(3, 0), 2)
        self.assertEqual(memory.memory.shape, (2, 3))
        torch.testing.assert_close(memory.memory, repeated.memory)
        scores = memory(torch.tensor([[0., 0., 1.], [0., 1., 0.]]))
        self.assertLess(float(scores[0]), float(scores[1]))


if __name__ == "__main__":
    unittest.main()
