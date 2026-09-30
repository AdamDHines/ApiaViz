import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from apiaviz.research.circuit import CircuitConfig, SparseCircuit
from apiaviz.research.encoders import VisualEncoder, EncoderConfig
from apiaviz.research.metrics import cluster_bootstrap, reward_bootstrap
from apiaviz.research.navigation import evaluate_route, choose_heading, make_memory
from apiaviz.research.stimuli import perturb
from apiaviz.research.study import fingerprint, flower_split
from apiaviz.research.compare import compare
from apiaviz.nav.retino_kc import AntiHebbianMBON
from apiaviz.mbant.config import NavigationConfig


class CircuitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.circuit = SparseCircuit(12, 64, 4, config=CircuitConfig(duration_ms=50))
        self.inputs = torch.rand(5, 12, generator=torch.Generator().manual_seed(9))

    def test_repeat_batch_and_order_independence(self):
        whole = self.circuit.simulate(self.inputs, record=True)
        repeated = self.circuit.simulate(self.inputs, record=True)
        self.assertTrue(torch.equal(whole["kc_spikes"], repeated["kc_spikes"]))
        separate = torch.cat([self.circuit.simulate(x[None])["counts"] for x in self.inputs])
        self.assertTrue(torch.equal(whole["counts"], separate))
        self.assertTrue(torch.equal(whole["counts"].flip(0), self.circuit.simulate(self.inputs.flip(0))["counts"]))

    def test_silence_and_count_conservation(self):
        quiet = self.circuit.simulate(torch.zeros_like(self.inputs), record=True)
        self.assertEqual(int(quiet["counts"].sum()), 0)
        self.assertTrue(torch.isinf(quiet["first_spike_ms"]).all())
        ordinary = self.circuit.simulate(self.inputs, record=True)
        shifted = self.circuit.simulate(self.inputs, perturbation="shuffle-times", record=True)
        self.assertTrue(torch.equal(ordinary["pn_counts"], shifted["pn_counts"]))
        self.assertTrue(torch.equal(ordinary["counts"], ordinary["kc_spikes"].sum(0)))
        self.assertFalse(torch.equal(ordinary["pn_spikes"], shifted["pn_spikes"]))

    def test_interventions_and_state_are_explicit(self):
        base = self.circuit.simulate(self.inputs, record=True)
        no_inhib = self.circuit.simulate(self.inputs, perturbation="no-inhibition", record=True)
        self.assertEqual(float(no_inhib["inhibition"].sum()), 0)
        self.assertGreater(float(base["inhibition"].sum()), 0)
        no_adapt = self.circuit.simulate(self.inputs, perturbation="no-adaptation")
        self.assertEqual(float(no_adapt["state"]["adapt"].sum()), 0)
        continued = self.circuit.simulate(self.inputs, state=base["state"])
        self.assertTrue(torch.isfinite(continued["counts"]).all())
        self.assertTrue(torch.equal(base["counts"], self.circuit.simulate(self.inputs)["counts"]))

    def test_connections_private_rng_and_frozen(self):
        torch.manual_seed(1)
        expected = torch.get_rng_state().clone()
        model = SparseCircuit(12, 64, 4)
        self.assertTrue(torch.equal(expected, torch.get_rng_state()))
        self.assertEqual(list(model.parameters()), [])
        for row in model.indices:
            self.assertEqual(len(torch.unique(row)), 4)
        before = fingerprint(model)
        model.simulate(self.inputs)
        self.assertEqual(before, fingerprint(model))

    def test_timestep_refinement_has_bounded_rate_change(self):
        # Deterministic tonic PN firing has an analytic rate. Check integration
        # convergence separately from recurrent threshold discontinuities.
        x = torch.full((1, 12), .8)
        rates = []
        for dt in (1., .5, .25):
            circuit = SparseCircuit(12, 64, 4, config=CircuitConfig(duration_ms=100, dt_ms=dt))
            rates.append(float(circuit.pn_spikes(x)[0].sum(0).float().mean()))
        self.assertLessEqual(abs(rates[-1] - rates[-2]), abs(rates[-1] - rates[0]) + 1)
        self.assertLess(abs(rates[-1] - rates[0]), 3)


class EncoderTests(unittest.TestCase):
    def test_shape_calibration_and_frozen_memory(self):
        images = torch.rand(4, 3, 19, 30, generator=torch.Generator().manual_seed(8))
        model = VisualEncoder(EncoderConfig(code_dim=64, pool_hw=(2, 4), fan_in=4))
        report = model.calibrate(images, candidates=5)
        self.assertIn("target_error", report)
        before = fingerprint(model)
        codes = model(images)
        self.assertEqual(codes.shape, (4, 64))
        for name in ("cosine", "single", "population"):
            memory = make_memory(codes, name, 2)
            self.assertTrue(torch.isfinite(memory(codes)).all())
        self.assertEqual(before, fingerprint(model))
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_kwta_total_budget_with_silent_colour(self):
        model = VisualEncoder(EncoderConfig(representation="gray", mode="kwta", code_dim=100,
                                           sparsity=.1, pool_hw=(2, 4), fan_in=4))
        codes = model(torch.ones(3, 3, 19, 30) * .5)
        self.assertTrue(torch.equal(codes.sum(1), torch.full((3,), 10.)))
        zero = model(torch.zeros(1, 3, 19, 30))
        self.assertEqual(float(zero.sum()), 0)

    def test_legacy_matches_existing_projection(self):
        model = VisualEncoder(EncoderConfig(mode="legacy", code_dim=64, pool_hw=(2, 4), fan_in=4))
        images = torch.rand(2, 3, 19, 30)
        form, colour = model.maps(images)
        expected = torch.cat([torch.nn.functional.normalize(model.legacy[0](form), dim=1),
                              torch.nn.functional.normalize(model.legacy[1](colour[:, 1:]), dim=1)], 1)
        self.assertTrue(torch.equal(model(images), expected))

    def test_silent_memory_is_unfamiliar(self):
        memory = AntiHebbianMBON(4)
        memory.store(torch.eye(4))
        self.assertEqual(float(memory(torch.zeros(1, 4))), 1.)
        self.assertLess(float(memory(torch.ones(1, 4))), 1.)

    def test_uniform_achromatic_images_provide_no_contrast_evidence(self):
        model = VisualEncoder(EncoderConfig(code_dim=64, pool_hw=(2, 4), fan_in=4))
        images = torch.stack([torch.full((3, 19, 30), level) for level in (0., .5, 1.)])
        codes = model(images)
        self.assertTrue(torch.isfinite(codes).all())
        self.assertEqual(float(codes.sum()), 0.)


class EvaluationTests(unittest.TestCase):
    def test_perturbation_order_and_clipping_measurement(self):
        images = torch.ones(3, 3, 10, 12) * .5
        images[:, 1] += .1
        images[:, 2] -= .1
        keys = [21, 5, 99]
        out, report = perturb(images, "luminance", .02, 7, keys)
        reversed_out, _ = perturb(images.flip(0), "luminance", .02, 7, keys[::-1])
        self.assertTrue(torch.equal(out.flip(0), reversed_out))
        self.assertLess(report["opponent_change_mae"], 1e-6)
        _, clipped = perturb(images, "luminance", 20., 7, keys)
        self.assertGreater(clipped["clipped_fraction"], 0)
        self.assertGreater(clipped["opponent_change_mae"], 0)

    def test_silent_and_tied_scans_do_not_turn_arbitrarily(self):
        offsets = np.array([20., 10., 0., -10., -20.])
        self.assertEqual(choose_heading(np.full(5, np.inf), offsets), (2, True))
        self.assertEqual(choose_heading(np.zeros(5), offsets), (2, True))

    def test_navigation_protocols_and_metrics(self):
        class FakeScorer:
            world = SimpleNamespace(nav=NavigationConfig())
            def __call__(self, pos, headings):
                return np.abs(headings)
        positions = np.array([[0., 0.], [.1, 0.], [.2, 0.], [.3, 0.], [.4, 0.]])
        for protocol in ("reset", "free"):
            result = evaluate_route(positions, np.zeros(4), FakeScorer(), protocol, 10)
            self.assertTrue(result["reached_nest"])
            self.assertEqual(result["corrective_resets"], 0)
            self.assertAlmostEqual(result["path_length_m"], result["steps"] * .1)
        offline = evaluate_route(positions, np.zeros(4), FakeScorer(), "offline", offline_stride=1)
        self.assertEqual(offline["scans"], 12)
        self.assertEqual(offline["heading_error_deg"], 0)

    def test_bootstrap_unit_and_repeat_determinism(self):
        self.assertIsNone(cluster_bootstrap([1, 0], [1, 1])["ci95"])
        report = cluster_bootstrap([1, 1, 0, 0], [1, 1, 2, 2], samples=50)
        self.assertEqual(report["clusters"], 2)
        self.assertEqual(report, cluster_bootstrap([1, 1, 0, 0], [1, 1, 2, 2], samples=50))
        result = reward_bootstrap([0, 1, 0, 1], [0., 1., 0., 1.], [0, 1, 0, 1], [0, 1, 0, 1], samples=30)
        self.assertEqual(result["auc"]["value"], 1.)

    def test_resets_are_counted_without_inflating_travel(self):
        class WrongHeading:
            world = SimpleNamespace(nav=NavigationConfig())
            def __call__(self, pos, headings):
                return np.arange(len(headings), dtype=float)
        positions = np.column_stack([np.arange(21) * .1, np.zeros(21)])
        reset = evaluate_route(positions, np.zeros(20), WrongHeading(), "reset", 10)
        free = evaluate_route(positions, np.zeros(20), WrongHeading(), "free", 10)
        self.assertGreater(reset["corrective_resets"], 0)
        self.assertEqual(free["corrective_resets"], 0)
        self.assertAlmostEqual(reset["path_length_m"], 1.)

    def test_flower_calibration_excludes_every_cv_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a", "b"):
                folder = Path(tmp) / name
                folder.mkdir()
                for index in range(10):
                    (folder / f"{index}.jpg").touch()
            args = SimpleNamespace(flower_dir=Path(tmp), split_seed=7, folds=2,
                                   development_per_class=2, per_class=6)
            paths, labels, development, classes = flower_split(args)
            self.assertFalse(set(paths) & set(development))
            self.assertEqual(len(development), 4)
            self.assertEqual(len(paths), 12)

    def test_paired_comparison_resamples_aligned_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for ant in (1, 2):
                for encoder, reached in (("a", False), ("b", True)):
                    rows.append({"task": "nav", "encoder": encoder, "ant": ant, "route": 1,
                                 "environment_seed": 7, "reached_nest": reached})
            path = Path(tmp)
            (path / "results.jsonl").write_text("\n".join(map(json.dumps, rows)))
            result = compare(path, "a", "b", samples=30)[0]["candidate_minus_baseline"]["reached_nest"]
            self.assertEqual(result["mean"], 1.)
            self.assertEqual(result["ci95"], [1., 1.])


if __name__ == "__main__":
    unittest.main()
