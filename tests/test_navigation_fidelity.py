import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from apiaviz.research.dual_camera import sensor_views
from apiaviz.research.uv_trial_report import frame_display,display_response
from apiaviz.research.navigation_fidelity import protocol
from apiaviz.research.uv_trials import configuration,navigation_components
from apiaviz.research.confirmed_exploration import ConfirmedExplorationController
from apiaviz.research.familiarity_controller import SensorView


class FidelityTests(unittest.TestCase):
    def test_serial_and_parallel_dispatch_share_versioned_controller_for_every_model(self):
        from apiaviz.research.familiarity_controller import FamiliarityController
        from apiaviz.research.coherent_navigation import evaluate
        for method in ('apiaviz_uv','sobel_colour','ardin_input'):
            c,e=navigation_components(dict(navigation_revision='direction-before-cast-v1',method=method))
            self.assertIs(c,ConfirmedExplorationController);self.assertIs(e,evaluate)
        self.assertIs(navigation_components({})[0],FamiliarityController)
        # Historical wrappers can still supply their own explicit controls.
        sentinel=object();self.assertIs(navigation_components({},sentinel)[0],sentinel)
        with self.assertRaises(ValueError):navigation_components(dict(navigation_revision='typo'))

    def test_matched_protocol_preserves_models_budgets_and_requires_reteaching(self):
        parent=configuration(False)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory);(source/'protocol.json').write_text(json.dumps(parent))
            new=protocol(source)
        self.assertEqual(new['trials'],18)
        self.assertEqual(new['methods'],parent['methods'])
        self.assertEqual(new['sensor_settings'],parent['sensor_settings'])
        self.assertEqual(new['evaluation'],parent['evaluation'])
        self.assertEqual(new['navigation_revision'],'direction-before-cast-v1')
        self.assertEqual(new['render']['retinal_sampling'],'solid-angle-box-before-response-v1')
        self.assertNotIn('retinal_sampling',parent['render'])
        self.assertFalse(new['readiness_rule']['larger_study_authorized'])

    def test_report_uses_exact_policy_retina_before_display_only_transfer(self):
        raw=np.random.default_rng(2).uniform(0,4,(144,480,6)).astype('float32')
        config=dict(retinal_sampling='solid-angle-box-before-response-v1',elevation_deg=[90,-20],
                    retina_shape=[51,199],visible_white=1)
        uv,rgb=frame_display(raw,12,config)
        actual=sensor_views(raw,[12],config)[0].permute(1,2,0).numpy()[:,::-1]
        np.testing.assert_array_equal(rgb,display_response(actual))
        self.assertGreater(float(abs(rgb-actual).max()),.1)
        raw[:,:,:3]*=1000
        np.testing.assert_array_equal(rgb,frame_display(raw,12,config)[1])
        with self.assertRaises(ValueError):sensor_views(raw,[0],dict(config,retinal_sampling='typo'))

    def test_periodic_check_charges_observations_for_both_phases_and_preserves_memory_reference(self):
        for phase in (-1,1):
            policy=ConfirmedExplorationController(dict(scale=1),phase=phase)
            policy.state='follow';policy.anchor=policy.reference=policy.travel=0.;policy.last_check=11
            calls=[];moves=[]
            def observe(h):calls.append(h);return .9-.01*abs(h),h,.05*len(calls)
            sensor=SensorView(0,0,1.1,observe,lambda h:(moves.append(h) is None,h,.05*len(calls)),lambda:None,lambda _:None)
            self.assertTrue(policy.step(sensor,12))
            self.assertEqual(policy.travel,0);self.assertEqual(policy.cast_attempts,0)
            self.assertGreaterEqual(len(calls),5);self.assertEqual(policy.pending['reference'],0)
            self.assertAlmostEqual(policy.decisions[-1]['time_end_s'],.05*len(calls))

    def test_exhausted_observation_budget_prevents_blind_cast_or_movement(self):
        policy=ConfirmedExplorationController(dict(scale=1));policy.state='follow'
        policy.anchor=policy.reference=policy.travel=0.;policy.last_check=11
        moves=[]
        sensor=SensorView(0,360,1.1,lambda h:(None,h,360),lambda h:(moves.append(h),h,360),lambda:None,lambda _:None)
        self.assertFalse(policy.step(sensor,12));self.assertEqual(moves,[])
        self.assertEqual(policy.cast_attempts,0);self.assertIsNone(policy.pending)


if __name__=='__main__':unittest.main()
