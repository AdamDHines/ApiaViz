import unittest
import torch
from apiaviz.research.graded_uv import AdaptationConfig, initialize, advance, response_sequence, opponent_maps, GradedUVEncoder


class GradedUVTests(unittest.TestCase):
    def test_constant_exposure_subdivision_and_no_mutation(self):
        initial=torch.ones(2,3,6,8,dtype=torch.float64)
        x=initial*4;state=initialize(initial)
        full,final=advance(x,state,.1)
        partial=state
        for _ in range(10):_,partial=advance(x,partial,.01)
        torch.testing.assert_close(final.irradiance,partial.irradiance)
        torch.testing.assert_close(final.background,partial.background)
        self.assertTrue(torch.equal(state.irradiance,initial))
        self.assertTrue((full>.5).all())

    def test_adaptation_preserves_spatial_contrast_and_recovers_exposure(self):
        x=torch.ones(1,3,6,8);x[:,:,:,4:]=2
        baseline,_=advance(x,initialize(x),1.)
        y,state=advance(4*x,initialize(x),.05)
        self.assertGreater(float(y.mean()),float(baseline.mean()))
        settled,_=advance(4*x,state,20.)
        torch.testing.assert_close(settled,baseline)
        self.assertGreater(float(settled[:,:,:,4:].mean()),float(settled[:,:,:,:4].mean()))

    def test_batch_rows_have_independent_histories(self):
        initial=torch.ones(2,3,6,8);initial[1]*=4
        x=torch.ones_like(initial)*2
        both,_=advance(x,initialize(initial),.1)
        for i in range(2):
            one,_=advance(x[i:i+1],initialize(initial[i:i+1]),.1)
            torch.testing.assert_close(one,both[i:i+1])
        self.assertFalse(torch.equal(both[0],both[1]))

    def test_combined_opponency_sign_and_achromatic_cancellation(self):
        white=torch.ones(1,3,6,8)*.5
        self.assertEqual(float(opponent_maps(white,'combined').abs().sum()),0.)
        uv=white.clone();uv[:,0]=.8
        self.assertTrue((opponent_maps(uv,'combined')[:,0]>0).all())
        bg=white.clone();bg[:,1:]=.8
        self.assertTrue((opponent_maps(bg,'combined')[:,1]>0).all())

    def test_sequence_prefix_is_causal(self):
        initial=torch.ones(1,3,6,8)
        images=torch.stack([initial,initial*2,initial*.1])
        full,_=response_sequence(images,initial,.05,adaptive=True)
        prefix,_=response_sequence(images[:2],initial,.05,adaptive=True)
        self.assertTrue(torch.equal(prefix,full[:2]))

    def test_invalid_time_radiance_and_config(self):
        x=torch.ones(1,3,6,8)
        for t in [0,-1,float('nan')]:
            with self.assertRaises(ValueError):advance(x,initialize(x),t)
        with self.assertRaises(ValueError):advance(-x,initialize(x),1)
        with self.assertRaises(ValueError):AdaptationConfig(gain_tau_s=0)

    def test_neutral_field_silent_and_opponent_choice_preserves_visible(self):
        torch.set_num_threads(2)
        model=GradedUVEncoder()
        neutral=torch.full((1,3,51,199),.5)
        pooled=model.pooled(neutral,'combined')
        self.assertTrue(all(torch.count_nonzero(x)==0 for x in pooled))
        self.assertTrue(all(torch.count_nonzero(x)==0 for x in model.readouts(pooled).values()))
        image=torch.rand(2,3,51,199,generator=torch.Generator().manual_seed(19))
        a=model.pooled(image,'pairwise');b=model.pooled(image,'combined')
        for x,y in zip(a[:2],b[:2]):self.assertTrue(torch.equal(x,y))
        ra=model.readouts(a);rb=model.readouts(b)
        for key in ('graded_no_uv','spikes_no_uv','production_spikes_no_uv'):
            self.assertTrue(torch.equal(ra[key],rb[key]))


if __name__=='__main__':unittest.main()
