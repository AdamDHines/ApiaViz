import unittest
import numpy as np
from apiaviz.research.statistics import sign_flip_pvalue,holm_adjust,paired_effect

class PairedStatistics(unittest.TestCase):
    def test_exact_small_sample_resolution(self):
        self.assertEqual(sign_flip_pvalue(np.ones(8)),2/256)
        self.assertEqual(sign_flip_pvalue(np.zeros(8)),1.)
        self.assertEqual(sign_flip_pvalue([1,-1,2,-2]),1.)

    def test_holm_order_and_monotonicity(self):
        np.testing.assert_allclose(holm_adjust([.03,.001,.04,.02]),[.06,.004,.06,.06])

    def test_repeated_seeds_do_not_inflate_independent_sample_count(self):
        a=paired_effect(np.ones((8,2)),repeats=100)
        b=paired_effect(np.ones((8,20)),repeats=100)
        self.assertEqual(a["p_raw"],b["p_raw"])
        self.assertEqual(a["route_ci95"],[1.,1.])
        self.assertEqual(b["crossed_ci95"],[1.,1.])

    def test_sign_reversal_and_missing_data(self):
        x=np.arange(40).reshape(8,5)/10
        a,b=paired_effect(x,repeats=1000),paired_effect(-x,repeats=1000)
        self.assertEqual(a["p_raw"],b["p_raw"])
        np.testing.assert_allclose(a["crossed_ci95"],-np.array(b["crossed_ci95"])[::-1])
        with self.assertRaises(ValueError):paired_effect([[1,np.nan],[2,3]])

if __name__=="__main__":unittest.main()
