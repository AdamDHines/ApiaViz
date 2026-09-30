import unittest

from apiaviz.research.controller_full_report import (invalid_blocks, paired_effect,
                                                    retention, world_interval)


def row(world,seed,method,controller,phase,success,scenario='aligned',termination='arrival'):
    return dict(world=world,seed=seed,method=method,controller=controller,phase=phase,
                reached_nest=success,scenario=scenario,termination=termination)


class FullControllerStatisticsTests(unittest.TestCase):
    def test_average_phases_before_paired_world_comparison(self):
        rows=[row('a',1,'api','familiarity',1,True),row('a',1,'api','familiarity',-1,False),
              row('a',1,'api','active',1,True),row('a',2,'api','familiarity',1,True),
              row('a',2,'api','familiarity',-1,True),row('a',2,'api','active',1,False),
              row('b',1,'api','familiarity',1,False),row('b',1,'api','familiarity',-1,False),
              row('b',1,'api','active',1,True)]
        result=paired_effect(rows,'reached_nest',('familiarity',None),('active',None))
        self.assertEqual(result['by_world'],{'a':.25,'b':-1.})
        self.assertEqual(result['mean'],-.375)
        self.assertEqual(result['paired_cases'],3)
        self.assertEqual(result['worlds'],2)

    def test_preprocessing_comparison_pairs_methods_without_mixing_seeds(self):
        rows=[row('a',1,'api','familiarity',1,True),row('a',1,'api','familiarity',-1,False),
              row('a',1,'sobel','familiarity',1,False),row('a',1,'sobel','familiarity',-1,False),
              row('a',2,'api','familiarity',1,False)]
        result=paired_effect(rows,'reached_nest',('familiarity','api'),('familiarity','sobel'))
        self.assertEqual(result['mean'],.5)
        self.assertEqual(result['paired_cases'],1)
        self.assertEqual(result['missing_pairs'],1)

    def test_intervals_resample_worlds_not_duplicate_rows(self):
        result=world_interval([-.5,0.,.5])
        self.assertEqual(result,dict(mean=0.,low=-.5,high=.5,worlds=3))
        self.assertEqual(world_interval([])['mean'],None)

    def test_invalid_kick_marks_complete_paired_block_only(self):
        rows=[row('a',1,'api','familiarity',1,False,'kick_left50','displacement_into_rock'),
              row('a',1,'sobel','active',1,True,'kick_left50'),
              row('b',1,'api','active',1,False,'left50','rock_collision')]
        blocked=invalid_blocks(rows)
        self.assertEqual(blocked,{('a',1,'kick_left50')})
        valid=[r for r in rows if (r['world'],r['seed'],r['scenario']) not in blocked]
        self.assertEqual(len(valid),1)
        self.assertEqual(valid[0]['termination'],'rock_collision')

    def test_return_and_renewed_loss_require_sustained_positions(self):
        distances=[.5,.1,.09,.08,.3,.3,.1,.3,.3,.3]
        trace=[dict(step=i+1,polyline_m=x) for i,x in enumerate(distances)]
        self.assertEqual(retention(trace,'left50'),dict(return_found=True,lost_after_return=True))
        self.assertEqual(retention(trace[:7],'left50'),dict(return_found=True,lost_after_return=False))
        self.assertEqual(retention(trace,'aligned'),dict(return_found=False,lost_after_return=False))

    def test_pre_kick_route_following_does_not_count_as_recovery(self):
        trace=[dict(step=i,polyline_m=0.) for i in range(1,36)]
        self.assertFalse(retention(trace,'kick_right50')['return_found'])
        trace.extend(dict(step=i,polyline_m=.5) for i in range(36,40))
        self.assertFalse(retention(trace,'kick_right50')['return_found'])


if __name__=='__main__':unittest.main()
