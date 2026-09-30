import copy
import json
from pathlib import Path
import unittest

from apiaviz.research.route_full import cases
from apiaviz.research.route_full_audit import check_motion


class FullSuiteTests(unittest.TestCase):
    def test_matrix_has_all_378_unique_conditions_and_both_phases(self):
        p = json.loads(Path('docs/route-continuous-full/archive/protocol.json').read_text())
        p['controllers'] = [c for c in p['controllers'] if c['name'] == 'familiarity']
        scheduled = list(cases(p))
        self.assertEqual(len(scheduled), 378)
        self.assertEqual(len({c[0] for c in scheduled}), 378)
        self.assertEqual({c[-1] for c in scheduled}, {-1, 1})

    def test_kick_is_separate_from_walking_even_at_same_timestamp(self):
        p = dict(evaluation=dict(kick_before_step=2), avoidance_settings=dict(stride_m=.1))
        base = dict(world_bounds_m=[-1, 1, -1, 1], headings=[0, 0], route=[[0, 0], [.2, .5]])
        detail = dict(events=[dict(kind='displacement', time_s=1, position=[.1, .5], displacement=[0, .5])],
            microtrace=[dict(time_s=1, position=[.1, 0], stride_m=.1, heading=0, path_m=.1),
                        dict(time_s=2, position=[.2, .5], stride_m=.1, heading=0, path_m=.2)])
        row = dict(disturbance_applied=True, path_length_m=.2, final_nest_distance_m=0)
        self.assertEqual(check_motion([0, 0], detail, row, p, base, [], dict(kick=.5)), (.2, 1))
        corrupt = copy.deepcopy(detail)
        corrupt['microtrace'][1]['position'][0] = .3
        with self.assertRaises(AssertionError):
            check_motion([0, 0], corrupt, row, p, base, [], dict(kick=.5))


if __name__ == '__main__':
    unittest.main()
