import os
import sys
from os import path

import numpy as np
import torch

sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), '..'))
from Experiments.regret_lib import save_dataset, load_dataset


def main():
    entry = {
        'true_costs': [5.0, 1.0, 2.0],
        'customer_info': {'base_util': 0.0, 'home_util': 3.2,
                          'incentive_sensitivity': -0.25, 'revenue': 50.0},
        'pps_info': [{'util': -1.0}, {'util': -0.5}],
        'prices': np.zeros(3, dtype=np.float32),
        'feature_info': {'base_feat': torch.zeros((1, 3, 10, 10)), 'home_cell': (2, 3),
                         'pp_cells': [(4, 5), (6, 7)], 'time_int': 1,
                         'pp_caps': [10.0, 10.0], 'theta': 0.7,
                         'ins_cost_home': 100.0, 'ins_cost_pps': [50.0, 60.0],
                         'service_time': 120.0},
    }
    meta = {'instance': 'RC', 'mnl_dist_scaler': 1800.0, 'outside_option_util': 0.0,
            'min_price': -10.0, 'max_price': 10.0, 'home_failure': 0.0,
            'seed': 9999, 'episodes': 20, 'policy': 'static_home2.0_pp-5.0'}
    fp = path.join(path.dirname(path.abspath(__file__)), '_tmp_regret_ds.pt')
    save_dataset(fp, meta, [entry])
    m2, e2 = load_dataset(fp, expect_meta={'mnl_dist_scaler': 1800.0,
                                           'max_price': 10.0, 'instance': 'RC'})
    assert m2 == meta
    assert len(e2) == 1
    assert e2[0]['true_costs'] == entry['true_costs']
    assert torch.equal(e2[0]['feature_info']['base_feat'],
                       entry['feature_info']['base_feat'])
    try:
        load_dataset(fp, expect_meta={'max_price': 2.0})
        raise AssertionError('fingerprint mismatch not detected')
    except ValueError:
        pass
    os.remove(fp)
    print('OK test_regret_dataset_roundtrip')


if __name__ == '__main__':
    main()
