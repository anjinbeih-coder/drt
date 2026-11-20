import sys
from os import path
import numpy as np
import torch

sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), '..'))
from Experiments.regret_lib import (make_bare_oracle_agent, expected_profit_under,
                                    regret_for_sample)


def manual_expected_profit(prices, c, base_util, home_util, pp_utils, s, rev, v0):
    """独立的 numpy 实现:MNL 概率 + 期望利润,与库实现互为对照。"""
    utils = [base_util + home_util + s * prices[0]]
    for j, pu in enumerate(pp_utils):
        utils.append(pu + s * prices[j + 1])
    utils.append(v0)
    ex = np.exp(np.asarray(utils, dtype=np.float64))
    probs = ex / ex.sum()
    r = 0.0
    for k in range(len(prices)):
        r += probs[k] * (rev + prices[k] - c[k])
    return r


def main():
    agent = make_bare_oracle_agent(min_p=-10.0, max_p=10.0, base_util=0.0,
                                   revenue=50.0, outside_option_util=0.0)
    c = [12.0, 3.0]
    entry = {
        'true_costs': c,
        'customer_info': {'base_util': 0.0, 'home_util': 3.2,
                          'incentive_sensitivity': -0.25, 'revenue': 50.0},
        'pps_info': [{'util': -1.0}],
    }
    c_t = torch.tensor(c, dtype=torch.float32)

    # 1) 手算对照:库的期望利润 == 独立 numpy 实现(同一价格向量下)
    o = agent._pricing_oracle_lifted_torch(c_t, entry['customer_info'], entry['pps_info'])
    assert o is not None
    prices = o['prices'].cpu().numpy().tolist()
    r_lib = expected_profit_under(o, c_t)
    r_manual = manual_expected_profit(prices, c, 0.0, 3.2, [-1.0], -0.25, 50.0, 0.0)
    assert abs(r_lib - r_manual) < 1e-6, (r_lib, r_manual)

    # 2) oracle 近似最优性:0.05 步长网格搜索有界价格域,oracle 不明显劣于网格最优
    grid = np.arange(-10.0, 10.0 + 1e-9, 0.05)
    best = -1e18
    for a0 in grid:
        for a1 in grid:
            best = max(best, manual_expected_profit([a0, a1], c, 0.0, 3.2, [-1.0],
                                                    -0.25, 50.0, 0.0))
    assert r_lib >= best - 0.05, 'oracle profit {} << grid max {}'.format(r_lib, best)

    # 3) regret(c, c) == 0 精确成立(确定性 oracle 同输入同输出)
    res = regret_for_sample(agent, entry, c_t.clone())
    assert res is not None
    assert res['regret'] == 0.0, res['regret']
    assert res['mse'] == 0.0

    # 4) 压扁成本差的失真预测,遗憾严格为正
    res_bad = regret_for_sample(agent, entry, torch.tensor([8.0, 7.0]))
    assert res_bad is not None and res_bad['regret'] > 0.0, res_bad

    print('OK test_regret_toy')


if __name__ == '__main__':
    main()
