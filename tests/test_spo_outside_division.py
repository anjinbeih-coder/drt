import sys
from os import path
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
import numpy as np
import torch
from types import SimpleNamespace
from Src.Algorithms.DSPO import DSPO
from Src.Algorithms.DSPO_plus_SPO import DSPO_plus_SPO
from Environments.OOH.containers import Location, ParcelPoint, Customer
from Experiments.regret_lib import make_bare_oracle_agent

def stub_common(obj, outside_util):
    obj.initial_phase = False
    obj.load_data = False
    obj.init_theta = 0.75
    obj.cool_theta = 0.0
    obj.cost_multiplier = 0.01
    obj.base_util = 0.0
    obj.revenue = 50.0
    obj.min_p, obj.max_p = -1e9, 1e9
    obj.config = SimpleNamespace(outside_option_util=outside_util)
    obj.get_feature_rep_infer = lambda fleet: torch.zeros((1, 1, 2, 2))
    obj.get_prediction = lambda feat, home, pps: [0.0, 7.0, 3.0]
    obj.cheapestInsertionCosts = lambda loc, fleet: 600.0
    obj.mnl = lambda cust, pp: 0.0
    obj.adjacency = None
    obj.interval = 3600
    obj.n_layers = 1
    obj.customer_cell = {0: [0, 0], 1: [0, 0], 2: [0, 0]}  # minimal stub for cell locations

def make_state():
    home = Location(0.0, 0.0, 1, 0)
    cust = Customer(home, -0.25, 3.2, 120.0, 1)
    pp = ParcelPoint(Location(5.0, 5.0, 2, 0), 1000000, 2)
    return [cust, {"fleet": None}, {"parcelpoints": np.array([pp])}, 0]

def parent_prices(outside_util):
    obj = DSPO.__new__(DSPO)
    stub_common(obj, outside_util)
    return DSPO.get_action_pricing(obj, make_state(), training=False)

def child_prices(outside_util):
    obj = DSPO_plus_SPO.__new__(DSPO_plus_SPO)
    stub_common(obj, outside_util)
    obj.get_prediction_with_q = lambda feat, home, pps: (
        obj.get_prediction(feat, home, pps), [0.0] * (len(pps) + 2))
    obj.beta_ivt = 0.0
    obj.vehicle_speed = 1.0
    obj.spo_w0_band = False
    obj.spo_enabled = True
    obj.l0_home = 0.0
    obj.l_mp = 0.0
    obj.use_travel_time_prediction = False
    obj.spo_current_episode_data = {'pricing_decisions': [], 'predicted_costs': [],
                                    'true_costs': [], 'customer_ids': [], 'arrival_times': []}
    return DSPO_plus_SPO.get_action_pricing(obj, make_state(), training=False)

def main():
    # 锚点 1:V0=None 时子类(spo_enabled)与父类定价一致(l0=l_mp=0)
    assert np.allclose(parent_prices(None), child_prices(None), atol=0.011), \
        f"{parent_prices(None)} vs {child_prices(None)}"
    # 锚点 2:V0=0 与 None 一致(除法形式)
    assert np.allclose(child_prices(0.0), child_prices(None), atol=1e-9)
    # 方向:V0=2 时子类与父类同向、同幅统一下移
    p2, c2 = parent_prices(2.0), child_prices(2.0)
    assert np.allclose(p2, c2, atol=0.011), f"{p2} vs {c2}"
    shift = child_prices(0.0) - c2
    assert np.all(shift > 0.1), "prices must drop when outside option strengthens"

    # oracle:价格不动点除法 + 概率 softmax 保留外部项(P 之和 < 1)
    obj = make_bare_oracle_agent(min_p=-1e9, max_p=1e9, outside_option_util=0.0)
    costs = torch.tensor([7.0, 3.0], dtype=torch.float64)
    cust_info = {'base_util': 0.0, 'home_util': 3.2, 'incentive_sensitivity': -0.25, 'revenue': 50.0}
    out0 = DSPO_plus_SPO._pricing_oracle_lifted_torch(obj, costs, cust_info, [{'util': 0.0}])
    obj.config = SimpleNamespace(outside_option_util=None)
    out_none = DSPO_plus_SPO._pricing_oracle_lifted_torch(obj, costs, cust_info, [{'util': 0.0}])
    # 锚点:oracle 在 V0=0 与 None 下价格一致
    assert torch.allclose(out0['prices'], out_none['prices'], atol=1e-6), \
        f"{out0['prices']} vs {out_none['prices']}"
    # V0=0 时概率含外部项:sum(P) < 1;V0=None 时 sum(P) == 1
    assert float(out0['probs'].sum()) < 0.999, f"probs must leave room for outside: {out0['probs']}"
    assert abs(float(out_none['probs'].sum()) - 1.0) < 1e-6
    # 方向:V0=2 → oracle 价格统一低于 V0=0
    obj.config = SimpleNamespace(outside_option_util=2.0)
    out2 = DSPO_plus_SPO._pricing_oracle_lifted_torch(obj, costs, cust_info, [{'util': 0.0}])
    assert bool((out0['prices'] - out2['prices'] > 0.05).all()), f"{out0['prices']} vs {out2['prices']}"

    # 训练与部署共用同一预测器:helper 的单位结构(mltplr 只乘秒级插入,CNN 差值原样)
    # 用 float64 tensor 避免 float32 精度(~1e-7)在 1e-9 容差下的误报
    import torch as _t
    got = DSPO_plus_SPO._predicted_option_cost(
        120.0, 0.01, 0.75, 600.0,
        _t.tensor(7.0, dtype=_t.float64), _t.tensor(0.0, dtype=_t.float64))
    assert abs(float(got) - (120.0*0.01 + 0.25*0.01*600.0 + 0.75*7.0)) < 1e-9  # 1.2+1.5+5.25=7.95
    print("OK")

if __name__ == "__main__":
    main()
