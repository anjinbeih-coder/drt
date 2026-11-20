import sys
from os import path
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
import numpy as np
from Src.Algorithms.DSPO import DSPO
from Environments.OOH.containers import Location, ParcelPoint, Customer

def main():
    # 桩掉 CNN 与插入成本,验证 theta 混合的单位:mltplr 只乘秒级插入成本,
    # CNN 预测已是货币单位不再乘。a_hat[0]-a_hat[1] 中 Lambert 项相消,只剩成本差。
    obj = DSPO.__new__(DSPO)
    obj.initial_phase = False
    obj.load_data = False
    obj.init_theta = 0.75
    obj.cool_theta = 0.0  # theta 恒 0.75
    obj.cost_multiplier = 0.01
    obj.base_util = 0.0
    obj.revenue = 50.0
    obj.min_p, obj.max_p = -1e9, 1e9
    obj.get_feature_rep_infer = lambda fleet: None
    obj.get_prediction = lambda feat, home, pps: [0.0, 7.0, 3.0]  # [参考, home CNN, pp CNN](货币)
    obj.cheapestInsertionCosts = lambda loc, fleet: 600.0  # 秒
    obj.mnl = lambda cust, pp: 0.0

    home = Location(0.0, 0.0, 1, 0)
    cust = Customer(home, -0.25, 3.2, 120.0, 1)
    pp = ParcelPoint(Location(5.0, 5.0, 2, 0), 1000000, 2)
    state = [cust, {"fleet": []}, {"parcelpoints": np.array([pp])}, 0]

    a_hat = DSPO.get_action_pricing(obj, state, training=False)

    theta = 0.75
    home_costs = 120.0 * 0.01 + (1 - theta) * 0.01 * 600.0 + theta * (7.0 - 0.0)  # 1.2+1.5+5.25=7.95
    pp_costs = (1 - theta) * 0.01 * 600.0 + theta * (3.0 - 0.0)  # 1.5+2.25=3.75
    got = float(a_hat[0] - a_hat[1])
    want = home_costs - pp_costs  # 4.2
    assert abs(got - want) < 0.011, f"blend units broken: a_hat diff {got}, want {want}"
    print("OK")

if __name__ == "__main__":
    main()
