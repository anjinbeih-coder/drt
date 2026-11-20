import sys
from os import path
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
import numpy as np
import numpy.ma as ma
from Environments.OOH.customerchoice import customerchoicemodel
from Environments.OOH.containers import Location, ParcelPoint, Customer

def euclid_stub(a, b):
    return ((a.x-b.x)**2 + (a.y-b.y)**2) ** 0.5

def make_model(outside_util):
    return customerchoicemodel(base_util=0.0, dist_scaler=100.0, euclidean=euclid_stub,
                               dist_mat=[], n_cust=5, beta_k=0.02, outside_util=outside_util)

def make_choice_args():
    home = Location(10.0, 10.0, 1, 0)
    cust = Customer(home, -0.25, 3.2, 120.0, 1)
    pps = np.array([ParcelPoint(Location(20.0, 20.0, 5, 0), 1000000, 5)])
    masked = ma.masked_array(pps, mask=[1])
    action = np.zeros(2)
    return cust, masked, action

def main():
    cust, masked, action = make_choice_args()
    # V0=None:关闭,绝不返回退出哨兵
    m = make_model(None)
    for _ in range(200):
        _, _, idx, _ = m.customerchoice_pricing(cust, action, masked)
        assert idx != -2
    # V0 极大:必退出,返回约定 (None, False, -2, 0)
    m = make_model(50.0)
    for _ in range(50):
        loc, accepted, idx, price = m.customerchoice_pricing(cust, action, masked)
        assert loc is None and accepted is False and idx == -2 and price == 0
    # V0 极小:从不退出
    m = make_model(-50.0)
    for _ in range(200):
        _, _, idx, _ = m.customerchoice_pricing(cust, action, masked)
        assert idx != -2

    # env 层:退出步不产生停靠、不计价格统计,stats[8] 计数;守卫生效
    from Environments.OOH.Parcelpoint_py import Parcelpoint_py
    from Src.Utils.Utils import generate_demand_data
    coords = generate_demand_data(100)
    env = Parcelpoint_py(model='Baseline', max_steps_r=5, max_steps_p=0.5, pricing=True,
                         load_data=False, coords=coords, dist_matrix=[], n_parcelpoints=6,
                         adjacency=np.ones((100*100, 6)), service_times=np.ones(100*100),
                         outside_option_util=50.0)
    n_ids_before = np.size(env.data['id'])
    action = np.zeros(7)  # home + 6 个自提点
    _, _, stats, data = env.step(action)
    assert stats[8] == 1, "quit must be counted in stats[8]"
    assert np.size(data['id']) == n_ids_before, "quit must not add a stop"
    assert len(env.total_prices) == 0 and len(env.total_discounts) == 0, "quit must not pollute price stats"
    # 守卫:V0 + pricing=False → ValueError
    try:
        Parcelpoint_py(model='Baseline', max_steps_r=5, max_steps_p=0.5, pricing=False,
                       load_data=False, coords=coords, dist_matrix=[], n_parcelpoints=6,
                       adjacency=np.ones((100*100, 6)), service_times=np.ones(100*100),
                       outside_option_util=0.0)
        assert False, "expected ValueError (pricing=False)"
    except ValueError:
        pass
    # home_only / ooh_only + V0:乘客可退出(与 all 对齐)
    env_home = Parcelpoint_py(model='Baseline', max_steps_r=5, max_steps_p=0.5, pricing=True,
                              load_data=False, coords=coords, dist_matrix=[], n_parcelpoints=6,
                              adjacency=np.ones((100*100, 6)), service_times=np.ones(100*100),
                              offer_mode='home_only', outside_option_util=50.0)
    action_home = np.zeros(7)
    _, _, stats_h, data_h = env_home.step(action_home)
    assert stats_h[8] == 1, "home_only + large V0 must quit"
    assert np.size(data_h['id']) == n_ids_before, "home_only quit must not add a stop"
    env_ooh = Parcelpoint_py(model='Baseline', max_steps_r=5, max_steps_p=0.5, pricing=True,
                             load_data=False, coords=coords, dist_matrix=[], n_parcelpoints=6,
                             adjacency=np.ones((100*100, 6)), service_times=np.ones(100*100),
                             offer_mode='ooh_only', outside_option_util=50.0)
    _, _, stats_o, data_o = env_ooh.step(action_home)
    assert stats_o[8] == 1, "ooh_only + large V0 must quit"
    assert np.size(data_o['id']) == n_ids_before, "ooh_only quit must not add a stop"
    # choice-level: home_only / ooh_only with V0=None keep legacy no-quit behaviour
    m_home = make_model(None)
    for _ in range(50):
        loc, accepted, idx, price = m_home.customerchoice_home_only(cust, action, masked)
        assert loc is cust.home and idx == -1 and accepted is False
    m_home_v0 = make_model(50.0)
    for _ in range(30):
        loc, accepted, idx, price = m_home_v0.customerchoice_home_only(cust, action, masked)
        assert loc is None and idx == -2
    print("OK")

if __name__ == "__main__":
    main()
