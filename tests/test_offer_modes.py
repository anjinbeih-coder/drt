import sys
from os import path
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
import numpy as np
import numpy.ma as ma
from Environments.OOH.customerchoice import customerchoicemodel
from Environments.OOH.containers import Location, ParcelPoint, Customer

def euclid_stub(a, b):
    return ((a.x-b.x)**2 + (a.y-b.y)**2) ** 0.5

def make_fixture():
    m = customerchoicemodel(base_util=0.0, dist_scaler=100.0, euclidean=euclid_stub,
                            dist_mat=[], n_cust=5, beta_k=0.02)
    home = Location(10.0, 10.0, 1, 0)
    cust = Customer(home, -0.25, 3.2, 120.0, 1)
    pps = np.array([ParcelPoint(Location(20.0, 20.0, 5, 0), 1000000, 5),
                    ParcelPoint(Location(30.0, 30.0, 6, 0), 0, 6)])  # 第二个无容量
    # 与 get_delivery_loc_pricing 相同的掩码形式:mask=1 表示提供
    masked = ma.masked_array(pps, mask=[1, 1])
    action = np.zeros(3)
    return m, cust, masked, action

def main():
    m, cust, masked, action = make_fixture()
    # home_only:恒选 home
    for _ in range(20):
        loc, accepted, idx, price = m.customerchoice_home_only(cust, action, masked)
        assert accepted is False and idx == -1 and price == 0
        assert loc is cust.home
    # ooh_only:恒选 OOH,且永不选无容量的点
    for _ in range(200):
        loc, accepted, idx, price = m.customerchoice_ooh_only(cust, action, masked)
        assert accepted is True and price == 0
        assert idx == 5, f"must never pick zero-capacity pp, got {idx}"

    # 全部无容量:必须显式报错,而不是静默选中无容量点
    pps_dead = np.array([ParcelPoint(Location(20.0, 20.0, 5, 0), 0, 5),
                         ParcelPoint(Location(30.0, 30.0, 6, 0), 0, 6)])
    masked_dead = ma.masked_array(pps_dead, mask=[1, 1])
    try:
        m.customerchoice_ooh_only(cust, action, masked_dead)
        assert False, "expected RuntimeError for all-zero-capacity choice set"
    except RuntimeError as e:
        assert 'zero remaining capacity' in str(e)

    # ooh_only + pricing=False 必须在构造时报错
    from Environments.OOH.Parcelpoint_py import Parcelpoint_py
    from Src.Utils.Utils import generate_demand_data
    coords = generate_demand_data(100)
    try:
        Parcelpoint_py(model='Baseline', max_steps_r=5, max_steps_p=0.5, pricing=False,
                       load_data=False, coords=coords, dist_matrix=[], n_parcelpoints=6,
                       adjacency=np.ones(6), service_times=np.ones(100*100),
                       offer_mode='ooh_only')
        assert False, "expected ValueError for ooh_only with pricing=False"
    except ValueError as e:
        assert 'ooh_only' in str(e)

    print("OK")

if __name__ == "__main__":
    main()
