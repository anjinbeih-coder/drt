import sys
from os import path
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
from math import exp
from Environments.OOH.customerchoice import customerchoicemodel

def euclid_stub(a, b):
    return abs(a - b)

def main():
    m = customerchoicemodel(base_util=0.0, dist_scaler=100.0, euclidean=euclid_stub,
                            dist_mat=[], n_cust=10, beta_k=0.02)
    # 论文式(5):u = -beta_k * exp(d/scaler),越远效用越低
    u_near = m.ooh_utility(10.0)
    u_far = m.ooh_utility(200.0)
    assert u_near > u_far, f"utility must decrease with distance: {u_near} vs {u_far}"
    assert abs(u_near - (0.0 - 0.02 * exp(10.0 / 100.0))) < 1e-12
    assert abs(u_far - (0.0 - 0.02 * exp(200.0 / 100.0))) < 1e-12
    # 超大距离不能溢出(校准会探索小 scaler)
    u_huge = m.ooh_utility(1e9)
    assert u_huge < -1e6 and u_huge == u_huge, "must be finite, very negative"
    # base_util 是可加常数
    m2 = customerchoicemodel(base_util=1.5, dist_scaler=100.0, euclidean=euclid_stub,
                             dist_mat=[], n_cust=10, beta_k=0.02)
    assert abs(m2.ooh_utility(10.0) - (u_near + 1.5)) < 1e-12
    # DSPO 的 MNL 与成本缩放必须解耦:mnl 用 mnl_dist_scaler,addedcosts 除以 1
    import numpy as np
    from Src.Algorithms.DSPO import DSPO
    from Environments.OOH.containers import Location, Customer
    obj = DSPO.__new__(DSPO)
    obj.base_util = 0.0
    obj.beta_k = 0.02
    obj.mnl_dist_scaler = 100.0
    obj.dist_scaler = 1
    obj.dist_matrix = np.array([[0.0, 10.0, 200.0], [10.0, 0.0, 50.0], [200.0, 50.0, 0.0]])
    cust = Customer(Location(0.0, 0.0, 0, 0), -0.25, 3.2, 120.0, 0)
    class PP:
        pass
    pp_near, pp_far = PP(), PP()
    pp_near.id_num, pp_far.id_num = 1, 2
    u_near = DSPO.mnl_distmat(obj, cust, pp_near)
    u_far = DSPO.mnl_distmat(obj, cust, pp_far)
    assert u_near > u_far, "DSPO mnl must decrease with distance"
    assert abs(u_near - (0.0 - 0.02 * exp(10.0 / 100.0))) < 1e-12
    # addedcosts_distmat 不受 mnl_dist_scaler 影响(除以 dist_scaler=1)
    route = [Location(0.0, 0.0, 0, 0), Location(0.0, 0.0, 2, 0)]
    loc = Location(0.0, 0.0, 1, 0)
    added = DSPO.addedcosts_distmat(obj, route, 1, loc)
    assert abs(added - (10.0 + 50.0 - 200.0)) < 1e-12, f"addedcosts must be raw (scaler 1), got {added}"
    print("OK")

if __name__ == "__main__":
    main()
