import sys
from os import path
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
import numpy as np
from Src.Algorithms.DSPO import DSPO
from Environments.OOH.containers import Location, Vehicle, Fleet

def main():
    # 不跑 __init__,手工填 get_per_customer_costs 用到的属性
    obj = DSPO.__new__(DSPO)
    n = 5  # ids: 0=depot, 1-2=customers, 3-4=OOH
    obj.dist_matrix = np.full((n, n), 100.0)
    np.fill_diagonal(obj.dist_matrix, 0.0)
    obj.service_times = np.array([0.0, 300.0, 240.0, 120.0, 60.0])
    obj.cost_multiplier = 1.0  # 便于验证
    obj.first_parcelpoint_id = 3
    # 一辆车,路线:home客户1 -> OOH点3(被2个客户使用,route 里出现两次)
    stops = [Location(1.0,1.0,1,0), Location(2.0,2.0,3,1), Location(2.0,2.0,3,2)]
    fleet = Fleet([Vehicle(stops, 10, 0)])
    id_counts = {1: 1, 3: 2}
    costs = DSPO.get_per_customer_costs(obj, fleet, id_counts)
    assert len(costs) == 3
    # 半边成本:首停 0.5*d(0,1)+0.5*d(1,3)=100;中停 0.5*d(1,3)+0.5*d(3,3)=50(对角线0);
    # 末停 0.5*d(3,3)+0.5*d(3,0)=50
    # 服务时间:home客户1 加 300;OOH 3 每次出现摊 120/2=60
    expect = {0: 100.0 + 300.0, 1: 50.0 + 60.0, 2: 50.0 + 60.0}
    for t, c in costs:
        assert abs(c - expect[t]) < 1e-9, f"stop t={t}: got {c}, want {expect[t]}"

    # 子类 DSPO_plus_SPO 继承同一方法,新签名必须同样可用
    from Src.Algorithms.DSPO_plus_SPO import DSPO_plus_SPO
    obj2 = DSPO_plus_SPO.__new__(DSPO_plus_SPO)
    obj2.dist_matrix = obj.dist_matrix
    obj2.service_times = obj.service_times
    obj2.cost_multiplier = obj.cost_multiplier
    obj2.first_parcelpoint_id = obj.first_parcelpoint_id
    costs2 = DSPO_plus_SPO.get_per_customer_costs(obj2, fleet, id_counts)
    assert costs2 is not None and len(costs2) == 3

    print("OK")

if __name__ == "__main__":
    main()
