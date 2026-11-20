import sys
from os import path

import numpy as np

sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), '..'))
from Experiments.yanjiao_build.select_stations import greedy_maxmin
from Experiments.yanjiao_build.geo import haversine_m, shrunk_hull_path


def main():
    # 已知答案的玩具集:一维直线上 5 点,选 3 个,起点=0
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [10.0, 0.0]])
    sel = greedy_maxmin(pts, k=3, start_idx=0)
    assert sel[0] == 0 and sel[1] == 4, sel          # 最远点 10 先入选
    assert sel[2] == 2 or sel[2] == 3, sel           # 中段点次之(3 与已选集最小距离 3;2 为 2)
    assert sel[2] == 3, sel                          # maxmin: idx3 的 min-dist=3 > idx2 的 2

    # 确定性:两次调用同输出
    assert greedy_maxmin(pts, 3, 0) == greedy_maxmin(pts, 3, 0)

    # haversine 量级:燕郊经度 0.01 度 ~ 850m 上下
    d = haversine_m(116.80, 39.95, 116.81, 39.95)
    assert 700 < d < 1000, d

    # 收缩凸包:方形四角(边长~1km),收缩 200m 后中心在内、原角点在外
    sq = np.array([[116.80, 39.95], [116.8117, 39.95], [116.80, 39.959], [116.8117, 39.959]])
    hull = shrunk_hull_path(sq, 200.0)
    assert hull.contains_point((116.8058, 39.9545))
    assert not hull.contains_point((116.80, 39.95))

    print('OK test_yanjiao_station_select')


if __name__ == '__main__':
    main()
