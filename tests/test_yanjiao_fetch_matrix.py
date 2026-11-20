import sys
from os import path

import numpy as np

sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), '..'))
from Experiments.yanjiao_build.fetch_distances import build_matrix
from Experiments.yanjiao_build.geo import haversine_m


class FakeClient(object):
    """Driving distance = haversine * 1.4; one pair configured to fail."""
    def __init__(self, fail_pair=None):
        self.fail_pair = fail_pair
        self.new_calls = 0

    def batch_distance(self, origins, dest):
        self.new_calls += 1
        out = []
        for o in origins:
            if self.fail_pair and (tuple(o), tuple(dest)) == self.fail_pair:
                out.append(None)
            else:
                d = haversine_m(o[0], o[1], dest[0], dest[1]) * 1.4
                out.append((d, d / 30000.0 * 3600.0))
        return out


def main():
    nodes = [(116.80 + 0.005 * i, 39.95 + 0.002 * (i % 3)) for i in range(6)]
    fail = (tuple(nodes[2]), tuple(nodes[4]))
    cli = FakeClient(fail_pair=fail)
    mat, rep = build_matrix(cli, nodes)

    assert mat.shape == (6, 6)
    assert np.all(np.diag(mat) == 0.0)
    assert np.all(mat >= 0)
    # 失败对被 haversine*中位绕行系数填补(系数=1.4)
    exp = haversine_m(nodes[2][0], nodes[2][1], nodes[4][0], nodes[4][1]) * 1.4
    assert abs(mat[2][4] - exp) / exp < 0.05, (mat[2][4], exp)
    assert rep['failed_pairs'] == 1
    assert rep['sanity_ge_haversine_pct'] > 95.0
    assert rep['n_nodes'] == 6

    print('OK test_yanjiao_fetch_matrix')


if __name__ == '__main__':
    main()
