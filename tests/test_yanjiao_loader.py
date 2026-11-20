import shutil
import sys
from os import makedirs, path

import numpy as np

sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), '..'))
from Src.Utils.Utils import load_demand_data

ROOT = path.join(path.dirname(path.abspath(__file__)), '..')
INST_DIR = path.join(ROOT, 'Environments', 'OOH', 'yanjiao_instance', 'YJ_test')


def make_fixture():
    makedirs(INST_DIR, exist_ok=True)
    n = 471
    rng = np.random.RandomState(0)
    lon = 116.80 + rng.rand(n) * 0.05
    lat = 39.93 + rng.rand(n) * 0.04
    base = path.join(INST_DIR, 'YJ_test_471_0')
    with open(base + '_coords.txt', 'w') as f:
        f.write('NODE\tX\tY\n')
        for i in range(n):
            f.write('{}\t{:.6f}\t{:.6f}\n'.format(i, lon[i], lat[i]))
    m = rng.randint(60, 3600, size=(n, n))
    m = ((m + m.T) // 2).astype(int)
    np.fill_diagonal(m, 0)
    with open(base + '_dist_matrix.txt', 'w') as f:
        f.write('EDGE_WEIGHTS\n')
        for row in m:
            f.write('\t'.join(str(int(x)) for x in row) + '\n')
    adj = np.zeros((301, 170))
    for i in range(1, 301):
        adj[i][np.argsort(m[i][301:])[:10]] = 1
    np.save(base + '_adjacency10.npy', adj)
    return m, adj


def main():
    m, adj = make_fixture()
    coords, dm, n_pp, adjacency, st = load_demand_data(ROOT, 'YJ_test', 0, 10.0, 30.0)
    assert len(coords) == 471
    assert n_pp == 170
    assert dm.shape == (471, 471) and np.all(np.diag(dm) == 0) and np.all(dm >= 0)
    assert np.array_equal(dm, m)
    assert adjacency.shape == (301, 170)
    assert np.array_equal(adjacency, adj)
    assert all(int(r.sum()) == 10 for r in adjacency[1:301])
    assert len(st) == 471 and np.all(st >= 60)          # service_times in seconds, >=1min
    assert abs(coords[0].x - 116.80) < 0.06             # lon/lat passthrough
    shutil.rmtree(INST_DIR)
    print('OK test_yanjiao_loader')


if __name__ == '__main__':
    main()
