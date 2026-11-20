"""Assemble loader-format instance files for one variant from the fetched matrix.

Writes, for data_seed 0 and 1 (identical content -- real coordinates do not
change with the seed; documented deviation from RC's per-seed coords):
  <variant>_471_<seed>_coords.txt / _dist_matrix.txt / _adjacency10.npy
Usage: python Experiments/yanjiao_build/assemble_instance.py --variant YJ_random
"""
import argparse
import csv
import sys
from os import makedirs, path

import numpy as np

sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), '..', '..'))
from Experiments.yanjiao_build.fetch_distances import (HOMES_CSV, default_depot,  # noqa: E402
                                                       read_lonlat)

ROOT = path.join(path.dirname(path.abspath(__file__)), '..', '..')
YJ = path.join(ROOT, 'Environments', 'OOH', 'yanjiao')
SPEED_M_S = 30000.0 / 3600.0  # 30 km/h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=list(HOMES_CSV))
    ap.add_argument('--depot-lon', type=float, default=None)
    ap.add_argument('--depot-lat', type=float, default=None)
    opt = ap.parse_args()

    stations = read_lonlat('stations_170.csv')
    homes = read_lonlat(HOMES_CSV[opt.variant])
    depot = ((opt.depot_lon, opt.depot_lat)
             if opt.depot_lon is not None else default_depot(stations))
    nodes = [depot] + homes + stations
    mat_m = np.load(path.join(YJ, 'matrix_{}.npy'.format(opt.variant)))
    assert mat_m.shape == (471, 471) and len(nodes) == 471

    sec = np.rint(mat_m / SPEED_M_S).astype(int)
    np.fill_diagonal(sec, 0)

    adj = np.zeros((301, 170))
    for i in range(1, 301):
        adj[i][np.argsort(sec[i][301:471], kind='stable')[:10]] = 1

    out_dir = path.join(ROOT, 'Environments', 'OOH', 'yanjiao_instance', opt.variant)
    makedirs(out_dir, exist_ok=True)
    for seed in (0, 1):
        base = path.join(out_dir, '{}_471_{}'.format(opt.variant, seed))
        with open(base + '_coords.txt', 'w') as f:
            f.write('NODE\tX\tY\n')
            for i, (lon, lat) in enumerate(nodes):
                f.write('{}\t{:.6f}\t{:.6f}\n'.format(i, lon, lat))
        with open(base + '_dist_matrix.txt', 'w') as f:
            f.write('EDGE_WEIGHTS\n')
            for row in sec:
                f.write('\t'.join(str(int(x)) for x in row) + '\n')
        np.save(base + '_adjacency10.npy', adj)
    print('assembled {} (2 seeds) -> {}'.format(opt.variant, out_dir))


if __name__ == '__main__':
    main()
