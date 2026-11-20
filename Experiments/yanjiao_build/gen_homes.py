"""Generate the two 300-point home variants inside the (shrunk) station-pool hull.

V1 YJ_random: uniform in the boundary.  V2 YJ_poi: cluster_weight-weighted centers
+ N(0, 300m) scatter, rejection-sampled into the same boundary.  seed=42 each.
Usage: python Experiments/yanjiao_build/gen_homes.py
"""
import csv
import sys
from os import path

import numpy as np

sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), '..', '..'))
from Experiments.yanjiao_build.geo import meters_per_degree, shrunk_hull_path  # noqa: E402

ROOT = path.join(path.dirname(path.abspath(__file__)), '..', '..')
YJ = path.join(ROOT, 'Environments', 'OOH', 'yanjiao')
N_HOMES = 300
SIGMA_M = 300.0
SHRINK_M = 200.0


def load_pool_points():
    pts = []
    with open(path.join(YJ, 'yanjiao_meeting_points_pool.csv'), encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if str(r['is_stopped_route']).strip().lower() != 'true':
                pts.append([float(r['lon']), float(r['lat'])])
    return np.array(pts)


def write_homes(fname, homes, source):
    with open(path.join(YJ, fname), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['home_id', 'lon', 'lat', 'source'])
        for i, (lon, lat) in enumerate(homes):
            w.writerow(['hm_{:04d}'.format(i), '{:.6f}'.format(lon), '{:.6f}'.format(lat), source])


def main():
    pool = load_pool_points()
    hull = shrunk_hull_path(pool, SHRINK_M)
    lo, hi = pool.min(axis=0), pool.max(axis=0)

    rng = np.random.RandomState(42)
    uni = []
    while len(uni) < N_HOMES:
        p = lo + rng.rand(2) * (hi - lo)
        if hull.contains_point(tuple(p)):
            uni.append(p)
    write_homes('homes_random_300.csv', uni, 'uniform')

    clusters = []
    with open(path.join(YJ, 'yanjiao_home_cluster_candidates.csv'), encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            clusters.append((float(r['lon']), float(r['lat']), float(r['cluster_weight'])))
    centers = np.array([[c[0], c[1]] for c in clusters])
    weights = np.array([c[2] for c in clusters])
    weights = weights / weights.sum()
    m_lon, m_lat = meters_per_degree(float(centers[:, 1].mean()))

    rng = np.random.RandomState(42)
    poi = []
    while len(poi) < N_HOMES:
        c = centers[rng.choice(len(centers), p=weights)]
        p = c + rng.randn(2) * np.array([SIGMA_M / m_lon, SIGMA_M / m_lat])
        if hull.contains_point(tuple(p)):
            poi.append(p)
    write_homes('homes_poi_300.csv', poi, 'poi_weighted')
    print('wrote homes_random_300.csv and homes_poi_300.csv')


if __name__ == '__main__':
    main()
