"""Fetch the full driving-distance matrix (meters) for one home variant.

Node order: [depot] + homes(300) + stations(170).  All pairs, chunked 100
origins per request per destination column.  Failed pairs are filled with
haversine * (median detour ratio of successful pairs) and counted.

Usage:
  set AMAP_KEY=...          (environment variable)
  python Experiments/yanjiao_build/fetch_distances.py --variant YJ_random
  python Experiments/yanjiao_build/fetch_distances.py --variant YJ_poi
Options: --depot-lon/--depot-lat override; --max-new-calls 8000; --dry-run
(prints projected new calls from cache state and exits).
"""
import argparse
import csv
import json
import os
import sys
from os import path

import numpy as np

sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), '..', '..'))
from Experiments.yanjiao_build.amap_client import AmapDistanceClient  # noqa: E402
from Experiments.yanjiao_build.geo import haversine_m  # noqa: E402

ROOT = path.join(path.dirname(path.abspath(__file__)), '..', '..')
YJ = path.join(ROOT, 'Environments', 'OOH', 'yanjiao')
CACHE = path.join(YJ, 'cache', 'amap_cache.sqlite')
HOMES_CSV = {'YJ_random': 'homes_random_300.csv', 'YJ_poi': 'homes_poi_300.csv'}


def read_lonlat(fname, lon_key='lon', lat_key='lat'):
    out = []
    with open(path.join(YJ, fname), encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            out.append((float(r[lon_key]), float(r[lat_key])))
    return out


def default_depot(stations):
    pts = np.array(stations)
    c = pts.mean(axis=0)
    d = [haversine_m(p[0], p[1], c[0], c[1]) for p in stations]
    return stations[int(np.argmin(d))]


def build_matrix(client, nodes):
    n = len(nodes)
    mat = np.zeros((n, n), dtype=np.float64)
    failed = []
    for j, dest in enumerate(nodes):
        col = []
        for s in range(0, n, 100):
            col.extend(client.batch_distance(nodes[s:s + 100], dest))
        for i, r in enumerate(col):
            if i == j:
                mat[i][j] = 0.0
            elif r is None:
                failed.append((i, j))
            else:
                mat[i][j] = r[0]
    # fill failures with haversine * median detour ratio of successes
    ratios = []
    for i in range(n):
        for j in range(n):
            if i != j and mat[i][j] > 0:
                h = haversine_m(nodes[i][0], nodes[i][1], nodes[j][0], nodes[j][1])
                if h > 0:
                    ratios.append(mat[i][j] / h)
    med = float(np.median(ratios)) if ratios else 1.4
    for i, j in failed:
        mat[i][j] = haversine_m(nodes[i][0], nodes[i][1], nodes[j][0], nodes[j][1]) * med
    # sanity: driving >= haversine
    ok = tot = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                tot += 1
                if mat[i][j] >= haversine_m(nodes[i][0], nodes[i][1],
                                            nodes[j][0], nodes[j][1]) - 1.0:
                    ok += 1
    report = {'n_nodes': n, 'api_calls': int(client.new_calls),
              'failed_pairs': len(failed),
              'fill_ratio': len(failed) / float(max(1, n * n - n)),
              'median_detour': med,
              'sanity_ge_haversine_pct': 100.0 * ok / max(1, tot)}
    return mat, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=list(HOMES_CSV))
    ap.add_argument('--depot-lon', type=float, default=None)
    ap.add_argument('--depot-lat', type=float, default=None)
    ap.add_argument('--max-new-calls', type=int, default=8000)
    ap.add_argument('--dry-run', action='store_true')
    opt = ap.parse_args()

    stations = read_lonlat('stations_170.csv')
    homes = read_lonlat(HOMES_CSV[opt.variant])
    depot = ((opt.depot_lon, opt.depot_lat)
             if opt.depot_lon is not None else default_depot(stations))
    nodes = [depot] + homes + stations
    assert len(nodes) == 471, len(nodes)

    key = os.environ.get('AMAP_KEY', '')
    assert key or opt.dry_run, 'AMAP_KEY environment variable not set'
    os.makedirs(path.dirname(CACHE), exist_ok=True)
    client = AmapDistanceClient(key, CACHE, qps=3.0, max_new_calls=opt.max_new_calls)

    n_missing = 0
    for j, dest in enumerate(nodes):
        for i, o in enumerate(nodes):
            if i != j and client.cached(o, dest) is None:
                n_missing += 1
    projected = int(np.ceil(n_missing / 100.0)) if n_missing else 0
    print('missing pairs: {}, projected new calls (upper bound): ~{}'.format(
        n_missing, int(np.ceil(n_missing / 100.0)) + 471))
    if opt.dry_run:
        return
    assert projected + 471 <= opt.max_new_calls, \
        'projected calls exceed budget; raise --max-new-calls only after checking quota'

    mat, report = build_matrix(client, nodes)
    np.save(path.join(YJ, 'matrix_{}.npy'.format(opt.variant)), mat)
    with open(path.join(YJ, 'matrix_{}_report.json'.format(opt.variant)), 'w') as f:
        json.dump(report, f, indent=1)
    print('variant {} -> matrix saved; report: {}'.format(opt.variant, report))
    assert report['sanity_ge_haversine_pct'] >= 95.0, 'sanity check failed'
    if report['fill_ratio'] > 0.01:
        print('WARNING: >1% pairs filled by approximation')


if __name__ == '__main__':
    main()
