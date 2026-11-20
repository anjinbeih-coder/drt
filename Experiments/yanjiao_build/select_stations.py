"""Select 170 meeting-point stations from the raw pool (greedy max-min coverage).

Usage: python Experiments/yanjiao_build/select_stations.py
Deterministic: numpy argmax tie-break, start = station nearest to pool centroid.
"""
import csv
import sys
from os import path

import numpy as np

sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), '..', '..'))
from Experiments.yanjiao_build.geo import haversine_m  # noqa: E402

ROOT = path.join(path.dirname(path.abspath(__file__)), '..', '..')
POOL = path.join(ROOT, 'Environments', 'OOH', 'yanjiao', 'yanjiao_meeting_points_pool.csv')
OUT = path.join(ROOT, 'Environments', 'OOH', 'yanjiao', 'stations_170.csv')
K = 170


def greedy_maxmin(points, k, start_idx):
    """points: ndarray (N,2) lon/lat. Returns k indices, deterministic."""
    n = len(points)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_m(points[i][0], points[i][1], points[j][0], points[j][1])
            dist[i][j] = dist[j][i] = d
    selected = [int(start_idx)]
    min_d = dist[start_idx].copy()
    for _ in range(k - 1):
        min_d[selected] = -1.0
        nxt = int(np.argmax(min_d))
        selected.append(nxt)
        min_d = np.minimum(min_d, dist[nxt])
    return selected


def main():
    rows = []
    with open(POOL, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if str(r['is_stopped_route']).strip().lower() == 'true':
                continue
            rows.append(r)
    assert len(rows) >= K, 'pool too small after filtering: {}'.format(len(rows))
    pts = np.array([[float(r['lon']), float(r['lat'])] for r in rows])
    centroid = pts.mean(axis=0)
    d2c = [haversine_m(p[0], p[1], centroid[0], centroid[1]) for p in pts]
    sel = greedy_maxmin(pts, K, int(np.argmin(d2c)))
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['station_id', 'lon', 'lat', 'name'])
        for rank, i in enumerate(sel):
            w.writerow(['st_{:04d}'.format(rank), rows[i]['lon'], rows[i]['lat'], rows[i]['name']])
    print('selected {} stations -> {}'.format(K, OUT))


if __name__ == '__main__':
    main()
