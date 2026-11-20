# 燕郊实例数据管线与接入 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用高德驾车距离构造燕郊–国贸 DRT 实例(两个 home 变体),接入现有 load_demand_data,并完成 MNL 校准与冒烟。

**Architecture:** 管线四段(站点贪心选择 → home 双变体生成 → 高德批量采集[缓存/限速/断点] → 实例装配),产出 Amazon_data 同格式的静态实例文件;loader 加 `YJ_` 分支;校准脚本对数二分 `mnl_dist_scaler`。实验阶段只读文件、零 API 调用。

**Tech Stack:** Python 3.9(conda py39),numpy/scipy/matplotlib(均已装),sqlite3+urllib(标准库,高德客户端不引第三方依赖),测试为裸 assert 脚本。

## Global Constraints

- 分支 `yanjiao-instance`;解释器 `C:\Users\wwpst\miniforge3\envs\py39\python.exe`;测试单独运行打印 `OK <名字>`。
- 手稿 §5.3 参数(CLI 传,零算法改动):`n_vehicles=25, veh_capacity=12, max_steps_r=240, max_steps_p=0.5, revenue=30, fuel_cost=0.6, min_price=-10.0, max_price=10.0, outside_option_util=-1.0, home_util=1.4, incentive_sens=-0.25, home_failure=0.0, truck_speed=30`。
- 节点约定(已核实 Parcelpoint_py.py:59-66):**index 0=depot,1..300=homes,471−170..470=stations(coords 尾部 170 个)**;env 抽客户 `randint(1, num_cust_loc)`,num_cust_loc=471−170−1=300 → 实际可抽 index 1..299(继承的 off-by-one,文档注明,不改)。
- 邻接语义(已核实 DSPO.get_action_pricing + RC 的 np.ones):**adjacency 值 1=该站对该客户可报价**;形状 `(301, 170)`(行=id_num,0..300,行 0=depot 未用),每个 home 行恰好 10 个 1(最近 10 站,按驾车秒)。
- 距离单位:高德返回**米**;矩阵落盘为**秒**=round(米/30000×3600)(30 km/h,与手稿一致);对角 0;HGS/成本全程用矩阵(已核实 Baseline.py:77/DSPO.py:257/Parcelpoint_py.py:200)。
- 文件命名(loader 约定):`Environments/OOH/yanjiao_instance/<INSTANCE>/<INSTANCE>_471_<seed>_coords.txt|_dist_matrix.txt|_adjacency10.npy`,seed∈{0,1} 内容相同;coords 首行以 `NODE` 开头,数据行 `id\tlon\tlat`;dist 首行以 `EDGE` 开头,数据行为制表符分隔整数。
- INSTANCE 名:`YJ_random`、`YJ_poi`。
- API:环境变量 `AMAP_KEY`;距离测量 API `https://restapi.amap.com/v3/distance`,type=1,≤100 起点/次;3 QPS;新调用预算硬顶 8,000 次;配额类 infocode 触发即停;所有应答持久缓存(sqlite,gitignore)。
- 数据/缓存目录 gitignore:`Environments/OOH/yanjiao_instance/`、`Environments/OOH/yanjiao/cache/`。
- 输入 CSV 有 UTF-8 BOM,读取用 `encoding='utf-8-sig'`。
- 固定种子:站点贪心 tie-break 与 home 采样均 seed=42。

---

### Task 1: 高德客户端(缓存 + 限速 + 配额守卫)

**Files:**
- Create: `Experiments/yanjiao_build/__init__.py`(空文件)
- Create: `Experiments/yanjiao_build/amap_client.py`
- Test: `tests/test_yanjiao_amap_cache.py`

**Interfaces:**
- Produces: `AmapDistanceClient(key, cache_path, qps=3.0, max_new_calls=8000, http_get=None)`;
  `client.batch_distance(origins: list[(lon,lat)], dest: (lon,lat)) -> list[(dist_m, dur_s)|None]`(≤100 起点,缓存优先,一次最多 1 个 API 调用);
  `client.new_calls: int`(本进程新发起的调用数);`QuotaExceeded(RuntimeError)`。
  `http_get` 参数接受 `f(url)->dict` 以便测试桩替换。

- [ ] **Step 1: 写失败测试**

`tests/test_yanjiao_amap_cache.py`:

```python
import os
import sys
import time
from os import path

sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), '..'))
from Experiments.yanjiao_build.amap_client import AmapDistanceClient, QuotaExceeded


def make_fake(counter, fail_infocode=None):
    def fake_http_get(url):
        counter['n'] += 1
        if fail_infocode:
            return {'status': '0', 'infocode': fail_infocode, 'results': []}
        import urllib.parse
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        origins = q['origins'][0].split('|')
        results = [{'origin_id': str(i + 1), 'dest_id': '1',
                    'dist': str(1000.0 + 10 * i), 'duration': str(120.0 + i)}
                   for i in range(len(origins))]
        return {'status': '1', 'infocode': '10000', 'results': results}
    return fake_http_get


def main():
    fp = path.join(path.dirname(path.abspath(__file__)), '_tmp_amap_cache.sqlite')
    if path.isfile(fp):
        os.remove(fp)
    origins = [(116.80 + 0.001 * i, 39.95) for i in range(150)]
    dest = (116.90, 39.90)

    # 1) 150 个起点 -> 2 次调用(100+50),返回全非 None
    c1 = {'n': 0}
    cli = AmapDistanceClient('k', fp, qps=1000.0, http_get=make_fake(c1))
    out = []
    for chunk in (origins[:100], origins[100:]):
        out.extend(cli.batch_distance(chunk, dest))
    assert c1['n'] == 2, c1
    assert all(r is not None for r in out)
    assert out[0][0] == 1000.0 and out[1][0] == 1010.0

    # 2) 断点续采:新客户端同缓存,零新调用
    c2 = {'n': 0}
    cli2 = AmapDistanceClient('k', fp, qps=1000.0, http_get=make_fake(c2))
    out2 = cli2.batch_distance(origins[:100], dest)
    assert c2['n'] == 0 and all(r is not None for r in out2)
    assert cli2.new_calls == 0

    # 3) 限速:qps=5,3 次未缓存调用耗时 >= 0.4s
    c3 = {'n': 0}
    cli3 = AmapDistanceClient('k', fp, qps=5.0, http_get=make_fake(c3))
    t0 = time.time()
    for j in range(3):
        cli3.batch_distance([(120.0 + j, 30.0)], (121.0, 31.0))
    assert time.time() - t0 >= 0.4, 'throttle not enforced'

    # 4) 配额 infocode 即刻抛 QuotaExceeded
    c4 = {'n': 0}
    cli4 = AmapDistanceClient('k', fp, qps=1000.0, http_get=make_fake(c4, fail_infocode='10003'))
    try:
        cli4.batch_distance([(122.0, 30.0)], (121.0, 31.0))
        raise AssertionError('QuotaExceeded not raised')
    except QuotaExceeded:
        pass

    # 5) 新调用预算硬顶
    c5 = {'n': 0}
    cli5 = AmapDistanceClient('k', fp, qps=1000.0, max_new_calls=1, http_get=make_fake(c5))
    cli5.batch_distance([(123.0, 30.0)], (121.0, 31.0))
    try:
        cli5.batch_distance([(124.0, 30.0)], (121.0, 31.0))
        raise AssertionError('max_new_calls not enforced')
    except QuotaExceeded:
        pass

    os.remove(fp)
    print('OK test_yanjiao_amap_cache')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `"C:\Users\wwpst\miniforge3\envs\py39\python.exe" tests/test_yanjiao_amap_cache.py`(工作目录仓库根)
Expected: `ModuleNotFoundError: No module named 'Experiments.yanjiao_build'`

- [ ] **Step 3: 实现 amap_client.py**

```python
"""AMap driving-distance client: sqlite cache, 3 QPS throttle, quota guards.

Every (origin,dest) pair result is cached persistently; reruns cost zero calls.
The distance-measurement API (v3/distance, type=1) accepts up to 100 origins
per request and bills per REQUEST (probed at execution time before bulk use).
"""
import json
import sqlite3
import time
import urllib.parse
import urllib.request

AMAP_DISTANCE_URL = 'https://restapi.amap.com/v3/distance'
QUOTA_INFOCODES = {'10003', '10009', '10044'}  # quota exhausted family


class QuotaExceeded(RuntimeError):
    pass


class AmapDistanceClient(object):
    def __init__(self, key, cache_path, qps=3.0, max_new_calls=8000, http_get=None):
        self.key = key
        self.qps = float(qps)
        self.max_new_calls = int(max_new_calls)
        self.new_calls = 0
        self._last_ts = 0.0
        self._http_get = http_get or self._default_http_get
        self.conn = sqlite3.connect(cache_path)
        self.conn.execute('CREATE TABLE IF NOT EXISTS dist_cache ('
                          'okey TEXT, dkey TEXT, dist_m REAL, dur_s REAL, ts REAL, '
                          'PRIMARY KEY (okey, dkey))')
        self.conn.commit()

    @staticmethod
    def _k(lonlat):
        return '{:.6f},{:.6f}'.format(float(lonlat[0]), float(lonlat[1]))

    @staticmethod
    def _default_http_get(url):
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read().decode('utf-8'))

    def _throttle(self):
        wait = self._last_ts + 1.0 / self.qps - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last_ts = time.time()

    def cached(self, origin, dest):
        return self.conn.execute(
            'SELECT dist_m, dur_s FROM dist_cache WHERE okey=? AND dkey=?',
            (self._k(origin), self._k(dest))).fetchone()

    def batch_distance(self, origins, dest):
        """origins (<=100 lon-lat pairs) -> list of (dist_m, dur_s) or None each.

        Cache-first: only cache-missing origins go to the API (single request).
        Raises QuotaExceeded on quota infocodes or when max_new_calls is hit.
        """
        assert len(origins) <= 100, 'at most 100 origins per request'
        out = [self.cached(o, dest) for o in origins]
        missing = [i for i, r in enumerate(out) if r is None]
        if not missing:
            return out
        if self.new_calls >= self.max_new_calls:
            raise QuotaExceeded('new-call budget exhausted ({})'.format(self.max_new_calls))
        params = {'key': self.key, 'type': '1',
                  'origins': '|'.join(self._k(origins[i]) for i in missing),
                  'destination': self._k(dest)}
        url = AMAP_DISTANCE_URL + '?' + urllib.parse.urlencode(params)
        resp = None
        for attempt in range(5):
            self._throttle()
            try:
                resp = self._http_get(url)
            except Exception:
                resp = None
            if resp is not None and resp.get('status') == '1':
                break
            if resp is not None and resp.get('infocode') in QUOTA_INFOCODES:
                raise QuotaExceeded('AMap quota infocode ' + str(resp.get('infocode')))
            time.sleep(min(2 ** attempt, 20))
        self.new_calls += 1
        got = {}
        if resp is not None and resp.get('status') == '1':
            for res in resp.get('results', []):
                oid = int(res.get('origin_id', 0)) - 1  # 1-based within this request
                if 0 <= oid < len(missing) and res.get('dist') not in (None, ''):
                    got[missing[oid]] = (float(res['dist']), float(res.get('duration', 0.0)))
        for i in missing:
            if i in got:
                self.conn.execute('INSERT OR REPLACE INTO dist_cache VALUES (?,?,?,?,?)',
                                  (self._k(origins[i]), self._k(dest),
                                   got[i][0], got[i][1], time.time()))
                out[i] = got[i]
        self.conn.commit()
        return out
```

同时创建空的 `Experiments/yanjiao_build/__init__.py`。

- [ ] **Step 4: 跑测试确认通过**

Run: `"C:\Users\wwpst\miniforge3\envs\py39\python.exe" tests/test_yanjiao_amap_cache.py`
Expected: `OK test_yanjiao_amap_cache`

- [ ] **Step 5: Commit**

```bash
git add Experiments/yanjiao_build/__init__.py Experiments/yanjiao_build/amap_client.py tests/test_yanjiao_amap_cache.py
git commit -m "feat: 高德距离客户端(sqlite 缓存/3QPS 限速/配额守卫)"
```

---

### Task 2: 站点选择与 home 双变体生成

**Files:**
- Create: `Experiments/yanjiao_build/geo.py`
- Create: `Experiments/yanjiao_build/select_stations.py`
- Create: `Experiments/yanjiao_build/gen_homes.py`
- Test: `tests/test_yanjiao_station_select.py`

**Interfaces:**
- Consumes: 输入 CSV `Environments/OOH/yanjiao/yanjiao_meeting_points_pool.csv`(列 meeting_id,raw_id,name,lon,lat,address,adname,business_area,is_stopped_route)、`yanjiao_home_cluster_candidates.csv`(列 cluster_id,raw_id,name,search_keyword,type,lon,lat,address,cluster_weight)。
- Produces:
  - `geo.haversine_m(lon1, lat1, lon2, lat2) -> float`(米);
  - `geo.shrunk_hull_path(points_lonlat: np.ndarray, shrink_m: float) -> matplotlib.path.Path`(凸包向质心收缩 shrink_m 米);
  - `select_stations.greedy_maxmin(points: np.ndarray[N,2], k: int, start_idx: int) -> list[int]`(确定性);
  - 产物 CSV:`Environments/OOH/yanjiao/stations_170.csv`(station_id,lon,lat,name)、`homes_random_300.csv`、`homes_poi_300.csv`(home_id,lon,lat,source)。

- [ ] **Step 1: 写失败测试**

`tests/test_yanjiao_station_select.py`:

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `"C:\Users\wwpst\miniforge3\envs\py39\python.exe" tests/test_yanjiao_station_select.py`
Expected: `ModuleNotFoundError`(select_stations 不存在)

- [ ] **Step 3: 实现 geo.py**

```python
"""Shared geometry helpers for the Yanjiao pipeline (lon/lat in degrees)."""
import numpy as np
from math import asin, cos, radians, sin, sqrt
from matplotlib.path import Path
from scipy.spatial import ConvexHull

EARTH_R = 6371000.0


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, (lon1, lat1, lon2, lat2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * asin(sqrt(a))


def meters_per_degree(lat_deg):
    """(m/deg_lon, m/deg_lat) at the given latitude."""
    return 111320.0 * cos(radians(lat_deg)), 110540.0


def shrunk_hull_path(points_lonlat, shrink_m):
    """Convex hull of lon/lat points, shrunk toward the centroid by shrink_m meters."""
    pts = np.asarray(points_lonlat, dtype=float)
    hull = ConvexHull(pts)
    verts = pts[hull.vertices]
    centroid = verts.mean(axis=0)
    m_lon, m_lat = meters_per_degree(centroid[1])
    out = []
    for v in verts:
        d_m = sqrt(((v[0] - centroid[0]) * m_lon) ** 2 + ((v[1] - centroid[1]) * m_lat) ** 2)
        f = max(0.0, 1.0 - shrink_m / d_m) if d_m > 0 else 0.0
        out.append(centroid + (v - centroid) * f)
    return Path(np.asarray(out))
```

- [ ] **Step 4: 实现 select_stations.py**

```python
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
```

- [ ] **Step 5: 实现 gen_homes.py**

```python
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
```

- [ ] **Step 6: 跑测试 + 真实运行两脚本**

Run: `"C:\Users\wwpst\miniforge3\envs\py39\python.exe" tests/test_yanjiao_station_select.py`
Expected: `OK test_yanjiao_station_select`

Run: `"C:\Users\wwpst\miniforge3\envs\py39\python.exe" Experiments/yanjiao_build/select_stations.py`
Expected: `selected 170 stations -> ...stations_170.csv`(383 池去停运后约 340+,足够)

Run: `"C:\Users\wwpst\miniforge3\envs\py39\python.exe" Experiments/yanjiao_build/gen_homes.py`
Expected: `wrote homes_random_300.csv and homes_poi_300.csv`

- [ ] **Step 7: 把三个产物 CSV 加入 git(小文件,构建输入的一部分)并提交**

```bash
git add Experiments/yanjiao_build/geo.py Experiments/yanjiao_build/select_stations.py Experiments/yanjiao_build/gen_homes.py tests/test_yanjiao_station_select.py "Environments/OOH/yanjiao/stations_170.csv" "Environments/OOH/yanjiao/homes_random_300.csv" "Environments/OOH/yanjiao/homes_poi_300.csv"
git commit -m "feat: 站点贪心选择(170) + home 双变体生成(各300, seed=42)"
```

---

### Task 3: 距离矩阵采集与合理性检查

**Files:**
- Create: `Experiments/yanjiao_build/fetch_distances.py`
- Test: `tests/test_yanjiao_fetch_matrix.py`

**Interfaces:**
- Consumes: Task 1 `AmapDistanceClient`;Task 2 产物 CSV;`geo.haversine_m`。
- Produces: `fetch_distances.build_matrix(client, nodes: list[(lon,lat)]) -> (mat_m: ndarray[N,N], report: dict)`
  (report 键:`n_nodes, api_calls, cache_hits_pairs, failed_pairs, fill_ratio, sanity_ge_haversine_pct`);
  CLI 产物:`Environments/OOH/yanjiao/matrix_YJ_random.npy`、`matrix_YJ_poi.npy`(米,float64)+ 对应 `matrix_*_report.json`;
  节点顺序=**[depot] + homes(300) + stations(170)**,depot 在 CLI 由 `--depot-lon/--depot-lat` 传入,默认取 stations_170.csv 质心最近站(与 spec C1 一致)。

- [ ] **Step 1: 写失败测试**

`tests/test_yanjiao_fetch_matrix.py`:

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `"C:\Users\wwpst\miniforge3\envs\py39\python.exe" tests/test_yanjiao_fetch_matrix.py`
Expected: `ModuleNotFoundError`(fetch_distances 不存在)

- [ ] **Step 3: 实现 fetch_distances.py**

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `"C:\Users\wwpst\miniforge3\envs\py39\python.exe" tests/test_yanjiao_fetch_matrix.py`
Expected: `OK test_yanjiao_fetch_matrix`(真实 API 采集在 Task 5 执行,本任务不触网)

- [ ] **Step 5: Commit**

```bash
git add Experiments/yanjiao_build/fetch_distances.py tests/test_yanjiao_fetch_matrix.py
git commit -m "feat: 全矩阵采集(分块/失败填补/合理性检查/干跑预算)"
```

---

### Task 4: 实例装配 + loader YJ_ 分支

**Files:**
- Create: `Experiments/yanjiao_build/assemble_instance.py`
- Modify: `Src/Utils/Utils.py:292-340`(load_demand_data 分支)
- Modify: `.gitignore`(追加两行)
- Test: `tests/test_yanjiao_loader.py`

**Interfaces:**
- Consumes: Task 3 的 `matrix_<variant>.npy` + Task 2 的 CSV。
- Produces: 实例文件三件套 × 2 seeds × 2 变体,目录 `Environments/OOH/yanjiao_instance/<variant>/`;
  loader:`load_demand_data(root, 'YJ_random'|'YJ_poi', data_seed, clip, speed)` 返回
  `(coords[471], dist_matrix[471,471]秒, n_parcelpoints=170, adjacency[301,170], service_times[471])`。

- [ ] **Step 1: 写失败测试**

`tests/test_yanjiao_loader.py`(自造 471 节点微型 fixture,不依赖真实采集):

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `"C:\Users\wwpst\miniforge3\envs\py39\python.exe" tests/test_yanjiao_loader.py`
Expected: 失败——现行 load_demand_data 把 YJ_test 落入 Homberger 分支(路径不存在 → ValueError,或 n_parcelpoints=10)

- [ ] **Step 3: 修改 load_demand_data(Src/Utils/Utils.py)**

三处最小修改:

(a) 目录分支(:292-297)改为:

```python
    if instance=='Austin' or instance=='Seattle':
        instance_folder = 'Amazon_data'
        instance_size = '_700_'
    elif instance.startswith('YJ_'):
        instance_folder = 'yanjiao_instance'
        instance_size = '_471_'
    else:
        instance_folder = 'HombergerGehring_data'
        instance_size = '_90_'
```

(b) 矩阵读取分支(:312)改为:

```python
        if instance_folder=='Amazon_data' or instance_folder=='yanjiao_instance':
```

(c) n_parcelpoints/adjacency 块(:332-340)改为 elif 链(注:原代码 Austin 分支会被后面的 else 覆盖为 10,是既有 bug,改成 elif 顺带修正,RC 行为不变):

```python
    if instance=='Austin':
        n_parcelpoints=278
        adjacency = np.load(pathh+"_adjacency20.npy")
    elif instance=='Seattle':
        n_parcelpoints=299
        adjacency = np.load(pathh+"_adjacency20.npy")
    elif instance.startswith('YJ_'):
        n_parcelpoints=170
        adjacency = np.load(pathh+"_adjacency10.npy")
    else:
        n_parcelpoints=10
        adjacency = np.ones(shape=(100-n_parcelpoints,n_parcelpoints))
```

- [ ] **Step 4: 跑 loader 测试 + 全量回归**

Run: `"C:\Users\wwpst\miniforge3\envs\py39\python.exe" tests/test_yanjiao_loader.py`
Expected: `OK test_yanjiao_loader`

逐个运行 tests/ 下全部既有测试(现在共 15 个含本文件),全部 OK(RC 路径不得受影响)。

- [ ] **Step 5: 实现 assemble_instance.py**

```python
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
```

- [ ] **Step 6: .gitignore 追加**

在 `.gitignore` 末尾追加两行:

```
Environments/OOH/yanjiao_instance/
Environments/OOH/yanjiao/cache/
```

- [ ] **Step 7: Commit**

```bash
git add Experiments/yanjiao_build/assemble_instance.py Src/Utils/Utils.py tests/test_yanjiao_loader.py .gitignore
git commit -m "feat: 实例装配 + load_demand_data 增加 YJ_ 分支(附带修正 Austin elif)"
```

---

### Task 5: 校准脚本 + 真实采集与验收(操作性)

**Files:**
- Create: `Experiments/run_req4_calibrate.py`
- 产物: `Environments/OOH/yanjiao/build_report.md`(入 git)、`docs/req4_yanjiao_calibration.md`(入 git)

**Interfaces:**
- Consumes: Tasks 1–4 全部;`run_paper_repro.run_one`。
- Produces: 两变体的校准 scaler + NoPricing/Static 冒烟表。

- [ ] **Step 1: 写校准脚本**

```python
"""Requirement-4 calibration: log-space bisection of mnl_dist_scaler on the
Yanjiao instance, targeting NoPricing home% = 98 +- 3 (manuscript 5.3 anchor:
without incentives nearly everyone picks door-to-door, which maximises the
steering room for pricing).  Then a StaticPricing (+2/-5) smoke at the chosen
scaler.  eval-only (0 training episodes).

Usage: python Experiments/run_req4_calibrate.py --instance YJ_random
Options: --lo 1e3 --hi 1e6 --iters 7 --seeds 1234 --eval_episodes 10
"""
import argparse
import sys
from os import path

sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), '..'))
from Experiments.run_paper_repro import run_one  # noqa: E402

YJ_EXTRA_BASE = {
    'load_data': 'True', 'n_vehicles': '25', 'veh_capacity': '12',
    'max_steps_r': '240', 'max_steps_p': '0.5', 'revenue': '30',
    'fuel_cost': '0.6', 'truck_speed': '30', 'min_price': '-10.0',
    'max_price': '10.0', 'outside_option_util': '-1.0', 'home_util': '1.4',
    'incentive_sens': '-0.25', 'home_failure': '0.0',
}
TARGET, TOL = 0.98, 0.03


def yj_extra(instance, folder_tag):
    extra = dict(YJ_EXTRA_BASE)
    extra['instance'] = instance
    extra['folder_suffix'] = 'req4cal_{}_{}'.format(instance, folder_tag)
    return extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance', required=True, choices=['YJ_random', 'YJ_poi'])
    ap.add_argument('--lo', type=float, default=1e3)
    ap.add_argument('--hi', type=float, default=1e6)
    ap.add_argument('--iters', type=int, default=7)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1234])
    ap.add_argument('--eval_episodes', type=int, default=10)
    opt = ap.parse_args()

    lo, hi, best = opt.lo, opt.hi, None
    for it in range(opt.iters):
        mid = (lo * hi) ** 0.5
        m = run_one('NoPricing', opt.seeds[0], mid, 0, opt.eval_episodes, 10000,
                    extra=yj_extra(opt.instance, 'bis{}'.format(it)))
        print('iter {}: scaler={:.1f} home%={:.2f} quit%={:.2f}'.format(
            it, mid, 100 * m['pct_home'], 100 * m['quit_rate']), flush=True)
        best = (mid, m)
        if abs(m['pct_home'] - TARGET) <= TOL:
            break
        # higher scaler weakens the walk-distance penalty -> OOH more attractive -> less home
        if m['pct_home'] > TARGET:
            lo = mid
        else:
            hi = mid

    scaler = best[0]
    print('chosen scaler = {:.1f}'.format(scaler))
    for seed in opt.seeds:
        mn = run_one('NoPricing', seed, scaler, 0, opt.eval_episodes, 10000,
                     extra=yj_extra(opt.instance, 'np_{}'.format(seed)))
        ms = run_one('StaticPricing', seed, scaler, 0, opt.eval_episodes, 10000,
                     extra=yj_extra(opt.instance, 'sp_{}'.format(seed)))
        print('SMOKE {} seed {}: NoPricing home%={:.2f} quit%={:.2f} net={:.2f} | '
              'Static home%={:.2f} quit%={:.2f} net={:.2f}'.format(
                  opt.instance, seed, 100 * mn['pct_home'], 100 * mn['quit_rate'],
                  mn['net_profit'], 100 * ms['pct_home'], 100 * ms['quit_rate'],
                  ms['net_profit']), flush=True)


if __name__ == '__main__':
    main()
```

注:方向判断与 RC 校准相反侧写法一致——scaler 越大,步行距离惩罚越弱,集合点越有吸引力,home% 越低。
若冒烟显示 HGS 1.1s/1.5s 时限对 240 节点路线质量不足(路线明显交叉/成本异常),如实记录到校准文档,连同
`--hgs_reopt_time/--hgs_final_time` 的上调建议留给 B 阶段决策,本阶段不改默认值。

- [ ] **Step 2: 提交校准脚本**

```bash
git add Experiments/run_req4_calibrate.py
git commit -m "feat: 需求4 燕郊校准脚本(NoPricing home%=98±3 对数二分 + Static 冒烟)"
```

- [ ] **Step 3: 计费口径试探(硬门槛,需用户参与)**

```bash
set AMAP_KEY=<用户提供> 
"C:\Users\wwpst\miniforge3\envs\py39\python.exe" Experiments/yanjiao_build/fetch_distances.py --variant YJ_random --max-new-calls 3
```
(预算 3 次调用后被 QuotaExceeded 停下,属预期。)
**暂停,请用户到高德控制台确认用量增加了 ~3 次而不是 ~300 次**;确认按请求计费后再继续。若按起点计费:停止,回报控制器走降级方案(混合拟合,另行确认)。

- [ ] **Step 4: 全量采集(两变体,约 4,400 次调用 / 30–45 分钟)**

```bash
"C:\Users\wwpst\miniforge3\envs\py39\python.exe" Experiments/yanjiao_build/fetch_distances.py --variant YJ_random
"C:\Users\wwpst\miniforge3\envs\py39\python.exe" Experiments/yanjiao_build/fetch_distances.py --variant YJ_poi
```
Expected: 两个 matrix_*.npy + report json;`sanity_ge_haversine_pct >= 95`;fill_ratio ≤ 0.01;
第二个变体因站–站块缓存命中,调用数明显少于第一个。

- [ ] **Step 5: 装配 + 校准 + 冒烟**

```bash
"C:\Users\wwpst\miniforge3\envs\py39\python.exe" Experiments/yanjiao_build/assemble_instance.py --variant YJ_random
"C:\Users\wwpst\miniforge3\envs\py39\python.exe" Experiments/yanjiao_build/assemble_instance.py --variant YJ_poi
"C:\Users\wwpst\miniforge3\envs\py39\python.exe" Experiments/run_req4_calibrate.py --instance YJ_random --seeds 1234 2345 3456
"C:\Users\wwpst\miniforge3\envs\py39\python.exe" Experiments/run_req4_calibrate.py --instance YJ_poi --seeds 1234 2345 3456
```
(240 客/episode 的 eval 每次约 3–6 分钟,两变体校准+冒烟合计约 2–4 小时,可两变体并行。)

- [ ] **Step 6: 写报告并提交**

`Environments/OOH/yanjiao/build_report.md`:节点数、每变体 API 调用数与缓存命中、失败填补对数与比率、
sanity 百分比、depot 坐标与名称、站点选择/采样种子。
`docs/req4_yanjiao_calibration.md`:两变体的 scaler、逐 seed NoPricing/Static 表、与 98% 锚点偏差、
HGS 时限观察、异常如实记录。按 spec §1 六条验收逐条判定写入文档结尾。

```bash
git add Environments/OOH/yanjiao/build_report.md docs/req4_yanjiao_calibration.md
git commit -m "docs: 燕郊实例构建报告与校准结果(spec §1 验收判定)"
```

---

## Self-Review 记录

- Spec 覆盖:C1→Task 2(select_stations),C2→Task 2(gen_homes),C3→Task 1+3(客户端/采集,含计费试探=Task 5 Step 3、
  预算硬顶、缓存断点),C4→Task 4(装配+命名),§3 loader→Task 4,§4 校准→Task 5,§5 三个测试→Tasks 1/2/4
  (外加 Task 3 的矩阵装配测试,比 spec 多一个),§1 验收→Task 5 Step 6 逐条判定,gitignore→Task 4 Step 6。
- 占位符扫描:无 TBD;全部代码块完整可运行。
- 类型/签名一致性:`AmapDistanceClient.batch_distance(origins, dest)` 在 Task 1 定义、Task 3 的 FakeClient 同签名;
  `read_lonlat/default_depot/HOMES_CSV` 由 Task 3 定义、Task 4 复用;节点顺序 [depot]+homes+stations 在
  Task 3/Task 4/Global Constraints 三处一致;`greedy_maxmin(points, k, start_idx)` 测试与实现一致。
- 已知取舍:env 抽客户 randint(1,300) 使 index 300 的 home 永不被抽(继承 off-by-one),两文档处注明;
  校准二分方向(scaler↑→home%↓)已在脚本注释与 RC 经验对齐。
