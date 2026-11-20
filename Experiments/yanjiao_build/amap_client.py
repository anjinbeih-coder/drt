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

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    @staticmethod
    def _k(lonlat):
        return '{:.6f},{:.6f}'.format(float(lonlat[0]), float(lonlat[1]))

    @staticmethod
    def _default_http_get(url):
        try:
            import certifi
            import ssl
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = None
        with urllib.request.urlopen(url, timeout=30, context=ctx) as r:
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
        # counts logical batches; each internal retry is also a billed HTTP request (undercounts under flaky networking)
        self.new_calls += 1
        got = {}
        if resp is not None and resp.get('status') == '1':
            for res in resp.get('results', []):
                oid = int(res.get('origin_id', 0)) - 1  # 1-based within this request
                dist_val = res.get('distance', res.get('dist'))
                if 0 <= oid < len(missing) and dist_val not in (None, ''):
                    got[missing[oid]] = (float(dist_val), float(res.get('duration', 0.0)))
        for i in missing:
            if i in got:
                self.conn.execute('INSERT OR REPLACE INTO dist_cache VALUES (?,?,?,?,?)',
                                  (self._k(origins[i]), self._k(dest),
                                   got[i][0], got[i][1], time.time()))
                out[i] = got[i]
        self.conn.commit()
        return out
