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
                    'distance': str(1000.0 + 10 * i), 'duration': str(120.0 + i)}
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
        cli4.batch_distance([(125.0, 30.0)], (121.0, 31.0))
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

    # Close all clients to release SQLite locks before removing cache file
    cli.close()
    cli2.close()
    cli3.close()
    cli4.close()
    cli5.close()

    os.remove(fp)
    print('OK test_yanjiao_amap_cache')


if __name__ == '__main__':
    main()
