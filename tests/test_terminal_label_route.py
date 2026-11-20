import sys
from os import path
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
import numpy as np
from Src.Algorithms.DSPO_plus_SPO import DSPO_plus_SPO

def main():
    obj = DSPO_plus_SPO.__new__(DSPO_plus_SPO)
    sentinel = ({'fleet': ['v0']}, 123.4)
    obj.reopt_HGS_final = lambda data: sentinel
    data = {'id': np.array([0, 1, 2])}
    # 委托:返回 reopt_HGS_final 的结果原样
    out = DSPO_plus_SPO._get_terminal_label_route(obj, data, np.array([0, 1, 2]))
    assert out is sentinel or (out[0] is sentinel[0] and out[1] == sentinel[1])
    # route_ids 与 data['id'] 不一致 → AssertionError
    try:
        DSPO_plus_SPO._get_terminal_label_route(obj, data, np.array([0, 1]))
        assert False, "expected AssertionError on route_ids mismatch"
    except AssertionError as ex:
        assert 'route_ids' in str(ex)
    print("OK")

if __name__ == "__main__":
    main()
