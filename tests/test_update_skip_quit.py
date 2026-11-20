import sys
from os import path
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
import numpy as np
from Src.Algorithms.DSPO import DSPO

def make_obj():
    obj = DSPO.__new__(DSPO)
    obj.n_layers = 1
    obj.grid_dim = 2
    obj.interval = 10
    obj.customer_cell = np.zeros((10, 2), dtype=int)
    obj.features = np.empty((0, 4))
    obj.cap_features = np.empty((0, 1))
    obj._last_data_len = 1
    return obj

def main():
    obj = make_obj()
    state = [None, None, {"parcelpoints": np.array([])}, 0]
    # 成交步:data['id'] 长度增长 -> 存 1 行特征
    data = {'id': np.array([0, 1]), 'time': np.array([0, 1])}
    DSPO.update(obj, data, state, done=False)
    assert obj.features.shape[0] == 1 and obj._last_data_len == 2
    # 退出步:data 未增长 -> 不存特征
    DSPO.update(obj, data, state, done=False)
    assert obj.features.shape[0] == 1, "quit step must not store a feature"
    # 再次成交:继续存
    data2 = {'id': np.array([0, 1, 2]), 'time': np.array([0, 1, 2])}
    DSPO.update(obj, data2, state, done=False)
    assert obj.features.shape[0] == 2 and obj._last_data_len == 3
    print("OK")

if __name__ == "__main__":
    main()
