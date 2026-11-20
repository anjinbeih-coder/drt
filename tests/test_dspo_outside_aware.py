import sys
from os import path
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
import numpy as np
from types import SimpleNamespace
from Src.Algorithms.DSPO import DSPO
from Environments.OOH.containers import Location, ParcelPoint, Customer

def make_obj(outside_util):
    obj = DSPO.__new__(DSPO)
    obj.initial_phase = False
    obj.load_data = False
    obj.init_theta = 0.75
    obj.cool_theta = 0.0
    obj.cost_multiplier = 0.01
    obj.base_util = 0.0
    obj.revenue = 50.0
    obj.min_p, obj.max_p = -1e9, 1e9
    obj.config = SimpleNamespace(outside_option_util=outside_util)
    obj.get_feature_rep_infer = lambda fleet: None
    obj.get_prediction = lambda feat, home, pps: [0.0, 7.0, 3.0]
    obj.cheapestInsertionCosts = lambda loc, fleet: 600.0
    obj.mnl = lambda cust, pp: 0.0
    return obj

def price_vec(outside_util):
    obj = make_obj(outside_util)
    home = Location(0.0, 0.0, 1, 0)
    cust = Customer(home, -0.25, 3.2, 120.0, 1)
    pp = ParcelPoint(Location(5.0, 5.0, 2, 0), 1000000, 2)
    state = [cust, {"fleet": []}, {"parcelpoints": np.array([pp])}, 0]
    return DSPO.get_action_pricing(obj, state, training=False)

def main():
    a_none = price_vec(None)
    a_zero = price_vec(0.0)
    # 一致性锚点:V0=0(除以 e^0=1)必须与 None 路径完全一致
    assert np.allclose(a_none, a_zero, atol=1e-9), f"{a_none} vs {a_zero}"
    # 方向:V0 越大(外部选项越诱人),所有价格越低,且为统一平移(只有 Lambert 项变)
    a_hi = price_vec(2.0)
    shift = a_zero - a_hi
    assert np.all(shift > 0.1), f"prices must drop when outside option strengthens: {a_zero} -> {a_hi}"
    assert abs(shift[0] - shift[1]) < 0.011, "shift must be uniform across options (2-decimal rounding tolerance)"
    print("OK")

if __name__ == "__main__":
    main()
