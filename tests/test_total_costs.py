import sys
from os import path
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
from types import SimpleNamespace
from Src.Utils.Utils import total_costs

def main():
    cfg = SimpleNamespace(driver_wage=30.0, fuel_cost=0.3, truck_speed=30.0,
                          home_failure=0.1, failure_cost=10.0)
    # (service 3600s + travel 3600s) * (30+0.3*30)/3600 = 2*39 = 78;
    # surcharges 5 抵成本,discounts -3 加成本;10 个上门 → 10*0.1*10 = 10
    got = total_costs(10, 3600.0, 3600.0, [5.0], [-3.0], cfg)
    want = 78.0 - 5.0 + 3.0 + 10.0
    assert abs(got - want) < 1e-9, f"got {got}, want {want}"
    print("OK")

if __name__ == "__main__":
    main()
