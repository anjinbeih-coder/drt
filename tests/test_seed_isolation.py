import sys
from os import path
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
import numpy as np
from Environments.OOH.env_utils import utils_env


def main():
    # generate_fixed_list 不得覆写全局 RNG:否则 Config 设的 --seed 会在环境构造时被抹掉,
    # 多 seed 实验坍缩成同一条客户流(2026-07-25 全量跑批实测暴露的 bug)。
    u = utils_env.__new__(utils_env)

    np.random.seed(123)
    expected = np.random.rand()

    np.random.seed(123)
    lst1 = utils_env.generate_fixed_list(u, 10, 0.38)
    got = np.random.rand()
    assert got == expected, f"global RNG clobbered by generate_fixed_list: {got} != {expected}"

    # 列表本身仍然是确定性的(同参数两次调用结果一致)
    lst2 = utils_env.generate_fixed_list(u, 10, 0.38)
    assert lst1 == lst2, "capacitated list must stay deterministic for fixed (length, fraction)"
    assert sum(lst1) == 3 and len(lst1) == 10
    print("OK")


if __name__ == "__main__":
    main()
