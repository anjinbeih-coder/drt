# 决策遗憾评测 + oracle 上界 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在共享留出标签集上离线计算任意 checkpoint 的决策遗憾 R(a*(c);c) − R(a*(ĉ);c) 与 oracle 利润上界，配套逐客户配对 Wilcoxon 检验。

**Architecture:** 三个新文件共用一个核心库:`Experiments/regret_lib.py`(ĉ 重建、oracle 利润、遗憾、数据集读写),`Experiments/gen_regret_dataset.py`(StaticPricing rollout + 现有标签机制收集共享集),`Experiments/eval_regret.py`(加载 checkpoint 批量评测并写报告)。`run.py` 补一行训练结束保存模型。

**Tech Stack:** Python 3.9(conda py39),torch,numpy,scipy 1.11.4(已核实存在),测试为仓库惯例的裸 assert 脚本(无 pytest)。

## Global Constraints

- 分支 `regret-metric`;解释器 **必须** 用 `C:\Users\wwpst\miniforge3\envs\py39\python.exe`(唯一装有 torch/hygese 的环境)。
- 实验口径(与实验 2/2b 一致):V0=0,home_failure=0,mnl_dist_scaler=1800.0,min_price=-10.0,**max_price=10.0**,instance RC,revenue=50,incentive_sens=-0.25,home_util=3.2。
- 共享集:StaticPricing(home **+2.0**,pp **-5.0**),seed **9999**,**20** 个标签收集 episode。
- 数据集样本 schema **必须** 与 `DSPO_plus_SPO.spo_training_data` 条目完全一致(键:`true_costs`,`customer_info`,`pps_info`,`prices`,`feature_info`;`feature_info` 内键:`base_feat`,`home_cell`,`pp_cells`,`time_int`,`pp_caps`,`theta`,`ins_cost_home`,`ins_cost_pps`,`service_time`)。
- 算法代码零改动;仓库代码唯一改动是 `run.py` `Solver.train()` 末尾加 `self.model.save()`。
- 生成产物(数据集 .pt、checkpoint)放 `Experiments/Parcelpoint_py/` 下(.gitignore 已忽略该目录,已核实)。
- 测试是普通 python 脚本,单独运行,成功打印 `OK <名字>`,失败靠 assert 崩溃。
- checkpoint 文件路径:`<run目录>/<seed>/Checkpoints/supervised_ml.pt`(Config paths + Agent.save 拼接,已核实)。
- 遗憾符号约定:regret = R(a*(c);c) − R(a*(ĉ);c);R(a;c) = r0 − probs·c,其中 r0/probs 取自 `_pricing_oracle_lifted_torch` 返回值。
- 负遗憾按诊断处理(spec §1.1):负值占比 >5% 或负值幅度中位数 > 平均遗憾的 10% → 标记 blocker。

---

### Task 1: 核心库 regret_lib + 玩具正确性测试 + 数据集读写往返测试

**Files:**
- Create: `Experiments/regret_lib.py`
- Test: `tests/test_regret_toy.py`
- Test: `tests/test_regret_dataset_roundtrip.py`

**Interfaces:**
- Consumes: `Src.Algorithms.DSPO_plus_SPO.DSPO_plus_SPO` 的 `_pricing_oracle_lifted_torch(costs_t, customer_info, pps_info)`(实例方法,依赖 `self.min_p/max_p/base_util/revenue/config.outside_option_util`,返回 dict 含 `prices/probs/r0/lifted` 或 None)、静态方法 `_predicted_option_cost(fixed_term, mltplr, theta, insertion_cost, cnn_out, cnn_base)`、实例属性 `device/supervised_ml/cost_multiplier/l0_home/l_mp`。
- Produces(后续任务与测试依赖的精确签名):
  - `make_bare_oracle_agent(min_p=-10.0, max_p=10.0, base_util=0.0, revenue=50.0, outside_option_util=0.0) -> DSPO_plus_SPO`(绕过 `__init__` 的裸实例,仅够跑 oracle);
  - `expected_profit_under(oracle_out: dict, c_true_t: torch.Tensor) -> float`;
  - `regret_for_sample(agent, entry: dict, c_hat_t: torch.Tensor) -> dict|None`(键 `regret/r_oracle/r_model/mse`);
  - `reconstruct_c_hat(agent, entry: dict) -> torch.Tensor|None`(长度 K 的 float32);
  - `save_dataset(filepath, meta: dict, entries: list)` / `load_dataset(filepath, expect_meta: dict|None) -> (meta, entries)`(指纹不符抛 `ValueError`);
  - 常量 `DATASET_META_KEYS = ['instance','mnl_dist_scaler','outside_option_util','min_price','max_price','home_failure','seed','episodes','policy']`。

- [ ] **Step 1: 写玩具测试(先失败)**

`tests/test_regret_toy.py`(参考既有 tests/ 的裸对象模式):

```python
import sys
from os import path
import numpy as np
import torch

sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), '..'))
from Experiments.regret_lib import (make_bare_oracle_agent, expected_profit_under,
                                    regret_for_sample)


def manual_expected_profit(prices, c, base_util, home_util, pp_utils, s, rev, v0):
    """独立的 numpy 实现:MNL 概率 + 期望利润,与库实现互为对照。"""
    utils = [base_util + home_util + s * prices[0]]
    for j, pu in enumerate(pp_utils):
        utils.append(pu + s * prices[j + 1])
    utils.append(v0)
    ex = np.exp(np.asarray(utils, dtype=np.float64))
    probs = ex / ex.sum()
    r = 0.0
    for k in range(len(prices)):
        r += probs[k] * (rev + prices[k] - c[k])
    return r


def main():
    agent = make_bare_oracle_agent(min_p=-10.0, max_p=10.0, base_util=0.0,
                                   revenue=50.0, outside_option_util=0.0)
    c = [12.0, 3.0]
    entry = {
        'true_costs': c,
        'customer_info': {'base_util': 0.0, 'home_util': 3.2,
                          'incentive_sensitivity': -0.25, 'revenue': 50.0},
        'pps_info': [{'util': -1.0}],
    }
    c_t = torch.tensor(c, dtype=torch.float32)

    # 1) 手算对照:库的期望利润 == 独立 numpy 实现(同一价格向量下)
    o = agent._pricing_oracle_lifted_torch(c_t, entry['customer_info'], entry['pps_info'])
    assert o is not None
    prices = o['prices'].cpu().numpy().tolist()
    r_lib = expected_profit_under(o, c_t)
    r_manual = manual_expected_profit(prices, c, 0.0, 3.2, [-1.0], -0.25, 50.0, 0.0)
    assert abs(r_lib - r_manual) < 1e-6, (r_lib, r_manual)

    # 2) oracle 近似最优性:0.05 步长网格搜索有界价格域,oracle 不明显劣于网格最优
    grid = np.arange(-10.0, 10.0 + 1e-9, 0.05)
    best = -1e18
    for a0 in grid:
        for a1 in grid:
            best = max(best, manual_expected_profit([a0, a1], c, 0.0, 3.2, [-1.0],
                                                    -0.25, 50.0, 0.0))
    assert r_lib >= best - 0.05, 'oracle profit {} << grid max {}'.format(r_lib, best)

    # 3) regret(c, c) == 0 精确成立(确定性 oracle 同输入同输出)
    res = regret_for_sample(agent, entry, c_t.clone())
    assert res is not None
    assert res['regret'] == 0.0, res['regret']
    assert res['mse'] == 0.0

    # 4) 压扁成本差的失真预测,遗憾严格为正
    res_bad = regret_for_sample(agent, entry, torch.tensor([8.0, 7.0]))
    assert res_bad is not None and res_bad['regret'] > 0.0, res_bad

    print('OK test_regret_toy')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `"C:\Users\wwpst\miniforge3\envs\py39\python.exe" tests/test_regret_toy.py`(工作目录 `D:\project\python\ooh_code`)
Expected: `ModuleNotFoundError: No module named 'Experiments.regret_lib'`

- [ ] **Step 3: 写 regret_lib.py**

```python
"""Shared core for decision-regret evaluation (spec 2026-07-26-regret-metric-design.md).

Entry schema == one value of DSPO_plus_SPO.spo_training_data (see plan Global Constraints).
All agent-taking functions accept either a full DSPO_plus_SPO or the bare oracle agent.
"""
import sys
from os import path

import numpy as np
import torch

sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), '..'))
from Src.Algorithms.DSPO_plus_SPO import DSPO_plus_SPO  # noqa: E402


class _BareConfig(object):
    def __init__(self, outside_option_util):
        self.outside_option_util = outside_option_util


def make_bare_oracle_agent(min_p=-10.0, max_p=10.0, base_util=0.0, revenue=50.0,
                           outside_option_util=0.0):
    """DSPO_plus_SPO instance without __init__ (no env); enough for the pricing oracle."""
    agent = object.__new__(DSPO_plus_SPO)
    agent.min_p = float(min_p)
    agent.max_p = float(max_p)
    agent.base_util = float(base_util)
    agent.revenue = float(revenue)
    agent.config = _BareConfig(float(outside_option_util))
    return agent


def expected_profit_under(oracle_out, c_true_t):
    """R(a; c) = sum_k P_k(a) * (rev + a_k - c_k) = r0 - probs . c."""
    probs = oracle_out['probs'].detach().cpu()
    r0 = float(oracle_out['r0'].detach().cpu())
    return r0 - float(torch.dot(probs, c_true_t))


def regret_for_sample(agent, entry, c_hat_t):
    c_true_t = torch.tensor(np.asarray(entry['true_costs'], dtype=np.float32))
    if c_hat_t is None or c_hat_t.numel() != c_true_t.numel() or c_true_t.numel() <= 1:
        return None
    if not (torch.isfinite(c_hat_t).all() and torch.isfinite(c_true_t).all()):
        return None
    o_true = agent._pricing_oracle_lifted_torch(
        c_true_t, entry['customer_info'], entry['pps_info'])
    o_hat = agent._pricing_oracle_lifted_torch(
        c_hat_t.detach().cpu(), entry['customer_info'], entry['pps_info'])
    if o_true is None or o_hat is None:
        return None
    r_oracle = expected_profit_under(o_true, c_true_t)
    r_model = expected_profit_under(o_hat, c_true_t)
    return {'regret': r_oracle - r_model, 'r_oracle': r_oracle, 'r_model': r_model,
            'mse': float(torch.mean((c_hat_t - c_true_t) ** 2))}


def reconstruct_c_hat(agent, entry):
    """Model's predicted option-cost vector for one stored sample.

    Mirrors DSPO_plus_SPO._calculate_spo_loss_for_batch phases 1-3 for a single
    sample, eval mode, no grad. The stored per-sample theta is the deployment-time
    within-day schedule value (state[3] is the intra-episode step counter), identical
    for every model -- reusing it is both faithful and fair.
    """
    fi = entry.get('feature_info', None)
    if fi is None:
        return None
    base_feat = fi['base_feat']
    pp_cells = fi['pp_cells']
    time_int = fi['time_int']
    pp_caps = fi.get('pp_caps', [1000000.0] * len(pp_cells))
    K = 1 + len(pp_cells)
    if K <= 1 or len(entry['true_costs']) < K:
        return None
    feats = [base_feat]
    caps = [1000000.0]
    home_aug = base_feat.clone()
    home_aug[0][time_int][fi['home_cell'][0]][fi['home_cell'][1]] += 1
    feats.append(home_aug)
    caps.append(1000000.0)
    for j, pc in enumerate(pp_cells):
        aug = base_feat.clone()
        aug[0][time_int][pc[0]][pc[1]] += 1
        feats.append(aug)
        caps.append(float(pp_caps[j]) if j < len(pp_caps) else 1000000.0)
    batch = torch.cat(feats, dim=0).to(agent.device)
    caps_t = torch.tensor(caps, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        out = agent.supervised_ml(batch, caps_t).squeeze(-1).cpu()
    mltplr = agent.cost_multiplier
    theta = fi['theta']
    c_hat = torch.zeros(K, dtype=torch.float32)
    c_hat[0] = float(agent._predicted_option_cost(
        agent.l0_home + fi['service_time'], mltplr, theta,
        fi['ins_cost_home'], out[1], out[0]))
    for j in range(K - 1):
        c_hat[j + 1] = float(agent._predicted_option_cost(
            agent.l_mp, mltplr, theta, fi['ins_cost_pps'][j], out[2 + j], out[0]))
    return c_hat


DATASET_META_KEYS = ['instance', 'mnl_dist_scaler', 'outside_option_util', 'min_price',
                     'max_price', 'home_failure', 'seed', 'episodes', 'policy']


def save_dataset(filepath, meta, entries):
    missing = [k for k in DATASET_META_KEYS if k not in meta]
    assert not missing, 'dataset meta missing keys: {}'.format(missing)
    torch.save({'meta': dict(meta), 'entries': list(entries)}, filepath)


def load_dataset(filepath, expect_meta=None):
    blob = torch.load(filepath, map_location='cpu')
    meta, entries = blob['meta'], blob['entries']
    if expect_meta:
        for k, v in expect_meta.items():
            got = meta.get(k, None)
            if isinstance(v, (int, float)) and got is not None:
                same = abs(float(got) - float(v)) < 1e-9
            else:
                same = (got == v)
            if not same:
                raise ValueError('dataset fingerprint mismatch on {}: dataset={} expected={}'
                                 .format(k, got, v))
    return meta, entries
```

- [ ] **Step 4: 跑玩具测试确认通过**

Run: `"C:\Users\wwpst\miniforge3\envs\py39\python.exe" tests/test_regret_toy.py`
Expected: `OK test_regret_toy`(网格搜索约 16 万次求值,允许运行几十秒)

- [ ] **Step 5: 写往返测试并跑通**

`tests/test_regret_dataset_roundtrip.py`:

```python
import os
import sys
from os import path

import numpy as np
import torch

sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), '..'))
from Experiments.regret_lib import save_dataset, load_dataset


def main():
    entry = {
        'true_costs': [5.0, 1.0, 2.0],
        'customer_info': {'base_util': 0.0, 'home_util': 3.2,
                          'incentive_sensitivity': -0.25, 'revenue': 50.0},
        'pps_info': [{'util': -1.0}, {'util': -0.5}],
        'prices': np.zeros(3, dtype=np.float32),
        'feature_info': {'base_feat': torch.zeros((1, 3, 10, 10)), 'home_cell': (2, 3),
                         'pp_cells': [(4, 5), (6, 7)], 'time_int': 1,
                         'pp_caps': [10.0, 10.0], 'theta': 0.7,
                         'ins_cost_home': 100.0, 'ins_cost_pps': [50.0, 60.0],
                         'service_time': 120.0},
    }
    meta = {'instance': 'RC', 'mnl_dist_scaler': 1800.0, 'outside_option_util': 0.0,
            'min_price': -10.0, 'max_price': 10.0, 'home_failure': 0.0,
            'seed': 9999, 'episodes': 20, 'policy': 'static_home2.0_pp-5.0'}
    fp = path.join(path.dirname(path.abspath(__file__)), '_tmp_regret_ds.pt')
    save_dataset(fp, meta, [entry])
    m2, e2 = load_dataset(fp, expect_meta={'mnl_dist_scaler': 1800.0,
                                           'max_price': 10.0, 'instance': 'RC'})
    assert m2 == meta
    assert len(e2) == 1
    assert e2[0]['true_costs'] == entry['true_costs']
    assert torch.equal(e2[0]['feature_info']['base_feat'],
                       entry['feature_info']['base_feat'])
    try:
        load_dataset(fp, expect_meta={'max_price': 2.0})
        raise AssertionError('fingerprint mismatch not detected')
    except ValueError:
        pass
    os.remove(fp)
    print('OK test_regret_dataset_roundtrip')


if __name__ == '__main__':
    main()
```

Run: `"C:\Users\wwpst\miniforge3\envs\py39\python.exe" tests/test_regret_dataset_roundtrip.py`
Expected: `OK test_regret_dataset_roundtrip`

- [ ] **Step 6: Commit**

```bash
git add Experiments/regret_lib.py tests/test_regret_toy.py tests/test_regret_dataset_roundtrip.py
git commit -m "feat: regret_lib 核心(oracle 利润/遗憾/c_hat 重建/数据集读写) + 玩具与往返测试"
```

---

### Task 2: 共享标签集生成脚本 gen_regret_dataset.py

**Files:**
- Create: `Experiments/gen_regret_dataset.py`

**Interfaces:**
- Consumes: Task 1 的 `save_dataset`;`Experiments.run_paper_repro.build_argv(policy, seed, scaler, train_episodes, buffer_size, extra)`;`Experiments.run_req2_profit.req2_extra(v0)`;`Experiments.run_req3_spo.SCALER, V0`;`run.Solver`;`Src.parser.Parser`;`Src.config.Config`。
- Produces: 数据集文件 `Experiments/Parcelpoint_py/regret_dataset_RC_seed9999.pt`(meta 含 Global Constraints 的九个指纹键)。

背景事实(已核实,实现者不必再查):
- `run.Solver.train()` 每 episode 调 `model.get_action(state, training=True)` → `DSPO.get_action` 内部经 `self.get_action_pricing` 派发(实例属性可覆写);episode 结束调 `model.update(route_data, state, True)`,子类在其中做终局标签收集并写入 `agent.spo_training_data`(dict,值即数据集样本 schema)。
- 初始阶段在 `memory.length >= config.buffer_size` 时结束(`DSPO.update` :271-273);传 buffer_size=90 → 约 1 个 episode 后进入正常阶段。初始阶段收敛判据基于 `self_supervised_update` 返回的损失,桩成 0.0 后 20 个 epoch 内自动 break。
- 训练入口只有两个:`agent.optimize()`(每 episode)与 `agent.self_supervised_update(...)`(初始阶段内循环)。都桩掉即"不训练任何模型"(spec C1)。
- `spo_buffer_size=100000` 显式传入 → `spo_training_data` 不淘汰(0 才回退到 buffer_size)。
- `spo_label_failures` 计数器为 0 是数据集有效的门槛(spec F1 机制)。

- [ ] **Step 1: 写脚本**

```python
"""Generate the shared held-out regret dataset (spec C1).

StaticPricing rollout (home +2.0 / pp -5.0) with the terminal-route counterfactual
label machinery of DSPO_plus_SPO. No model is trained: optimize() and
self_supervised_update() are stubbed to no-ops (the collector's CNN weights never
matter -- offline evaluation re-predicts from checkpoints). Episode 1 is a warm-up
that only serves to end the initial phase (buffer_size=90).
"""
import sys
from os import path

import numpy as np

ROOT = path.join(path.dirname(path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)

from Src.parser import Parser                        # noqa: E402
from Src.config import Config                        # noqa: E402
from run import Solver                               # noqa: E402
from Experiments.run_paper_repro import build_argv   # noqa: E402
from Experiments.run_req2_profit import req2_extra   # noqa: E402
from Experiments.run_req3_spo import SCALER, V0      # noqa: E402
from Experiments.regret_lib import save_dataset      # noqa: E402

SEED = 9999
EPISODES = 20
STATIC_HOME = 2.0
STATIC_PP = -5.0
OUT = path.join(ROOT, 'Experiments', 'Parcelpoint_py',
                'regret_dataset_RC_seed{}.pt'.format(SEED))


def main():
    extra = dict(req2_extra(V0))
    extra.update({
        'algo_name': 'DSPO_plus_SPO',
        'spo_loss_weight': '0.7',      # >0 so label collection is enabled
        'spo_buffer_size': '100000',   # keep every labelled sample
        'spo_label_sample_size': '0',  # label every priced customer
        'max_price': '10.0',
        'save_model': 'False',
        'folder_suffix': 'regretds_{}'.format(SEED),
    })
    argv = build_argv('DSPO', SEED, SCALER, EPISODES + 1, 90, extra=extra)
    args = Parser().get_parser().parse_args(argv)
    config = Config(args)
    solver = Solver(config)
    agent = solver.model

    # --- no training (spec C1): stub both training entry points ---
    agent.optimize = lambda *a, **k: 0.0
    agent.self_supervised_update = lambda *a, **k: 0.0

    # --- static pricing: parent call records the decision, then charge fixed prices ---
    orig_pricing = agent.get_action_pricing

    def static_pricing(state, training):
        a = np.asarray(orig_pricing(state, training), dtype=np.float64)
        a_static = np.full_like(a, STATIC_PP)
        a_static[0] = STATIC_HOME
        decs = agent.spo_current_episode_data['pricing_decisions']
        if decs:  # keep the recorded decision consistent with what is charged
            decs[-1]['prices'] = a_static.copy()
        return a_static

    agent.get_action_pricing = static_pricing

    solver.train()

    entries = list(agent.spo_training_data.values())
    assert len(entries) > 0, 'no labelled samples collected'
    assert agent.spo_label_failures == 0, \
        '{} label failures during collection'.format(agent.spo_label_failures)
    meta = {'instance': 'RC', 'mnl_dist_scaler': float(SCALER),
            'outside_option_util': float(V0), 'min_price': -10.0, 'max_price': 10.0,
            'home_failure': 0.0, 'seed': SEED, 'episodes': EPISODES,
            'policy': 'static_home{}_pp{}'.format(STATIC_HOME, STATIC_PP)}
    save_dataset(OUT, meta, entries)
    print('saved {} samples -> {}'.format(len(entries), OUT))


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 冒烟验证(临时缩短)**

临时把 `EPISODES = 20` 改为 `EPISODES = 2` 跑一遍:

Run: `"C:\Users\wwpst\miniforge3\envs\py39\python.exe" Experiments/gen_regret_dataset.py`(工作目录仓库根)
Expected: 数分钟内打印 `saved <N> samples -> ...regret_dataset_RC_seed9999.pt`,N 大约 120–200(≈2 个 episode 的非退出客户数);全程无 `[SPO+错误`。

再用一段临时 python 验证样本可评测(临时脚本或 REPL,不入库):

```python
import torch, sys; sys.path.insert(0, '.')
from Experiments.regret_lib import load_dataset, make_bare_oracle_agent, regret_for_sample
meta, entries = load_dataset('Experiments/Parcelpoint_py/regret_dataset_RC_seed9999.pt')
agent = make_bare_oracle_agent(outside_option_util=meta['outside_option_util'])
e = entries[0]
res = regret_for_sample(agent, e, torch.tensor(e['true_costs'], dtype=torch.float32))
assert res is not None and res['regret'] == 0.0, res
print('smoke OK, K =', len(e['true_costs']), ', samples =', len(entries))
```

Expected: `smoke OK, K = <2~11>, samples = <N>`

- [ ] **Step 3: 恢复 EPISODES = 20,删除冒烟产物**

把 `EPISODES` 改回 `20`;删除冒烟数据集文件(正式生成在 Task 5 做):

```bash
rm Experiments/Parcelpoint_py/regret_dataset_RC_seed9999.pt
```

- [ ] **Step 4: Commit**

```bash
git add Experiments/gen_regret_dataset.py
git commit -m "feat: 共享遗憾标签集生成脚本(StaticPricing rollout + 终局反事实标签,不训练)"
```

---

### Task 3: run.py 训练结束保存模型

**Files:**
- Modify: `run.py:56`(`Solver.train()` 循环结束后)

**Interfaces:**
- Consumes: `Agent.save()`(`Src/Algorithms/Agent.py:24-27`,内部有 `if self.config.save_model` 门控;`save_model=False` 时为纯 no-op,对既有一切运行零影响)。
- Produces: `save_model=True` 的训练 run 在 `<run>/<seed>/Checkpoints/supervised_ml.pt` 留下**最终**权重(现状:只有初始阶段结束时保存过一次,终态权重从不落盘——已核实 `.save()` 全库仅 DSPO.py:309/:318 两处调用)。

- [ ] **Step 1: 修改 run.py**

在 `Solver.train()` 的 for 循环结束后(与 `for` 对齐的缩进层级,`t0 = time()` 行之后)添加:

```python
        self.model.save()  # persist final weights when config.save_model is set
```

- [ ] **Step 2: 冒烟验证**

用 3 个 episode 的迷你 run 验证 checkpoint 落盘(约 2–3 分钟):

```python
import sys; sys.path.insert(0, '.')
from Experiments.run_paper_repro import run_one
from Experiments.run_req2_profit import req2_extra
from Experiments.run_req3_spo import SCALER, V0
extra = dict(req2_extra(V0)); extra.update({'save_model': 'True', 'max_price': '10.0',
                                            'folder_suffix': 'ckpt_smoke'})
m = run_one('DSPO', 1234, SCALER, 3, 1, 90, extra=extra)
from os import path
p = 'Experiments/Parcelpoint_py/pricing/DSPO/runckpt_smoke/1234/Checkpoints/supervised_ml.pt'
assert path.isfile(p), 'checkpoint missing: ' + p
print('ckpt smoke OK')
```

Run: 存成临时脚本用 py39 解释器执行(工作目录仓库根)
Expected: `ckpt smoke OK`

- [ ] **Step 3: 既有测试回归**

逐个运行 `tests/` 下全部 15 个既有测试脚本(py39 解释器),全部打印各自的 OK。

- [ ] **Step 4: Commit**

```bash
git add run.py
git commit -m "feat: Solver.train 结束时保存最终模型权重(save_model 门控)"
```

---

### Task 4: 离线评测脚本 eval_regret.py

**Files:**
- Create: `Experiments/eval_regret.py`

**Interfaces:**
- Consumes: Task 1 的 `load_dataset/reconstruct_c_hat/regret_for_sample`;`build_argv/req2_extra/SCALER/V0`(同 Task 2);`scipy.stats.wilcoxon`(scipy 1.11.4 已核实);checkpoint 文件 `<run目录参数>/Checkpoints/supervised_ml.pt`,经 `agent.supervised_ml.load(path)` 加载(`load_state_dict`,已核实存在)。
- Produces: CLI `python Experiments/eval_regret.py --dataset P --model NAME=RUN_SEED_DIR ... --pair A:B ... --out docs/req3_regret_results.md`;markdown 报告(每模型汇总表 + 配对检验表 + POOLED 行 + 指纹/口径说明)。

- [ ] **Step 1: 写脚本**

```python
"""Offline decision-regret evaluation (spec C2).

Usage (paths are run *seed* directories, i.e. the parent of Checkpoints/):
  python Experiments/eval_regret.py \
      --dataset Experiments/Parcelpoint_py/regret_dataset_RC_seed9999.pt \
      --model DSPO_1234=Experiments/Parcelpoint_py/pricing/DSPO/runreq3p10ck_DSPO_1234/1234 \
      --model SPOa07_1234=Experiments/Parcelpoint_py/pricing/DSPO_plus_SPO/runreq3p10ck_SPOa07_1234/1234 \
      --pair DSPO_1234:SPOa07_1234 \
      --out docs/req3_regret_results.md

Pair A:B reports diff = regret_A - regret_B (positive median => B has lower regret).
"""
import argparse
import sys
from os import path

import numpy as np
import torch
from scipy import stats

ROOT = path.join(path.dirname(path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)

from Src.parser import Parser                        # noqa: E402
from Src.config import Config                        # noqa: E402
from Experiments.run_paper_repro import build_argv   # noqa: E402
from Experiments.run_req2_profit import req2_extra   # noqa: E402
from Experiments.run_req3_spo import SCALER, V0      # noqa: E402
from Experiments.regret_lib import (load_dataset, reconstruct_c_hat,   # noqa: E402
                                    regret_for_sample)

EXPECT_META = {'instance': 'RC', 'mnl_dist_scaler': float(SCALER),
               'outside_option_util': float(V0), 'min_price': -10.0,
               'max_price': 10.0, 'home_failure': 0.0}


def build_agent():
    """One full agent (env + CNN skeleton) reused for every checkpoint."""
    extra = dict(req2_extra(V0))
    extra.update({'algo_name': 'DSPO_plus_SPO', 'spo_loss_weight': '0.0',
                  'max_price': '10.0', 'save_model': 'False',
                  'folder_suffix': 'regret_eval_tmp'})
    argv = build_argv('DSPO', 1234, SCALER, 0, 10000, extra=extra)
    args = Parser().get_parser().parse_args(argv)
    config = Config(args)
    return config.algo(config=config)


def eval_model(agent, run_seed_dir, entries):
    ckpt = path.join(run_seed_dir, 'Checkpoints', 'supervised_ml.pt')
    assert path.isfile(ckpt), 'missing checkpoint: ' + ckpt
    agent.supervised_ml.load(ckpt)
    agent.supervised_ml.eval()
    rows = []
    for entry in entries:
        c_hat = reconstruct_c_hat(agent, entry)
        rows.append(regret_for_sample(agent, entry, c_hat) if c_hat is not None else None)
    return rows


def summarize(name, rows):
    ok = [r for r in rows if r is not None]
    reg = np.array([r['regret'] for r in ok])
    ro = np.array([r['r_oracle'] for r in ok])
    rm = np.array([r['r_model'] for r in ok])
    mse = np.array([r['mse'] for r in ok])
    neg = reg[reg < 0]
    mean_reg = float(reg.mean())
    out = {'name': name, 'n': len(ok), 'skipped': len(rows) - len(ok),
           'mean_regret': mean_reg,
           'rel_regret_pct': 100.0 * mean_reg / float(ro.mean()),
           'mean_r_oracle': float(ro.mean()), 'mean_r_model': float(rm.mean()),
           'mse': float(mse.mean()),
           'neg_share_pct': 100.0 * len(neg) / max(1, len(reg)),
           'neg_median_mag': float(np.median(-neg)) if len(neg) else 0.0}
    out['blocker'] = bool(out['neg_share_pct'] > 5.0 or
                          (mean_reg > 0 and out['neg_median_mag'] > 0.10 * mean_reg))
    return out


def pair_stats(label, rows_a, rows_b):
    diffs = np.array([ra['regret'] - rb['regret'] for ra, rb in zip(rows_a, rows_b)
                      if ra is not None and rb is not None])
    if len(diffs) == 0 or not np.any(diffs != 0):
        return {'pair': label, 'n': len(diffs), 'median_diff': 0.0,
                'mean_diff': 0.0, 'p_value': 1.0}
    res = stats.wilcoxon(diffs)
    return {'pair': label, 'n': int(len(diffs)),
            'median_diff': float(np.median(diffs)),
            'mean_diff': float(diffs.mean()), 'p_value': float(res.pvalue)}


def write_report(out_path, meta, summaries, pairs):
    lines = ['# 决策遗憾评测结果(共享留出集)', '',
             '- 数据集:seed={seed},episodes={episodes},policy={policy},样本 n={n}'.format(
                 n=summaries[0]['n'] + summaries[0]['skipped'] if summaries else 0, **meta),
             '- 口径:scaler={mnl_dist_scaler},V0={outside_option_util},'
             '价格域 [{min_price},{max_price}],home_failure={home_failure}'.format(**meta),
             '- oracle 为"闭式 Lambert-W + 裁剪"近似(与训练一致);负遗憾为近似诊断量',
             '', '## 每模型汇总', '',
             '| 模型 | n | 平均遗憾 | 相对遗憾% | oracle 利润 | 模型利润 | MSE | 负遗憾占比% | blocker |',
             '|---|---|---|---|---|---|---|---|---|']
    for s in summaries:
        lines.append('| {name} | {n} | {mean_regret:.4f} | {rel_regret_pct:.2f} | '
                     '{mean_r_oracle:.3f} | {mean_r_model:.3f} | {mse:.3f} | '
                     '{neg_share_pct:.2f} | {blocker} |'.format(**s))
    lines += ['', '## 配对检验(diff = 左 − 右;中位差 >0 表示右侧遗憾更低)', '',
              '| 对比 | n | 配对差中位数 | 配对差均值 | Wilcoxon p |', '|---|---|---|---|---|']
    for p in pairs:
        lines.append('| {pair} | {n} | {median_diff:.4f} | {mean_diff:.4f} | '
                     '{p_value:.3g} |'.format(**p))
    lines.append('')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('report -> ' + out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--model', action='append', required=True,
                    help='NAME=run seed dir (parent of Checkpoints/)')
    ap.add_argument('--pair', action='append', default=[], help='NAME_A:NAME_B')
    ap.add_argument('--out', default=path.join(ROOT, 'docs', 'req3_regret_results.md'))
    opt = ap.parse_args()

    meta, entries = load_dataset(opt.dataset, expect_meta=EXPECT_META)
    agent = build_agent()

    all_rows, summaries = {}, []
    for spec in opt.model:
        name, run_dir = spec.split('=', 1)
        rows = eval_model(agent, run_dir, entries)
        all_rows[name] = rows
        s = summarize(name, rows)
        summaries.append(s)
        print('{name}: n={n} mean_regret={mean_regret:.4f} rel={rel_regret_pct:.2f}% '
              'mse={mse:.3f} neg%={neg_share_pct:.2f} blocker={blocker}'.format(**s))

    pairs = []
    pooled = []
    for spec in opt.pair:
        a, b = spec.split(':', 1)
        pairs.append(pair_stats('{} - {}'.format(a, b), all_rows[a], all_rows[b]))
        pooled.extend([(ra, rb) for ra, rb in zip(all_rows[a], all_rows[b])
                       if ra is not None and rb is not None])
    if len(opt.pair) > 1 and pooled:
        rows_a = [p[0] for p in pooled]
        rows_b = [p[1] for p in pooled]
        pairs.append(pair_stats('POOLED (all pairs)', rows_a, rows_b))
    for p in pairs:
        print('{pair}: n={n} median_diff={median_diff:.4f} p={p_value:.3g}'.format(**p))

    write_report(opt.out, meta, summaries, pairs)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 冒烟验证**

先临时生成迷你数据集(Task 2 的 EPISODES=2 流程,或直接复用 Task 2 冒烟没删的文件——若已删,重跑一次 EPISODES=2 再改回),再用 Task 3 冒烟产出的 checkpoint 跑:

```bash
"C:\Users\wwpst\miniforge3\envs\py39\python.exe" Experiments/eval_regret.py \
  --dataset Experiments/Parcelpoint_py/regret_dataset_RC_seed9999.pt \
  --model SMOKE=Experiments/Parcelpoint_py/pricing/DSPO/runckpt_smoke/1234 \
  --out Experiments/Parcelpoint_py/_smoke_regret.md
```

Expected: 打印 `SMOKE: n=... mean_regret=...`(3 episode 训出的模型,遗憾应明显 >0),生成 `_smoke_regret.md`;无异常。验证后删除迷你数据集与 `_smoke_regret.md`。

- [ ] **Step 3: Commit**

```bash
git add Experiments/eval_regret.py
git commit -m "feat: 离线决策遗憾评测 CLI(逐客户配对 Wilcoxon + markdown 报告)"
```

---

### Task 5: 执行——重训 checkpoint、生成正式数据集、评测出报告(操作性任务,无新代码)

**Files:**
- Create(临时,scratchpad 而非仓库): 单 seed 重训脚本(下方给出)
- 产物: `docs/req3_regret_results.md`(评测脚本生成 + 人工补判读段落)

**Interfaces:**
- Consumes: Task 2/3/4 的全部产物。
- Produces: spec §1.4 三分支判读所需的全部数字。

- [ ] **Step 1: 并行重训 6 个 run(约 40 分钟墙钟)**

单 seed 脚本(存 scratchpad,内容如下,`GROUP`/`SEED` 由命令行传入):

```python
import sys
from os import path
sys.path.insert(0, r'D:\project\python\ooh_code')
from Experiments.run_req3_spo import SCALER, V0
from Experiments.run_req2_profit import req2_extra
from Experiments.run_paper_repro import run_one

group, seed = sys.argv[1], int(sys.argv[2])   # group in {'DSPO', 'SPOa07'}
extra = dict(req2_extra(V0))
extra.update({'max_price': '10.0', 'save_model': 'True',
              'folder_suffix': 'req3p10ck_{}_{}'.format(group, seed)})
if group == 'SPOa07':
    extra.update({'algo_name': 'DSPO_plus_SPO', 'spo_loss_weight': '0.7'})
m = run_one('DSPO', seed, SCALER, 500, 20, 10000, extra=extra)
print('{} SEED {} -> net_profit={:.2f} quit={:.4f}'.format(
    group, seed, m['net_profit'], m['quit_rate']), flush=True)
```

6 个进程并行:`(DSPO|SPOa07) × (1234|2345|3456)`。

**一致性门槛**:新 run 的 net_profit 必须与实验 2/2b 同 seed **完全一致**(save_model 不影响训练轨迹):
DSPO → 1896.84 / 1990.53 / 2003.75;SPOa07 → 1939.02 / 1953.56 / 1953.62。不一致即停,先查再评。

- [ ] **Step 2: 生成正式共享集(约 5–10 分钟)**

Run: `"C:\Users\wwpst\miniforge3\envs\py39\python.exe" Experiments/gen_regret_dataset.py`
Expected: `saved ~1600-1800 samples`,零标签失败。

- [ ] **Step 3: 全量评测**

```bash
"C:\Users\wwpst\miniforge3\envs\py39\python.exe" Experiments/eval_regret.py \
  --dataset Experiments/Parcelpoint_py/regret_dataset_RC_seed9999.pt \
  --model DSPO_1234=Experiments/Parcelpoint_py/pricing/DSPO/runreq3p10ck_DSPO_1234/1234 \
  --model DSPO_2345=Experiments/Parcelpoint_py/pricing/DSPO/runreq3p10ck_DSPO_2345/2345 \
  --model DSPO_3456=Experiments/Parcelpoint_py/pricing/DSPO/runreq3p10ck_DSPO_3456/3456 \
  --model SPOa07_1234=Experiments/Parcelpoint_py/pricing/DSPO_plus_SPO/runreq3p10ck_SPOa07_1234/1234 \
  --model SPOa07_2345=Experiments/Parcelpoint_py/pricing/DSPO_plus_SPO/runreq3p10ck_SPOa07_2345/2345 \
  --model SPOa07_3456=Experiments/Parcelpoint_py/pricing/DSPO_plus_SPO/runreq3p10ck_SPOa07_3456/3456 \
  --pair DSPO_1234:SPOa07_1234 --pair DSPO_2345:SPOa07_2345 --pair DSPO_3456:SPOa07_3456 \
  --out docs/req3_regret_results.md
```

- [ ] **Step 4: 判读落袋并提交报告**

按 spec §1.4 三分支在 `docs/req3_regret_results.md` 末尾补判读段落(DSPO 相对遗憾 <1% → 无空间;SPO+ 显著更低 → 判据达成;可观但未降 → 方法问题),并检查负遗憾 blocker 字段。

```bash
git add docs/req3_regret_results.md
git commit -m "docs: 决策遗憾评测结果与 §1.4 判读"
```

---

## Self-Review 记录

- Spec 覆盖:C1→Task 2,C2→Task 4,C3→Task 3+Task 5 Step 1,§1.1 工具正确性→Task 1 测试,§1.2 可复现→eval 模式无随机 + 确定性 oracle,§1.3 产出→Task 4 报告字段,§1.4→Task 5 Step 4,§4 两个测试→Task 1,§5 风险(theta 公平性/负遗憾诊断/指纹)→已入代码与注释。спec 中数据集路径原写 `Experiments/regret_dataset_*.pt`,计划改放 `Experiments/Parcelpoint_py/`(该目录已被 .gitignore 忽略,满足"不入 git"意图,路径差异在此记录)。
- 占位符扫描:无 TBD/TODO;所有代码块完整。
- 类型/签名一致性:`regret_for_sample(agent, entry, c_hat_t)`、`load_dataset(filepath, expect_meta)`、`build_argv(policy, seed, scaler, train_episodes, buffer_size, extra)` 在 Task 1/2/4 间一致;checkpoint 路径三处一致(`Checkpoints/supervised_ml.pt`)。
