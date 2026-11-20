# SPO 正确性修复 + 有效性验证(需求 3)— 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 DSPO_plus_SPO 的四类缺陷(缺失方法/静默失败、外部项加法、缺 parser 参数、死代码),写对齐审计文档,用最小消融验证净利润排序 SPO > DSPO > StaticPricing > NoPricing。

**Architecture:** 修复集中在 `Src/Algorithms/DSPO_plus_SPO.py`(F1/F2/F4)与 `Src/parser.py`(F3);实验复用需求 2 的三条基线(逐 seed 数据内嵌),只新跑 SPO-ablation(α=0)与 SPO(α=0.7)两组;SPO+ 生效证据由响亮失败计数 + 日志 grep 提供,是一切对比的先决门槛。

**Tech Stack:** Python 3.9(conda py39)、PyTorch、hygese;纯 python assert 测试脚本。

**对应规格:** `docs/superpowers/specs/2026-07-25-req3-spo-alignment-design.md`(含 F2 范围修正与 α=0 委托父类的核实注记)。

## Global Constraints

- 分支 `req3-spo-alignment`;仓库根 `D:\project\python\ooh_code`;解释器 `C:\Users\wwpst\miniforge3\envs\py39\python.exe`(下文 `python` 均指它),测试从仓库根运行。
- 现有 9 个测试(tests/ 全部)每个任务提交前必须全过。
- 除法形式锚点:V₀=None 与 V₀=0 的定价必须逐位一致;子类(l0_home=l_mp=0、spo_enabled=True)与父类 DSPO 在 V₀=None 下定价一致。
- `sum_mnl_price`(:386/:555 区域)是选择概率分母记录量,**保留加法**,不得改动。
- α=0 时 `get_action_pricing`/`update` 开头委托父类的行为(:306-307/:614-615)不得破坏。
- 失败必须响亮:修复后不允许存在任何"吞掉异常只打一行"的路径;失败计数器必须可从日志核查。
- 实验参数沿用需求 2:V₀=0、home_failure=0、scaler=1800、seeds 1234/2345/3456、500 ep/seed、buffer 10000、评测 20 ep/seed。
- 如实报告,不调参硬凑。

---

### Task 0: 环境自检(控制器内联)

- [ ] **Step 1:** `git branch --show-current` → `req3-spo-alignment`;9 个既有测试全 `OK`。

---

### Task 1: F1 — `_get_terminal_label_route` + 响亮失败计数

**Files:**
- Modify: `Src/Algorithms/DSPO_plus_SPO.py`(__init__ 计数器、新方法、:777 与 :937 两处 except)
- Test: `tests/test_terminal_label_route.py`

**Interfaces:**
- Consumes: 父类 `DSPO.reopt_HGS_final(data) -> (fleet, cost)`。
- Produces: `DSPO_plus_SPO._get_terminal_label_route(self, data, route_ids) -> (fleet, cost)`;实例属性 `self.spo_label_failures: int`、`self.spo_loss_failures: int`(__init__ 置 0);失败打印格式 `[SPO+错误 #k] 标签收集失败(共{k}次): {ExceptionType}: {e}` 与 `[SPO+ warning #k] SPO+ loss computation failed(共{k}次): {e}`——Task 6 的脚本按 `SPO+错误` / `SPO+ warning` 子串 grep 判门槛。

- [ ] **Step 1: 写失败测试 `tests/test_terminal_label_route.py`**

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `python tests\test_terminal_label_route.py`
Expected: `AttributeError: ... has no attribute '_get_terminal_label_route'`(正是生产缺陷本身)。

- [ ] **Step 3: 实现**

(a) `__init__` 中 `self.use_travel_time_prediction = False`(:90)之后加:

```python
        self.spo_label_failures = 0
        self.spo_loss_failures = 0
```

(b) 在 `update` 方法定义之前(:607 上方)加新方法:

```python
    def _get_terminal_label_route(self, data, route_ids):
        """Terminal label route = the HGS re-optimized end-of-day route -- the same anchor
        the parent's Huber labels use (design decision 2026-07-25, spec F1 / plan A).
        route_ids is validated against data['id'] but takes no part in the solve."""
        assert np.array_equal(np.asarray(route_ids).ravel(), np.asarray(data['id']).ravel()), \
            "route_ids out of sync with data['id']"
        return self.reopt_HGS_final(data)
```

(c) :777–778 的 except 块改为(保留后面的 episode-data 重置):

```python
            except Exception as e:
                self.spo_label_failures += 1
                print(f"  [SPO+错误 #{self.spo_label_failures}] 标签收集失败(共{self.spo_label_failures}次): {type(e).__name__}: {e}")
```

(d) :937–940 的 except 块改为:

```python
            except Exception as e:
                self.spo_loss_failures += 1
                print(f"[SPO+ warning #{self.spo_loss_failures}] SPO+ loss computation failed(共{self.spo_loss_failures}次): {type(e).__name__}: {e}")
                spo_loss = None
                effective_spo_weight = 0.0
```

- [ ] **Step 4: 运行测试确认通过 + 回归**

Run: `python tests\test_terminal_label_route.py` → `OK`;9 个既有测试 → 全 `OK`。

- [ ] **Step 5: Commit**

```powershell
git add Src/Algorithms/DSPO_plus_SPO.py tests/test_terminal_label_route.py
git commit -m "Define _get_terminal_label_route (delegate to reopt_HGS_final); loud failure counters"
```

---

### Task 2: F2 — 外部项除法修正(两处价格不动点)

**Files:**
- Modify: `Src/Algorithms/DSPO_plus_SPO.py`(:491–495 与 `_pricing_oracle_lifted_torch` :158–172)
- Test: `tests/test_spo_outside_division.py`

**Interfaces:**
- Consumes: 父类 `DSPO.get_action_pricing`(除法形式,需求 2);`_safe_exp`、`lambertw_torch`。
- Produces: 子类非 initial 定价与 oracle 的价格不动点均为除法形式;`sum_mnl_price`(:386/:555)与 oracle 的 `choice_terms`(:194–195)保持加法**不动**。

- [ ] **Step 1: 写失败测试 `tests/test_spo_outside_division.py`**

```python
import sys
from os import path
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
import numpy as np
import torch
from types import SimpleNamespace
from Src.Algorithms.DSPO import DSPO
from Src.Algorithms.DSPO_plus_SPO import DSPO_plus_SPO
from Environments.OOH.containers import Location, ParcelPoint, Customer

def stub_common(obj, outside_util):
    obj.initial_phase = False
    obj.load_data = False
    obj.init_theta = 0.75
    obj.cool_theta = 0.0
    obj.cost_multiplier = 0.01
    obj.base_util = 0.0
    obj.revenue = 50.0
    obj.min_p, obj.max_p = -1e9, 1e9
    obj.config = SimpleNamespace(outside_option_util=outside_util)
    obj.get_feature_rep_infer = lambda fleet: torch.zeros((1, 1, 2, 2))
    obj.get_prediction = lambda feat, home, pps: [0.0, 7.0, 3.0]
    obj.cheapestInsertionCosts = lambda loc, fleet: 600.0
    obj.mnl = lambda cust, pp: 0.0

def make_state():
    home = Location(0.0, 0.0, 1, 0)
    cust = Customer(home, -0.25, 3.2, 120.0, 1)
    pp = ParcelPoint(Location(5.0, 5.0, 2, 0), 1000000, 2)
    return [cust, None, {"parcelpoints": np.array([pp])}, 0]

def parent_prices(outside_util):
    obj = DSPO.__new__(DSPO)
    stub_common(obj, outside_util)
    return DSPO.get_action_pricing(obj, make_state(), training=False)

def child_prices(outside_util):
    obj = DSPO_plus_SPO.__new__(DSPO_plus_SPO)
    stub_common(obj, outside_util)
    obj.spo_enabled = True
    obj.l0_home = 0.0
    obj.l_mp = 0.0
    obj.use_travel_time_prediction = False
    obj.spo_current_episode_data = {'pricing_decisions': [], 'predicted_costs': [],
                                    'true_costs': [], 'customer_ids': [], 'arrival_times': []}
    return DSPO_plus_SPO.get_action_pricing(obj, make_state(), training=False)

def main():
    # 锚点 1:V0=None 时子类(spo_enabled)与父类定价一致(l0=l_mp=0)
    assert np.allclose(parent_prices(None), child_prices(None), atol=0.011), \
        f"{parent_prices(None)} vs {child_prices(None)}"
    # 锚点 2:V0=0 与 None 一致(除法形式)
    assert np.allclose(child_prices(0.0), child_prices(None), atol=1e-9)
    # 方向:V0=2 时子类与父类同向、同幅统一下移
    p2, c2 = parent_prices(2.0), child_prices(2.0)
    assert np.allclose(p2, c2, atol=0.011), f"{p2} vs {c2}"
    shift = child_prices(0.0) - c2
    assert np.all(shift > 0.1), "prices must drop when outside option strengthens"

    # oracle:价格不动点除法 + 概率 softmax 保留外部项(P 之和 < 1)
    obj = DSPO_plus_SPO.__new__(DSPO_plus_SPO)
    stub_common(obj, 0.0)
    costs = torch.tensor([7.0, 3.0], dtype=torch.float64)
    cust_info = {'base_util': 0.0, 'home_util': 3.2, 'incentive_sensitivity': -0.25, 'revenue': 50.0}
    out0 = DSPO_plus_SPO._pricing_oracle_lifted_torch(obj, costs, cust_info, [{'util': 0.0}])
    obj.config = SimpleNamespace(outside_option_util=None)
    out_none = DSPO_plus_SPO._pricing_oracle_lifted_torch(obj, costs, cust_info, [{'util': 0.0}])
    # 锚点:oracle 在 V0=0 与 None 下价格一致
    assert torch.allclose(out0['prices'], out_none['prices'], atol=1e-6), \
        f"{out0['prices']} vs {out_none['prices']}"
    # V0=0 时概率含外部项:sum(P) < 1;V0=None 时 sum(P) == 1
    assert float(out0['probs'].sum()) < 0.999, f"probs must leave room for outside: {out0['probs']}"
    assert abs(float(out_none['probs'].sum()) - 1.0) < 1e-6
    # 方向:V0=2 → oracle 价格统一低于 V0=0
    obj.config = SimpleNamespace(outside_option_util=2.0)
    out2 = DSPO_plus_SPO._pricing_oracle_lifted_torch(obj, costs, cust_info, [{'util': 0.0}])
    assert bool((out0['prices'] - out2['prices'] > 0.05).all()), f"{out0['prices']} vs {out2['prices']}"
    print("OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行确认失败**

Run: `python tests\test_spo_outside_division.py`
Expected: 锚点 2 或 oracle 锚点失败(现实现为加法:V₀=0 与 None 结果不同)。
注意:若测试在子类的决策记录(feature_info 构建等)处因桩不足抛 AttributeError,查看该处实际代码并补最小桩
(断言本身不得改);child 的 `spo_current_episode_data` 相关键已在桩里给出。

- [ ] **Step 3: 修改非 initial 分支(:491–495)**

把:

```python
            outside_option_util = getattr(self.config, 'outside_option_util', None)
            if outside_option_util is not None:
                sum_mnl_pred += self._safe_exp(outside_option_util)
            
            lambertw0_pred = (lambertw(sum_mnl_pred/e).real + 1) / state[0].incentiveSensitivity
```

改为:

```python
            #outside option enters the price fixed point in DIVISION form (spec F2; same as DSPO.py):
            #(m-1)e^m = sum/e^{V0}; V0=None/0 are bit-identical anchors. The probability-side
            #sum_mnl_price bookkeeping below keeps the ADDITIVE outside term (true MNL denominator).
            outside_option_util = getattr(self.config, 'outside_option_util', None)
            if outside_option_util is not None:
                lambertw0_pred = (lambertw(sum_mnl_pred/(e*self._safe_exp(outside_option_util))).real + 1) / state[0].incentiveSensitivity
            else:
                lambertw0_pred = (lambertw(sum_mnl_pred/e).real + 1) / state[0].incentiveSensitivity
```

(注意::554–555 的 `if outside_option_util is not None: sum_mnl_price += ...` 依赖本处定义的变量,改动后变量仍在,**该处保留不动**。)

- [ ] **Step 4: 修改 oracle(:158–172)**

把 util_terms 构建与 lambert 行:

```python
        if outside_util is not None:
            util_terms.append(torch.tensor(outside_util, dtype=dtype, device=device))

        terms_t = torch.stack(util_terms)
        tmax = torch.max(terms_t)
        sum_mnl = torch.exp(torch.clamp(tmax, -80.0, 80.0)) * torch.sum(torch.exp(terms_t - tmax))
        if (not torch.isfinite(sum_mnl)) or float(sum_mnl.detach().cpu()) <= 0.0:
            return None

        lambertw0 = (self.lambertw_torch(sum_mnl / torch.tensor(e, dtype=dtype, device=device)) + 1.0) / s
```

改为(util_terms 不再注入外部项;Lambert 自变量除以 exp(V0);choice_terms 的外部项注入 :194–195 **保持不动**):

```python
        #price fixed point in DIVISION form (spec F2): the outside option does NOT join this sum;
        #it divides the Lambert argument instead. choice_terms below keeps the additive outside
        #term because that is the true MNL choice denominator.
        terms_t = torch.stack(util_terms)
        tmax = torch.max(terms_t)
        sum_mnl = torch.exp(torch.clamp(tmax, -80.0, 80.0)) * torch.sum(torch.exp(terms_t - tmax))
        if (not torch.isfinite(sum_mnl)) or float(sum_mnl.detach().cpu()) <= 0.0:
            return None

        lambert_arg = sum_mnl / torch.tensor(e, dtype=dtype, device=device)
        if outside_util is not None:
            lambert_arg = lambert_arg / torch.exp(torch.clamp(torch.tensor(outside_util, dtype=dtype, device=device), -80.0, 80.0))
        lambertw0 = (self.lambertw_torch(lambert_arg) + 1.0) / s
```

同时更新方法 docstring 中 "The outside option enters only through the denominator" 一句为:
`The outside option divides the Lambert argument of the price fixed point (division form) and joins the choice-probability denominator additively; it is not included in w.`

- [ ] **Step 5: 运行测试确认通过 + 回归**

Run: `python tests\test_spo_outside_division.py` → `OK`;既有 10 个测试 → 全 `OK`。

- [ ] **Step 6: Commit**

```powershell
git add Src/Algorithms/DSPO_plus_SPO.py tests/test_spo_outside_division.py
git commit -m "Outside option in DIVISION form at both price fixed points; keep additive choice denominator"
```

---

### Task 3: F3 — parser 补 spo_* 参数 + 权重调度测试

**Files:**
- Modify: `Src/parser.py`(6 个新参数)
- Modify: `Src/Algorithms/DSPO_plus_SPO.py:71-74`(spo_buffer_size 0→回退)
- Test: `tests/test_spo_weight_schedule.py`

**Interfaces:**
- Produces: parser 参数 `--spo_loss_weight 0.7 / --spo_warmup_episodes 5 / --spo_rampup_episodes 10 / --spo_batch_size 32 / --spo_buffer_size 0 / --spo_label_sample_size 0`(类型 float/int/int/int/int/int);`__init__` 中 `spo_buffer_size` 值 ≤0 时回退 `config.buffer_size`。

- [ ] **Step 1: 写失败测试 `tests/test_spo_weight_schedule.py`**

```python
import sys
from os import path
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
from Src.parser import Parser
from Src.Algorithms.DSPO_plus_SPO import DSPO_plus_SPO

def make_obj(warmup=5, rampup=10, max_w=0.7, data_len=1, episode=0):
    obj = DSPO_plus_SPO.__new__(DSPO_plus_SPO)
    obj.spo_training_data = {i: None for i in range(data_len)}
    obj.spo_warmup_episodes = warmup
    obj.spo_rampup_episodes = rampup
    obj.max_spo_loss_weight = max_w
    obj.spo_current_episode_count = episode
    return obj

def main():
    a = Parser().get_parser().parse_args([])
    assert (a.spo_loss_weight, a.spo_warmup_episodes, a.spo_rampup_episodes,
            a.spo_batch_size, a.spo_buffer_size, a.spo_label_sample_size) == (0.7, 5, 10, 32, 0, 0)
    # warmup 内为 0
    assert DSPO_plus_SPO._get_current_spo_weight(make_obj(episode=4)) == 0.0
    # rampup 中点线性
    w = DSPO_plus_SPO._get_current_spo_weight(make_obj(episode=10))
    assert abs(w - 0.7 * 0.5) < 1e-9, w
    # 封顶
    assert abs(DSPO_plus_SPO._get_current_spo_weight(make_obj(episode=100)) - 0.7) < 1e-9
    # 无数据恒 0
    assert DSPO_plus_SPO._get_current_spo_weight(make_obj(data_len=0, episode=100)) == 0.0
    print("OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行确认失败**

Run: `python tests\test_spo_weight_schedule.py`
Expected: `AttributeError: 'Namespace' object has no attribute 'spo_loss_weight'`。

- [ ] **Step 3: 实现**

`Src/parser.py` 在 `DSPO_parameters` 方法末尾(`--dropout` 行之后)加:

```python
        # SPO+ (DSPO_plus_SPO) training knobs -- defaults mirror the class's historical getattr fallbacks
        parser.add_argument("--spo_loss_weight", default=0.7, help="max SPO+ loss weight alpha in the Huber/SPO+ blend; 0 disables SPO+ (ablation switch)", type=float)
        parser.add_argument("--spo_warmup_episodes", default=5, help="episodes with alpha=0 before rampup", type=int)
        parser.add_argument("--spo_rampup_episodes", default=10, help="episodes to ramp alpha linearly to its max", type=int)
        parser.add_argument("--spo_batch_size", default=32, help="SPO+ loss batch size", type=int)
        parser.add_argument("--spo_buffer_size", default=0, help="SPO+ training data buffer; <=0 means use --buffer_size", type=int)
        parser.add_argument("--spo_label_sample_size", default=0, help="per-episode cap on labeled decisions; 0 = label all", type=int)
```

`DSPO_plus_SPO.__init__` 的 :71–74 改为(缺失→回退 变成 值≤0→回退):

```python
        spo_buffer_size = getattr(config, 'spo_buffer_size', None)
        if spo_buffer_size is None or int(spo_buffer_size) <= 0:
            spo_buffer_size = config.buffer_size
        self.spo_training_data_max_len = int(spo_buffer_size)
```

- [ ] **Step 4: 运行测试确认通过 + 回归**(12 个测试全 `OK`)

- [ ] **Step 5: Commit**

```powershell
git add Src/parser.py Src/Algorithms/DSPO_plus_SPO.py tests/test_spo_weight_schedule.py
git commit -m "Expose spo_* training knobs on the CLI; schedule test"
```

---

### Task 4: F4 — 死代码清理(建议 sonnet 实现者)

**Files:**
- Modify: `Src/Algorithms/DSPO_plus_SPO.py`(只删不加,除注释)

**Interfaces:**
- Consumes: Tasks 1–3 之后的文件状态。
- Produces: 下列符号从文件中消失(grep 零匹配):`_pricing_oracle_lifted_np`、`_calculate_pricing_torch_spo`、`SPOExperimentResult`、`spo_results`、`_calculate_spo_loss_statistics`、`max_discount`、`travel_time_weight`、`use_travel_time_prediction`、`get_travel_time_prediction`、`_utility_home_nonprice`、`set_travel_times`、`travel_times`。行为不变(测试与冒烟保证)。

- [ ] **Step 1: 删除自包含死代码**

逐项删除(先 grep 确认无其他引用,再删):
1. `SPOExperimentResult` dataclass(约 :21–35)及其 import 依赖(`dataclass`/`List` 若仅它使用);
2. `self.spo_results = []`(:61)与 `spo_result = self._calculate_spo_loss_statistics(data)` / `self.spo_results.append(...)` 两行(:707–709)及 `_calculate_spo_loss_statistics` 整个方法(约 :869–902);
3. `_pricing_oracle_lifted_np` 整个方法(约 :206–303,从未调用);
4. `_calculate_pricing_torch_spo` 整个方法(约 :1134–1191,从未调用);
5. 所有 `max_discount = self.min_p` 死赋值(约 :177、:499;:257 在被删的 np oracle 内)。

- [ ] **Step 2: 删除永假 travel-time 分支**

机械变换,每处相同:
- `if customerchoice_model is not None and getattr(customerchoice_model, 'travel_time_weight', None) is not None: <A> else: <B>` → 只留 `<B>`(约 8 处:348、371、392、450、473、520、544、563 附近;行号以 Task 1–3 后实际为准,按内容定位);
- `if self.use_travel_time_prediction: ...` 两个块(约 :331–335、:432 附近)整块删除;`travel_times = None` 的初始化与后续引用一并清理;`self.use_travel_time_prediction = False`(__init__)删除;
- `customerchoice_model` 的获取(:314–317)若清理后仍有读者则保留,无读者则删;
- `_utility_home_nonprice(...)` 调用全部内联为 `base_util + home_util`(或 `self.base_util + state[0].home_util`),模块级函数定义(:15–18)删除;oracle :157 处 `home_nonprice = _utility_home_nonprice(base_util, home_util, None, None)` → `home_nonprice = base_util + home_util`。

- [ ] **Step 3: 验证**

Run: `grep -n -E "_pricing_oracle_lifted_np|_calculate_pricing_torch_spo|SPOExperimentResult|spo_results|_calculate_spo_loss_statistics|max_discount|travel_time_weight|use_travel_time_prediction|_utility_home_nonprice|set_travel_times" Src/Algorithms/DSPO_plus_SPO.py`
Expected: 无输出。
Run: `python -c "import sys; sys.path.insert(0,r'D:\project\python\ooh_code'); from Src.Algorithms.DSPO_plus_SPO import DSPO_plus_SPO; print('import OK')"` → `import OK`
Run: 全部 12 个测试 → `OK`(尤其 test_spo_outside_division.py 的父子一致锚点,是清理未改变行为的最强证据)。

- [ ] **Step 4: Commit**

```powershell
git add Src/Algorithms/DSPO_plus_SPO.py
git commit -m "Remove dead code from DSPO_plus_SPO (unused oracles, legacy diagnostics, always-false branches)"
```

---

### Task 5: 对齐审计文档

**Files:**
- Create: `docs/req3_spo_alignment_audit.md`

- [ ] **Step 1: 写文档(完整内容如下,行号以 Task 4 完成后实际核对微调)**

```markdown
# SPO+ 实现与论文/手稿对齐审计(需求 3)

日期:2026-07-25。对象:`Src/Algorithms/DSPO_plus_SPO.py`(req3-spo-alignment 分支修复后)。
参照:Elmachtoub & Grigas (2022) "Smart Predict-then-Optimize"(paper/smart_predict_then_optimize_paper.md);
手稿 §4.2.4(paper/manuscript/part_c模版_DRT_0522_manuscript_working.md Eq.(20)–(32))。

## 1. 公式对照(结论:等价 ✓)

| 项 | 论文/手稿 | 代码 | 结论 |
|---|---|---|---|
| SPO+ 损失 | ℓ_SPO+(ĉ,c)=max_w{(c−2ĉ)ᵀw}+2ĉᵀw*(c)−z*(c) | `y·(w_pseudo−w_true)+2ŷ·(w_true−w_pseudo)`,w_pseudo=w*(2ĉ−c) | 代数等价(以 max_w(y−2ŷ)ᵀw=(y−2ŷ)ᵀw*(2ŷ−y) 代入即得) |
| lifted 决策 | w(a)=[P(a);−R0(a)],y=[c;1] | `lifted=cat([probs,−r0])`,y/ŷ 增广 1 | 一致(手稿 Eq.23–26) |
| 次梯度 | 2(w*(y)−w*(2ŷ−y))∈∂ℓ_SPO+ | 由 loss 表达式对 ŷ 自动微分,梯度只经 ŷ 项 | 一致(常数项不带梯度) |
| 混合目标 | L=α·ℓ_SPO+ + (1−α)·L_Huber(Eq.32) | `self_supervised_update` 分次 backward,权重同式 | 一致 |
| α 调度 | 预热 α=0 后逐步升高 | warmup(5 ep)→线性 rampup(10 ep)→封顶 0.7 | 一致(线性化实现) |
| 真实成本 c | option-level marginal cost(终局路线) | `_get_terminal_label_route`=HGS 终局重优化,选中项 HEP 摊销、未选项最便宜插入+服务成本 | 一致(方案 A,与 Huber 标签同锚;用户 2026-07-25 拍板) |
| oracle w*(·) | argmax_a R(a;c) | Lambert 闭式解(含外部项除法不动点)+ [min_p,max_p] 裁剪 + safety_margin | 边界内精确;近边界为近似(见 §2) |

## 2. 记录在案的偏差(不修,含理由)

1. **/(K+1) 归一化**:代码把 SPO+ 损失除以 (K+1)(选项数+1)。canonical 公式无此项;为纯缩放,
   与 α 合并等效(α_eff = α/(K+1) 相对 Huber)。保留,手稿实现说明中注明。
2. **oracle 边界近似**:价格不动点是无约束闭式解,随后 clip 到 [min_p,max_p] 并对负价乘 (1−safety_margin=0.9);
   价格落在界内时为精确 argmax,压边界时为近似。与部署定价规则完全同形,故"oracle 与部署规则一致"这一
   SPO 的关键前提成立。
3. **失败成本不对称**:终局标签的 home 选项含 home_failure×failure_cost,在线预测不含;本轮实验口径
   home_failure=0,无影响。若未来恢复失败成本,须先统一口径。

## 3. 修复摘要(本分支)

- F1:补定义 `_get_terminal_label_route`(委托 `reopt_HGS_final`,终局路线锚);修复前该方法缺失、异常被吞,
  **SPO+ 从未参与过任何梯度更新**(此前所有 DSPO_plus_SPO 结果实为纯 Huber);静默失败改为响亮计数。
- F2:外部选项在两处价格不动点改除法形式((m−1)e^m=Σ/e^{V0},推导见需求 2 规格);选择概率分母保留加法
  (真实 MNL);修复前加法形式方向相反(外部越诱人定价越高)。
- F3:spo_* 六个训练超参暴露到 CLI(此前只有 getattr 硬编码默认,无法消融)。
- F4:删除从未调用的重复 oracle/定价辅助、遗留诊断(自注"非论文级指标")、永假 travel-time 分支,
  缩小审计面;父子定价一致性测试锚定行为未变。

## 4. 生效证据要求(实验门槛)

任何 DSPO_plus_SPO 结果可信的前提:训练日志中 `[SPO+错误` 与 `[SPO+ warning` 计数为 0、
`spo_training_data populated` 出现、`spo_weight became positive` 出现且值达 α 上限。
`Experiments/run_req3_spo.py` 自动核查并写入结果文档。
```

- [ ] **Step 2: Commit**

```powershell
git add docs/req3_spo_alignment_audit.md
git commit -m "Add SPO+ alignment audit document"
```

---

### Task 6: 实验入口 `Experiments/run_req3_spo.py` + build_argv extra 后置

**Files:**
- Modify: `Experiments/run_paper_repro.py`(build_argv 中 extra 应用移到 run-specific 键之后)
- Create: `Experiments/run_req3_spo.py`

**Interfaces:**
- Consumes: `run_paper_repro.run_one/agg`;`run_req2_profit.req2_extra`;eval2 metrics 键。
- Produces: CLI `python Experiments\run_req3_spo.py --mode {smoke,full} [--train_episodes N] [--eval_episodes N] [--buffer_size N] [--seeds ...]`;full 写 `docs/req3_spo_results.md`。

- [ ] **Step 1: `run_paper_repro.build_argv` 调整**

把 `if extra: over.update(extra)` 从 `over.update(POLICIES[policy])` 之后**移动到** `over['folder_suffix'] = ...` 赋值行之后(即 run-specific 键之后、argv 构建之前),使 extra 可覆盖 folder_suffix/algo_name 等。既有调用(req2 的 extra 只含 home_failure/outside_option_util,不与 run-specific 键重叠)行为不变。

- [ ] **Step 2: 写 `Experiments/run_req3_spo.py`(完整内容)**

```python
"""Requirement 3 runner: SPO+ effectiveness with minimal ablation (spec:
docs/superpowers/specs/2026-07-25-req3-spo-alignment-design.md).

Reuses requirement-2 per-seed baseline metrics (embedded below, source: req2 full run
2026-07-25, V0=0, seeds 1234/2345/3456) and runs only the two DSPO_plus_SPO variants.
"""
import sys
import re
import argparse
from os import path

ROOT = path.abspath(path.join(path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from Experiments.run_paper_repro import run_one, agg  # noqa: E402
from Experiments.run_req2_profit import req2_extra, SCALER  # noqa: E402

# per-seed metrics from docs/req2_profit_results.md's underlying run (seeds 1234, 2345, 3456)
REQ2_BASELINES = {
    'NoPricing': [
        {'net_profit': 1897.33, 'total_costs': 2270.17, 'quit_rate': 0.0308, 'pct_home': 0.852},
        {'net_profit': 1829.89, 'total_costs': 2160.11, 'quit_rate': 0.0339, 'pct_home': 0.872},
        {'net_profit': 1859.34, 'total_costs': 2190.66, 'quit_rate': 0.0334, 'pct_home': 0.861}],
    'StaticPricing': [
        {'net_profit': 1978.39, 'total_costs': 2179.11, 'quit_rate': 0.0331, 'pct_home': 0.570},
        {'net_profit': 1877.54, 'total_costs': 2089.96, 'quit_rate': 0.0393, 'pct_home': 0.559},
        {'net_profit': 1920.15, 'total_costs': 2127.35, 'quit_rate': 0.0340, 'pct_home': 0.575}],
    'DSPO': [
        {'net_profit': 1905.60, 'total_costs': 2021.90, 'quit_rate': 0.0559, 'pct_home': 0.742},
        {'net_profit': 1950.66, 'total_costs': 2066.84, 'quit_rate': 0.0586, 'pct_home': 0.737},
        {'net_profit': 1957.58, 'total_costs': 2077.42, 'quit_rate': 0.0561, 'pct_home': 0.742}],
}
V0 = 0.0
VARIANTS = {
    'SPO-ablation': {'algo_name': 'DSPO_plus_SPO', 'spo_loss_weight': '0.0'},
    'SPO':          {'algo_name': 'DSPO_plus_SPO', 'spo_loss_weight': '0.7'},
}
ORDER_BY_PROFIT_DESC = ['SPO', 'DSPO', 'StaticPricing', 'NoPricing']


def variant_extra(name, seed):
    extra = dict(req2_extra(V0))
    extra.update(VARIANTS[name])
    extra['folder_suffix'] = 'req3_{}_{}'.format(name.replace('-', '_'), seed)
    return extra


def check_spo_gate(name, seed, expect_active):
    """SPO+ 生效门槛:日志无失败;active 变体须出现权重/数据两条 debug 行。"""
    log = path.join(ROOT, 'Experiments', 'Parcelpoint_py', 'pricing', 'DSPO_plus_SPO',
                    'runreq3_{}_{}'.format(name.replace('-', '_'), seed), str(seed), 'Logs', 'logfile.log')
    if not path.isfile(log):
        return {'ok': False, 'why': 'logfile missing: {}'.format(log)}
    text = open(log, 'r', encoding='utf-8', errors='replace').read()
    n_err = len(re.findall(r'\[SPO\+错误', text)) + len(re.findall(r'\[SPO\+ warning', text))
    has_data = 'spo_training_data populated' in text
    has_weight = 'spo_weight became positive' in text
    if n_err > 0:
        return {'ok': False, 'why': '{} failure lines in log'.format(n_err)}
    if expect_active and not (has_data and has_weight):
        return {'ok': False, 'why': 'missing debug evidence (data={}, weight={})'.format(has_data, has_weight)}
    if not expect_active and (has_data or has_weight):
        return {'ok': False, 'why': 'ablation unexpectedly ran SPO+ (data={}, weight={})'.format(has_data, has_weight)}
    return {'ok': True, 'why': 'clean'}


def run_variants(opt):
    results = {name: [] for name in VARIANTS}
    gates = {name: [] for name in VARIANTS}
    for seed in opt.seeds:
        for name in VARIANTS:
            print('running {} seed {} ...'.format(name, seed))
            m = run_one('DSPO', seed, SCALER, opt.train_episodes, opt.eval_episodes,
                        opt.buffer_size, extra=variant_extra(name, seed))
            g = check_spo_gate(name, seed, expect_active=(name == 'SPO'))
            print('  -> net_profit={:.2f} quit%={:.2f} home%={:.1f} gate={} ({}) ({:.0f}s)'.format(
                m['net_profit'], 100*m['quit_rate'], 100*m['pct_home'],
                'PASS' if g['ok'] else 'FAIL', g['why'], m['wall_total_s']))
            results[name].append(m)
            gates[name].append(g)
    return results, gates


def write_report(results, gates, opt):
    allrows = dict(REQ2_BASELINES)
    allrows.update(results)
    lines = []
    lines.append('# 需求 3 结果:SPO+ 有效性与最小消融(RC 合成案例)\n')
    lines.append('- 预设:需求 2 口径(V0=0, home_failure=0, scaler={});seeds={};DSPO_plus_SPO 训练 {} ep,buffer {};评测 {} ep/seed;NoPricing/StaticPricing/DSPO 三行复用需求 2 逐 seed 结果\n'.format(
        SCALER, opt.seeds, opt.train_episodes, opt.buffer_size, opt.eval_episodes))
    lines.append('| 策略 | 净利润 | 退出率% | home% | total costs |')
    lines.append('|---|---|---|---|---|')
    for p in ['SPO', 'SPO-ablation', 'DSPO', 'StaticPricing', 'NoPricing']:
        rows = allrows[p]
        def m(key):
            vals = [r[key] for r in rows]
            mu = sum(vals)/len(vals)
            sd = (sum((v-mu)**2 for v in vals)/len(vals)) ** 0.5
            return mu, sd
        np_m, np_s = m('net_profit')
        lines.append('| {} | {:.1f}±{:.1f} | {:.2f}±{:.2f} | {:.1f} | {:.1f}±{:.1f} |'.format(
            p, np_m, np_s, 100*m('quit_rate')[0], 100*m('quit_rate')[1],
            100*m('pct_home')[0], m('total_costs')[0], m('total_costs')[1]))
    def mean(p, key):
        rows = allrows[p]
        return sum(r[key] for r in rows)/len(rows)
    def std(p, key):
        rows = allrows[p]
        mu = mean(p, key)
        return (sum((r[key]-mu)**2 for r in rows)/len(rows)) ** 0.5
    # 排序判定(4 个策略,消融组不参与主排序)
    main_policies = ['SPO', 'DSPO', 'StaticPricing', 'NoPricing']
    means = {p: mean(p, 'net_profit') for p in main_policies}
    actual = sorted(means, key=means.get, reverse=True)
    lines.append('\n- 目标排序(净利润降序):{}'.format(' > '.join(ORDER_BY_PROFIT_DESC)))
    lines.append('- 实际排序(净利润降序):{}'.format(' > '.join(actual)))
    lines.append('- 排序判定:{}'.format('**成立**' if actual == ORDER_BY_PROFIT_DESC else '**不成立(如实报告)**'))
    # 消融判定
    diff_ab = abs(mean('SPO-ablation', 'net_profit') - mean('DSPO', 'net_profit'))
    noise = max(std('SPO-ablation', 'net_profit'), std('DSPO', 'net_profit'))
    gain = mean('SPO', 'net_profit') - mean('SPO-ablation', 'net_profit')
    lines.append('\n## 消融判定')
    lines.append('- |SPO-ablation − DSPO| = {:.1f},噪声级(较大 std)= {:.1f} → 管线一致性 {}'.format(
        diff_ab, noise, '**PASS**' if diff_ab < noise else '**FAIL(需先查管线差异)**'))
    lines.append('- SPO − SPO-ablation = {:+.1f} → SPO+ 增益 {}'.format(
        gain, '**PASS**' if gain > 0 else '**FAIL(如实报告)**'))
    # 生效门槛
    lines.append('\n## SPO+ 生效门槛(先决条件)')
    for name in VARIANTS:
        for seed, g in zip(opt.seeds, gates[name]):
            lines.append('- {} seed {}: {} ({})'.format(name, seed, 'PASS' if g['ok'] else 'FAIL', g['why']))
    out = path.join(ROOT, 'docs', 'req3_spo_results.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('report written to', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['smoke', 'full'])
    ap.add_argument('--train_episodes', type=int, default=500)
    ap.add_argument('--eval_episodes', type=int, default=20)
    ap.add_argument('--buffer_size', type=int, default=10000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1234, 2345, 3456])
    opt = ap.parse_args()
    if opt.mode == 'smoke':
        opt.seeds = opt.seeds[:1]
        opt.train_episodes = min(opt.train_episodes, 140)
        opt.eval_episodes = min(opt.eval_episodes, 2)
        results, gates = run_variants(opt)
        for name in VARIANTS:
            print('smoke {}: gate {}'.format(name, 'PASS' if all(g['ok'] for g in gates[name]) else 'FAIL'))
    else:
        results, gates = run_variants(opt)
        write_report(results, gates, opt)


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: 语法/导入检查 + 回归**

Run: `python -c "import ast; ast.parse(open(r'Experiments\run_req3_spo.py',encoding='utf-8').read()); print('syntax OK')"` → `syntax OK`
Run: `python -c "import sys; sys.path.insert(0,r'D:\project\python\ooh_code'); from Experiments.run_req3_spo import VARIANTS, REQ2_BASELINES, variant_extra; print(list(VARIANTS), len(REQ2_BASELINES['DSPO']), variant_extra('SPO', 1234)['folder_suffix'])"`
Expected: `['SPO-ablation', 'SPO'] 3 req3_SPO_1234`
Run: 全部 12 个测试 → `OK`。

- [ ] **Step 4: Commit**

```powershell
git add Experiments/run_paper_repro.py Experiments/run_req3_spo.py
git commit -m "Add requirement-3 runner (SPO variants + gate checks); build_argv extra applies last"
```

---

### Task 7: 冒烟 + 生效门槛验证(控制器执行)

- [ ] **Step 1:** Run: `python Experiments\run_req3_spo.py --mode smoke`(约 15–20 分钟:两个变体各 140 训练 ep)
Expected: 两行 gate PASS——SPO 变体日志出现 `spo_training_data populated` 与 `spo_weight became positive`、零失败行;ablation 变体零 SPO+ 痕迹。
- [ ] **Step 2:** 若 gate FAIL:按日志失败行诊断(常见:标签匹配零命中、oracle 返回 None 比例过高),修复后重跑冒烟;不得带病进全量。
- [ ] **Step 3:** 记录 SPO 变体单训练 episode 耗时(全量时长依据)。

### Task 8: 全量跑批 → `docs/req3_spo_results.md` → 验收(控制器执行)

- [ ] **Step 1:** Run: `python Experiments\run_req3_spo.py --mode full --train_episodes 500 --buffer_size 10000 --eval_episodes 20 --seeds 1234 2345 3456`(预计 3–4.5 小时)
- [ ] **Step 2:** 核对报告并手工追加:验收标准五条逐条判定(生效门槛/排序/退出率 SPO≤DSPO+2pt/消融两条/审计成文)、局限性段(快速档、消融组与 DSPO 逐位一致性核对结论)。
- [ ] **Step 3:** Commit 报告;按五条验收向用户汇报,排序或消融不成立时附机制分析,不调参硬凑。
