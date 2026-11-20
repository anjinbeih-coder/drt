# 外部效用 + 净利润验证(需求 2)— 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 MNL 选择模型中加入外部(退出)选项,删除上门失败成本,验证 3 seed 均值净利润 DSPO > StaticPricing > NoPricing。

**Architecture:** 退出选项经 `--outside_option_util`(默认 None=关闭,现有行为逐位不变)进入 `customerchoice_pricing` 与 `env.step`(退出哨兵 idx=-2);DSPO 定价以除法形式感知 V₀(Lambert 自变量 ÷ exp(V₀),V₀=0 与现有公式逐位等价);训练侧退出步跳过特征存储;eval2 新增 quit_rate/net_profit;薄壳入口复用需求 1 的 preset 与 run_one。

**Tech Stack:** Python 3.9(conda py39)、PyTorch、hygese、numpy;纯 python assert 测试脚本(无 pytest)。

**对应规格:** `docs/superpowers/specs/2026-07-25-req2-outside-option-design.md`(含 2026-07-25 除法形式修正)。

## Global Constraints

- 分支:`req2-outside-option`;仓库根 `D:\project\python\ooh_code`;解释器(唯一可用):`C:\Users\wwpst\miniforge3\envs\py39\python.exe`,下文所有 `python` 均指它;测试从仓库根运行。
- **默认关闭等价性**:`--outside_option_util` 缺省(None)时,一切现有行为逐位不变;需求 1 的 7 个既有测试(tests/ 下全部)每个任务提交前必须全过。
- 退出返回约定:`customerchoice_pricing` 退出时返回 `(None, False, -2, 0)`(idx=-2 哨兵;home 是 -1)。
- DSPO 感知形式(规格修正后):`lambertw0 = (lambertw(sum_mnl/(e*exp(V0))).real+1)/βᵈ`;V₀=0 必须与 None 路径结果完全一致。
- stats 元组只在**尾部追加** `count_quit`(索引 8),索引 0–7 不动(`run.py::train` 依赖 stats[1,2,3,6])。
- 守卫:V₀ 非 None 时 `pricing=False` 或 `offer_mode != 'all'` 在环境构造时 ValueError。
- 实验参数:scaler=1800、seeds 1234/2345/3456、DSPO 500 ep/seed、buffer 10000、评测 20 ep/seed、`home_failure=0.0`;V₀ 校准目标 = NoPricing 退出率 2%–5%。
- 报告实验结果如实呈现,排序不成立不得调参硬凑。

---

### Task 0: 环境自检(控制器内联执行)

**Files:** 无修改。

- [ ] **Step 1:** `git branch --show-current` → `req2-outside-option`;`git log --oneline -1` → d8a3543(或其后)。
- [ ] **Step 2:** 跑既有 7 个测试:`for t in tests/test_*.py; do python $t; done` → 全部 `OK`(基线绿)。

---

### Task 1: 退出选项(选择模型 + 环境 + 接线 + 守卫)

**Files:**
- Modify: `Environments/OOH/customerchoice.py`(构造签名 + customerchoice_pricing)
- Modify: `Environments/OOH/Parcelpoint_py.py`(构造签名、守卫、reset、step)
- Modify: `Src/parser.py`(新增 `--outside_option_util`)
- Modify: `Src/config.py`(get_domain 两处透传)
- Test: `tests/test_outside_option.py`

**Interfaces:**
- Consumes: 现有 `customerchoicemodel.__init__(self, base_util, dist_scaler, euclidean, dist_mat, n_cust, beta_k=0.02)`。
- Produces: `customerchoicemodel.__init__(..., beta_k=0.02, outside_util=None)`;`customerchoice_pricing` 退出时返回 `(None, False, -2, 0)`;`Parcelpoint_py.__init__(..., outside_option_util=None)`,quit 步 stats=(steps, count_home, service_time, total_prices, pps, 0, total_discounts, 0, count_quit),正常步 stats 尾部追加 count_quit;parser 参数 `--outside_option_util`(float,默认 None)。

- [ ] **Step 1: 写失败测试 `tests/test_outside_option.py`**

```python
import sys
from os import path
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
import numpy as np
import numpy.ma as ma
from Environments.OOH.customerchoice import customerchoicemodel
from Environments.OOH.containers import Location, ParcelPoint, Customer

def euclid_stub(a, b):
    return ((a.x-b.x)**2 + (a.y-b.y)**2) ** 0.5

def make_model(outside_util):
    return customerchoicemodel(base_util=0.0, dist_scaler=100.0, euclidean=euclid_stub,
                               dist_mat=[], n_cust=5, beta_k=0.02, outside_util=outside_util)

def make_choice_args():
    home = Location(10.0, 10.0, 1, 0)
    cust = Customer(home, -0.25, 3.2, 120.0, 1)
    pps = np.array([ParcelPoint(Location(20.0, 20.0, 5, 0), 1000000, 5)])
    masked = ma.masked_array(pps, mask=[1])
    action = np.zeros(2)
    return cust, masked, action

def main():
    cust, masked, action = make_choice_args()
    # V0=None:关闭,绝不返回退出哨兵
    m = make_model(None)
    for _ in range(200):
        _, _, idx, _ = m.customerchoice_pricing(cust, action, masked)
        assert idx != -2
    # V0 极大:必退出,返回约定 (None, False, -2, 0)
    m = make_model(50.0)
    for _ in range(50):
        loc, accepted, idx, price = m.customerchoice_pricing(cust, action, masked)
        assert loc is None and accepted is False and idx == -2 and price == 0
    # V0 极小:从不退出
    m = make_model(-50.0)
    for _ in range(200):
        _, _, idx, _ = m.customerchoice_pricing(cust, action, masked)
        assert idx != -2

    # env 层:退出步不产生停靠、不计价格统计,stats[8] 计数;守卫生效
    from Environments.OOH.Parcelpoint_py import Parcelpoint_py
    from Src.Utils.Utils import generate_demand_data
    coords = generate_demand_data(100)
    env = Parcelpoint_py(model='Baseline', max_steps_r=5, max_steps_p=0.5, pricing=True,
                         load_data=False, coords=coords, dist_matrix=[], n_parcelpoints=6,
                         adjacency=np.ones((100*100, 6)), service_times=np.ones(100*100),
                         outside_option_util=50.0)
    n_ids_before = np.size(env.data['id'])
    action = np.zeros(7)  # home + 6 个自提点
    _, _, stats, data = env.step(action)
    assert stats[8] == 1, "quit must be counted in stats[8]"
    assert np.size(data['id']) == n_ids_before, "quit must not add a stop"
    assert len(env.total_prices) == 0 and len(env.total_discounts) == 0, "quit must not pollute price stats"
    # 守卫:V0 + pricing=False → ValueError
    try:
        Parcelpoint_py(model='Baseline', max_steps_r=5, max_steps_p=0.5, pricing=False,
                       load_data=False, coords=coords, dist_matrix=[], n_parcelpoints=6,
                       adjacency=np.ones((100*100, 6)), service_times=np.ones(100*100),
                       outside_option_util=0.0)
        assert False, "expected ValueError (pricing=False)"
    except ValueError:
        pass
    # 守卫:V0 + offer_mode!=all → ValueError
    try:
        Parcelpoint_py(model='Baseline', max_steps_r=5, max_steps_p=0.5, pricing=True,
                       load_data=False, coords=coords, dist_matrix=[], n_parcelpoints=6,
                       adjacency=np.ones((100*100, 6)), service_times=np.ones(100*100),
                       offer_mode='home_only', outside_option_util=0.0)
        assert False, "expected ValueError (offer_mode)"
    except ValueError:
        pass
    print("OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行确认失败**

Run: `python tests\test_outside_option.py`
Expected: `TypeError: __init__() got an unexpected keyword argument 'outside_util'`

- [ ] **Step 3: 改 `customerchoice.py`**

(a) 构造签名:`beta_k=0.02` 之后加 `outside_util=None,` 参数;`self.beta_k = beta_k` 之后加:

```python
        self.outside_util = outside_util
```

(b) `customerchoice_pricing` 整体替换为(在价格项之后、Gumbel 之前插入退出行;判定顺序:先查退出哨兵):

```python
    def customerchoice_pricing(self,customer,action,parcelpoints):
        """
        Customer choice model for the pricing decision, i.e., action is vector of prices for all PPs and home delivery
        """
        pps = parcelpoints[parcelpoints.mask].data
        shape = (len(pps)+1, 1)
        utils= np.empty(shape)
        utils[0]=self.base_util+customer.home_util
        for idx,pp in enumerate(pps):
            utils[idx+1] = self.mnl(customer,pp)
        utils = np.add(utils,customer.incentiveSensitivity*action.reshape((len(action),1)))#incentive
        if self.outside_util is not None:#outside option (requirement 2): utility V0, no price term
            utils = np.vstack((utils, [[self.outside_util]]))
        utils = np.add(utils,gumbel(0,1, np.shape(utils)))#mu=0,beta=1 (std Gumbel)

        idx = np.argmax(utils)
        if self.outside_util is not None and idx == len(pps)+1:
            return None, False, -2, 0#customer quits (outside option), -2 sentinel
        if idx==0:
            return customer.home, False, -1, action[0]#home delivery
        else:
            return pps[idx-1].location, True, pps[idx-1].id_num,action[idx]#accept offer
```

- [ ] **Step 4: 改 `Parcelpoint_py.py`**

(a) 构造签名在 `offer_mode='all',` 之后加 `outside_option_util=None,`。

(b) 两处 `customerchoicemodel(...)` 构造的 `beta_k=mnl_beta_k` 后追加 `,outside_util=outside_option_util`。

(c) offer_mode 覆盖块之后追加守卫:

```python
        #outside option (requirement 2): quit semantics only defined for the pricing variant with the full choice set
        self.outside_option_util = outside_option_util
        if outside_option_util is not None:
            if not pricing:
                raise ValueError("outside_option_util requires pricing=True (offering variant has no quit semantics)")
            if offer_mode != 'all':
                raise ValueError("outside_option_util requires offer_mode='all' (NoOOH/OnlyOOH exclude the quit option)")
```

(d) `reset()` 里 `self.used_pps = set()` 之后加:

```python
        self.count_quit = 0
```

(e) `step()` 里紧跟 `loc,accepted_pp,idx,price = self.get_delivery_loc(action)` 之后、`if price>0:` 之前插入退出分支:

```python
        if idx == -2:#customer quits via the outside option: no booking, no stop, no price stats
            self.count_quit += 1
            stats = self.steps,self.count_home_delivery,self.service_time,self.total_prices,self.parcelPoints["parcelpoints"],0,self.total_discounts,0,self.count_quit
            self.curr_state = self.make_state()
            return self.curr_state.copy(), self.is_terminal(), stats, self.data
```

(f) 正常路径的 `stats = self.steps,...,price` 行尾追加 `,self.count_quit`(成为索引 8)。

- [ ] **Step 5: parser / config 接线**

`Src/parser.py` 的 `environment_parameters`,`--offer_mode` 之后:

```python
        parser.add_argument("--outside_option_util", default=None, type=float, help="utility V0 of the outside (quit) option in the pricing MNL; omit/None disables quitting (requirement 2)")
```

`Src/config.py::get_domain` 两处 `obj(...)` 里(offer_mode 透传旁)加:

```python
                      outside_option_util=args.outside_option_util,
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python tests\test_outside_option.py`
Expected: `OK`

- [ ] **Step 7: 回归**

Run: `for t in tests/test_*.py; do python $t; done`(bash)或逐个运行全部 8 个测试
Expected: 全部 `OK`

- [ ] **Step 8: Commit**

```powershell
git add Environments/OOH/customerchoice.py Environments/OOH/Parcelpoint_py.py Src/parser.py Src/config.py tests/test_outside_option.py
git commit -m "Add outside (quit) option to pricing MNL and env step, default off"
```

---

### Task 2: DSPO 感知 V₀(除法形式)+ 训练跳过退出步

**Files:**
- Modify: `Src/Algorithms/DSPO.py`(__init__、get_action_pricing、update)
- Test: `tests/test_dspo_outside_aware.py`、`tests/test_update_skip_quit.py`

**Interfaces:**
- Consumes: Task 1 的 parser 参数 `outside_option_util`(经 Config 成为 `config.outside_option_util`);env 退出步不增长 `data['id']` 的事实。
- Produces: `DSPO.get_action_pricing` 在 V₀ 非 None 时用 `lambertw(sum_mnl/(e*exp(V0)))`;`DSPO.update` 非终止步在 `np.size(data['id'])` 未增长时直接 `return 0.0`(不存特征);`DSPO.__init__` 新属性 `self._last_data_len = 1`。

- [ ] **Step 1: 写失败测试 `tests/test_dspo_outside_aware.py`**

```python
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
    state = [cust, None, {"parcelpoints": np.array([pp])}, 0]
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
```

- [ ] **Step 2: 运行确认失败**

Run: `python tests\test_dspo_outside_aware.py`
Expected: AssertionError(`shift` 全为 0——现实现不感知 V₀,a_zero 与 a_hi 相同)。注意:V0=None 与 V0=0 的第一条断言在改动前就会通过(getattr 尚未存在也没关系,`SimpleNamespace` 已带该属性但代码未读它),真正失败在方向断言。

- [ ] **Step 3: 改 `DSPO.py` 的 get_action_pricing**

把 `#2 obtain lambert w0` 附近的一行:

```python
            lambertw0 = (lambertw(sum_mnl/e).real+1)/state[0].incentiveSensitivity
```

替换为:

```python
            #2 obtain lambert w0. Adapted DSPO (requirement 2): an outside option with utility V0
            #  divides the Lambert argument -- (m-1)e^m = sum/e^{V0}; identical to the original
            #  no-outside formula when V0 is None or 0. Direction check: V0 up -> prices down.
            outside_util = getattr(self.config, 'outside_option_util', None)
            if outside_util is not None:
                lambertw0 = (lambertw(sum_mnl/(e*exp(outside_util))).real+1)/state[0].incentiveSensitivity
            else:
                lambertw0 = (lambertw(sum_mnl/e).real+1)/state[0].incentiveSensitivity
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python tests\test_dspo_outside_aware.py`
Expected: `OK`

- [ ] **Step 5: 写失败测试 `tests/test_update_skip_quit.py`**

```python
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
```

- [ ] **Step 6: 运行确认失败**

Run: `python tests\test_update_skip_quit.py`
Expected: `AttributeError`(`_last_data_len` 未被 update 使用)或第二次 update 后 `features.shape[0] == 2` 的断言失败。

- [ ] **Step 7: 改 `DSPO.py` 的 __init__ 与 update**

(a) `__init__` 中 `self.features = np.empty(...)` 之后加:

```python
        self._last_data_len = 1#depot only; used to detect quit steps (no new stop -> no label -> no feature)
```

(b) `update` 的非终止分支开头(`if not done:` 之后、第一行 vstack 之前)插入:

```python
            n_stops = np.size(data['id'])
            if n_stops == self._last_data_len:#customer quit via outside option: no stop, no cost label
                return 0.0
            self._last_data_len = n_stops
```

(c) done 分支里 `self.features = np.empty(...)`/`self.cap_features = np.empty(...)` 重置处之后加:

```python
            self._last_data_len = 1#next episode's data starts with the depot only
```

- [ ] **Step 8: 运行测试确认通过 + 回归**

Run: `python tests\test_update_skip_quit.py` → `OK`;随后跑全部测试(现在共 10 个)→ 全部 `OK`。

- [ ] **Step 9: 核对 DSPO_plus_SPO 的 update 覆写**

Run: `grep -n "def update\|features = np.vstack\|super().update" Src/Algorithms/DSPO_plus_SPO.py | head`
判定:若其 update 覆写在非终止步直接调用 `super().update(...)`,则自动继承跳过逻辑,无需改动(在报告里记一句);若它**复制**了特征存储逻辑,则在其非终止分支应用与 Step 7(b) 相同的四行守卫(用 `self._last_data_len`,父类 __init__ 已初始化)。V₀ 默认 None 时两种情况行为都不变;需求 3 会实际跑到该类,此处必须留下准确结论。

- [ ] **Step 10: Commit**

```powershell
git add Src/Algorithms/DSPO.py Src/Algorithms/DSPO_plus_SPO.py tests/test_dspo_outside_aware.py tests/test_update_skip_quit.py
git commit -m "DSPO: outside-aware Lambert pricing (division form) and quit-step skip in training"
```

(若 Step 9 判定无需改 DSPO_plus_SPO,则 git add 去掉该文件。)

---

### Task 3: eval2 新指标(quit_rate / net_profit)+ run_one 支持 extra 覆盖

**Files:**
- Modify: `run.py:100-184`(eval2)
- Modify: `Experiments/run_paper_repro.py`(build_argv/run_one 加 `extra=None`)

**Interfaces:**
- Consumes: Task 1 的 stats[8]=count_quit。
- Produces: `Solver.eval2` 返回 dict 新增键 `quit_rate`(float)与 `net_profit`(float);`run_paper_repro.build_argv(policy, seed, scaler, train_episodes, buffer_size, extra=None)` 与 `run_one(policy, seed, scaler, train_episodes, eval_episodes, buffer_size, extra=None)`——`extra` 为 dict,叠加在 PAPER_PRESET+POLICIES 之上(可覆盖任意 parser 参数,值为字符串)。

- [ ] **Step 1: 改 `run.py::eval2`**

(a) 统计列表区(`count_home_delivery = []` 之后)加:

```python
            count_quit = []
```

(b) episode 收尾区(`accepted_discount.append(stats[6])` 之后)加:

```python
                count_quit.append(stats[8])
```

(c) metrics 构建:`d_list = np.concatenate(accepted_discount)` 之前加:

```python
            quits = sum(count_quit)
```

metrics dict 里 `'pct_home'` 行之后加:

```python
                'quit_rate': quits / len(home_delivery_loc),
```

(d) `metrics['total_costs'] = ...` 之后加:

```python
            served = len(home_delivery_loc) - quits
            metrics['net_profit'] = (self.config.revenue * served / episodes
                                     + metrics['charge_revenue'] - metrics['discount_costs']
                                     - metrics['travel_costs'] - metrics['service_costs']
                                     - metrics['failure_costs'])
```

(e) 打印区 `print('total costs: ', ...)` 之后加:

```python
            print('quit rate: ', metrics['quit_rate'])
            print('net profit: ', metrics['net_profit'])
```

- [ ] **Step 2: 改 `Experiments/run_paper_repro.py`**

`def build_argv(policy, seed, scaler, train_episodes, buffer_size):` 改为:

```python
def build_argv(policy, seed, scaler, train_episodes, buffer_size, extra=None):
```

其函数体 `over.update(POLICIES[policy])` 之后加:

```python
    if extra:
        over.update(extra)
```

`def run_one(policy, seed, scaler, train_episodes, eval_episodes, buffer_size):` 改为:

```python
def run_one(policy, seed, scaler, train_episodes, eval_episodes, buffer_size, extra=None):
```

其中 `argv = build_argv(policy, seed, scaler, train_episodes, buffer_size)` 改为:

```python
    argv = build_argv(policy, seed, scaler, train_episodes, buffer_size, extra)
```

(smoke/calibrate/full 三个既有调用不传 extra,行为不变。)

- [ ] **Step 3: 语法与回归**

Run: `python -c "import ast; ast.parse(open('run.py',encoding='utf-8').read()); ast.parse(open(r'Experiments\run_paper_repro.py',encoding='utf-8').read()); print('syntax OK')"` → `syntax OK`;跑全部 10 个测试 → `OK`。

- [ ] **Step 4: Commit**

```powershell
git add run.py Experiments/run_paper_repro.py
git commit -m "eval2: quit_rate and net_profit metrics; run_one accepts extra overrides"
```

---

### Task 4: 实验入口 `Experiments/run_req2_profit.py`

**Files:**
- Create: `Experiments/run_req2_profit.py`

**Interfaces:**
- Consumes: Task 3 的 `run_one(..., extra=...)`、`agg(results, p, key)`(均从 `Experiments.run_paper_repro` 导入);eval2 的 `net_profit`/`quit_rate` 键。
- Produces: CLI `python Experiments\run_req2_profit.py --mode {calibrate_v0,full} [--v0 X] [--v0_grid ...] [--train_episodes N] [--eval_episodes N] [--buffer_size N] [--seeds ...]`;full 模式写 `docs/req2_profit_results.md`。

- [ ] **Step 1: 写脚本(完整内容)**

```python
"""Requirement 2 runner: outside option + net-profit comparison (spec:
docs/superpowers/specs/2026-07-25-req2-outside-option-design.md).

Modes:
  calibrate_v0 -- sweep V0 under NoPricing, report quit rates, target band 2%-5%
  full         -- NoPricing/StaticPricing/DSPO x seeds, writes docs/req2_profit_results.md
"""
import sys
import argparse
from os import path

ROOT = path.abspath(path.join(path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from Experiments.run_paper_repro import run_one, agg  # noqa: E402

SCALER = 1800.0  # requirement-1 MNL calibration, user-approved
POLICIES2 = ['NoPricing', 'StaticPricing', 'DSPO']
ORDER_BY_PROFIT_DESC = ['DSPO', 'StaticPricing', 'NoPricing']  # expected: DSPO highest net profit


def req2_extra(v0):
    return {'home_failure': '0.0', 'outside_option_util': str(v0)}


def mode_calibrate_v0(opt):
    print('== calibrate_v0: NoPricing quit-rate sweep, target band 2%-5% ==')
    best = None
    for v0 in opt.v0_grid:
        m = run_one('NoPricing', opt.seeds[0], SCALER, 0, opt.eval_episodes, opt.buffer_size,
                    extra=req2_extra(v0))
        print('V0={:6.2f} -> quit%={:5.2f} home%={:5.1f} net_profit={:9.2f}'.format(
            v0, 100*m['quit_rate'], 100*m['pct_home'], m['net_profit']))
        err = abs(m['quit_rate'] - 0.035)
        if best is None or err < best[1]:
            best = (v0, err, m['quit_rate'])
    print('closest to band center 3.5%: V0={} (quit {:.2f}%)'.format(best[0], 100*best[2]))
    print('acceptance: {}'.format('PASS' if 0.02 <= best[2] <= 0.05 else 'FAIL (densify grid near best V0)'))


def mode_full(opt):
    assert opt.v0 is not None, 'full mode requires --v0 from calibrate_v0'
    results = {p: [] for p in POLICIES2}
    for seed in opt.seeds:
        for p in POLICIES2:
            print('running {} seed {} ...'.format(p, seed))
            m = run_one(p, seed, SCALER, opt.train_episodes if p == 'DSPO' else 0,
                        opt.eval_episodes, opt.buffer_size, extra=req2_extra(opt.v0))
            print('  -> net_profit={:.2f} total_costs={:.2f} quit%={:.2f} home%={:.1f} ({:.0f}s)'.format(
                m['net_profit'], m['total_costs'], 100*m['quit_rate'], 100*m['pct_home'], m['wall_total_s']))
            results[p].append(m)
    write_report(results, opt)


def write_report(results, opt):
    lines = []
    lines.append('# 需求 2 结果:外部效用 + 净利润对比(RC 合成案例)\n')
    lines.append('- 预设:需求 1 论文预设 + home_failure=0 + outside_option_util={};scaler={};seeds={};DSPO 训练 {} ep,buffer {};评测 {} ep/seed\n'.format(
        opt.v0, SCALER, opt.seeds, opt.train_episodes, opt.buffer_size, opt.eval_episodes))
    lines.append('| 策略 | 净利润 | 退出率% | home% | travel | service | 折扣成本(收费收入) | total costs |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for p in ORDER_BY_PROFIT_DESC:
        pm, ps = agg(results, p, 'net_profit')
        lines.append('| {} | {:.1f}±{:.1f} | {:.2f}±{:.2f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} ({:.1f}) | {:.1f}±{:.1f} |'.format(
            p, pm, ps,
            100*agg(results, p, 'quit_rate')[0], 100*agg(results, p, 'quit_rate')[1],
            100*agg(results, p, 'pct_home')[0],
            agg(results, p, 'travel_costs')[0], agg(results, p, 'service_costs')[0],
            agg(results, p, 'discount_costs')[0], agg(results, p, 'charge_revenue')[0],
            agg(results, p, 'total_costs')[0], agg(results, p, 'total_costs')[1]))
    means = {p: agg(results, p, 'net_profit')[0] for p in POLICIES2}
    actual = sorted(means, key=means.get, reverse=True)
    lines.append('\n- 目标排序(净利润降序):{}'.format(' > '.join(ORDER_BY_PROFIT_DESC)))
    lines.append('- 实际排序(净利润降序):{}'.format(' > '.join(actual)))
    lines.append('- 排序判定:{}\n'.format('**成立**' if actual == ORDER_BY_PROFIT_DESC else '**不成立(如实报告,见上表)**'))
    out = path.join(ROOT, 'docs', 'req2_profit_results.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('report written to', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['calibrate_v0', 'full'])
    ap.add_argument('--v0', type=float, default=None)
    ap.add_argument('--v0_grid', type=float, nargs='+', default=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
    ap.add_argument('--train_episodes', type=int, default=500)
    ap.add_argument('--eval_episodes', type=int, default=20)
    ap.add_argument('--buffer_size', type=int, default=10000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1234, 2345, 3456])
    opt = ap.parse_args()
    {'calibrate_v0': mode_calibrate_v0, 'full': mode_full}[opt.mode](opt)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 语法 + 导入检查**

Run: `python -c "import ast; ast.parse(open(r'Experiments\run_req2_profit.py',encoding='utf-8').read()); print('syntax OK')"` → `syntax OK`
Run: `python -c "import sys; sys.path.insert(0,r'D:\project\python\ooh_code'); from Experiments.run_req2_profit import POLICIES2, ORDER_BY_PROFIT_DESC, req2_extra; print(POLICIES2, ORDER_BY_PROFIT_DESC, req2_extra(0.5))"`
Expected: 打印三策略列表、期望排序、`{'home_failure': '0.0', 'outside_option_util': '0.5'}`。

- [ ] **Step 3: 默认关闭等价性冒烟(全局约束验收)**

Run: `python Experiments\run_paper_repro.py --mode smoke --eval_episodes 2 --train_episodes 3`
Expected: 与需求 1 冒烟同形的 5 行输出、无异常(V₀ 未传 → None → 行为不变;数值应与需求 1 分支尾部的冒烟一致——NoOOH home%=100、OnlyOOH home%=0)。

- [ ] **Step 4: Commit**

```powershell
git add Experiments/run_req2_profit.py
git commit -m "Add requirement-2 runner (calibrate_v0/full, net-profit report)"
```

---

### Task 5: V₀ 校准运行(控制器执行)

**Files:** 无代码变更。

**Interfaces:**
- Consumes: Task 4 的 `--mode calibrate_v0`。
- Produces: 选定的 V₀ 值(NoPricing 退出率 2%–5%,取最接近 3.5% 者),供 Task 6 `--v0` 使用。

- [ ] **Step 1:** Run: `python Experiments\run_req2_profit.py --mode calibrate_v0 --eval_episodes 20`
Expected: 每个 V₀ 一行 quit%/home%/net_profit;末尾 `closest to band center` 与 PASS/FAIL。
- [ ] **Step 2:** 若 FAIL(整格都不进带),在最优点附近加密网格重跑(如 `--v0_grid -0.3 -0.1 0.1 0.3`),直到进带;记录最终 V₀ 与初算的偏差(初算 V₀≈0 附近:P(quit)=e^{V₀}/(e^{3.2}+S+e^{V₀}),S≈6.2)。

---

### Task 6: 全量跑批 → `docs/req2_profit_results.md` → 验收判定(控制器执行)

**Files:**
- Create(脚本生成): `docs/req2_profit_results.md`

**Interfaces:**
- Consumes: Task 5 的 V₀。
- Produces: 3 策略 × 3 seed 净利润结果 + 排序判定,提交到分支。

- [ ] **Step 1:** Run(V₀ 以 Task 5 结果为准):

```powershell
python Experiments\run_req2_profit.py --mode full --v0 <Task5的值> --train_episodes 500 --buffer_size 10000 --eval_episodes 20 --seeds 1234 2345 3456
```

Expected: 9 次运行(DSPO 每 seed 约 20 分钟,Baseline 类约 25 秒),末尾 `report written to ...`。

- [ ] **Step 2:** 核对并补充报告:排序判定如实;手工追加"验收标准逐条判定"与"局限性"段(对照规格 §1 的 4 条:排序 PASS/FAIL 及逐 seed 翻转、V₀ 校准值与退出率、定价策略退出率是否失控 >15% 需分析、默认关闭等价性引用 Task 4 Step 3 的冒烟结论)。

- [ ] **Step 3:** Commit:

```powershell
git add docs/req2_profit_results.md
git commit -m "Add requirement-2 results: outside option, net-profit ordering (3 seeds)"
```

- [ ] **Step 4:** 按规格 §1 验收标准逐条向用户汇报,附逐 seed 数据;排序不成立时给分析(退出损失分解、DSPO 定价行为对比),不调参硬凑。
