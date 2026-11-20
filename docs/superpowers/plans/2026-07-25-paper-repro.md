# 论文原文配置 5 策略成本对比复现 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `paper-repro` 分支上,按 Akkerman-TS24 论文原文参数复现 RC 合成案例的 NoOOH / NoPricing / StaticPricing / DSPO / OnlyOOH 五策略成本对比(3 评测 seed),验证成本排序 NoOOH > NoPricing > StaticPricing > DSPO > OnlyOOH。

**Architecture:** 三处结构性修复(MNL 效用方向、OOH 服务时间、成本乘数统一)改在环境/Utils/DSPO 层,五策略共用;NoOOH/OnlyOOH 通过环境层新增 `offer_mode` 选择集开关实现(不新写算法类);一个统一实验脚本 `Experiments/run_paper_repro.py` 内置论文参数预设,支持 smoke / calibrate / full 三种模式。

**Tech Stack:** Python 3.10、PyTorch、hygese(HGS-CVRP)、numpy;无测试框架,用纯 python assert 脚本(`tests/*.py`,直接 `python tests/test_x.py` 运行)。

**对应规格:** `docs/superpowers/specs/2026-07-24-paper-repro-design.md`(含 2026-07-25 证据修正)。

## Global Constraints

- 分支:`paper-repro`;仓库根:`D:\project\python\ooh_code`(目录名必须是 `ooh_code`,`dynamic_load` 依赖)。
- **不改 `Src/parser.py` 现有参数的默认值**,只允许新增参数;论文值全部通过 `run_paper_repro.py` 预设传入。
- 论文参数(Table A.1/B.1/F.1,合成案例):`Cw=30, Cf=0.3, Cm=10, r(revenue)=50, Pm=0.1, V=9, K=10, 价格∈[-10,2], E[D]=90(负二项 r=90,p=0.5), βᵏ=0.02, u₀⁺=3.2, βᵈ=-0.25, StaticPricing 折扣5/收费2, grid 10×10, Dᵀ=3, batch 64, lr 1e-3, filters 32, dropout 0.0, Huber δ=1.0, FC 128`。
- `dist_matrix` 语义保持"行驶时间(秒)";评测口径 `run.py::eval2` 的乘数 `(Cw+Cf·v)/3600` 是论文报数口径,**不改**。
- 单位:服务时间为秒(`calculate_service_time` ×60);距离矩阵为秒。
- 所有命令在仓库根目录执行;Python 解释器用 Task 0 确定的那个(下文统一写 `python`)。
- 报告实验结果时如实呈现,排序不成立不得调参硬凑。

---

### Task 0: 环境自检

**Files:** 无修改,只验证。

**Interfaces:**
- Produces: 可用的 Python 解释器路径(后续所有任务用它);hygese 可导入的确认。

- [ ] **Step 1: 确定解释器**

```powershell
cd D:\project\python\ooh_code
if (Test-Path venv\Scripts\python.exe) { venv\Scripts\python.exe --version } else { python --version }
```

Expected: 打出 Python 3.x 版本号。记录用哪个解释器,后续统一使用。

- [ ] **Step 2: 依赖与数据自检**

```powershell
python -c "import torch, hygese, numpy, scipy, matplotlib; print('deps OK')"
python -c "from os import path; assert path.isfile(r'Environments\OOH\HombergerGehring_data\RC\RC_90_0_coords.txt'), 'RC seed0 missing'; assert path.isfile(r'Environments\OOH\HombergerGehring_data\RC\RC_90_1_coords.txt'), 'RC seed1 missing'; print('data OK')"
```

Expected: `deps OK`、`data OK`。若 RC 文件名不符,先 `ls Environments\OOH\HombergerGehring_data\RC\` 确认真实命名并在后续任务沿用(代码里的命名规则是 `{instance}_90_{seed}_coords.txt`,见 `Utils.load_demand_data`)。

- [ ] **Step 3: 确认分支**

```powershell
git branch --show-current
```

Expected: `paper-repro`。

---

### Task 1: 论文式 MNL 效用(方向修复 + 参数化)

**Files:**
- Modify: `Environments/OOH/customerchoice.py`
- Modify: `Src/parser.py`(新增 2 个参数)
- Modify: `Src/config.py:98-111`(get_domain 透传)
- Modify: `Environments/OOH/Parcelpoint_py.py`(构造签名 + dist_scaler + 透传)
- Modify: `Src/Algorithms/DSPO.py:58-73, 217-225`(MNL 同步)
- Test: `tests/test_mnl_utility.py`

**Interfaces:**
- Produces: `customerchoicemodel.__init__(self, base_util, dist_scaler, euclidean, dist_mat, n_cust, beta_k=0.02)`;新方法 `ooh_utility(self, distance) -> float`(= `base_util - beta_k*exp(distance/dist_scaler)`);parser 新参数 `--mnl_beta_k`(float, 默认 0.02)、`--mnl_dist_scaler`(float, 默认 0.0,0 表示自动取 `np.amax(dist_matrix)`);`Parcelpoint_py.__init__` 新增关键字参数 `mnl_beta_k=0.02, mnl_dist_scaler=0.0`。
- Consumes: 无(首任务)。

- [ ] **Step 1: 写失败测试 `tests/test_mnl_utility.py`**

```python
import sys
from os import path
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
from math import exp
from Environments.OOH.customerchoice import customerchoicemodel

def euclid_stub(a, b):
    return abs(a - b)

def main():
    m = customerchoicemodel(base_util=0.0, dist_scaler=100.0, euclidean=euclid_stub,
                            dist_mat=[], n_cust=10, beta_k=0.02)
    # 论文式(5):u = -beta_k * exp(d/scaler),越远效用越低
    u_near = m.ooh_utility(10.0)
    u_far = m.ooh_utility(200.0)
    assert u_near > u_far, f"utility must decrease with distance: {u_near} vs {u_far}"
    assert abs(u_near - (0.0 - 0.02 * exp(10.0 / 100.0))) < 1e-12
    assert abs(u_far - (0.0 - 0.02 * exp(200.0 / 100.0))) < 1e-12
    # 超大距离不能溢出(校准会探索小 scaler)
    u_huge = m.ooh_utility(1e9)
    assert u_huge < -1e6 and u_huge == u_huge, "must be finite, very negative"
    # base_util 是可加常数
    m2 = customerchoicemodel(base_util=1.5, dist_scaler=100.0, euclidean=euclid_stub,
                             dist_mat=[], n_cust=10, beta_k=0.02)
    assert abs(m2.ooh_utility(10.0) - (u_near + 1.5)) < 1e-12
    print("OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行确认失败**

Run: `python tests\test_mnl_utility.py`
Expected: `TypeError: __init__() got an unexpected keyword argument 'beta_k'`(或 `AttributeError: ... no attribute 'ooh_utility'`)。

- [ ] **Step 3: 改 `customerchoice.py`**

把文件开头到 `mnl_distmat` 结束(第 1–36 行)替换为:

```python
from math import exp
from numpy.random import gumbel
import numpy as np

class customerchoicemodel(object):
    def __init__(self,
                 base_util,
                 dist_scaler,
                 euclidean,
                 dist_mat,
                 n_cust,
                 beta_k=0.02):
        self.euclidean_distance = euclidean
        self.dist_scaler = dist_scaler
        self.base_util = base_util
        self.dist_mat = dist_mat
        self.n_cust = n_cust
        self.beta_k = beta_k
        if len(self.dist_mat)>0:
            self.mnl = self.mnl_distmat
        else:
            self.mnl = self.mnl_euclid

    def ooh_utility(self,distance):
        """paper Eq.(5): u_k = -beta^k * exp[d]; distance divided by dist_scaler to keep exp bounded.
        Exponent capped at 50 to avoid overflow when calibration probes small scalers
        (utility ~ -1e20, effectively minus infinity)."""
        return self.base_util - self.beta_k*exp(min(distance/self.dist_scaler,50.0))

    def mnl_euclid(self,customer,parcelpoint):
        """
        multi-nomial logit model calculating euclidean distance
        """
        distance = self.euclidean_distance(customer.home,parcelpoint.location)#distance from parcelpoint to home
        return self.ooh_utility(distance)

    def mnl_distmat(self,customer,parcelpoint):
        """
        multi-nomial logit model using distance matrix
        """
        distance = self.dist_mat[customer.id_num][parcelpoint.id_num]#distance from parcelpoint to home
        return self.ooh_utility(distance)
```

`customerchoice_offer` / `customerchoice_pricing` 不动。

- [ ] **Step 4: 运行测试确认通过**

Run: `python tests\test_mnl_utility.py`
Expected: `OK`

- [ ] **Step 5: parser 新增参数**

在 `Src/parser.py` 的 `environment_parameters` 中,`--home_util` 那行之后插入:

```python
        parser.add_argument("--mnl_beta_k", default=0.02, help="MNL distance sensitivity beta^k, paper Table B.1", type=float)
        parser.add_argument("--mnl_dist_scaler", default=0.0, help="distance divisor inside exp of MNL utility; 0.0 = auto (max of dist matrix)", type=float)
```

- [ ] **Step 6: `Parcelpoint_py.py` 接线**

构造签名(第 12–32 行)在 `dissatisfaction=False,` 之前加两个参数:

```python
                 mnl_beta_k=0.02,
                 mnl_dist_scaler=0.0,
```

`load_data` 分支里(现第 63 行)把 `self.dist_scaler = np.amax(self.dist_matrix)` 改为:

```python
            self.dist_scaler = mnl_dist_scaler if mnl_dist_scaler > 0 else np.amax(self.dist_matrix)
```

else 分支(现第 72 行)把 `self.dist_scaler = 10` 改为:

```python
            self.dist_scaler = mnl_dist_scaler if mnl_dist_scaler > 0 else 10
```

两处 `customerchoicemodel(...)` 构造(现第 85、90 行)都改为:

```python
            self.customerchoice = customerchoicemodel(base_util,self.dist_scaler,self.utils.getdistance_euclidean,self.dist_matrix,self.n_unique_customer_locs,beta_k=mnl_beta_k)
```

- [ ] **Step 7: `config.py` 透传**

`get_domain` 的两处 `obj(...)` 调用(test_env 与非 test_env)都在 `dissatisfaction=self.dissatisfaction,` 前加:

```python
                      mnl_beta_k=args.mnl_beta_k,mnl_dist_scaler=args.mnl_dist_scaler,
```

- [ ] **Step 8: DSPO 的 MNL 同步**

`Src/Algorithms/DSPO.py`:

第 65 行 `self.dist_scaler = 1#np.amax(self.dist_matrix)` 改为:

```python
            self.dist_scaler = config.mnl_dist_scaler if config.mnl_dist_scaler > 0 else np.amax(self.dist_matrix)
```

第 69 行(else 分支)`self.dist_scaler = 1#10` 改为:

```python
            self.dist_scaler = config.mnl_dist_scaler if config.mnl_dist_scaler > 0 else 10
```

第 73 行 `self.base_util = config.base_util` 之后加一行:

```python
        self.beta_k = config.mnl_beta_k
```

第 217–225 行两个 mnl 函数体中的 `beta_p = -exp(-distance/self.dist_scaler)` / `return self.base_util + beta_p` 都改为(与 customerchoice 同款,含防溢出上限):

```python
        return self.base_util - self.beta_k*exp(min(distance/self.dist_scaler,50.0))
```

- [ ] **Step 9: 冒烟导入检查**

Run: `python -c "from Src.parser import Parser; a=Parser().get_parser().parse_args([]); print(a.mnl_beta_k, a.mnl_dist_scaler)"`
Expected: `0.02 0.0`

- [ ] **Step 10: Commit**

```powershell
git add Environments/OOH/customerchoice.py Environments/OOH/Parcelpoint_py.py Src/parser.py Src/config.py Src/Algorithms/DSPO.py tests/test_mnl_utility.py
git commit -m "Fix MNL utility to paper Eq.(5) form, parameterize beta_k and dist scaler"
```

---

### Task 2: offer_mode 选择集开关(NoOOH / OnlyOOH)

**Files:**
- Modify: `Environments/OOH/customerchoice.py`(新增两个选择函数)
- Modify: `Src/parser.py`(新增 `--offer_mode`)
- Modify: `Src/config.py`(透传)
- Modify: `Environments/OOH/Parcelpoint_py.py`(构造参数 + 选择函数覆盖)
- Test: `tests/test_offer_modes.py`

**Interfaces:**
- Consumes: Task 1 的 `customerchoicemodel`(含 `beta_k`、`ooh_utility`)。
- Produces: `customerchoicemodel.customerchoice_home_only(customer, action, parcelpoints) -> (Location, bool, int, float)`(恒返回 home);`customerchoicemodel.customerchoice_ooh_only(customer, action, parcelpoints) -> (Location, bool, int, float)`(恒返回某个有容量的 OOH,价格 0);parser 参数 `--offer_mode`(choices `['all','home_only','ooh_only']`,默认 `'all'`);`Parcelpoint_py.__init__` 新增关键字参数 `offer_mode='all'`。

- [ ] **Step 1: 写失败测试 `tests/test_offer_modes.py`**

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

def make_fixture():
    m = customerchoicemodel(base_util=0.0, dist_scaler=100.0, euclidean=euclid_stub,
                            dist_mat=[], n_cust=5, beta_k=0.02)
    home = Location(10.0, 10.0, 1, 0)
    cust = Customer(home, -0.25, 3.2, 120.0, 1)
    pps = np.array([ParcelPoint(Location(20.0, 20.0, 5, 0), 1000000, 5),
                    ParcelPoint(Location(30.0, 30.0, 6, 0), 0, 6)])  # 第二个无容量
    # 与 get_delivery_loc_pricing 相同的掩码形式:mask=1 表示提供
    masked = ma.masked_array(pps, mask=[1, 1])
    action = np.zeros(3)
    return m, cust, masked, action

def main():
    m, cust, masked, action = make_fixture()
    # home_only:恒选 home
    for _ in range(20):
        loc, accepted, idx, price = m.customerchoice_home_only(cust, action, masked)
        assert accepted is False and idx == -1 and price == 0
        assert loc is cust.home
    # ooh_only:恒选 OOH,且永不选无容量的点
    for _ in range(200):
        loc, accepted, idx, price = m.customerchoice_ooh_only(cust, action, masked)
        assert accepted is True and price == 0
        assert idx == 5, f"must never pick zero-capacity pp, got {idx}"
    print("OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行确认失败**

Run: `python tests\test_offer_modes.py`
Expected: `AttributeError: 'customerchoicemodel' object has no attribute 'customerchoice_home_only'`

- [ ] **Step 3: `customerchoice.py` 文件末尾追加两个方法**

```python
    def customerchoice_home_only(self,customer,action,parcelpoints):
        """
        NoOOH baseline: no OOH offered, customer always takes home delivery
        """
        return customer.home, False, -1, 0

    def customerchoice_ooh_only(self,customer,action,parcelpoints):
        """
        OnlyOOH baseline: home not in the choice set, MNL over offered OOH points, no pricing
        """
        pps = parcelpoints[parcelpoints.mask].data
        shape = (len(pps), 1)
        utils = np.empty(shape)
        for idx,pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                utils[idx] = self.mnl(customer,pp)
            else:
                utils[idx] = -1e9#infeasible, never chosen
        utils = np.add(utils,gumbel(0,1, np.shape(utils)))#mu=0,beta=1 (std Gumbel)

        idx = np.argmax(utils)
        return pps[idx].location, True, pps[idx].id_num, 0
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python tests\test_offer_modes.py`
Expected: `OK`

- [ ] **Step 5: parser / config / env 接线**

`Src/parser.py` 的 `environment_parameters`,紧跟 `--mnl_dist_scaler` 之后:

```python
        parser.add_argument("--offer_mode", default='all', help="choice set restriction: all (paper default), home_only (NoOOH), ooh_only (OnlyOOH)", choices=['all','home_only','ooh_only'])
```

`Src/config.py::get_domain` 两处 `obj(...)` 里加(与 Task 1 的透传放一起):

```python
                      offer_mode=args.offer_mode,
```

`Environments/OOH/Parcelpoint_py.py` 构造签名加 `offer_mode='all',`;`__init__` 里现有的 pricing/offer 分支(第 83–92 行)**之后**追加:

```python
        #choice-set restriction for the NoOOH / OnlyOOH baselines
        self.offer_mode = offer_mode
        if offer_mode == 'home_only':
            self.customerChoice = self.customerchoice.customerchoice_home_only
        elif offer_mode == 'ooh_only':
            self.customerChoice = self.customerchoice.customerchoice_ooh_only
```

- [ ] **Step 6: 冒烟导入检查**

Run: `python -c "from Src.parser import Parser; a=Parser().get_parser().parse_args(['--offer_mode','ooh_only']); print(a.offer_mode)"`
Expected: `ooh_only`

- [ ] **Step 7: Commit**

```powershell
git add Environments/OOH/customerchoice.py Environments/OOH/Parcelpoint_py.py Src/parser.py Src/config.py tests/test_offer_modes.py
git commit -m "Add offer_mode choice-set switch for NoOOH and OnlyOOH baselines"
```

---

### Task 3: OOH 服务时间(环境结算 + DSPO 标签摊销)

**Files:**
- Modify: `Environments/OOH/Parcelpoint_py.py:113, 197-199`(reset + step)
- Modify: `Src/Algorithms/DSPO.py:63, 228-247, 327-347`(first_parcelpoint_id 修正、update 传 id 计数、标签加服务时间)
- Test: `tests/test_label_service_time.py`

**Interfaces:**
- Consumes: Task 1 后的 DSPO(`self.beta_k` 等已存在,但本任务不依赖)。
- Produces: `Parcelpoint_py` 每 episode 维护 `self.used_pps: set`,每个被使用的 OOH 点的 lᵢ 计一次进 `self.service_time`;`DSPO.get_per_customer_costs(self, fleet, id_counts) -> list[[time, cost]]`(签名新增 `id_counts: dict[int,int]`,home 停靠加 `mltplr*lᵢ`,OOH 停靠加 `mltplr*lᵢ/id_counts[id]`);`DSPO.first_parcelpoint_id = len(dist_matrix[0]) - n_parcelpoints`(修正 off-by-one)。

- [ ] **Step 1: 写失败测试 `tests/test_label_service_time.py`**

```python
import sys
from os import path
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
import numpy as np
from Src.Algorithms.DSPO import DSPO
from Environments.OOH.containers import Location, Vehicle, Fleet

def main():
    # 不跑 __init__,手工填 get_per_customer_costs 用到的属性
    obj = DSPO.__new__(DSPO)
    n = 5  # ids: 0=depot, 1-2=customers, 3-4=OOH
    obj.dist_matrix = np.full((n, n), 100.0)
    np.fill_diagonal(obj.dist_matrix, 0.0)
    obj.service_times = np.array([0.0, 300.0, 240.0, 120.0, 60.0])
    obj.cost_multiplier = 1.0  # 便于验证
    obj.first_parcelpoint_id = 3
    # 一辆车,路线:home客户1 -> OOH点3(被2个客户使用,route 里出现两次)
    stops = [Location(1.0,1.0,1,0), Location(2.0,2.0,3,1), Location(2.0,2.0,3,2)]
    fleet = Fleet([Vehicle(stops, 10, 0)])
    id_counts = {1: 1, 3: 2}
    costs = DSPO.get_per_customer_costs(obj, fleet, id_counts)
    assert len(costs) == 3
    # 半边成本:首停 0.5*d(0,1)+0.5*d(1,3)=100;中停 0.5*d(1,3)+0.5*d(3,3)=50(对角线0);
    # 末停 0.5*d(3,3)+0.5*d(3,0)=50
    # 服务时间:home客户1 加 300;OOH 3 每次出现摊 120/2=60
    expect = {0: 100.0 + 300.0, 1: 50.0 + 60.0, 2: 50.0 + 60.0}
    for t, c in costs:
        assert abs(c - expect[t]) < 1e-9, f"stop t={t}: got {c}, want {expect[t]}"
    print("OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行确认失败**

Run: `python tests\test_label_service_time.py`
Expected: `TypeError: get_per_customer_costs() takes 2 positional arguments but 3 were given`

- [ ] **Step 3: 改 `DSPO.py`**

(a) 文件顶部 import 区加:

```python
from collections import Counter
```

(b) 第 63 行 off-by-one 修正:

```python
            self.first_parcelpoint_id = len(self.dist_matrix[0])-config.n_parcelpoints
```

(c) `update()` 的 done 分支,`target = self.get_per_customer_costs(fleet)` 改为:

```python
            id_counts = Counter(np.asarray(data['id']).tolist())
            target = self.get_per_customer_costs(fleet,id_counts)
```

(d) `get_per_customer_costs` 整体替换为:

```python
    def get_per_customer_costs(self,fleet,id_counts):
        mltplr = self.cost_multiplier
        costs = []
        for v in fleet["fleet"]:
            if len(v["routePlan"])==1:#when only 1 customer is visited
                costs.append( [v["routePlan"][0].time,mltplr * (self.dist_matrix[0][v["routePlan"][0].id_num])] )
                self._add_service_time_to_label(costs,v["routePlan"][0],mltplr,id_counts)
            else:
                for i in range(0,len(v["routePlan"])):
                    #costs is composed of distance*mltplr
                    if i==0:
                        costs.append( [v["routePlan"][i].time,mltplr * (0.5*self.dist_matrix[0][v["routePlan"][i].id_num] + 0.5*self.dist_matrix[v["routePlan"][i].id_num][v["routePlan"][i+1].id_num])] )
                    elif i==len(v["routePlan"])-1:
                        costs.append( [v["routePlan"][i].time,mltplr * (0.5*self.dist_matrix[v["routePlan"][i-1].id_num][v["routePlan"][i].id_num] + 0.5*self.dist_matrix[v["routePlan"][i].id_num][0])] )
                    else:
                        costs.append( [v["routePlan"][i].time,mltplr * (0.5*self.dist_matrix[v["routePlan"][i-1].id_num][v["routePlan"][i].id_num] + 0.5*self.dist_matrix[v["routePlan"][i].id_num][v["routePlan"][i+1].id_num])] )
                    self._add_service_time_to_label(costs,v["routePlan"][i],mltplr,id_counts)

        return costs

    def _add_service_time_to_label(self,costs,stop,mltplr,id_counts):
        """paper 4.1.3: home stop carries its full service time; an OOH stop's service
        time is amortized over the customers using that OOH point this episode"""
        if stop.id_num >= self.first_parcelpoint_id:#OOH location
            costs[-1][1] += mltplr*self.service_times[stop.id_num]/max(1,id_counts.get(stop.id_num,1))
        else:#home delivery customer
            costs[-1][1] += mltplr*self.service_times[stop.id_num]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python tests\test_label_service_time.py`
Expected: `OK`

- [ ] **Step 5: 环境结算改动 `Parcelpoint_py.py`**

`reset()` 里 `self.service_time = 0` 之后加:

```python
        self.used_pps = set()
```

`step()` 里 accepted_pp 分支(现第 197–199 行)改为:

```python
        if accepted_pp:
            self.parcelPoints["parcelpoints"][idx-self.n_unique_customer_locs].remainingCapacity -= 1
            if idx not in self.used_pps:#each used OOH stop incurs its service time once per episode
                self.used_pps.add(idx)
                self.service_time += self.service_times[idx]
```

- [ ] **Step 6: Commit**

```powershell
git add Environments/OOH/Parcelpoint_py.py Src/Algorithms/DSPO.py tests/test_label_service_time.py
git commit -m "Account OOH service time: once per used point in env stats, amortized in DSPO labels"
```

---

### Task 4: 成本乘数统一 + CNN FC 宽度 128

**Files:**
- Modify: `Src/Utils/Utils.py:56-63`(total_costs)
- Modify: `Src/Algorithms/DSPO.py:74`(cost_multiplier)
- Modify: `Src/Utils/Predictors.py:57-61`(FC 128)
- Test: `tests/test_total_costs.py`

**Interfaces:**
- Consumes: 无新依赖。
- Produces: `Utils.total_costs` 与 `DSPO.cost_multiplier` 均使用 `(driver_wage + fuel_cost*truck_speed)/3600`(与 `run.py::eval2` 一致);CNN FC 层 128/128(论文 Table F.1 合成案例)。

- [ ] **Step 1: 写失败测试 `tests/test_total_costs.py`**

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `python tests\test_total_costs.py`
Expected: AssertionError(现实现漏乘 truck_speed,got 约 70.36)。

- [ ] **Step 3: 改 `Utils.total_costs`**

第 59 行改为:

```python
    cost_multiplier = (config.driver_wage+config.fuel_cost*config.truck_speed) / 3600
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python tests\test_total_costs.py`
Expected: `OK`

- [ ] **Step 5: DSPO 与 Predictors 同步**

`Src/Algorithms/DSPO.py` 第 74 行改为:

```python
        self.cost_multiplier =  (config.driver_wage+config.fuel_cost*config.truck_speed) / 3600
```

`Src/Utils/Predictors.py` 第 57–61 行改为:

```python
        self.fc1 = nn.Linear(in_features=int(1+2*out_channels*h3*w3), out_features=128)#+1 for capacity of OOH

        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
```

- [ ] **Step 6: 回归跑全部 tests**

Run: `python tests\test_mnl_utility.py; python tests\test_offer_modes.py; python tests\test_label_service_time.py; python tests\test_total_costs.py`
Expected: 4 个 `OK`。

- [ ] **Step 7: Commit**

```powershell
git add Src/Utils/Utils.py Src/Algorithms/DSPO.py Src/Utils/Predictors.py tests/test_total_costs.py
git commit -m "Unify cost multiplier with eval accounting; set CNN FC width to paper's 128"
```

---

### Task 5: eval2 返回 metrics 字典

**Files:**
- Modify: `run.py:93-168`(eval2 尾部)、`run.py:170-184`(main)

**Interfaces:**
- Consumes: 无新依赖。
- Produces: `Solver.eval2(episodes) -> dict`,键:`pct_home, travel_costs, service_costs, failure_costs, avg_charge, std_charge, avg_discount, std_discount, charge_revenue, discount_costs, total_costs`(均为 float;charge/discount 为空时相应值取 0.0)。打印行为保留。

- [ ] **Step 1: 替换 eval2 尾部**

`run.py` 中从 `d_list = np.concatenate(accepted_discount)`(现第 155 行)到 `return total_cost, accepted_price, step_time`(现第 168 行)整体替换为:

```python
            d_list = np.concatenate(accepted_discount)
            r_list = np.concatenate(accepted_price)
            metrics = {
                'pct_home': cnt / len(home_delivery_loc),
                'travel_costs': trvl / episodes,
                'service_costs': srvc / episodes,
                'failure_costs': fail / episodes,
                'avg_charge': float(np.mean(r_list)) if len(r_list) else 0.0,
                'std_charge': float(np.std(r_list)) if len(r_list) else 0.0,
                'avg_discount': float(-np.mean(d_list)) if len(d_list) else 0.0,
                'std_discount': float(np.std(d_list)) if len(d_list) else 0.0,
                'charge_revenue': float(np.sum(r_list)) / episodes,
                'discount_costs': float(-np.sum(d_list)) / episodes,
            }
            metrics['total_costs'] = (metrics['travel_costs'] + metrics['service_costs']
                                      + metrics['failure_costs'] + metrics['discount_costs']
                                      - metrics['charge_revenue'])
            print('percentage home delivery: ', metrics['pct_home'])
            print('travel costs: ', metrics['travel_costs'])
            print('service costs: ', metrics['service_costs'])
            print('failure costs: ', metrics['failure_costs'])
            print('Avg. Charge: ', metrics['avg_charge'], 'std.: ', metrics['std_charge'])
            print('Avg. Discount: ', metrics['avg_discount'], 'std.: ', metrics['std_discount'])
            print('Charge revenue: ', metrics['charge_revenue'])
            print('Discount costs: ', metrics['discount_costs'])
            print('total costs: ', metrics['total_costs'])
            if count_pp > 0:
                print('average travelled by customers: ', distance / count_pp)

            return metrics
```

- [ ] **Step 2: 改 main()**

`rewards,prices,step_time = solver.eval2(20)` 改为:

```python
    metrics = solver.eval2(20)
```

- [ ] **Step 3: 语法检查**

Run: `python -c "import ast; ast.parse(open('run.py',encoding='utf-8').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 4: Commit**

```powershell
git add run.py
git commit -m "eval2 returns a metrics dict for programmatic use"
```

---

### Task 6: 实验脚本 `Experiments/run_paper_repro.py`

**Files:**
- Create: `Experiments/run_paper_repro.py`

**Interfaces:**
- Consumes: `run.py::Solver`(Task 5 的 `eval2 -> dict`)、`Src.parser.Parser`、`Src.config.Config`、Task 1–2 的新 parser 参数。
- Produces: CLI `python Experiments\run_paper_repro.py --mode {smoke,calibrate,full} [--scaler X] [--train_episodes N] [--eval_episodes N] [--buffer_size N] [--seeds 1234 2345 3456] [--policies ...]`;full 模式写 `docs/paper_repro_results.md`。

- [ ] **Step 1: 写脚本(完整内容)**

```python
"""Paper-faithful reproduction runner for the 5-policy comparison (spec:
docs/superpowers/specs/2026-07-24-paper-repro-design.md).

Modes:
  smoke     -- tiny run of all policies to validate the pipeline and time one episode
  calibrate -- bisection on the MNL distance scaler targeting paper App. B (80% / 60% home)
  full      -- 5 policies x N seeds, writes docs/paper_repro_results.md
"""
import sys
import argparse
import time
from os import path

ROOT = path.abspath(path.join(path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from Src.parser import Parser  # noqa: E402
from Src.config import Config  # noqa: E402
from run import Solver         # noqa: E402

# paper Table A.1 / B.1 / F.1 / Sec. 5.2.1 -- synthetic RC case
PAPER_PRESET = {
    'instance': 'RC', 'data_seed': '0', 'data_seed_test': '1',
    'max_steps_r': '90', 'max_steps_p': '0.5',
    'n_vehicles': '9', 'veh_capacity': '10',
    'fraction_capacitated': '0.0',
    'fuel_cost': '0.3', 'driver_wage': '30', 'truck_speed': '30',
    'failure_cost': '10.0', 'home_failure': '0.1',
    'revenue': '50', 'clip_service_time': '10',
    'min_price': '-10.0', 'max_price': '2.0',
    'pricing': 'True',
    'incentive_sens': '-0.25', 'base_util': '0.0', 'home_util': '3.2',
    'mnl_beta_k': '0.02',
    # DSPO hyperparameters (paper Table F.1, synthetic)
    'grid_dim': '10', 'n_input_layers': '3', 'batch_size': '64',
    'learning_rate': '1e-3', 'n_filters': '32', 'dropout': '0.0',
    'save_model': 'False', 'debug': 'False',
}

POLICIES = {
    'NoOOH':         {'algo_name': 'Baseline', 'offer_mode': 'home_only', 'price_home': '0.0', 'price_pp': '0.0'},
    'NoPricing':     {'algo_name': 'Baseline', 'offer_mode': 'all',       'price_home': '0.0', 'price_pp': '0.0'},
    'StaticPricing': {'algo_name': 'Baseline', 'offer_mode': 'all',       'price_home': '2.0', 'price_pp': '-5.0'},
    'DSPO':          {'algo_name': 'DSPO',     'offer_mode': 'all'},
    'OnlyOOH':       {'algo_name': 'Baseline', 'offer_mode': 'ooh_only',  'price_home': '0.0', 'price_pp': '0.0'},
}
ORDER = ['NoOOH', 'NoPricing', 'StaticPricing', 'DSPO', 'OnlyOOH']


def build_argv(policy, seed, scaler, train_episodes, buffer_size):
    over = dict(PAPER_PRESET)
    over.update(POLICIES[policy])
    over['seed'] = str(seed)
    over['mnl_dist_scaler'] = str(scaler)
    over['max_episodes'] = str(max(1, train_episodes))
    over['buffer_size'] = str(buffer_size)
    over['folder_suffix'] = 'paper_{}_{}'.format(policy, seed)
    argv = []
    for k, v in over.items():
        argv += ['--' + k, v]
    return argv


def run_one(policy, seed, scaler, train_episodes, eval_episodes, buffer_size):
    argv = build_argv(policy, seed, scaler, train_episodes, buffer_size)
    orig_stdout = sys.stdout
    t0 = time.time()
    try:
        args = Parser().get_parser().parse_args(argv)
        config = Config(args)
        solver = Solver(config)
        if policy == 'DSPO' and train_episodes > 0:
            solver.train()
        t_train = time.time() - t0
        metrics = solver.eval2(eval_episodes)
    finally:
        sys.stdout = orig_stdout
    metrics['wall_train_s'] = t_train
    metrics['wall_total_s'] = time.time() - t0
    return metrics


def mode_smoke(opt):
    print('== smoke: all policies, seed {}, {} train ep (DSPO), {} eval ep =='.format(
        opt.seeds[0], opt.train_episodes, opt.eval_episodes))
    for p in opt.policies:
        m = run_one(p, opt.seeds[0], opt.scaler, opt.train_episodes, opt.eval_episodes, opt.buffer_size)
        print('{:>14s}: total={:9.2f} home%={:5.1f} travel={:8.2f} service={:7.2f} '
              'fail={:6.2f} disc={:7.2f} chrg={:6.2f} [train {:.1f}s, total {:.1f}s]'.format(
                  p, m['total_costs'], 100*m['pct_home'], m['travel_costs'], m['service_costs'],
                  m['failure_costs'], m['discount_costs'], m['charge_revenue'],
                  m['wall_train_s'], m['wall_total_s']))


def mode_calibrate(opt):
    lo, hi = opt.scaler_lo, opt.scaler_hi
    print('== calibrate: bisection on mnl_dist_scaler in [{}, {}] targeting 80% home (NoPricing) =='.format(lo, hi))
    best = None
    for it in range(opt.calib_iters):
        mid = (lo * hi) ** 0.5  # log-space bisection
        m = run_one('NoPricing', opt.seeds[0], mid, 0, opt.eval_episodes, opt.buffer_size)
        print('iter {:2d}: scaler={:12.2f} -> home%={:5.1f}'.format(it, mid, 100*m['pct_home']))
        best = (mid, m['pct_home'])
        if abs(m['pct_home'] - 0.80) <= 0.01:
            break
        if m['pct_home'] > 0.80:
            lo = mid   # too much home -> make OOH more attractive -> larger scaler
        else:
            hi = mid
    scaler = best[0]
    ms = run_one('StaticPricing', opt.seeds[0], scaler, 0, opt.eval_episodes, opt.buffer_size)
    print('chosen scaler = {:.2f}: NoPricing home%={:5.1f}, StaticPricing home%={:5.1f} (targets 80/60 +-3)'.format(
        scaler, 100*best[1], 100*ms['pct_home']))
    print('acceptance: NoPricing {}  StaticPricing {}'.format(
        'PASS' if abs(best[1]-0.80) <= 0.03 else 'FAIL',
        'PASS' if abs(ms['pct_home']-0.60) <= 0.03 else 'FAIL'))


def mode_full(opt):
    assert opt.scaler > 0, 'full mode requires --scaler from calibration'
    results = {p: [] for p in opt.policies}
    for seed in opt.seeds:
        for p in opt.policies:
            print('running {} seed {} ...'.format(p, seed))
            m = run_one(p, seed, opt.scaler, opt.train_episodes if p == 'DSPO' else 0,
                        opt.eval_episodes, opt.buffer_size)
            print('  -> total={:.2f} home%={:.1f} ({:.0f}s)'.format(
                m['total_costs'], 100*m['pct_home'], m['wall_total_s']))
            results[p].append(m)
    write_report(results, opt)


def agg(results, p, key):
    vals = [m[key] for m in results[p]]
    n = len(vals)
    mean = sum(vals) / n
    std = (sum((v-mean)**2 for v in vals) / n) ** 0.5
    return mean, std


def write_report(results, opt):
    base_mean, _ = agg(results, 'NoOOH', 'total_costs')
    lines = []
    lines.append('# 论文配置复现结果(RC 合成案例)\n')
    lines.append('- 预设:论文 Table A.1/B.1/F.1;scaler={};seeds={};DSPO 训练 {} episodes,buffer {};评测 {} episodes/seed\n'.format(
        opt.scaler, opt.seeds, opt.train_episodes, opt.buffer_size, opt.eval_episodes))
    lines.append('| 策略 | home% | travel | service | failure | 折扣成本(收费收入) | 平均折扣(平均收费) | total | 相对NoOOH节省% |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for p in [x for x in ORDER if x in results]:
        tm, ts = agg(results, p, 'total_costs')
        row = '| {} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} ({:.1f}) | {:.1f}±{:.1f} ({:.1f}±{:.1f}) | {:.1f}±{:.1f} | {:.1f} |'.format(
            p, 100*agg(results, p, 'pct_home')[0],
            agg(results, p, 'travel_costs')[0], agg(results, p, 'service_costs')[0],
            agg(results, p, 'failure_costs')[0],
            agg(results, p, 'discount_costs')[0], agg(results, p, 'charge_revenue')[0],
            agg(results, p, 'avg_discount')[0], agg(results, p, 'std_discount')[0],
            agg(results, p, 'avg_charge')[0], agg(results, p, 'std_charge')[0],
            tm, ts, 100*(1 - tm/base_mean))
        lines.append(row)
    means = {p: agg(results, p, 'total_costs')[0] for p in results}
    expected = [p for p in ORDER if p in results]
    actual = sorted(means, key=means.get, reverse=True)
    lines.append('\n- 目标排序(成本降序):{}'.format(' > '.join(expected)))
    lines.append('- 实际排序(成本降序):{}'.format(' > '.join(actual)))
    lines.append('- 排序判定:{}\n'.format('**成立**' if actual == expected else '**不成立(如实报告,见上表)**'))
    out = path.join(ROOT, 'docs', 'paper_repro_results.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('report written to', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['smoke', 'calibrate', 'full'])
    ap.add_argument('--scaler', type=float, default=0.0, help='mnl_dist_scaler; 0=auto(amax)')
    ap.add_argument('--scaler_lo', type=float, default=200.0)
    ap.add_argument('--scaler_hi', type=float, default=250000.0)
    ap.add_argument('--calib_iters', type=int, default=12)
    ap.add_argument('--train_episodes', type=int, default=3, help='DSPO training episodes')
    ap.add_argument('--eval_episodes', type=int, default=2)
    ap.add_argument('--buffer_size', type=int, default=500)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1234])
    ap.add_argument('--policies', nargs='+', default=ORDER, choices=ORDER)
    opt = ap.parse_args()
    {'smoke': mode_smoke, 'calibrate': mode_calibrate, 'full': mode_full}[opt.mode](opt)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 语法检查**

Run: `python -c "import ast; ast.parse(open(r'Experiments\run_paper_repro.py',encoding='utf-8').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 3: Commit**

```powershell
git add Experiments/run_paper_repro.py
git commit -m "Add paper-preset experiment runner (smoke/calibrate/full)"
```

---

### Task 7: 冒烟运行

**Files:** 无新文件;可能触发 hygese 处置。

**Interfaces:**
- Consumes: Task 1–6 全部。
- Produces: 每策略单 episode 耗时数据(用于 Task 9 定规模);全链路跑通的确认。

- [ ] **Step 1: 跑冒烟**

Run: `python Experiments\run_paper_repro.py --mode smoke --eval_episodes 2 --train_episodes 3`
Expected: 5 行结果(NoOOH home%=100,OnlyOOH home%=0,其余介于中间),无异常栈。注意:DSPO 在 3 个 episode 内不会完成初始训练(buffer 500 未满),其行为等价 NoPricing——这是预期,冒烟只验证管线。

- [ ] **Step 2: 若 hygese 抛负坐标 assertion**

按 `Parcelpoint_py.py:57` 的提示,找到已安装的 `hygese.py`(`python -c "import hygese; print(hygese.__file__)"`),注释掉其中对坐标 `>= 0` 的 assert 行,重跑 Step 1。在最终报告中记录做过此处置。

- [ ] **Step 3: 记录耗时**

从输出记录每策略 `total X.Xs`,估算:Baseline 单 eval episode 耗时、DSPO 单训练 episode 耗时。写进临时笔记(Task 9 要用)。

- [ ] **Step 4: Commit(若有 hygese 之外的小修)**

无代码变更则跳过。

---

### Task 8: MNL 校准

**Files:** 无代码变更(只跑脚本、记录 scaler)。

**Interfaces:**
- Consumes: Task 7 跑通的管线。
- Produces: 校准出的 `mnl_dist_scaler` 数值(供 Task 9 `--scaler` 使用),及 NoPricing/StaticPricing 的 home% 验收结果。

- [ ] **Step 1: 跑校准**

Run: `python Experiments\run_paper_repro.py --mode calibrate --eval_episodes 20`
Expected: 若干次 bisection 迭代后打印 `chosen scaler = ...` 与两个 PASS/FAIL。目标:NoPricing home% ∈ [77,83],StaticPricing home% ∈ [57,63]。

- [ ] **Step 2: 若 StaticPricing 验收 FAIL**

规格约定只动 scaler,不动 βᵏ/βᵈ/u₀⁺。此时选择使两目标联合误差(|home%₁-80|+|home%₂-60|)最小的 scaler:在 chosen scaler 附近手工再试 2–3 个值(如 ×0.7 / ×1.4),用 `--mode calibrate --scaler_lo X --scaler_hi X` 收窄区间观察。最终值与偏差如实记录,进入结果文档的"局限性"。

- [ ] **Step 3: 记录**

把最终 scaler 值和两个 home% 写进临时笔记(Task 9 报告要引用)。

---

### Task 9: 定规模(用户拍板)→ 全量跑批 → 结果文档

**Files:**
- Create(由脚本生成): `docs/paper_repro_results.md`

**Interfaces:**
- Consumes: Task 7 的耗时数据、Task 8 的 scaler。
- Produces: 3 seed 全量结果表 + 排序判定,提交到分支。

- [ ] **Step 1: 向用户提出规模建议并等确认**

用 Task 7 实测耗时估算总时长,给出建议(例如:DSPO `--train_episodes` 取过夜可完成的量、`--buffer_size` 相应放大到几千、`--eval_episodes 20`、`--seeds 1234 2345 3456`),报给用户拍板。**未确认前不启动全量。**

- [ ] **Step 2: 全量跑批**

Run(episodes 数以用户确认值为准):

```powershell
python Experiments\run_paper_repro.py --mode full --scaler <Task8的值> --train_episodes <确认值> --buffer_size <确认值> --eval_episodes 20 --seeds 1234 2345 3456
```

Expected: 15 次运行(5 策略 × 3 seed)逐行打印,末尾 `report written to ...\docs\paper_repro_results.md`。

- [ ] **Step 3: 核对结果文档**

打开 `docs/paper_repro_results.md`:
- 表格 5 行齐全,NoOOH home%=100、OnlyOOH home%=0;
- 排序判定行如实反映(成立/不成立);
- 把 Task 7 hygese 处置(如有)、Task 8 校准偏差(如有)、训练量缩尺说明补进文档"局限性"段(手工编辑追加)。

- [ ] **Step 4: 提交**

```powershell
git add docs/paper_repro_results.md
git commit -m "Add paper-faithful 5-policy reproduction results (3 seeds, RC)"
```

- [ ] **Step 5: 对照验收标准逐条报告**

按规格 §1 的 4 条验收标准逐条给出 PASS/FAIL 与证据,向用户汇报;排序不成立时附分析(训练量、收敛、各成本分量对比),不调参硬凑。
