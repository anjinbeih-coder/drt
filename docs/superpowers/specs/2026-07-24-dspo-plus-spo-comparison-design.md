# DSPO_plus_SPO：让四方对比（NoPricing / StaticPricing / DSPO / DSPO_plus_SPO）可运行

## 背景

用户已经实现了 `Src/Algorithms/DSPO_plus_SPO.py`（在 `DSPO` 基础上叠加 "lifted SPO+" 训练损失），假设是：如果 SPO+ 这个决策感知（decision-aware）训练目标真的有效，`DSPO_plus_SPO` 的总成本应该优于 `DSPO`，两者又都应该优于 `NoPricing`/`StaticPricing`。但目前 `DSPO_plus_SPO.py` 根本无法运行，也不确定它内部的成本计算是否正确，需要先讨论清楚再动手，尤其是成本计算相关的部分。

## 现状排查

`DSPO_plus_SPO.py` 目前无法实例化/运行，原因有三层，层层深入：

1. **表层**：顶部 `from Src.Utils.passenger_utility import utility_home_nonprice`，但仓库里根本没有 `Src/Utils/passenger_utility.py` 这个文件——`ImportError`，模块都导不进来。
2. **属性缺失**：`self.l0_home`、`self.l_mp`、`self.use_travel_time_prediction` 在 `__init__`（包括父类 `DSPO.__init__`）里都从未被赋值，一旦执行到 `get_action_pricing`/`update` 会抛 `AttributeError`。
3. **更深层、之前没发现的问题**：交叉核对 `docs/manuscript/part_c模版_DRT_0522_manuscript_working.md` 后发现，`utility_home_nonprice`、`l0_home`/`l_mp`、`travel_time_weight`、`use_travel_time_prediction` 这些命名，对应的是用户新章节里描述的 **DRT 载客场景**（home pickup vs. 会合点 meeting point，效用含车内行程时间 τ_tk）。但 `DSPO_plus_SPO` 目前挂在的仍是**包裹环境** `Environments/OOH/Parcelpoint_py.py` + `customerchoice.py`：
   - `customerchoice.py` 里的 `customerchoicemodel` 类**没有** `travel_time_weight` 属性；
   - 其 `mnl_distmat`/`mnl_euclid` 方法**不接受** `travel_time` 关键字参数；
   - `self.get_travel_time_prediction(...)` 方法在 `DSPO_plus_SPO`/`DSPO` 里都**没有实现**。

   也就是说，`DSPO_plus_SPO.py` 是按照"未来要接入的 DRT 环境"设计的，但现在跑在旧的包裹环境上，这是比"缺几个初始化"更大的一处设计/环境不匹配。

## 与用户确认的决策

通过 brainstorming 逐一确认：

1. **运行环境范围**：这次对比就跑在**当前的包裹 OOH 环境**（`Parcelpoint_py`）上，不接入 DRT/会合点环境——DRT 相关逻辑先关闭/兜底，不是本次任务范围。
2. **`l0_home`/`l_mp` 语义**：用户确认"先设为 0，别的地方不动"——不引入新的未验证常数，让 `DSPO_plus_SPO` 在包裹场景下的定价公式退化成与基类 `DSPO` 完全一致（这两项目前在基类里就是隐式的 0）。
3. **`travel_time_weight`/`mnl(travel_time=...)` 的处理方式**：为了完全不碰 `customerchoice.py`/`DSPO.py`，改成让 `DSPO_plus_SPO.py` 内部判断"当前 `customerchoice_model` 是否真的有 `travel_time_weight` 属性"，没有就走代码里本来就存在、逻辑正确的 fallback 分支（用继承自 `DSPO` 的 `self.mnl(...)`，不含 travel_time）。这个 fallback 分支已经写好，只是触发条件写反了（原来写成"只要 customerchoice_model 不是 None"，应该是"只要它真的配了 travel_time_weight"）。
4. **实验规模**：先用当前默认超参数（`--instance RC`、`max_episodes=10`、单 seed 1234）快速看趋势，不追求论文级别的严谨复现（多 seeds/更多 episode 留到后续单独任务）。

## 设计：Part 1 — 让 `DSPO_plus_SPO` 可运行

**改动范围严格限制在两个文件：`Src/Algorithms/DSPO_plus_SPO.py`（主要改动）和 `Src/parser.py`（一行，加 choices）。不改 `DSPO.py`、`customerchoice.py`、`Utils.py` 等其他任何文件。**

在 `DSPO_plus_SPO.py` 内：

1. 移除对不存在模块的 import，改为在本文件内定义等价的私有函数：
   ```python
   def _utility_home_nonprice(base_util, home_util, travel_time_weight, home_travel_time):
       if travel_time_weight is not None and home_travel_time is not None:
           return base_util + home_util - travel_time_weight * home_travel_time
       return base_util + home_util
   ```
   所有原来调用 `utility_home_nonprice(...)` 的地方改为调用这个新函数。当前场景下 `travel_time_weight`/`home_travel_time` 恒为 `None`（见第 3 点），效果等价于 `base_util + home_util`，与 `DSPO.py`/`customerchoice.py` 里 home 效用的算法一致。

2. `__init__` 新增：
   ```python
   self.l0_home = 0.0
   self.l_mp = 0.0
   self.use_travel_time_prediction = False
   ```
   `use_travel_time_prediction=False` 使得所有 `if self.use_travel_time_prediction:` 分支（调用未实现的 `get_travel_time_prediction`/`set_travel_times`）整体跳过。

3. 把文件里**全部 8 处**
   ```python
   if customerchoice_model is not None:
   ```
   （`get_action_pricing` 的 `initial_phase` 分支和非 `initial_phase` 分支各有 4 处：home 效用一处、pp/OOH 的 `mnl` 调用一处、`pps_info` 构建一处，外加各自的重复计算——具体行号以实际改动时的 grep 结果为准，不要按行号硬编，逐处确认改全）统一改为：
   ```python
   if customerchoice_model is not None and getattr(customerchoice_model, 'travel_time_weight', None) is not None:
   ```
   当前 `customerchoicemodel` 没有该属性，条件恒为假，8 处都会自动走已有的、正确的 `else` 分支。**改的时候要用 grep 数一遍确认 8 处全改到，不能漏。**

4. `Src/parser.py` 第 25 行 `--algo_name` 的 `choices` 列表加入 `'DSPO_plus_SPO'`。

**改动之后要验证的点**（不是要修的新 bug，是确认修复到位）：
- `DSPO_plus_SPO(config=config)` 能正常实例化，不抛 `ImportError`/`AttributeError`。
- `get_action_pricing` 在 `initial_phase` 和非 `initial_phase` 两个分支下都能正常返回价格向量。
- `update(...)` 能正常跑完一个 episode 不报错。
- SPO+ 损失确实在跑（日志里能看到 `[SPO+ debug] spo_training_data populated` 和 `[SPO+ debug] spo_weight became positive` 这两行，说明训练数据积累和权重升温机制都被触发了，不是静默 no-op）。

## 设计：Part 2 — 对比实验

跑 4 组，全部用 `run.py` 现成的 `eval2()` 输出做对比（跟本次会话之前验证过的口径一致）：

| 组别 | 命令 |
|---|---|
| NoPricing | `python run.py --algo_name Baseline --price_home 0.0 --price_pp -0.0` |
| StaticPricing | `python run.py --algo_name Baseline --price_home 1.0 --price_pp -2.0` |
| DSPO | `python run.py --algo_name DSPO` |
| DSPO_plus_SPO | `python run.py --algo_name DSPO_plus_SPO` |

（NoPricing/DSPO 两组其实就是当前 parser.py 的默认值，写全参数只是为了显式、可复现。StaticPricing 的具体价格是本次快速趋势验证的示意值，非论文调优值。）

其余参数保持当前默认（`--instance RC --max_episodes 10 --seed 1234`），跟之前 smoke test 一致，图快出结果。

**验收标准**：
1. 四组全部跑完不崩溃。
2. `DSPO_plus_SPO` 日志确认 SPO+ 损失被实际触发（见上）。
3. 汇总四者的 `total costs`（以及 `percentage home delivery`、`Avg. Charge`/`Avg. Discount` 等辅助指标）做一次定性对比，如果 `DSPO_plus_SPO` 出现明显反常（比如比 NoPricing 还差很多），记录下来作为后续排查线索，而不是当场下"SPO+ 有效/无效"的结论——单 seed、10 episode 在统计上不严谨，且 `docs/cost_alignment_audit.md` 里还有两个已知未修的成本口径问题（OOH 服务时间恒为 0、Homberger 数据集燃油按时间计费），这些对四组是同等影响，不会系统性偏向某一组，但会影响绝对数值的可信度。

## 明确不做的事（本次任务范围之外）

- 不接入 DRT/会合点环境，不实现 `get_travel_time_prediction`/`set_travel_times`。
- 不修 `docs/cost_alignment_audit.md` 里剩下的两个已知问题（OOH 服务时间、Homberger 燃油单位）。
- 不追求论文级别的多 seed 严谨对比（留作后续任务）。
- 不改 `DSPO.py`、`customerchoice.py`、`Utils.py`。
