# 设计:按论文原文配置复现 5 策略成本对比(需求 1)

日期:2026-07-24
对应需求:《整个项目需求总结7.23.md》第 1 条 —— 用原文配置和参数复现 Akkerman-TS24(理论排序:成本 NoOOH > NoPricing > StaticPricing > DSPO > OnlyOOH)。

## 1. 范围与目标

在 `paper-repro` 分支上,复现论文(Akkerman, Dieter, Mes, Transportation Science 2024)**合成案例 RC 实例**的 5 个策略对比:

| 策略 | 论文名 | 含义 |
|---|---|---|
| NoOOH | NoOOH | 只上门(需求里的 onlyhome),成本近似上界 |
| NoPricing | NoPricing | 可选 OOH 但无定价 |
| StaticPricing | StaticPricing | 固定 OOH 折扣 5 / 上门收费 2 |
| DSPO | DSPO | 论文主方法(CNN 成本预测 + Lambert-W 定价) |
| OnlyOOH | OnlyOOH | 只自提点(需求里的 onlymeetingpoint),成本近似下界 |

**成功标准(验收):**

1. 3 个评测 seed 均值上,总成本排序 NoOOH > NoPricing > StaticPricing > DSPO > OnlyOOH 成立;
2. 各策略上门率与论文 Table 2 方向一致(100 / ~82 / ~59 / ~70 / 0);
3. MNL 校准通过:NoPricing 下 home% ≈ 80%,StaticPricing(5/2)下 home% ≈ 60%(论文 Appendix B 的调参目标,容差 ±3pt);
4. 绝对成本数值**不要求**对上论文(训练量缩尺,见 §6)。

**不在范围内**:Hindsight / Foresight / Linear / PPO / DSPO_plus_SPO(需求 2/3 处理);Seattle/Amazon 案例;有限容量实验。

## 2. 参数对齐

论文出处:Table A.1(问题参数)、§5.2.1(实例与到达分布)、Table B.1(MNL)、§5.2.4(StaticPricing 取值)、Table F.1 + Appendix F 正文(DSPO 超参)、Appendix D(N=20、初始训练集 100 万条)。

所有值写在 `Experiments/run_paper_repro.py` 的预设 dict 里覆盖 parser,**不改 `Src/parser.py` 的默认值**(避免影响仓库里其他在途工作)。

| parser 参数 | 论文值 | 当前默认 | 备注 |
|---|---|---|---|
| instance | RC | RC | 数据 seed:train=0,test=1(现状不变) |
| max_steps_r / max_steps_p | 90 / 0.5 | 700 / 0.5 | 负二项 E[D]=90 |
| n_vehicles / veh_capacity | 9 / 10 | 15 / 25 | |
| fuel_cost | 0.3 | 0.6 | |
| driver_wage | 30 | 30 | |
| failure_cost | 10 | 20 | |
| home_failure | 0.1 | 0.1 | |
| revenue | 50 | 50 | |
| truck_speed | 30 | 30 | |
| clip_service_time | 10(下界 1 也要保证) | 10 | lᵢ ∈ [1,10] 分钟,six-hump camel |
| min_price / max_price | -10 / 2 | -6 / 5 | |
| k(提供 OOH 数) | 20,但 RC 只有 10 个 OOH → 实际全部提供 | 8 | 需处理 k ≥ 可选数时 `argpartition` 越界 |
| fraction_capacitated | 0.0(无限容量) | 0.38 | |
| incentive_sens(βᵈ) | -0.25 | -0.25 | |
| home_util(u₀⁺) | 3.2 | 3.2 | |
| base_util | 0(论文式(5)无基础效用,被 β 吸收) | -2.0 | 见 §3.1 |
| grid_dim(M=100) | 10 | 11 | |
| n_input_layers(Dᵀ) | 3 | 2 | |
| batch_size | 64 | 256 | |
| learning_rate | 1e-3 | 1e-3 | |
| n_filters | 32(第二层 64) | 16 | kernel 3×3、pool 2×2 代码已与 Appendix F 一致,无需改 |
| dropout | 0.0 | 0.05 | |
| Huber δ | 1.0 | 1.0(`DSPO.py:51`,已一致) | |

策略专属参数:

- NoPricing:`Baseline`,price_home=0.0,price_pp=0.0;
- StaticPricing:`Baseline`,price_home=+2.0,price_pp=-5.0(注意符号约定按现有代码:pp 折扣为负);
- NoOOH / OnlyOOH:`Baseline`(价格全 0)+ 环境 `offer_mode`(见 §4);
- DSPO:`DSPO`,超参如上表。

## 3. 三处结构性修复(环境/Utils 层,5 个策略共用同一口径)

代码是原实现丢失后重写的,以下三处与论文机制不一致,本次一并修复。

### 3.1 MNL 距离效用方向(`Environments/OOH/customerchoice.py`)

- 现状:`beta_p = -exp(-d/dist_scaler)`,距离越远效用越高,方向与论文相反;
- 论文式(5):`u_k = -βᵏ·exp[d_{0,k}] + βᵈ·a_k^pricing + ε`,βᵏ=0.02,越远效用越低;home 用单独效用项 u₀⁺=3.2(Lyu & Teo 2022 做法);ε ~ Gumbel(0,1);
- 改法:OOH 效用 = `-βᵏ·exp(d/scaler)`;home 效用 = u₀⁺;base_util 置 0;价格项 `βᵈ·a` 保持现有接线;
- d 的量纲论文未写明(RC 距离可达上百,裸 exp 溢出,原文必有缩放)。**scaler 是唯一的校准自由度**,按论文 Appendix B 自己的流程标定:NoPricing → home% ≈ 80%,StaticPricing(5/2)→ home% ≈ 60%。校准只动 scaler,不动 βᵏ/βᵈ/u₀⁺。

### 3.2 OOH 服务时间(`Environments/OOH/Parcelpoint_py.py` 等)

- 现状:顾客选 OOH 时 `service_time += 0`,最终路线里 OOH 停靠不计服务时间 → OnlyOOH 服务成本恒为 0,而论文 Table 2 是 15;
- 论文口径:路线中每个**被实际使用的** OOH 停靠计一次该点的 lᵢ(式(11)同一函数生成);DSPO 训练标签中,OOH 停靠的 lᵢ 成本按使用该点的顾客数摊销(§4.1.3 原文);
- 改法:环境为每个 OOH 点生成 lᵢ;结算(含 `service_time` 统计与最终路线成本)对使用中的 OOH 停靠计一次 lᵢ;DSPO 标签生成处按人数摊销。

### 3.3 成本乘数统一(`Src/Utils/Utils.py::total_costs`、`Src/Algorithms/DSPO.py`)

- 证据修正(2026-07-25):核对论文 Table 2,NoOOH 服务成本 523.3 只有在服务时间按 `(Cw+Cf·v)/3600` 计费时才成立(纯工资口径下人均服务时长须 698 秒,超过 600 秒上限,不可能)。即论文报数口径 = 现有 `run.py::eval2` 的实现:`dist_matrix` 存时间(秒),乘数 `(Cw+Cf·v)/3600`,在恒速 v=30 下等价于"燃油按距离计费"。**eval2 不改**;
- 真正要修的是两处**乘数不一致**:`Utils.total_costs`(训练曲线)与 `DSPO.cost_multiplier`(定价用的内部成本估计)都用了 `(Cw+Cf)/3600`,漏乘 truck_speed,与评测口径差约 9 倍的燃油分量;
- 改法:两处统一为 `(driver_wage + fuel_cost*truck_speed)/3600`;dist_matrix 语义(时间,秒)保持不变。

## 4. 新增 choice-set 开关(方案 A)

`Parcelpoint_py`(经 config/parser 透传)加参数 `offer_mode ∈ {all, home_only, ooh_only}`,默认 `all`(现有行为完全不变):

- `home_only`(NoOOH):不提供任何 OOH,顾客必选上门,跳过 MNL 抽样;
- `ooh_only`(OnlyOOH):选择集去掉 home,顾客在提供的 OOH 中按 MNL(含 Gumbel 噪声)选择;价格全 0;无限容量下无需回退逻辑。

选择集约束放在环境层(它本来就属于这里),`Baseline`/`DSPO` 算法代码零改动即可复用;**不用**价格推到边界的 hack(Gumbel 噪声下不能严格保证选边,口径含糊)。

## 5. 实验脚本 `Experiments/run_paper_repro.py`

一条命令跑全部:

1. 内置 §2 预设 dict,逐策略覆盖 parser 参数;
2. 循环 5 策略 × 3 评测 seed(不同 `--seed`;数据 seed 不变 train=0/test=1);
3. 4 个无学习策略只跑评测;DSPO 先训练(episode 数为脚本参数)再评测;
4. 汇总输出论文 Table 2 同列格式对比表 → `docs/paper_repro_results.md`:home%、travel / service / failure 成本、折扣成本(收费收入)、平均折扣(平均收费)、相对 NoOOH 的节省 %(3 seed 均值 ± 标准差);
5. 冒烟与全量共用此脚本,只差 episode 参数。

## 6. 执行流程

1. **冒烟**:每策略 2–3 episode 跑通全链路,记录单 episode 耗时;确认 hygese 负坐标 assertion 在 RC 上是否触发(CLAUDE.md 已知坑);
2. **MNL 校准**:NoPricing / StaticPricing 各跑足量 episode 标定 scaler,直到 80% / 60% 验收通过;
3. **定规模**:论文原始规模(初始 100 万样本 + 10 万训练 episode,128 核 HPC)单机不可行;根据实测单 episode 耗时给出缩尺建议(过夜量级,例如初始阶段几百 episode + 训练几千 episode),**报用户拍板后**再跑全量;
4. **全量跑批**:5 策略 × 3 seed,出汇总表,对照验收标准逐条判定。

## 7. 风险与如实报告原则

- **DSPO 可能打不过 StaticPricing**:论文里两者只差 0.9pt,缩尺训练下排序可能翻转。若排序不成立,如实报告数据与分析(训练量、收敛曲线),不通过调参硬凑排序;
- ~~`k=20 > |L|=10` 的 `argpartition` 越界~~(2026-07-25 核实:定价模式选择集走 adjacency 掩码,RC 上全 1 → 10 个 OOH 全部提供,`k`/`argpartition` 只在 offering 模式使用,本实验不涉及,无需处理);
- 三处修复会改变所有后续实验的成本口径:需求 2/3 依赖本次口径,合并前需用户确认;
- 本仓库无测试套件:修复处配最小 sanity 断言(如 MNL 效用随距离单调递减、`offer_mode` 选择集正确),不引入完整测试框架。

## 8. 交付物

- 代码:三处结构性修复 + `offer_mode` + `run_paper_repro.py`(均在 `paper-repro` 分支);
- 结果:`docs/paper_repro_results.md`(对比表 + 排序判定 + 与论文 Table 2 的方向性对照);
- 结论:验收标准 §1 逐条判定,不达标项附原因分析。
