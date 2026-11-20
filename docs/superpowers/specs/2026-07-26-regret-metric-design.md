# 设计:决策遗憾评测 + 真实成本定价上界(oracle benchmark)

日期:2026-07-26
目的:为 SPO+ 有效性提供第 1 层判据(机制级)。在共享留出标签集上,对任意已训练模型计算
决策遗憾 = R(a*(c);c) − R(a*(ĉ);c),并输出真成本定价上界 R(a*(c);c),配套逐客户配对统计检验。
背景:净利润在快速档三轮实验(±2 空间 α=0.7;±10 空间 α=0.7/1.5/3.0)均无法区分 SPO+ 与 DSPO;
α≥1.5 在宽价格空间训练崩塌(角点吸收)。遗憾指标剔除到达/选择/路由噪声,是坐实或证伪有效性的决定性工具。
基线:master(含需求 1–3 成果),新分支 `regret-metric`。

## 0. 用户已拍板的决策

- 评测集来源:**共享留出集**(固定策略 + 固定种子生成一份,所有模型同一份样本,逐客户配对)。
- 评测时机:**存模型 + 离线脚本**(今后 run 传 `save_model=True`;独立脚本加载 checkpoint 评测)。
- 本轮范围:重训 DSPO(p10) 与 SPO+ α=0.7(p10) 两组 × 3 seed 拿 checkpoint;seed 扩容、燕郊数据均不做。

## 1. 验收标准

1. **工具正确性**:玩具案例(2 选项,手算 MNL 期望利润)上遗憾数值与手算一致(容差 1e-6);
   regret(c,c)=0 精确成立(同输入同 oracle 必同输出;不为 0 即实现 bug)。全数据集上遗憾**原则上** ≥0,
   但 oracle 是近似解(闭式+裁剪),边界饱和区可能出现小幅负值——按诊断处理:统计负值样本数与幅度并写入报告,
   若负值占比 >5% 或幅度中位数超过平均遗憾的 10%,说明 oracle 近似误差掩盖信号,升级为阻断问题。
2. **可复现**:同一 checkpoint + 同一数据集文件,两次运行输出逐位一致(无随机性;torch 推理 eval 模式)。
3. **产出完整**:每模型报告——平均遗憾、相对遗憾(遗憾/oracle 利润)、oracle 利润上界、逐点 MSE;
   两模型间——逐客户配对差、Wilcoxon 符号秩检验 p 值。输出 markdown 表(`docs/req3_regret_results.md`)。
4. **判读预案三选一落袋**(写入结果文档):
   (a) DSPO 相对遗憾 < 1% → 判"本合成场景无改进空间",净利润平局定性为场景性质;
   (b) DSPO 遗憾可观且 SPO+ 显著更低(Wilcoxon p<0.05)→ 有效性第 1 层判据达成;
   (c) DSPO 遗憾可观但 SPO+ 未降 → 训练方法问题坐实,后续启动稳定化改造(镜像截断优先,另立需求)。

## 2. 组件

### C1 共享标签集生成:`Experiments/gen_regret_dataset.py`

- 固定策略 **StaticPricing**(与所有被评模型无关,保证公平;复用 `run_paper_repro.POLICIES['StaticPricing']` 配置)
  + 测试数据种子(`data_seed_test=1` 口径)+ 固定 run seed(9999,与训练 seed 池不相交),
  跑 20 个评测 episode(约 1800 客户)。
- 每位客户记录一条样本:
  - **CNN 输入**:该客户全部候选选项的 φ(s,k) 输入张量(feat / cap_feat,与训练缓冲同构,
    参照 `Src/Utils/Utils.py` DataSet 的 features/capacity_features 存储格式);
  - **真实成本向量 c**:终局路线反事实标签,复用 `DSPO_plus_SPO` 现有标签机制
    (`_get_terminal_label_route` → HEP 边际插入,与 `spo_training_data` 的 `true_costs` 同一口径);
  - **选项属性**:计算 MNL 效用所需字段(选项数 K、步行距离项、服务时间/τ 代理、home 标志),
    与 `spo_training_data` 条目的 `customer_info` 同 schema(写计划时核实该 schema 是否已含
    oracle `_pricing_oracle_lifted_torch` 所需全部字段——SPO+ 损失已在用它重建 R,应当齐备;缺则补记)。
- 存盘:单文件 `Experiments/regret_dataset_RC_seed9999.pt`(torch.save 的 dict:样本列表 + 生成配置元数据)。
  生成一次,全部模型共用;文件带配置指纹(scaler/V0/价格域/instance),评测脚本校验指纹防口径混用。
- 实现方式约束:标签收集挂在 StaticPricing rollout 上,**不训练任何模型**;
  可通过实例化 DSPO_plus_SPO(spo_loss_weight>0 以启用标签收集、跳过 optimize)或提取标签函数复用实现,
  写计划时选定一种,以改动最小者优先。

### C2 离线遗憾评测:`Experiments/eval_regret.py`

- 输入:`--checkpoint <run目录>`(可多个,依次评测)+ `--dataset <数据集文件>`;
  从 run 目录的 config namespace 恢复模型结构超参并加载 CNN 权重(Predictors 的 load 接口)。
- 对每位客户:
  1. 前向得 ĉ(eval 模式,无梯度);
  2. 同一有界 oracle(闭式 Lambert-W + 裁剪到 [min_price, max_price],与训练口径一致,
     复用/提取 `_pricing_oracle_lifted_torch` 的求解路径)分别解 a*(ĉ) 与 a*(c);
  3. 在**真实成本 c** 下算期望利润 R(a*(ĉ);c) 与 R(a*(c);c),遗憾取差;逐点误差取 MSE(ĉ,c)。
- 输出:终端表 + `docs/req3_regret_results.md`(每模型汇总行 + 模型对之间的配对统计)。
  配对检验用 scipy.stats.wilcoxon(py39 环境已有 scipy;写计划时核实版本接口)。
- 价格域取自数据集元数据(本轮 [−10,10] 口径),与 checkpoint config 不一致时响亮报错。

### C3 重训拿 checkpoint(不改算法代码)

- 2 组 × 3 seed,全部并行(6 进程,~40 分钟墙钟):
  - DSPO,max_price=10,`save_model=True`,folder `req3p10ck_DSPO_{seed}`;
  - DSPO_plus_SPO,α=0.7,max_price=10,`save_model=True`,folder `req3p10ck_SPOa07_{seed}`;
- 其余配置与实验 2/2b 完全一致(V0=0,home_failure=0,scaler=1800,500 ep,buffer 10000,eval 20 ep)。
- 附带核对:新 run 的净利润应与实验 2/2b 同 seed 结果一致(同配置同种子,允许保存模型带来的
  无关差异为零——save_model 不影响训练路径;若不一致,先查再评)。

## 3. 统计与报告口径

- 主对比:同 seed 的 DSPO vs SPO+ checkpoint 在同一数据集上——遗憾对(每客户一对),
  Wilcoxon 符号秩(双边);3 个 seed 各自报告 + 合并(逐客户配对合并前提是同一数据集,成立)。
- 报告字段:平均遗憾、相对遗憾%、oracle 利润、模型利润、MSE、配对差中位数、p 值。
- 叙事对照:MSE 与遗憾的方向若相反(SPO+ MSE 更高而遗憾更低),明确标注为决策导向签名。
- 如实报告红线不变:三种判读预案(§1.4)均为合法结论,不调参硬凑。

## 4. 测试

- `tests/test_regret_toy.py`:2 选项手算案例——给定 c、ĉ、MNL 参数,手算 a*(c)/a*(ĉ)/两个 R 值与遗憾,
  与脚本核心函数输出对齐(容差 1e-6);并断言 regret(c,c)=0。
- `tests/test_regret_dataset_roundtrip.py`:生成迷你数据集(1 episode)→ 存盘 → 载入 → 指纹校验通过、
  样本字段齐全、c 向量长度与选项数一致。
- 既有 15 个测试全绿(本需求不改算法代码,理论上零影响;跑一遍确认)。

## 5. 风险与说明

- **分布偏移**:共享集由 StaticPricing 生成,与各模型部署分布不同——文中注明;这是配对公平与
  on-policy 真实性的标准权衡,主目的(区分预测质量的决策后果)不受影响。
- **oracle 近似**:闭式 + 裁剪不是有界问题精确解——两模型同一口径,遗憾差不受系统偏差影响;
  绝对遗憾值含近似误差,报告中注明(与审计文档偏差项一致)。
- **标签噪声**:c 本身是事后样本,遗憾的"真值"含标签噪声;逐客户配对 + 大样本(~1800)摊薄。
- checkpoint 保存的是全部 modules;评测只需 CNN 分支,加载时按名取(写计划时核实 module 命名)。

## 6. 交付物

- 代码:`Experiments/gen_regret_dataset.py`、`Experiments/eval_regret.py`(分支 `regret-metric`);
- 数据:`Experiments/regret_dataset_RC_seed9999.pt`(生成产物,不入 git;.gitignore 核实);
- checkpoint:C3 两组 6 run(产物目录,不入 git);
- 测试:§4 两个新测试;
- 文档:`docs/req3_regret_results.md`(结果 + §1.4 判读落袋);
- 结论:验收标准 §1 四条逐条判定。
