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
| α 调度 | 预热 α=0 后逐步升高 | warmup(5 ep)→线性 rampup(10 ep)→封顶 0.7,自首个有标签 episode 起计(2026-07-25 重锚定) | 一致(线性化实现) |
| 真实成本 c | option-level marginal cost(终局路线) | `_get_terminal_label_route`=HGS 终局重优化,选中项 HEP 摊销、未选项最便宜插入+服务成本 | 一致(方案 A,与 Huber 标签同锚;用户 2026-07-25 拍板) |
| oracle w*(·) | argmax_a R(a;c) | Lambert 闭式解(含外部项除法不动点)+ [min_p,max_p] 裁剪;与部署定价规则同形(safety_margin 启发式已于本分支移除) | 边界内精确;近边界为近似(见 §2) |

## 2. 记录在案的偏差(不修,含理由)

1. **/(K+1) 归一化**:代码把 SPO+ 损失除以 (K+1)(选项数+1)。canonical 公式无此项;为纯缩放,
   与 α 合并等效(α_eff = α/(K+1) 相对 Huber)。保留,手稿实现说明中注明。
2. **oracle 边界近似**:价格不动点是无约束闭式解,随后 clip 到 [min_p,max_p];
   价格落在界内时为精确 argmax,压边界时为近似。原实现中的负价 safety_margin 缩放已移除,
   使 oracle 与部署规则严格同形,故"oracle 与部署规则一致"这一 SPO 的关键前提成立。
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
