# 设计:SPO 正确性修复 + 有效性验证(需求 3)

日期:2026-07-25
对应需求:《整个项目需求总结7.23.md》第 3 条 —— 检查 SPO 方法的正确性、与文章对齐,验证正确性与有效性(净利润 SPO 最高、退出率不过高、消融证明 SPO+ 有效)。
排序约定:需求链"差→好",即数值上净利润 **SPO > DSPO > StaticPricing > NoPricing**。
基线:master(含需求 1/2 成果),分支 `req3-spo-alignment`。

## 0. 审计结论(设计输入,已完成的探查)

- **SPO+ 损失数学正确**:`DSPO_plus_SPO._calculate_spo_loss_for_batch`(:960–1132)实现的
  `spo_loss = y·(w_pseudo−w_true) + 2ŷ·(w_true−w_pseudo)`(w_pseudo=w*(2ĉ−c))与
  Elmachtoub & Grigas 的 `ℓ_SPO+(ĉ,c)=max_w{(c−2ĉ)ᵀw}+2ĉᵀw*(c)−z*(c)` 代数等价;
  lifted 决策 `w(a)=[P(a),−R0(a)]` 与手稿 Eq.(23)–(31) 一致;warmup/rampup 线性调度对应 Eq.(32)。
- **阻断缺陷**:`_get_terminal_label_route` 被调用(:638)但从未定义;AttributeError 被 :777 宽 except 吞掉
  → `spo_training_data` 恒空 → spo 权重恒 0 → **SPO+ 从未参与过梯度更新**(此前所有 DSPO_plus_SPO 结果实为纯 Huber)。
- **外部选项加法形式**(8 处,含 oracle):与除法形式的正确 FOC 不符(推导见需求 2 规格)。
- **spo_* 六个超参無 parser 表面**,只有 getattr 默认值。
- 偏差记录项:`/(K+1)` 归一化(canonical 公式没有,纯缩放);oracle 为"无约束闭式解 + [min_p,max_p] 裁剪 + safety_margin"的近似;
  终局标签含 `home_failure*failure_cost` 而在线预测不含(`home_failure=0` 口径下无影响)。

## 1. 目标与验收标准

1. **SPO+ 生效硬证据(先决门槛)**:全程训练 0 次 `[SPO+错误]`(或等价失败计数为 0);`spo_training_data` 非空;
   spo 权重按 warmup(5 ep)→rampup(10 ep)爬到 0.7(日志/指标可查)。此条不过,后续对比无效。
2. **净利润排序(3 seed 均值)**:SPO > DSPO > StaticPricing > NoPricing;逐 seed 翻转如实报告。
3. **退出率**:SPO 的退出率 ≤ DSPO + 2pt;超出需分析原因(手稿叙事中 DRPO 甚至降低退出率)。
4. **消融成立**:`SPO-ablation`(α=0,同管线纯 Huber)与 DSPO 的净利润均值差 **小于两者 seed 间标准差的较大者**
   (噪声级的操作性定义,证明子类管线无私货);`SPO`(α=0.7)净利润均值 > `SPO-ablation`(证明增益来自 SPO+ 损失本身)。
   若 α=0 与 DSPO 差距超噪声级,先查明管线差异并报告,再谈 SPO+。
   (写计划时核实:`get_action_pricing`/`update` 在 `spo_enabled=False` 时开头即委托父类,故 α=0 组与 DSPO
   同 seed 应**逐位一致或极接近**——消融门槛实际上强于噪声级定义,若不一致即为红旗。)
5. **对齐审计成文**:`docs/req3_spo_alignment_audit.md` 逐项对照论文/手稿公式,给出等价性结论与偏差清单。

## 2. 修复清单(全部在 `Src/Algorithms/DSPO_plus_SPO.py`,除 F3)

### F1 缺失方法 + 响亮失败

- 定义(方案 A,用户 2026-07-25 拍板):

  ```python
  def _get_terminal_label_route(self, data, route_ids):
      """Terminal label route = the HGS re-optimized end-of-day route (same anchor as the
      Huber labels). route_ids is validated against data['id'] but takes no part in the solve."""
      assert np.array_equal(np.asarray(route_ids).ravel(), np.asarray(data['id']).ravel()), \
          "route_ids out of sync with data['id']"
      return self.reopt_HGS_final(data)
  ```

  语义依据:《SPO方法的完整理解》——标签来自当天终局路线的边际插入成本;与 Huber 标签同锚(消融干净)。
- :777 宽 except 改为:捕获后 `self.spo_label_failures += 1`(__init__ 初始化 0),打印含 traceback 类型的一行,
  episode 继续;**训练结束时**(update done 分支或 eval 前)若 `spo_label_failures > 0`,打印醒目汇总行
  `SPO+ label collection FAILED n/N episodes`——实验脚本据此判验收标准 1。
- :937 宽 except 同样计数(`spo_loss_failures`)并汇总。不再有任何静默失败路径。

### F2 外部选项除法修正(区分两种用途)

- **最优价格不动点 → 除法(修正后仅两处,2026-07-25 写计划时核实)**:
  (i) 非 initial 分支 `sum_mnl_pred`(:491–495):删除加法注入,`lambertw0_pred` 的自变量改为
  `sum_mnl_pred/(e·exp(V0))`(与父类 DSPO.py:165–170 完全同形,V₀=None/0 等价锚点);
  (ii) `_pricing_oracle_lifted_torch` 的价格不动点求和(:163–164 向 util_terms 注入处):util_terms 不再含外部项,
  `lambertw_torch` 自变量整体除以 `exp(V0)`。
  **保留加法的两处(规格初稿误列,现更正)**::386 与 :555 的 `sum_mnl_price` 是**选择概率分母**的记录量
  (只写入 decision['sum_mnl'],全库无读者),外部项作为选择集成员加进分母在语义上本来就正确,不改。
- **选择概率 softmax → 保留加法**:oracle 中计算 `P(a)`/`r0` 的 softmax 分母(:194–195)**必须保留**外部项——
  那是真实 MNL 选择概率,外部选项本来就在选择集里。修正后 oracle 语义:价格由除法不动点解出,概率由含外部项的 softmax 算出。
- 一致性测试:V₀=None 时子类定价与父类逐位一致(l0_home/l_mp=0 前提);V₀=2 时两者同向、同幅平移(±舍入容差)。

### F3 parser 补 spo_* 参数(`Src/parser.py`)

新增 6 个,默认值 = 现行 getattr 默认,保证不传参时行为不变:
`--spo_loss_weight 0.7`、`--spo_warmup_episodes 5`、`--spo_rampup_episodes 10`、`--spo_batch_size 32`、
`--spo_buffer_size 0`(0=沿用 buffer_size)、`--spo_label_sample_size 0`。
消融开关即 `--spo_loss_weight 0.0`(spo_enabled=False,跳过标签收集与 SPO 损失)。
注意:parser 提供默认后 getattr 必然命中,`__init__` 中 `spo_buffer_size` 的读取须显式处理 0→`config.buffer_size` 的回退
(原先的"attr 缺失→buffer_size"语义改为"值为 0→buffer_size")。

### F4 死代码清理(缩小审计面)

删除:`_pricing_oracle_lifted_np`(:206–303,从未调用)、`_calculate_pricing_torch_spo`(:1134–1191,从未调用)、
`SPOExperimentResult` + `self.spo_results` + `_calculate_spo_loss_statistics`(自注"非论文级指标",无读者)、
`max_discount` 死变量(:177/:257/:499)、永假 travel-time 分支(`travel_time_weight`/`use_travel_time_prediction`/
`get_travel_time_prediction` 相关守卫与调用,customerchoicemodel 无该属性,恒 False/未定义)。
`_utility_home_nonprice` 若清理后仅剩平凡分支则一并内联。清理后全部现有测试 + 新增测试必须通过。

### 记录不修(写入审计文档)

`/(K+1)` 归一化(与 α 合并等效,保留并在手稿实现说明中注明);oracle 边界近似;失败成本不对称(本实验口径 home_failure=0 无影响)。

## 3. 对齐审计文档

`docs/req3_spo_alignment_audit.md`:公式逐条对照表(SPO+ 定义/次梯度/lifted 决策/调度 ↔ 代码行号),
等价性结论,偏差清单(含"记录不修"三项),修复前后行为差异摘要(SPO+ 从未生效 → 现已生效的证据)。

## 4. 实验设计

- 配置沿用需求 2:V₀=0、home_failure=0、scaler=1800、seeds 1234/2345/3456、DSPO 类 500 ep/seed、buffer 10000、评测 20 ep/seed;
- **复用** `docs/req2_profit_results.md` 的 NoPricing/StaticPricing/DSPO 三行(同配置同 seed,不重跑);
- **新跑** 2 组 × 3 seed:
  - `SPO-ablation`:algo_name=DSPO_plus_SPO,`--spo_loss_weight 0.0`;
  - `SPO`:algo_name=DSPO_plus_SPO,`--spo_loss_weight 0.7`(warmup 5 / rampup 10 默认);
  - 每 seed 预计 30–40 分钟(SPO+ 每次 optimize 多 2×batch 次 oracle),合计 3.5–4 小时;
- 入口 `Experiments/run_req3_spo.py`(薄壳,复用 run_paper_repro 的 run_one/agg + req2_extra):
  - 模式 `full`:跑 2 新组,读入 req2 三行(硬编码其均值±std 或重读其原始 metrics?**设计定为:重跑不必要、
    直接在脚本中内嵌 req2 的逐 seed metrics 常量**,来源注明 req2 运行日志,避免口径漂移);
  - 报告 `docs/req3_spo_results.md`:五行表(净利润、退出率、home%、成本分解)+ 排序判定(SPO>DSPO>Static>NoPricing)
    + 消融判定(|ablation−DSPO| 噪声级;SPO>ablation)+ SPO+ 生效证据(标签失败计数=0、权重轨迹、spo_training_data 规模);
  - SPO+ 生效证据的采集:实验脚本从 run 的 Logs/logfile.log 中 grep 失败汇总行与权重打印(F1 保证其存在)。

## 5. 测试

- `tests/test_terminal_label_route.py`:裸对象上 `_get_terminal_label_route` 委托(桩掉 reopt_HGS_final)+ route_ids 不一致断言;
- `tests/test_spo_outside_division.py`:V₀=None 时子类 get_action_pricing ≡ 父类;V₀=2 时同向统一平移;
  oracle 的 softmax 概率仍含外部项(P 之和 < 1,缺额=退出概率);
- `tests/test_spo_weight_schedule.py`:`_get_current_spo_weight` 在 warmup 内 0、rampup 线性、封顶 max;spo_loss_weight=0 → spo_enabled False;
- 既有 9 个测试全绿(F4 清理不得破坏)。

## 6. 风险与如实报告

- 快速档下 SPO 可能赢不了 DSPO(500 ep 中 SPO+ 实际生效约 380 ep:初始阶段 ~112 + warmup/rampup 15);如实报告,必要时建议加档;
- `lambertw_torch` Newton 迭代在极端成本下的数值稳定性:oracle 输出加有限性检查,NaN/Inf 触发响亮计数(并入 F1 的失败统计);
- 消融若显示 α=0 ≢ DSPO(差距超噪声),先诊断管线差异并报告,不得直接归因 SPO+;
- 报告结果如实呈现,不调参硬凑。

## 7. 交付物

- 代码:F1–F4(分支 `req3-spo-alignment`);
- 测试:§5 三个新测试文件;
- 文档:`docs/req3_spo_alignment_audit.md`(对齐审计)、`docs/req3_spo_results.md`(实验结果 + 验收判定);
- 结论:验收标准 §1 五条逐条判定。
