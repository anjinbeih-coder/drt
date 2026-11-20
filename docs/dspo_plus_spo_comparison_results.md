# NoPricing / StaticPricing / DSPO / DSPO_plus_SPO 对比结果

跑于 2026-07-24，`--instance RC --max_episodes 10 --seed 1234`（当前默认超参数，未做论文级对齐），四组结果均来自各自 `Logs/logfile.log` 末尾的 `eval2()` 汇总。

## 是否跑通

| 方法 | 是否跑完不崩 | SPO+ 是否实际触发（仅 DSPO_plus_SPO 适用） |
|---|---|---|
| NoPricing | 是 | - |
| StaticPricing | 是 | - |
| DSPO | 是 | - |
| DSPO_plus_SPO | 是（跑完不崩），但 **SPO+ 训练本身完全没有生效** | 否——10 个 episode 里 10 次都报错 `'DSPO_plus_SPO' object has no attribute '_get_terminal_label_route'`（`grep -c "SPO+错误" logfile.log` = 10） |

## 数值对比

| 指标 | NoPricing | StaticPricing | DSPO | DSPO_plus_SPO |
|---|---|---|---|---|
| percentage home delivery | 0.8291 | 0.6947 | 0.6092 | 0.6299 |
| travel costs | 3729.52 | 3702.88 | 3655.12 | 3615.12 |
| service costs | 1714.23 | 1436.17 | 1259.55 | 1302.40 |
| failure costs | 620.2 | 519.6 | 455.7 | 471.2 |
| Avg. Charge | nan (std nan) | 1.0 (std 0.0) | 2.764 (std 1.940) | 2.764 (std 1.937) |
| Avg. Discount | -0.0 (std 0.0) | 2.0 (std 0.0) | 3.123 (std 2.184) | 2.787 (std 2.036) |
| Charge revenue | 0.0 | 259.8 | 42.43 | 42.28 |
| Discount costs | -0.0 | 228.4 | 1120.09 | 999.52 |
| **total costs** | **6063.95** | **5627.25** | **6448.04** | **6345.95** |

（NoPricing 的 Avg. Charge 是 `nan` 属于预期行为，不是异常：`price_home=0.0`/`price_pp=-0.0` 时价格恒为 0，环境只在 `price>0` 时才把它记进 `total_prices`，所以这一整个列表在 20 个评估 episode 里始终为空，`np.mean([])`/`np.std([])` 自然是 `nan`。）

## 观察

1. **本次跑出来的四者成本排序是 StaticPricing(5627) < NoPricing(6064) < DSPO_plus_SPO(6346) < DSPO(6448)。** 这跟"整个项目需求总结7.23.md"里写的目标排序（NoPricing > StaticPricing > DSPO，即 NoPricing 成本最高、DSPO 最低）方向相反——当前默认超参数、单 seed、10 episode 的这次跑法，本来就不是为了验证那个排序，看这个数字目前没有意义，不能当结论用（见下面"局限性"）。
2. **DSPO_plus_SPO 的 total costs（6345.95）确实比 DSPO（6448.04）低约 102**，乍看像是"更好"，但这个差异**不能**归因于 SPO+ 方法生效——因为 SPO+ 训练全程没有跑起来（见下一条）。更可能的解释是：`DSPO_plus_SPO.get_action_pricing` 是重写的一份实现，即便数学上等价，也会消耗不同的随机数序列（`np.random`/`torch` 的调用顺序不同），导致哪怕逻辑等价也会产生不同的仿真轨迹，这属于随机噪声而非算法效果差异。
3. **关键发现（超出原计划范围，需要讨论）：`DSPO_plus_SPO.py:637` 调用了 `self._get_terminal_label_route(data, route_ids)`，但这个方法在 `DSPO_plus_SPO` 和父类 `DSPO` 里都**没有定义**。`update()` 里这处调用包在 `try/except` 里，异常被吞掉只打印一行 `[SPO+错误]`，不会让整个训练崩溃——但后果是**每个 episode 的 SPO+ 训练标签收集都失败**，`spo_training_data` 永远是空的，`_get_current_spo_weight()` 第一行 `if len(self.spo_training_data) == 0: return 0.0` 直接返回 0，SPO+ 损失全程权重为 0，等价于全程只用 Huber Loss 在训练——也就是说，**这次"对比"实际上根本没有测试到 SPO+ 方法本身，DSPO_plus_SPO 这一栏的数字目前只反映"重新实现了一遍 DSPO 的定价逻辑（l0_home/l_mp=0，等价于原版）+ 不同随机数序列"，不反映 SPO+ 是否有效**。
   - 这是本次执行计划（Tasks 1-4）的验证范围之外新发现的第五个缺陷：Tasks 1-4 用的冒烟测试只测到 `update(..., done=False)`（非终止步），没有测过 `done=True`（episode 结束时）才会触发的这条代码路径，所以之前没暴露出来。
   - 从代码看，`_get_terminal_label_route(data, route_ids)` 的调用方式（返回 `(fleet, final_cost)`，用于取代原本调用 `self.reopt_HGS_final(data)` 的位置）很像是想封装/替换基类 `DSPO.reopt_HGS_final(data)`，但 `route_ids` 参数具体要怎么用（是否要先按 `route_ids` 过滤/去重再传给 HGS，还是单纯传参占位）不清楚，**不应该在没有确认设计意图的情况下直接猜一个实现**。

## 局限性（据 spec 明确写明，不要省略）

- 单 seed、10 episode，统计上不严谨，不能当作显著性结论。
- `docs/cost_alignment_audit.md` 里仍有两个未修的成本口径问题（OOH 服务时间恒为 0、Homberger 数据集燃油按时间而非距离计费），同等影响四组的绝对数值。
- StaticPricing 的价格（home=1.0, pp=-2.0）是本次示意值，不是论文调优值。
- **最重要的一条**：DSPO_plus_SPO 这次的结果不能用来评价 SPO+ 方法本身是否有效——SPO+ 训练全程没有真正跑起来（见上文"关键发现"）。要想验证"DSPO_plus_SPO 是否比 DSPO 更好"这个原始假设，必须先修好 `_get_terminal_label_route` 缺失的问题，再重新跑一遍对比。
