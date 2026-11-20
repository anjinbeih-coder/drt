# 设计:外部效用 + 净利润验证(需求 2)

日期:2026-07-25
对应需求:《整个项目需求总结7.23.md》第 2 条 —— 在原始结果的基础上,加入外部效用,删除上门失败的成本,验证效果。
排序方向(用户 2026-07-25 裁决):需求链为"差→好"排列,即数值上**净利润 DSPO > StaticPricing > NoPricing**。
基线:需求 1 已合并的 master(652e78f),分支 `req2-outside-option`。

## 1. 目标与成功标准

RC 实例,3 评测 seed(1234/2345/3456),scaler=1800(沿用需求 1 校准),DSPO 快速档(500 ep/seed,buffer 10000),对比 **NoPricing / StaticPricing / DSPO** 三策略(NoOOH/OnlyOOH 不在需求 2 排序内,不跑)。

**验收标准:**

1. 3 seed 均值净利润 **DSPO > StaticPricing > NoPricing**;逐 seed 翻转情况如实报告(同需求 1 做法);
2. V₀ 校准:NoPricing 下退出率落在 **2%–5%**(对齐手稿主实验的低个位数退出率);
3. 定价策略(StaticPricing/DSPO)的退出率报告在案,不设硬门槛,但若失控(如 >15%)需在结果文档分析原因;
4. `--outside_option_util` 缺省(None)时,全部现有行为逐位不变(需求 1 的 6 个回归测试全过,且不加该参数跑 smoke 结果与需求 1 一致)。

## 2. 选择模型与环境改动

### 2.1 退出选项(外部效用)

- 新 parser 参数 `--outside_option_util`,类型 float,**默认 None(关闭)**。语义:退出选项的确定性效用 V₀。命名与 `DSPO_plus_SPO` 中既有的 `getattr(self.config, 'outside_option_util', None)` 完全一致,天然接通。
- `customerchoice_pricing` 在 V₀ 开启时,选择集追加退出项:效用 `u_quit = V₀ + ε`(ε 同款标准 Gumbel;**无价格项**——退出不直接受定价影响,只受其他选项相对吸引力影响)。客户选中退出项时返回退出信号。
- 返回约定:`(None, False, -2, 0)`,即 location=None、accepted_pp=False、idx=-2(退出哨兵,区别于 home 的 -1)、price=0。
- `customerchoice_offer`(offering 变体)本需求不接入退出选项;`offer_mode` 的 home_only/ooh_only 与 V₀ 组合不在需求 2 范围(NoOOH/OnlyOOH 不跑),实现处若 V₀ 开启且 offer_mode≠all 直接 ValueError,防止未定义语义。

### 2.2 env.step 退出分支

客户选择退出(idx == -2)时:

- 不 append 坐标/id/time 到 `data`,不做 cheapest insertion,不扣自提点容量;
- `self.count_quit += 1`(`reset()` 中初始化为 0);
- 价格统计不记录(price=0 也不进 total_prices/total_discounts,避免污染折扣均值);
- stats 元组**尾部追加** `count_quit`(索引 8;旧索引 0–7 不动,向后兼容)。

### 2.3 删除上门失败成本

纯预设:`home_failure=0.0`。不改代码,eval2 的 failure 项自然为 0。

## 3. DSPO 适配(感知 V₀)

- `DSPO.get_action_pricing`:当 `config.outside_option_util` 非 None 时,Lambert-W 的自变量除以 `exp(V₀)`:`lambertw0 = (W(sum_mnl/(e·exp(V₀))).real+1)/βᵈ`。
  - 推导修正(2026-07-25,取代本规格初稿的"`sum_mnl += exp(V₀)`"写法):对含外部选项的 logit 定价,FOC 给出统一加价 m 满足 `(m−1)e^m = Σ_k exp(u_k+βᵈ(C_k−r)) / e^{V₀}`——外部项以**除法**进入 Lambert 自变量。方向自检:V₀↑ ⇒ m↓ ⇒ 所有价格统一下移(外部选项越诱人定价越温和)✓;**V₀=0 与不带外部项的现有公式逐位等价**(天然回归锚点)。加法写法方向相反,是错的。这是"适配版 DSPO"——管线(CNN 成本预测、Huber 训练、选择启发式、闭式定价)不变,只有定价推导中的 MNL 分母与问题的选择模型保持一致(Dong et al. 2009 引理覆盖含不购买项的情形;Akkerman 论文对 Yang et al. 基线也是同样的"适配后引用"做法)。论文写作时注明一句。
- 用户裁决(2026-07-25):选"感知"。理由:需求 3 的 DSPO vs DSPO+SPO 消融要干净——DSPO_plus_SPO 定价已感知 V₀,基线 DSPO 若不感知,差异会混入"感知与否"这个无关变量。

### 3.1 训练数据对齐(实现要点)

退出客户没有路线停靠 → 没有成本标签。若 `DSPO.update` 照常为退出步存特征,episode 结束时 features 数 > targets 数,`MemoryBuffer.add` 抛 ValueError。

- 改法:`DSPO.update` 非终止分支通过 `data['id']` 长度是否增长判断本步是否成交(自持 `self._last_data_len`,`reset()`/episode 结束时归零),退出步跳过特征与容量特征存储;
- **不改 update 签名**,`DSPO_plus_SPO` 零波及(其 update 覆写调用 super 途径需在实现时核对一次)。

## 4. 指标与实验入口

### 4.1 eval2 新增指标(`run.py`)

- `quit_rate = 总退出数 / 总到达数`;
- `served =` 总到达数 − 退出数;
- `net_profit = revenue×served/episodes + charge_revenue − discount_costs − travel_costs − service_costs − failure_costs`(本实验 failure 恒 0;revenue 取 config.revenue=50);
- `pct_home` 分母维持"全部到达客户"(与 quit_rate 同底,pct_home + pct_ooh + quit_rate = 1);
- V₀ 关闭时 quit_rate=0、net_profit 照公式退化,不影响需求 1 语义。

### 4.2 实验入口 `Experiments/run_req2_profit.py`

薄壳脚本,从 `run_paper_repro.py` 导入 `PAPER_PRESET`/`run_one` 等并叠加:

- 覆盖:`home_failure=0.0`、`outside_option_util=V₀`、`mnl_dist_scaler=1800`;
- 策略集:NoPricing / StaticPricing / DSPO(价格配置同需求 1);
- 模式:
  - `calibrate_v0`:NoPricing 下扫 V₀(初值区间约 [-1.5, 1.0],粗扫+细化),选退出率进 2–5% 带的值(初算 V₀≈0 附近:P(quit)=e^{V₀}/(e^{3.2}+S+e^{V₀}),S≈6.2);
  - `full`:三策略 × 3 seed,输出 `docs/req2_profit_results.md` —— 净利润表(均值±标准差)、退出率列、home%、成本分解、**按净利润降序的排序判定**(目标 DSPO > StaticPricing > NoPricing)、局限性段。

## 5. 风险与如实报告

- NoPricing 无定价 → 退出损失最小;DSPO 需靠"省成本+收费"净胜,快速档下可能均值险胜/逐 seed 翻转(需求 1 中 DSPO vs StaticPricing 即 2/3 seed 反转),如实报告,不调参硬凑;
- V₀ 校准在 NoPricing 上做;定价策略退出率会更高(上门收费推人),报告并解释;
- DSPO 训练时初始阶段以零价收集数据(等价 NoPricing),彼时退出率即校准带水平,标签分布与评测时一致性可接受;
- 退出使成交量随机下降,车队容量上限(89 步)按"到达步数"计,语义与现状一致,不改。

## 6. 交付物

- 代码:退出选项(customerchoice/env/parser/config 透传)、DSPO 感知 V₀ + 训练对齐、eval2 新指标、`run_req2_profit.py`,均在 `req2-outside-option` 分支;
- 测试:退出选项行为(V₀=None 关闭等价性、退出返回约定、极端 V₀ 下必退出)、DSPO 感知(sum_mnl 含 exp(V₀) 的定价差异)、update 跳过退出步(features/targets 等长);
- 结果:`docs/req2_profit_results.md`(表 + 排序判定 + 校准记录 + 局限性);
- 结论:验收标准 §1 逐条判定。
