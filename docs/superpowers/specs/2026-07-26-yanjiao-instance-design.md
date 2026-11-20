# 设计:燕郊–国贸真实数据实例(需求 4 · Spec A:数据管线与接入)

日期:2026-07-26
对应需求:《整个项目需求总结7.23.md》第 4 条——用高德 API 构造燕郊–国贸通勤数据,meeting 点用公交站,home 点随机生成或用 POI(两变体都建,后续按"SPO+ 相对优势更大"选定)。
范围拆分:本 spec 只覆盖 **A:数据管线 + 实例接入 + 校准冒烟**;**B:燕郊实验**在 A 验收后另立 spec。
**总目标(用户 2026-07-26 明确)**:构造该场景的目的是**让 SPO+ 在此显著有效**——B 阶段验收 = SPO+ 遗憾显著更低
(逐客户配对检验)且净利润/退出率不劣化;**不以复现手稿 Table 2 数字为目标**,Table 2 仅作参数与量级参照。
场景设计自由度(home 变体、需求/供给松紧)用于扩大"可改进遗憾空间",属正当的案例设计并如实披露;
两方法的训练与评测口径始终对等,不得削弱基线。
基线:master(含需求 1–3 全部成果),新分支 `yanjiao-instance`。

## 0. 用户已拍板的决策(2026-07-26)

1. **参数基准 = 手稿 §5.3**:170 站、300 home、E[D]=240(负二项 r=240,p=0.5)、25 车 × 12 座、
   票价 f_e=30、调价域 [−10,10]、车速 30 km/h、司机工资 30/h、油耗 0.6/km、U₀=−1.0、ASC_home=1.4、β_price=0.25。
2. **距离 = 高德驾车距离**(距离测量 API,type=1),不用实时时长——距离(米)按 30 km/h 统一换算成秒,与手稿口径一致。
3. **线路语义**:depot=燕郊场站,域内闭合接驾巡回;燕郊→国贸干线段对所有策略是同一常量,不进矩阵,报告注明。
4. **home 变体选择准则**(B 阶段执行):两变体都建;用遗憾基建测"可改进遗憾空间",选空间更大、SPO+ vs DSPO 分离更清晰者为主场景,另一变体作鲁棒性附注;论文如实说明选择依据。
5. **API 配额**:key 经环境变量 `AMAP_KEY` 传入(用户已提供,剩余约 14.4 万次,3 QPS);预算约 4,400 次批量调用(约 3%)。

## 1. 验收标准

1. **实例文件齐全**:两变体(`YJ_random`/`YJ_poi`)各含 coords、dist_matrix、adjacency 文件,
   train/test 两个 data_seed 指向同一套真实坐标(真实位置不随 seed 变;与 RC 每 seed 换坐标的差异在文档注明)。
2. **数据合理性**:≥95% 节点对满足 驾车距离 ≥ 直线(haversine)距离;矩阵非负、对角为 0;全部坐标落在燕郊建成区边界内(站点按池子原坐标)。
3. **loader 单测通过**:节点总数 471(1 depot + 300 home + 170 站)、n_parcelpoints=170、邻接每个 home 恰好 10 个可报价站、服务时长机制正常。
4. **校准达标**:`mnl_dist_scaler` 对数二分后,NoPricing 的 home% = 98±3%(锚点=手稿 Table 2 NoPricing 行 98.18%),退出率 <2%;校准过程与结果写入 `docs/req4_yanjiao_calibration.md`。
5. **冒烟通过**:两变体各跑 NoPricing、StaticPricing(+2/−5) eval-only × 3 seed,无崩溃、指标量级合理(如实记录,不设排序门槛——排序验证属 B)。
6. **配额安全**:实际 API 调用数 ≤ 预算 1.5 倍(≈6,600);实验阶段零调用(只读静态文件)。

## 2. 数据管线(新目录 `Experiments/yanjiao_build/`)

输入:`Environments/OOH/yanjiao/yanjiao_meeting_points_pool.csv`(383 站,含 is_stopped_route)、
`yanjiao_home_cluster_candidates.csv`(41 住宅簇,含 cluster_weight)、环境变量 `AMAP_KEY`。

### C1 站点选择 `select_stations.py`

- 剔除 `is_stopped_route=True` → 剩余池;
- **最大最小距离贪心**选 170 个(从池中依次选与已选集最小距离最大的点,起点取池子质心最近站),固定随机打破平局(seed=42),保证空间覆盖且完全可复现;
- depot:默认取距选定站点集质心最近的站点坐标作为"燕郊场站",CLI 可用 `--depot-lon/--depot-lat` 覆盖;
- 输出 `stations_170.csv`(station_id, lon, lat, name)。

### C2 home 生成 `gen_homes.py`

- **V1 `YJ_random`**:燕郊建成区边界内均匀随机 300 点(边界=站点池凸包收缩 200m,避免点落在域外),seed=42;
- **V2 `YJ_poi`**:按 `cluster_weight` 归一化加权抽 300 次簇中心,每次加二维高斯散布 σ=300m,并拒绝采样保证落在同一边界内,seed=42;
- 输出 `homes_random_300.csv` / `homes_poi_300.csv`(home_id, lon, lat, source)。

### C3 距离采集 `fetch_distances.py` + `amap_client.py`

- **计费口径试探(第一步,硬门槛)**:先发 3 次批量请求(各 100 起点),读取配额消耗口径;
  若距离测量 API 不可用或按起点数计费(意味着全矩阵 41 万次配额不够),**停止并报告**,
  降级方案(混合:真实采样 2,000 对 + 按"燕郊域内/跨区"两档拟合绕行系数 × haversine 推广)另行确认后执行;
- 正常路径:批量 100 起点 × 1 终点,type=1(驾车),3 QPS 节流(令牌桶,含指数退避重试 ≤5 次);
- **持久缓存**:`Environments/OOH/yanjiao/cache/amap_cache.sqlite`(gitignore),键=(起点坐标6位小数,终点坐标6位小数),
  值=(距离米,时长秒,采集时间);任何请求先查缓存,断点续采;脚本启动时打印"待采对数/预计新增调用/缓存命中",
  预计新增调用 >8,000 时中止并要求确认;
- 两变体共享"站–站 + depot"块(171×171),homes 块分变体采;总预算 ≈4,400 次;
- 失败对(API 返回无路线等):记录清单,用 haversine × 全局中位绕行系数填补,填补数写入构建报告,>1% 时警告;
- 输出:每变体一个 471×471 距离矩阵(米),存中间 npy。

### C4 实例装配 `assemble_instance.py`

- 节点顺序(**写计划时核实与 env 的切片约定一致**:`get_new_customer_from_data` 用 `randint(1, num_cust_loc)` 抽客户,
  故约定为 index 0=depot,1..300=homes,301..470=stations;parcelpoints 由 coords 尾部 n_parcelpoints 个构成的机制在
  `env_utils`/`Parcelpoint_py` 中核实后对齐);
- 距离矩阵:米 → 秒(÷30000×3600),取整,写 Amazon_data 同格式的 `_dist_matrix.txt`;
- 邻接:每 home 按**驾车距离**取最近 10 站,写 `_adjacency10.npy`(掩码语义——1/0 哪个表示可报价,
  **写计划时核实** `DSPO.get_action_pricing` 的 `ma.masked_array(..., mask=adjacency)` 用法后对齐 Amazon 的存法);
- coords 文件:`YJ_random_471_0_coords.txt` 与 `..._1_coords.txt` 内容相同(决策 §1.1),格式仿 Homberger/Amazon(制表符分隔);
- 输出目录:`Environments/OOH/yanjiao_instance/YJ_random/`、`.../YJ_poi/`(**目录 gitignore**,构建脚本可随时重建;
  仅提交一份小的构建摘要 `build_report.md`:节点数、调用数、缓存命中、填补对数、校验结果)。

## 3. 代码接入(仓库代码唯一改动点)

`Src/Utils/Utils.py::load_demand_data` 增加分支:instance 名以 `YJ_` 开头 →
`instance_folder='yanjiao_instance'`,`instance_size='_471_'`,距离矩阵从文件读(Amazon 模式),
`n_parcelpoints=170`,`adjacency = np.load(pathh+"_adjacency10.npy")`;service_times 沿用现有
`calculate_service_time`(合成 6-hump camelback;手稿未定义真实服务时长,文档注明)。
其余全部经 CLI 传参,零算法代码改动:
`--instance YJ_random --n_vehicles 25 --veh_capacity 12 --max_steps_r 240 --revenue 30 --fuel_cost 0.6
--min_price -10 --max_price 10 --outside_option_util -1.0 --home_util 1.4 --incentive_sens -0.25 --home_failure 0.0`。

已知近似(文档注明,不改):MNL 步行项 `mnl_distmat` 用的是 dist_matrix(驾车秒)作为"到集合点距离"的代理,
其尺度由重新校准的 `mnl_dist_scaler` 吸收;燕郊路网规整,驾车与步行距离高度相关,可接受。

## 4. 校准(`Experiments/run_req4_calibrate.py`,复用 run_paper_repro 的二分模式)

- 目标:NoPricing home% = 98±3%,对数空间二分 `mnl_dist_scaler`,eval-only(0 训练)。
  锚点取自手稿设定的行为含义——无激励时乘客几乎全选上门,这给定价留出最大的引导空间,正是 SPO+ 显效的前提;
  不是为了复现 Table 2 数字;
- 变体各自校准(home 分布不同,尺度可能不同);U₀=−1.0、ASC_home=1.4 固定不调;
- 产出 `docs/req4_yanjiao_calibration.md`:两变体的 scaler、NoPricing/Static 冒烟表(3 seed)、与 Table 2 锚点的偏差、
  失败对填补统计、如实记录任何异常。

## 5. 测试

- `tests/test_yanjiao_station_select.py`:玩具点集上贪心选择的确定性(两次运行同输出)与覆盖性(min-min 距离不小于随机选);
- `tests/test_yanjiao_amap_cache.py`:桩掉 HTTP 的缓存命中/断点续采(第二次运行零新请求)、限速器不越 3 QPS(时间戳断言);
- `tests/test_yanjiao_loader.py`:用微型合成 fixture(6 节点)走 load_demand_data 的 YJ_ 分支,断言节点数/邻接形状/矩阵对称非负;
- 既有 14 个测试全绿(loader 改动不得影响 RC/Amazon 路径)。

## 6. 风险与如实报告

- **计费口径不确定**是最大风险,已设硬门槛试探(C3 第一步),不确认不批量采;
- API key 只经环境变量,不落任何入 git 文件;对话中已暴露,项目结束后建议用户在高德控制台重置;
- E[D]=240、25 车的 episode 约为 RC 的 2.7 倍重,HGS 时限参数是否需要上调(1.1s/1.5s 对 240 节点可能偏紧)
  在冒烟中观察并如实记录,若路线质量明显劣化,调整建议写入校准报告供 B 阶段决策;
- B 阶段验收是"SPO+ 显著有效"而非复现 Table 2;若两种 home 变体的可改进遗憾空间都很小(重演 RC 的窘境),
  合法的加压手段是场景设计层面的(需求异质性、供给松紧、候选站密度),逐项披露;结果如实呈现,不调参凑显著。

## 7. 交付物

- 代码:`Experiments/yanjiao_build/`(4 脚本 + amap_client)、`Experiments/run_req4_calibrate.py`、
  `load_demand_data` 的 YJ_ 分支(分支 `yanjiao-instance`);
- 数据:两变体实例文件 + sqlite 缓存(均 gitignore,可重建);`build_report.md` + `docs/req4_yanjiao_calibration.md` 入 git;
- 测试:§5 三个新测试;
- 结论:§1 六条验收逐条判定。
