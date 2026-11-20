# 燕郊实例构建报告(2026-07-26)

## 输入与选择

- 站点池:`yanjiao_meeting_points_pool.csv` 383 站 → 剔除停运后 336 → 最大最小距离贪心选 **170**
  (起点=池质心最近站,tie-break 确定性;`stations_170.csv`)。
- home 变体(各 300 点,seed=42,边界=站点池凸包收缩 200m):
  - `YJ_random`:边界内均匀(`homes_random_300.csv`);
  - `YJ_poi`:41 住宅簇按 cluster_weight 加权 + N(0,300m) 散布(`homes_poi_300.csv`)。
- depot:站点集质心最近站(选择脚本默认,未覆盖)(116.89082, 39.950213), 东柳河屯(公交站)。
- 节点顺序:index 0=depot,1..300=homes,301..470=stations(共 471;env 抽样 randint(1,300) 使
  index 300 的 home 永不被抽,继承自代码既有约定,如实记录)。

## 距离采集(高德距离测量 API,type=1 驾车)

| 变体 | API 调用 | 失败对 | 填补率 | 中位绕行系数 | 驾车≥直线 |
|---|---|---|---|---|---|
| YJ_random | 2,350 | 0 | 0.0 | 1.372 | 99.56% |
| YJ_poi | 2,180 | 0 | 0.0 | 1.393 | 99.58% |

- 计费口径:按请求计费(用户 2026-07-26 控制台核实 +4);合计调用(含试探)≈4,537,
  预算上限 6,600(验收 §1.6 ✓);缓存 `cache/amap_cache.sqlite`(gitignore),断点续采,实验阶段零调用。
- 距离落盘:米 → 秒(÷30000×3600,车速 30 km/h,与手稿口径一致),对角 0。
- 执行中修正的两个缺陷(均已过复审):py39 Windows SSL 证书库 bug(certifi 上下文修复,已 pip 安装 certifi);
  真实应答字段为 `distance` 而非计划代码的 `dist`(不修则全矩阵被当失败对)。

## 实例文件

`Environments/OOH/yanjiao_instance/{YJ_random,YJ_poi}/<变体>_471_{0,1}_{coords.txt,dist_matrix.txt,adjacency10.npy}`
(两 data_seed 内容相同——真实坐标不随 seed 变,与 RC 每 seed 换坐标的做法不同,如实注明;
邻接 (301,170),每 home 恰好最近 10 站为 1;目录 gitignore,可由管线一键重建。)
