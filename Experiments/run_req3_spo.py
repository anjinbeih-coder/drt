"""Requirement 3 runner: SPO+ effectiveness with minimal ablation (spec:
docs/superpowers/specs/2026-07-25-req3-spo-alignment-design.md).

Reuses requirement-2 per-seed baseline metrics (embedded below, source: req2 full run
2026-07-25, V0=0, seeds 1234/2345/3456) and runs only the two DSPO_plus_SPO variants.
"""
import sys
import re
import argparse
from os import path

ROOT = path.abspath(path.join(path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from Experiments.run_paper_repro import run_one  # noqa: E402
from Experiments.run_req2_profit import req2_extra, SCALER  # noqa: E402

# per-seed metrics from docs/req2_profit_results.md's underlying run (seeds 1234, 2345, 3456)
REQ2_BASELINES = {
    'NoPricing': [
        {'net_profit': 1897.33, 'total_costs': 2270.17, 'quit_rate': 0.0308, 'pct_home': 0.852},
        {'net_profit': 1829.89, 'total_costs': 2160.11, 'quit_rate': 0.0339, 'pct_home': 0.872},
        {'net_profit': 1859.34, 'total_costs': 2190.66, 'quit_rate': 0.0334, 'pct_home': 0.861}],
    'StaticPricing': [
        {'net_profit': 1978.39, 'total_costs': 2179.11, 'quit_rate': 0.0331, 'pct_home': 0.570},
        {'net_profit': 1877.54, 'total_costs': 2089.96, 'quit_rate': 0.0393, 'pct_home': 0.559},
        {'net_profit': 1920.15, 'total_costs': 2127.35, 'quit_rate': 0.0340, 'pct_home': 0.575}],
    'DSPO': [
        {'net_profit': 1905.60, 'total_costs': 2021.90, 'quit_rate': 0.0559, 'pct_home': 0.742},
        {'net_profit': 1950.66, 'total_costs': 2066.84, 'quit_rate': 0.0586, 'pct_home': 0.737},
        {'net_profit': 1957.58, 'total_costs': 2077.42, 'quit_rate': 0.0561, 'pct_home': 0.742}],
}
V0 = 0.0
VARIANTS = {
    'SPO-ablation': {'algo_name': 'DSPO_plus_SPO', 'spo_loss_weight': '0.0'},
    'SPO':          {'algo_name': 'DSPO_plus_SPO', 'spo_loss_weight': '0.7'},
}
ORDER_BY_PROFIT_DESC = ['SPO', 'DSPO', 'StaticPricing', 'NoPricing']


def variant_extra(name, seed):
    extra = dict(req2_extra(V0))
    extra.update(VARIANTS[name])
    extra['folder_suffix'] = 'req3_{}_{}'.format(name.replace('-', '_'), seed)
    return extra


def check_spo_gate(name, seed, expect_active):
    """SPO+ 生效门槛:日志无失败;active 变体须出现权重/数据两条 debug 行。"""
    log = path.join(ROOT, 'Experiments', 'Parcelpoint_py', 'pricing', 'DSPO_plus_SPO',
                    'runreq3_{}_{}'.format(name.replace('-', '_'), seed), str(seed), 'Logs', 'logfile.log')
    if not path.isfile(log):
        return {'ok': False, 'why': 'logfile missing: {}'.format(log)}
    text = open(log, 'r', encoding='utf-8', errors='replace').read()
    n_err = len(re.findall(r'\[SPO\+错误', text)) + len(re.findall(r'\[SPO\+ warning', text))
    has_data = 'spo_training_data populated' in text
    has_weight = 'spo_weight became positive' in text
    if n_err > 0:
        return {'ok': False, 'why': '{} failure lines in log'.format(n_err)}
    if expect_active and not (has_data and has_weight):
        return {'ok': False, 'why': 'missing debug evidence (data={}, weight={})'.format(has_data, has_weight)}
    if not expect_active and (has_data or has_weight):
        return {'ok': False, 'why': 'ablation unexpectedly ran SPO+ (data={}, weight={})'.format(has_data, has_weight)}
    return {'ok': True, 'why': 'clean'}


def run_variants(opt):
    results = {name: [] for name in VARIANTS}
    gates = {name: [] for name in VARIANTS}
    for seed in opt.seeds:
        for name in VARIANTS:
            print('running {} seed {} ...'.format(name, seed))
            m = run_one('DSPO', seed, SCALER, opt.train_episodes, opt.eval_episodes,
                        opt.buffer_size, extra=variant_extra(name, seed))
            g = check_spo_gate(name, seed, expect_active=(name == 'SPO'))
            print('  -> net_profit={:.2f} quit%={:.2f} home%={:.1f} gate={} ({}) ({:.0f}s)'.format(
                m['net_profit'], 100*m['quit_rate'], 100*m['pct_home'],
                'PASS' if g['ok'] else 'FAIL', g['why'], m['wall_total_s']))
            results[name].append(m)
            gates[name].append(g)
    return results, gates


def write_report(results, gates, opt):
    allrows = dict(REQ2_BASELINES)
    allrows.update(results)
    lines = []
    lines.append('# 需求 3 结果:SPO+ 有效性与最小消融(RC 合成案例)\n')
    lines.append('- 预设:需求 2 口径(V0=0, home_failure=0, scaler={});seeds={};DSPO_plus_SPO 训练 {} ep,buffer {};评测 {} ep/seed;NoPricing/StaticPricing/DSPO 三行复用需求 2 逐 seed 结果\n'.format(
        SCALER, opt.seeds, opt.train_episodes, opt.buffer_size, opt.eval_episodes))
    lines.append('| 策略 | 净利润 | 退出率% | home% | total costs |')
    lines.append('|---|---|---|---|---|')
    for p in ['SPO', 'SPO-ablation', 'DSPO', 'StaticPricing', 'NoPricing']:
        rows = allrows[p]
        def m(key):
            vals = [r[key] for r in rows]
            mu = sum(vals)/len(vals)
            sd = (sum((v-mu)**2 for v in vals)/len(vals)) ** 0.5
            return mu, sd
        np_m, np_s = m('net_profit')
        lines.append('| {} | {:.1f}±{:.1f} | {:.2f}±{:.2f} | {:.1f} | {:.1f}±{:.1f} |'.format(
            p, np_m, np_s, 100*m('quit_rate')[0], 100*m('quit_rate')[1],
            100*m('pct_home')[0], m('total_costs')[0], m('total_costs')[1]))
    def mean(p, key):
        rows = allrows[p]
        return sum(r[key] for r in rows)/len(rows)
    def std(p, key):
        rows = allrows[p]
        mu = mean(p, key)
        return (sum((r[key]-mu)**2 for r in rows)/len(rows)) ** 0.5
    # 排序判定(4 个策略,消融组不参与主排序)
    main_policies = ['SPO', 'DSPO', 'StaticPricing', 'NoPricing']
    means = {p: mean(p, 'net_profit') for p in main_policies}
    actual = sorted(means, key=means.get, reverse=True)
    lines.append('\n- 目标排序(净利润降序):{}'.format(' > '.join(ORDER_BY_PROFIT_DESC)))
    lines.append('- 实际排序(净利润降序):{}'.format(' > '.join(actual)))
    lines.append('- 排序判定:{}'.format('**成立**' if actual == ORDER_BY_PROFIT_DESC else '**不成立(如实报告)**'))
    # 消融判定
    diff_ab = abs(mean('SPO-ablation', 'net_profit') - mean('DSPO', 'net_profit'))
    noise = max(std('SPO-ablation', 'net_profit'), std('DSPO', 'net_profit'))
    gain = mean('SPO', 'net_profit') - mean('SPO-ablation', 'net_profit')
    lines.append('\n## 消融判定')
    lines.append('- |SPO-ablation − DSPO| = {:.1f},噪声级(较大 std)= {:.1f} → 管线一致性 {}'.format(
        diff_ab, noise, '**PASS**' if diff_ab < noise else '**FAIL(需先查管线差异)**'))
    lines.append('- SPO − SPO-ablation = {:+.1f} → SPO+ 增益 {}'.format(
        gain, '**PASS**' if gain > 0 else '**FAIL(如实报告)**'))
    # 生效门槛
    lines.append('\n## SPO+ 生效门槛(先决条件)')
    for name in VARIANTS:
        for seed, g in zip(opt.seeds, gates[name]):
            lines.append('- {} seed {}: {} ({})'.format(name, seed, 'PASS' if g['ok'] else 'FAIL', g['why']))
    out = path.join(ROOT, 'docs', 'req3_spo_results.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('report written to', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['smoke', 'full'])
    ap.add_argument('--train_episodes', type=int, default=500)
    ap.add_argument('--eval_episodes', type=int, default=20)
    ap.add_argument('--buffer_size', type=int, default=10000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1234, 2345, 3456])
    opt = ap.parse_args()
    if opt.mode == 'smoke':
        opt.seeds = opt.seeds[:1]
        opt.train_episodes = min(opt.train_episodes, 140)
        opt.eval_episodes = min(opt.eval_episodes, 2)
        results, gates = run_variants(opt)
        for name in VARIANTS:
            print('smoke {}: gate {}'.format(name, 'PASS' if all(g['ok'] for g in gates[name]) else 'FAIL'))
    else:
        results, gates = run_variants(opt)
        write_report(results, gates, opt)


if __name__ == '__main__':
    main()
