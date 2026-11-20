"""Requirement 2 runner: outside option + net-profit comparison (spec:
docs/superpowers/specs/2026-07-25-req2-outside-option-design.md).

Modes:
  calibrate_v0 -- sweep V0 under NoPricing, report quit rates, target band 2%-5%
  full         -- NoPricing/StaticPricing/DSPO x seeds, writes docs/req2_profit_results.md
"""
import sys
import argparse
from os import path

ROOT = path.abspath(path.join(path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from Experiments.run_paper_repro import run_one, agg  # noqa: E402

SCALER = 1800.0  # requirement-1 MNL calibration, user-approved
POLICIES2 = ['NoPricing', 'StaticPricing', 'DSPO']
ORDER_BY_PROFIT_DESC = ['DSPO', 'StaticPricing', 'NoPricing']  # expected: DSPO highest net profit


def req2_extra(v0):
    return {'home_failure': '0.0', 'outside_option_util': str(v0)}


def mode_calibrate_v0(opt):
    print('== calibrate_v0: NoPricing quit-rate sweep, target band 2%-5% ==')
    best = None
    for v0 in opt.v0_grid:
        m = run_one('NoPricing', opt.seeds[0], SCALER, 0, opt.eval_episodes, opt.buffer_size,
                    extra=req2_extra(v0))
        print('V0={:6.2f} -> quit%={:5.2f} home%={:5.1f} net_profit={:9.2f}'.format(
            v0, 100*m['quit_rate'], 100*m['pct_home'], m['net_profit']))
        err = abs(m['quit_rate'] - 0.035)
        if best is None or err < best[1]:
            best = (v0, err, m['quit_rate'])
    print('closest to band center 3.5%: V0={} (quit {:.2f}%)'.format(best[0], 100*best[2]))
    print('acceptance: {}'.format('PASS' if 0.02 <= best[2] <= 0.05 else 'FAIL (densify grid near best V0)'))


def mode_full(opt):
    assert opt.v0 is not None, 'full mode requires --v0 from calibrate_v0'
    results = {p: [] for p in POLICIES2}
    for seed in opt.seeds:
        for p in POLICIES2:
            print('running {} seed {} ...'.format(p, seed))
            m = run_one(p, seed, SCALER, opt.train_episodes if p == 'DSPO' else 0,
                        opt.eval_episodes, opt.buffer_size, extra=req2_extra(opt.v0))
            print('  -> net_profit={:.2f} total_costs={:.2f} quit%={:.2f} home%={:.1f} ({:.0f}s)'.format(
                m['net_profit'], m['total_costs'], 100*m['quit_rate'], 100*m['pct_home'], m['wall_total_s']))
            results[p].append(m)
    write_report(results, opt)


def write_report(results, opt):
    lines = []
    lines.append('# 需求 2 结果:外部效用 + 净利润对比(RC 合成案例)\n')
    lines.append('- 预设:需求 1 论文预设 + home_failure=0 + outside_option_util={};scaler={};seeds={};DSPO 训练 {} ep,buffer {};评测 {} ep/seed\n'.format(
        opt.v0, SCALER, opt.seeds, opt.train_episodes, opt.buffer_size, opt.eval_episodes))
    lines.append('| 策略 | 净利润 | 退出率% | home% | travel | service | 折扣成本(收费收入) | total costs |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for p in ORDER_BY_PROFIT_DESC:
        pm, ps = agg(results, p, 'net_profit')
        lines.append('| {} | {:.1f}±{:.1f} | {:.2f}±{:.2f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} ({:.1f}) | {:.1f}±{:.1f} |'.format(
            p, pm, ps,
            100*agg(results, p, 'quit_rate')[0], 100*agg(results, p, 'quit_rate')[1],
            100*agg(results, p, 'pct_home')[0],
            agg(results, p, 'travel_costs')[0], agg(results, p, 'service_costs')[0],
            agg(results, p, 'discount_costs')[0], agg(results, p, 'charge_revenue')[0],
            agg(results, p, 'total_costs')[0], agg(results, p, 'total_costs')[1]))
    means = {p: agg(results, p, 'net_profit')[0] for p in POLICIES2}
    actual = sorted(means, key=means.get, reverse=True)
    lines.append('\n- 目标排序(净利润降序):{}'.format(' > '.join(ORDER_BY_PROFIT_DESC)))
    lines.append('- 实际排序(净利润降序):{}'.format(' > '.join(actual)))
    lines.append('- 排序判定:{}\n'.format('**成立**' if actual == ORDER_BY_PROFIT_DESC else '**不成立(如实报告,见上表)**'))
    out = path.join(ROOT, 'docs', 'req2_profit_results.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('report written to', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['calibrate_v0', 'full'])
    ap.add_argument('--v0', type=float, default=None)
    ap.add_argument('--v0_grid', type=float, nargs='+', default=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
    ap.add_argument('--train_episodes', type=int, default=500)
    ap.add_argument('--eval_episodes', type=int, default=20)
    ap.add_argument('--buffer_size', type=int, default=10000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1234, 2345, 3456])
    opt = ap.parse_args()
    {'calibrate_v0': mode_calibrate_v0, 'full': mode_full}[opt.mode](opt)


if __name__ == '__main__':
    main()
