"""Paper-faithful reproduction runner for the 5-policy comparison (spec:
docs/superpowers/specs/2026-07-24-paper-repro-design.md).

Modes:
  smoke     -- tiny run of all policies to validate the pipeline and time one episode
  calibrate -- bisection on the MNL distance scaler targeting paper App. B (80% / 60% home)
  full      -- 5 policies x N seeds, writes docs/paper_repro_results.md
"""
import sys
import argparse
import time
from os import path

ROOT = path.abspath(path.join(path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from Src.parser import Parser  # noqa: E402
from Src.config import Config  # noqa: E402
from run import Solver         # noqa: E402

# paper Table A.1 / B.1 / F.1 / Sec. 5.2.1 -- synthetic RC case
PAPER_PRESET = {
    'instance': 'RC', 'data_seed': '0', 'data_seed_test': '1',
    'max_steps_r': '90', 'max_steps_p': '0.5',
    'n_vehicles': '9', 'veh_capacity': '10',
    'fraction_capacitated': '0.0',
    'fuel_cost': '0.3', 'driver_wage': '30', 'truck_speed': '30',
    'failure_cost': '10.0', 'home_failure': '0.1',
    'revenue': '50', 'clip_service_time': '10',
    'min_price': '-10.0', 'max_price': '2.0',
    'pricing': 'True',
    'incentive_sens': '-0.25', 'base_util': '0.0', 'home_util': '3.2',
    'mnl_beta_k': '0.02',
    # DSPO hyperparameters (paper Table F.1, synthetic)
    'grid_dim': '10', 'n_input_layers': '3', 'batch_size': '64',
    'learning_rate': '1e-3', 'n_filters': '32', 'dropout': '0.0',
    'save_model': 'False', 'debug': 'False',
}

POLICIES = {
    'NoOOH':         {'algo_name': 'Baseline', 'offer_mode': 'home_only', 'price_home': '0.0', 'price_pp': '0.0'},
    'NoPricing':     {'algo_name': 'Baseline', 'offer_mode': 'all',       'price_home': '0.0', 'price_pp': '0.0'},
    'StaticPricing': {'algo_name': 'Baseline', 'offer_mode': 'all',       'price_home': '2.0', 'price_pp': '-5.0'},
    'DSPO':          {'algo_name': 'DSPO',     'offer_mode': 'all'},
    'OnlyOOH':       {'algo_name': 'Baseline', 'offer_mode': 'ooh_only',  'price_home': '0.0', 'price_pp': '0.0'},
}
ORDER = ['NoOOH', 'NoPricing', 'StaticPricing', 'DSPO', 'OnlyOOH']


def build_argv(policy, seed, scaler, train_episodes, buffer_size, extra=None):
    over = dict(PAPER_PRESET)
    over.update(POLICIES[policy])
    over['seed'] = str(seed)
    over['mnl_dist_scaler'] = str(scaler)
    over['max_episodes'] = str(max(1, train_episodes))
    over['buffer_size'] = str(buffer_size)
    over['folder_suffix'] = 'paper_{}_{}'.format(policy, seed)
    if extra:
        over.update(extra)
    argv = []
    for k, v in over.items():
        argv += ['--' + k, v]
    return argv


def run_one(policy, seed, scaler, train_episodes, eval_episodes, buffer_size, extra=None):
    argv = build_argv(policy, seed, scaler, train_episodes, buffer_size, extra)
    orig_stdout = sys.stdout
    t0 = time.time()
    try:
        args = Parser().get_parser().parse_args(argv)
        config = Config(args)
        solver = Solver(config)
        if policy == 'DSPO' and train_episodes > 0:
            solver.train()
        t_train = time.time() - t0
        metrics = solver.eval2(eval_episodes)
    finally:
        sys.stdout = orig_stdout
    metrics['wall_train_s'] = t_train
    metrics['wall_total_s'] = time.time() - t0
    return metrics


def mode_smoke(opt):
    print('== smoke: all policies, seed {}, {} train ep (DSPO), {} eval ep =='.format(
        opt.seeds[0], opt.train_episodes, opt.eval_episodes))
    for p in opt.policies:
        m = run_one(p, opt.seeds[0], opt.scaler, opt.train_episodes, opt.eval_episodes, opt.buffer_size)
        print('{:>14s}: total={:9.2f} home%={:5.1f} travel={:8.2f} service={:7.2f} '
              'fail={:6.2f} disc={:7.2f} chrg={:6.2f} [train {:.1f}s, total {:.1f}s]'.format(
                  p, m['total_costs'], 100*m['pct_home'], m['travel_costs'], m['service_costs'],
                  m['failure_costs'], m['discount_costs'], m['charge_revenue'],
                  m['wall_train_s'], m['wall_total_s']))


def mode_calibrate(opt):
    lo, hi = opt.scaler_lo, opt.scaler_hi
    print('== calibrate: bisection on mnl_dist_scaler in [{}, {}] targeting 80% home (NoPricing) =='.format(lo, hi))
    best = None
    for it in range(opt.calib_iters):
        mid = (lo * hi) ** 0.5  # log-space bisection
        m = run_one('NoPricing', opt.seeds[0], mid, 0, opt.eval_episodes, opt.buffer_size)
        print('iter {:2d}: scaler={:12.2f} -> home%={:5.1f}'.format(it, mid, 100*m['pct_home']))
        best = (mid, m['pct_home'])
        if abs(m['pct_home'] - 0.80) <= 0.01:
            break
        if m['pct_home'] > 0.80:
            lo = mid   # too much home -> make OOH more attractive -> larger scaler
        else:
            hi = mid
    scaler = best[0]
    ms = run_one('StaticPricing', opt.seeds[0], scaler, 0, opt.eval_episodes, opt.buffer_size)
    print('chosen scaler = {:.2f}: NoPricing home%={:5.1f}, StaticPricing home%={:5.1f} (targets 80/60 +-3)'.format(
        scaler, 100*best[1], 100*ms['pct_home']))
    print('acceptance: NoPricing {}  StaticPricing {}'.format(
        'PASS' if abs(best[1]-0.80) <= 0.03 else 'FAIL',
        'PASS' if abs(ms['pct_home']-0.60) <= 0.03 else 'FAIL'))


def mode_full(opt):
    assert opt.scaler > 0, 'full mode requires --scaler from calibration'
    assert 'NoOOH' in opt.policies, 'full mode requires NoOOH in --policies (baseline for savings %)'
    results = {p: [] for p in opt.policies}
    for seed in opt.seeds:
        for p in opt.policies:
            print('running {} seed {} ...'.format(p, seed))
            m = run_one(p, seed, opt.scaler, opt.train_episodes if p == 'DSPO' else 0,
                        opt.eval_episodes, opt.buffer_size)
            print('  -> total={:.2f} home%={:.1f} ({:.0f}s)'.format(
                m['total_costs'], 100*m['pct_home'], m['wall_total_s']))
            results[p].append(m)
    write_report(results, opt)


def agg(results, p, key):
    vals = [m[key] for m in results[p]]
    n = len(vals)
    mean = sum(vals) / n
    std = (sum((v-mean)**2 for v in vals) / n) ** 0.5
    return mean, std


def write_report(results, opt):
    base_mean, _ = agg(results, 'NoOOH', 'total_costs')
    lines = []
    lines.append('# 论文配置复现结果(RC 合成案例)\n')
    lines.append('- 预设:论文 Table A.1/B.1/F.1;scaler={};seeds={};DSPO 训练 {} episodes,buffer {};评测 {} episodes/seed\n'.format(
        opt.scaler, opt.seeds, opt.train_episodes, opt.buffer_size, opt.eval_episodes))
    lines.append('| 策略 | home% | travel | service | failure | 折扣成本(收费收入) | 平均折扣(平均收费) | total | 相对NoOOH节省% |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for p in [x for x in ORDER if x in results]:
        tm, ts = agg(results, p, 'total_costs')
        row = '| {} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} ({:.1f}) | {:.1f}±{:.1f} ({:.1f}±{:.1f}) | {:.1f}±{:.1f} | {:.1f} |'.format(
            p, 100*agg(results, p, 'pct_home')[0],
            agg(results, p, 'travel_costs')[0], agg(results, p, 'service_costs')[0],
            agg(results, p, 'failure_costs')[0],
            agg(results, p, 'discount_costs')[0], agg(results, p, 'charge_revenue')[0],
            agg(results, p, 'avg_discount')[0], agg(results, p, 'std_discount')[0],
            agg(results, p, 'avg_charge')[0], agg(results, p, 'std_charge')[0],
            tm, ts, 100*(1 - tm/base_mean))
        lines.append(row)
    means = {p: agg(results, p, 'total_costs')[0] for p in results}
    expected = [p for p in ORDER if p in results]
    actual = sorted(means, key=means.get, reverse=True)
    lines.append('\n- 目标排序(成本降序):{}'.format(' > '.join(expected)))
    lines.append('- 实际排序(成本降序):{}'.format(' > '.join(actual)))
    lines.append('- 排序判定:{}\n'.format('**成立**' if actual == expected else '**不成立(如实报告,见上表)**'))
    out = path.join(ROOT, 'docs', 'paper_repro_results.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('report written to', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['smoke', 'calibrate', 'full'])
    ap.add_argument('--scaler', type=float, default=0.0, help='mnl_dist_scaler; 0=auto(amax)')
    ap.add_argument('--scaler_lo', type=float, default=200.0)
    ap.add_argument('--scaler_hi', type=float, default=250000.0)
    ap.add_argument('--calib_iters', type=int, default=12)
    ap.add_argument('--train_episodes', type=int, default=3, help='DSPO training episodes')
    ap.add_argument('--eval_episodes', type=int, default=2)
    ap.add_argument('--buffer_size', type=int, default=500)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1234])
    ap.add_argument('--policies', nargs='+', default=ORDER, choices=ORDER)
    opt = ap.parse_args()
    if opt.mode == 'calibrate' and opt.calib_iters < 1:
        ap.error('--calib_iters must be >= 1')
    {'smoke': mode_smoke, 'calibrate': mode_calibrate, 'full': mode_full}[opt.mode](opt)


if __name__ == '__main__':
    main()
