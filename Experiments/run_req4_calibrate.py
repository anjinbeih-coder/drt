"""Requirement-4 calibration: log-space bisection of mnl_dist_scaler on the
Yanjiao instance, targeting NoPricing home% = 98 +- 3 (manuscript 5.3 anchor:
without incentives nearly everyone picks door-to-door, which maximises the
steering room for pricing).  Then a StaticPricing (+2/-5) smoke at the chosen
scaler.  eval-only (0 training episodes).

Usage: python Experiments/run_req4_calibrate.py --instance YJ_random
Options: --lo 2 --hi 2000 --iters 7 --seeds 1234 --eval_episodes 10
"""
import argparse
import sys
from os import path

sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), '..'))
from Experiments.run_paper_repro import run_one  # noqa: E402

YJ_EXTRA_BASE = {
    'load_data': 'True', 'n_vehicles': '25', 'veh_capacity': '12',
    'max_steps_r': '240', 'max_steps_p': '0.5', 'revenue': '30',
    'fuel_cost': '0.6', 'truck_speed': '30', 'min_price': '-10.0',
    'max_price': '10.0', 'outside_option_util': '-1.0',
    # ASC_home: manuscript 5.3 states 1.4, but with U0=-1.0 the MNL ceiling is
    # P(home) <= e^1.4/(e^1.4+e^-1) = 91.7% (quit ~8.3% at the ceiling), so the
    # (98% home, <1% quit) anchor is unreachable. User decision 2026-07-26:
    # raise to 3.2 (ceiling 98.5%, quit <1.5%); manuscript param table to follow.
    'home_util': '3.2',
    'incentive_sens': '-0.25', 'home_failure': '0.0',
}
TARGET, TOL = 0.98, 0.03


def yj_extra(instance, folder_tag):
    extra = dict(YJ_EXTRA_BASE)
    extra['instance'] = instance
    extra['folder_suffix'] = 'req4cal_{}_{}'.format(instance, folder_tag)
    return extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance', required=True, choices=['YJ_random', 'YJ_poi'])
    ap.add_argument('--lo', type=float, default=2.0)
    ap.add_argument('--hi', type=float, default=2000.0)
    ap.add_argument('--iters', type=int, default=7)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1234])
    ap.add_argument('--eval_episodes', type=int, default=10)
    opt = ap.parse_args()

    lo, hi, best = opt.lo, opt.hi, None
    for it in range(opt.iters):
        mid = (lo * hi) ** 0.5
        m = run_one('NoPricing', opt.seeds[0], mid, 0, opt.eval_episodes, 10000,
                    extra=yj_extra(opt.instance, 'bis{}'.format(it)))
        print('iter {}: scaler={:.1f} home%={:.2f} quit%={:.2f}'.format(
            it, mid, 100 * m['pct_home'], 100 * m['quit_rate']), flush=True)
        if best is None or abs(m['pct_home'] - TARGET) < abs(best[1]['pct_home'] - TARGET):
            best = (mid, m)
        if abs(m['pct_home'] - TARGET) <= TOL:
            break
        # higher scaler weakens the walk-distance penalty -> OOH more attractive -> less home
        if m['pct_home'] > TARGET:
            lo = mid
        else:
            hi = mid

    scaler = best[0]
    print('chosen scaler = {:.1f}'.format(scaler))
    for seed in opt.seeds:
        mn = run_one('NoPricing', seed, scaler, 0, opt.eval_episodes, 10000,
                     extra=yj_extra(opt.instance, 'np_{}'.format(seed)))
        ms = run_one('StaticPricing', seed, scaler, 0, opt.eval_episodes, 10000,
                     extra=yj_extra(opt.instance, 'sp_{}'.format(seed)))
        print('SMOKE {} seed {}: NoPricing home%={:.2f} quit%={:.2f} net={:.2f} | '
              'Static home%={:.2f} quit%={:.2f} net={:.2f}'.format(
                  opt.instance, seed, 100 * mn['pct_home'], 100 * mn['quit_rate'],
                  mn['net_profit'], 100 * ms['pct_home'], 100 * ms['quit_rate'],
                  ms['net_profit']), flush=True)


if __name__ == '__main__':
    main()
