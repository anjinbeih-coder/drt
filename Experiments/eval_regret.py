"""Offline decision-regret evaluation (spec C2).

Usage (paths are run *seed* directories, i.e. the parent of Checkpoints/):
  python Experiments/eval_regret.py \
      --dataset Experiments/Parcelpoint_py/regret_dataset_RC_seed9999.pt \
      --model DSPO_1234=Experiments/Parcelpoint_py/pricing/DSPO/runreq3p10ck_DSPO_1234/1234 \
      --model SPOa07_1234=Experiments/Parcelpoint_py/pricing/DSPO_plus_SPO/runreq3p10ck_SPOa07_1234/1234 \
      --pair DSPO_1234:SPOa07_1234 \
      --out docs/req3_regret_results.md

Pair A:B reports diff = regret_A - regret_B (positive median => B has lower regret).

Note: console prints are redirected by Config into the runregret_eval_tmp logfile; rely on --out.
"""
import argparse
import sys
from os import path

import numpy as np
from scipy import stats

ROOT = path.join(path.dirname(path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)

from Src.parser import Parser                        # noqa: E402
from Src.config import Config                        # noqa: E402
from Experiments.run_paper_repro import build_argv   # noqa: E402
from Experiments.run_req2_profit import req2_extra   # noqa: E402
from Experiments.run_req3_spo import SCALER, V0      # noqa: E402
from Experiments.regret_lib import (load_dataset, reconstruct_c_hat,   # noqa: E402
                                    regret_for_sample)

EXPECT_META = {'instance': 'RC', 'mnl_dist_scaler': float(SCALER),
               'outside_option_util': float(V0), 'min_price': -10.0,
               'max_price': 10.0, 'home_failure': 0.0}


def build_agent():
    """One full agent (env + CNN skeleton) reused for every checkpoint."""
    extra = dict(req2_extra(V0))
    extra.update({'algo_name': 'DSPO_plus_SPO', 'spo_loss_weight': '0.0',
                  'max_price': '10.0', 'save_model': 'False',
                  'folder_suffix': 'regret_eval_tmp'})
    argv = build_argv('DSPO', 1234, SCALER, 0, 10000, extra=extra)
    args = Parser().get_parser().parse_args(argv)
    config = Config(args)
    return config.algo(config=config)


def eval_model(agent, run_seed_dir, entries):
    ckpt = path.join(run_seed_dir, 'Checkpoints', 'supervised_ml.pt')
    assert path.isfile(ckpt), 'missing checkpoint: ' + ckpt
    agent.supervised_ml.load(ckpt)
    agent.supervised_ml.eval()
    rows = []
    for entry in entries:
        c_hat = reconstruct_c_hat(agent, entry)
        rows.append(regret_for_sample(agent, entry, c_hat) if c_hat is not None else None)
    return rows


def summarize(name, rows):
    ok = [r for r in rows if r is not None]
    assert ok, '{}: no evaluable samples (wrong checkpoint dir?)'.format(name)
    reg = np.array([r['regret'] for r in ok])
    ro = np.array([r['r_oracle'] for r in ok])
    rm = np.array([r['r_model'] for r in ok])
    mse = np.array([r['mse'] for r in ok])
    neg = reg[reg < 0]
    mean_reg = float(reg.mean())
    out = {'name': name, 'n': len(ok), 'skipped': len(rows) - len(ok),
           'mean_regret': mean_reg,
           'rel_regret_pct': 100.0 * mean_reg / float(ro.mean()),
           'mean_r_oracle': float(ro.mean()), 'mean_r_model': float(rm.mean()),
           'mse': float(mse.mean()),
           'neg_share_pct': 100.0 * len(neg) / max(1, len(reg)),
           'neg_median_mag': float(np.median(-neg)) if len(neg) else 0.0}
    out['blocker'] = bool(out['neg_share_pct'] > 5.0 or
                          (mean_reg > 0 and out['neg_median_mag'] > 0.10 * mean_reg))
    return out


def pair_stats(label, rows_a, rows_b):
    diffs = np.array([ra['regret'] - rb['regret'] for ra, rb in zip(rows_a, rows_b)
                      if ra is not None and rb is not None])
    if len(diffs) == 0 or not np.any(diffs != 0):
        return {'pair': label, 'n': len(diffs), 'median_diff': 0.0,
                'mean_diff': 0.0, 'p_value': 1.0}
    res = stats.wilcoxon(diffs)
    return {'pair': label, 'n': int(len(diffs)),
            'median_diff': float(np.median(diffs)),
            'mean_diff': float(diffs.mean()), 'p_value': float(res.pvalue)}


def write_report(out_path, meta, summaries, pairs):
    lines = ['# 决策遗憾评测结果(共享留出集)', '',
             '- 数据集:seed={seed},episodes={episodes},policy={policy},样本 n={n}'.format(
                 n=summaries[0]['n'] + summaries[0]['skipped'] if summaries else 0, **meta),
             '- 口径:scaler={mnl_dist_scaler},V0={outside_option_util},'
             '价格域 [{min_price},{max_price}],home_failure={home_failure}'.format(**meta),
             '- oracle 为"闭式 Lambert-W + 裁剪"近似(与训练一致);负遗憾为近似诊断量',
             '- POOLED p 值因同一客户在多个 seed 对中重复出现而偏乐观,仅作参考',
             '', '## 每模型汇总', '',
             '| 模型 | n | 平均遗憾 | 相对遗憾% | oracle 利润 | 模型利润 | MSE | 负遗憾占比% | blocker |',
             '|---|---|---|---|---|---|---|---|---|']
    for s in summaries:
        lines.append('| {name} | {n} | {mean_regret:.4f} | {rel_regret_pct:.2f} | '
                     '{mean_r_oracle:.3f} | {mean_r_model:.3f} | {mse:.3f} | '
                     '{neg_share_pct:.2f} | {blocker} |'.format(**s))
    lines += ['', '## 配对检验(diff = 左 − 右;中位差 >0 表示右侧遗憾更低)', '',
              '| 对比 | n | 配对差中位数 | 配对差均值 | Wilcoxon p |', '|---|---|---|---|---|']
    for p in pairs:
        lines.append('| {pair} | {n} | {median_diff:.4f} | {mean_diff:.4f} | '
                     '{p_value:.3g} |'.format(**p))
    lines.append('')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('report -> ' + out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--model', action='append', required=True,
                    help='NAME=run seed dir (parent of Checkpoints/)')
    ap.add_argument('--pair', action='append', default=[], help='NAME_A:NAME_B')
    ap.add_argument('--out', default=path.join(ROOT, 'docs', 'req3_regret_results.md'))
    opt = ap.parse_args()

    meta, entries = load_dataset(opt.dataset, expect_meta=EXPECT_META)
    agent = build_agent()

    all_rows, summaries = {}, []
    for spec in opt.model:
        name, run_dir = spec.split('=', 1)
        rows = eval_model(agent, run_dir, entries)
        all_rows[name] = rows
        s = summarize(name, rows)
        summaries.append(s)
        print('{name}: n={n} mean_regret={mean_regret:.4f} rel={rel_regret_pct:.2f}% '
              'mse={mse:.3f} neg%={neg_share_pct:.2f} blocker={blocker}'.format(**s))

    pairs = []
    pooled = []
    for spec in opt.pair:
        a, b = spec.split(':', 1)
        pairs.append(pair_stats('{} - {}'.format(a, b), all_rows[a], all_rows[b]))
        pooled.extend([(ra, rb) for ra, rb in zip(all_rows[a], all_rows[b])
                       if ra is not None and rb is not None])
    if len(opt.pair) > 1 and pooled:
        rows_a = [p[0] for p in pooled]
        rows_b = [p[1] for p in pooled]
        pairs.append(pair_stats('POOLED (all pairs)', rows_a, rows_b))
    for p in pairs:
        print('{pair}: n={n} median_diff={median_diff:.4f} p={p_value:.3g}'.format(**p))

    write_report(opt.out, meta, summaries, pairs)


if __name__ == '__main__':
    main()
