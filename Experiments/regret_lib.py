"""Shared core for decision-regret evaluation (spec 2026-07-26-regret-metric-design.md).

Entry schema == one value of DSPO_plus_SPO.spo_training_data (see plan Global Constraints).
All agent-taking functions accept either a full DSPO_plus_SPO or the bare oracle agent.
"""
import sys
from os import path

import numpy as np
import torch

sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), '..'))
from Src.Algorithms.DSPO_plus_SPO import DSPO_plus_SPO  # noqa: E402


class _BareConfig(object):
    def __init__(self, outside_option_util):
        self.outside_option_util = outside_option_util


def make_bare_oracle_agent(min_p=-10.0, max_p=10.0, base_util=0.0, revenue=50.0,
                           outside_option_util=0.0):
    """DSPO_plus_SPO instance without __init__ (no env); enough for the pricing oracle."""
    agent = object.__new__(DSPO_plus_SPO)
    agent.min_p = float(min_p)
    agent.max_p = float(max_p)
    agent.base_util = float(base_util)
    agent.revenue = float(revenue)
    agent.beta_ivt = 0.0
    agent.spo_w0_band = False
    agent.config = _BareConfig(float(outside_option_util))
    return agent


def expected_profit_under(oracle_out, c_true_t):
    """R(a; c) = sum_k P_k(a) * (rev + a_k - c_k) = r0 - probs . c."""
    probs_f32 = oracle_out['probs'].detach().cpu()
    prices_f32 = oracle_out['prices'].detach().cpu()
    r0_f32 = oracle_out['r0'].detach().cpu()
    # Extract revenue from oracle output (computed in float32)
    revenue = float(r0_f32 - torch.dot(probs_f32, prices_f32))
    # Recompute r0 in float64 for higher precision
    probs = probs_f32.double()
    prices = prices_f32.double()
    c_true_t = c_true_t.cpu().double()
    r0 = revenue + float(torch.dot(probs, prices))
    return r0 - float(torch.dot(probs, c_true_t))


def regret_for_sample(agent, entry, c_hat_t):
    c_true_t = torch.tensor(np.asarray(entry['true_costs'], dtype=np.float32))
    if c_hat_t is None or c_hat_t.numel() != c_true_t.numel() or c_true_t.numel() <= 1:
        return None
    if not (torch.isfinite(c_hat_t).all() and torch.isfinite(c_true_t).all()):
        return None
    o_true = agent._pricing_oracle_lifted_torch(
        c_true_t, entry['customer_info'], entry['pps_info'])
    o_hat = agent._pricing_oracle_lifted_torch(
        c_hat_t.detach().cpu(), entry['customer_info'], entry['pps_info'])
    if o_true is None or o_hat is None:
        return None
    r_oracle = expected_profit_under(o_true, c_true_t)
    r_model = expected_profit_under(o_hat, c_true_t)
    return {'regret': r_oracle - r_model, 'r_oracle': r_oracle, 'r_model': r_model,
            'mse': float(torch.mean((c_hat_t - c_true_t) ** 2))}


def reconstruct_c_hat(agent, entry):
    """Model's predicted option-cost vector for one stored sample.

    Mirrors DSPO_plus_SPO._calculate_spo_loss_for_batch phases 1-3 for a single
    sample, eval mode, no grad. The stored per-sample theta is the deployment-time
    within-day schedule value (state[3] is the intra-episode step counter), identical
    for every model -- reusing it is both faithful and fair.
    """
    fi = entry.get('feature_info', None)
    if fi is None:
        return None
    base_feat = fi['base_feat']
    pp_cells = fi['pp_cells']
    time_int = fi['time_int']
    pp_caps = fi.get('pp_caps', [1000000.0] * len(pp_cells))
    K = 1 + len(pp_cells)
    if K <= 1 or len(entry['true_costs']) < K:
        return None
    feats = [base_feat]
    caps = [1000000.0]
    home_aug = base_feat.clone()
    home_aug[0][time_int][fi['home_cell'][0]][fi['home_cell'][1]] += 1
    feats.append(home_aug)
    caps.append(1000000.0)
    for j, pc in enumerate(pp_cells):
        aug = base_feat.clone()
        aug[0][time_int][pc[0]][pc[1]] += 1
        feats.append(aug)
        caps.append(float(pp_caps[j]) if j < len(pp_caps) else 1000000.0)
    batch = torch.cat(feats, dim=0).to(agent.device)
    caps_t = torch.tensor(caps, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        out = agent.supervised_ml(batch, caps_t).squeeze(-1).cpu()
    mltplr = agent.cost_multiplier
    theta = fi['theta']
    c_hat = torch.zeros(K, dtype=torch.float32)
    c_hat[0] = float(agent._predicted_option_cost(
        agent.l0_home + fi['service_time'], mltplr, theta,
        fi['ins_cost_home'], out[1], out[0]))
    for j in range(K - 1):
        c_hat[j + 1] = float(agent._predicted_option_cost(
            agent.l_mp, mltplr, theta, fi['ins_cost_pps'][j], out[2 + j], out[0]))
    return c_hat


DATASET_META_KEYS = ['instance', 'mnl_dist_scaler', 'outside_option_util', 'min_price',
                     'max_price', 'home_failure', 'seed', 'episodes', 'policy']


def save_dataset(filepath, meta, entries):
    missing = [k for k in DATASET_META_KEYS if k not in meta]
    assert not missing, 'dataset meta missing keys: {}'.format(missing)
    torch.save({'meta': dict(meta), 'entries': list(entries)}, filepath)


def load_dataset(filepath, expect_meta=None):
    blob = torch.load(filepath, map_location='cpu', weights_only=False)
    meta, entries = blob['meta'], blob['entries']
    if expect_meta:
        for k, v in expect_meta.items():
            got = meta.get(k, None)
            if isinstance(v, (int, float)) and got is not None:
                same = abs(float(got) - float(v)) < 1e-9
            else:
                same = (got == v)
            if not same:
                raise ValueError('dataset fingerprint mismatch on {}: dataset={} expected={}'
                                 .format(k, got, v))
    return meta, entries
