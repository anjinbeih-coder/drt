import numpy as np
import numpy.ma as ma
import torch
import torch.nn as nn
from torch import float32
from math import sqrt, e
from operator import itemgetter
from collections import Counter

from Src.Algorithms.DSPO import DSPO
from scipy.special import lambertw
from Src.Utils.Utils import get_dist_mat_HGS


class DSPO_plus_SPO(DSPO):
    """
    DSPO算法 + lifted SPO+方法（Smart Predict-then-Optimize Plus）
    
    核心设计：
    1. 使用与论文改写一致的 lifted linear-objective SPO+ 损失：
       - y = [c, 1] 为增广成本向量
       - w(a) = [P(a), -R0(a)] 为由定价决策诱导的 lifted decision
       - MNL 非线性通过 pricing oracle 映射进 lifted feasible decision set
       - SPO+ 次梯度只通过预测成本 c_hat 回传
    
    2. 渐进式训练：
       - Warmup阶段：只用Huber Loss（学习预测）
       - Rampup阶段：逐渐引入SPO+损失（学习决策）
       - 稳定阶段：混合Loss（既学习预测，又学习决策）
    
    注意：
    - 该SPO+项作用于单乘客局部定价oracle；完整动态DRT系统仍通过仿真和
      episode结束后的路由成本标签提供训练数据。
    - pricing oracle 与线上部署共用 clipped Lambert-W 定价规则，保证训练和部署一致。
    """
    
    def __init__(self, config):
        super(DSPO_plus_SPO, self).__init__(config)
        # ---- dual-head CNN: service-position proxy q_tk (steps 1/2) ----
        # --use_q_head swaps the single-head CNN_2d for CNN_2d_dual: fc3 stays
        # the cost head, fc3_q is the new service-position head. The main
        # memory target grows a second column (cost, q) so the Huber branch
        # trains both heads. Gated by the config flag: without it everything
        # stays bit-for-bit with the baseline. The q label is the remaining
        # route distance from the stop to the route end (incl. depot return) --
        # the in-vehicle travel distance the parcel would ride (manuscript
        # Sec. 4.2.2). Step 3/4 consume q in the SPO+ losses and serving.
        self.use_q_head = bool(getattr(config, 'use_q_head', False))
        # Step 4: displayed in-vehicle time in the MNL utility (manuscript
        # Eq. 7): tau_tk = q_hat_tk / vehicle_speed is computed from the
        # predicted service position and enters the passenger's utility as
        # -beta_ivt * tau_tk. beta_ivt = 0.0 keeps the baseline bit-for-bit
        # (no tau anywhere); beta_ivt > 0 needs the q head (the only tau
        # source), so it auto-forces use_q_head on.
        self.beta_ivt = float(getattr(config, 'beta_ivt', 0.0) or 0.0)
        self.vehicle_speed = float(getattr(config, 'vehicle_speed', 1.0) or 1.0)
        if self.beta_ivt > 0.0 and not self.use_q_head:
            self.use_q_head = True
            print("[q-head] beta_ivt > 0 auto-forces use_q_head (tau = q/speed)")
        self.q_huber_weight = float(getattr(config, 'q_huber_weight', 0.5))
        self.criterion_q = None
        self.spo_last_pure_q = None
        if self.use_q_head:
            if getattr(config, 'use3d_conv', False) or getattr(config, 'linearModel', False):
                print("[q-head] warning: use_q_head requires the 2D CNN, disabling q head")
                self.use_q_head = False
            else:
                from Src.Utils.Predictors import CNN_2d_dual
                from Src.Utils.Utils import MemoryBuffer
                self.supervised_ml = CNN_2d_dual(self.grid_dim, self.n_layers,
                                                 config.n_filters, config.dropout)
                self.optimizer = config.optim(self.supervised_ml.parameters(),
                                              lr=self.config.learning_rate)
                self.criterion_q = nn.HuberLoss(delta=1.0)
                self.memory = MemoryBuffer(max_len=self.config.buffer_size,
                                           time_intervals=self.n_layers,
                                           matrix_dim=self.grid_dim,
                                           target_dim=2, atype=float32, config=config)
                self.modules = [('supervised_ml', self.supervised_ml)]
                self.init()
                self.device = config.device
        self.spo_current_episode_data = {
            'pricing_decisions': [],
            'predicted_costs': [],
            'true_costs': [],
            'customer_ids': [],
            'arrival_times': []
        }
        self.spo_training_data = {}
        self.spo_training_data_next_key = 0
        spo_buffer_size = getattr(config, 'spo_buffer_size', None)
        if spo_buffer_size is None or int(spo_buffer_size) <= 0:
            spo_buffer_size = config.buffer_size
        self.spo_training_data_max_len = int(spo_buffer_size)
        self.spo_batch_size = int(getattr(config, 'spo_batch_size', 32))
        self.spo_label_sample_size = int(getattr(config, 'spo_label_sample_size', 0) or 0)
        self.spo_warmup_episodes = getattr(config, 'spo_warmup_episodes', 5)
        self.spo_rampup_episodes = getattr(config, 'spo_rampup_episodes', 10)
        self.max_spo_loss_weight = getattr(config, 'spo_loss_weight', 0.7)
        # Stage 2: differentiable profit fine-tuning (思路8.1 §二, direction 1)
        self.dspos_stage2_weight = float(getattr(config, 'dspos_stage2_weight', 0.0))
        self.dspos_freeze_conv = bool(getattr(config, 'dspos_freeze_conv', False))
        self.spo_enabled = self.max_spo_loss_weight > 0.0 or self.dspos_stage2_weight > 0.0
        self.spo_current_episode_count = 0
        self.spo_first_data_episode = None
        self.spo_huber_loss_history = []
        self.spo_stability_window = 10
        self.spo_stability_threshold = 0.05
        self.spo_training_stable = True
        self._spo_training_data_logged = False
        self._spo_positive_weight_logged = False
        self.l0_home = 0.0
        self.l_mp = 0.0
        self.spo_label_failures = 0
        self.spo_loss_failures = 0
        self.spo_probe_interval = int(getattr(config, 'spo_probe_interval', 50) or 0)
        self.spo_probe_updates = 0
        self.spo_probe_pseudo_oor = 0        # 伪成本分量落在真成本 [min,max] 外的个数
        self.spo_probe_pseudo_total = 0
        self.spo_probe_corner_pseudo = 0     # 伪成本 oracle 贴边价格个数
        self.spo_probe_corner_true = 0       # 真成本 oracle 贴边价格个数
        self.spo_probe_price_total = 0
        self.spo_probe_gradnorm_ratios = []
        self.spo_probe_loss_attempts = 0    # 尝试计算SPO+损失的次数
        self.spo_probe_loss_ok = 0          # 成功得到有效损失的次数
        self.spo_pseudo_clip_spreads = float(getattr(config, 'spo_pseudo_clip_spreads', 0.0) or 0.0)
        # ---- gradient matching (GradNorm-style) ----
        # When > 0, replace the additive loss = Huber + alpha*SPO with a rescaled
        # combination: the raw SPO+ gradient is scaled by lambda so that
        # ||lambda * grad_SPO|| = spo_grad_ratio * ||grad_Huber||. SPO+ then always
        # contributes a fixed fraction of the update regardless of loss scale --
        # curing the no-win zone (alpha too small -> SPO+ grad drowned by Huber,
        # gradnorm_ratio=0.0000; alpha too large -> corner collapse). 0.0 = legacy
        # additive path.
        self.spo_grad_ratio = float(getattr(config, 'spo_grad_ratio', 0.0) or 0.0)
        # Profit-regret objective (2026-08-05): when True the SPO+ branch of the
        # update uses the expected-net-profit regret R(a(chat);c_true) vs the
        # oracle R(a*(c_true)) instead of the lifted cost-SPO+ loss. Rationale:
        # the lifted objective minimizes predicted *cost* -- which trained the
        # CNN to quote high prices (quit 9%->16%, revenue loss > cost savings).
        # The profit-regret objective is the same quantity the offline evaluation
        # measures, so SPO+ optimizes the deployed metric.
        self.spo_profit_regret = bool(getattr(config, 'spo_profit_regret', False))
        self.spo_probe_last_lam = 0.0
        self.spo_probe_last_spo_gn = 0.0
        self.spo_probe_last_huber_gn = 0.0
        # ---- stabilization round (2026-08-06), three independent toggles ----
        # T1: sample the SPO+ batch proportional to the deployed decision's
        # profit regret (stored at label time) instead of uniform over the
        # recent pool -- focuses the decision-oriented loss on the samples
        # whose pricing was most wrong.
        self.spo_regret_weighted = bool(getattr(config, 'spo_regret_weighted', False))
        # T2: clamp the shared Lambert-W markup into the per-sample band that
        # keeps every unconstrained price inside the price box, so prices are
        # decided by the markup, not by post-hoc clipping (no joint saturation).
        self.spo_w0_band = bool(getattr(config, 'spo_w0_band', False))
        # T3: label source for the SPO+ training-data path only ('counterfactual'
        # also labels quitters); the Huber branch keeps --label_mode untouched.
        self.spo_label_mode = getattr(config, 'spo_label_mode', 'hep')

    def _cnn_forward(self, feats, caps):
        """CNN forward tolerant of both the single-head CNN_2d (returns one
        tensor) and the dual-head CNN_2d_dual (returns (cost, q)). Returns
        (cost_out, q_out_or_None)."""
        out = self.supervised_ml(feats, caps)
        if isinstance(out, (tuple, list)):
            return out[0], out[1]
        return out, None

    def _tau_from_q(self, q_val):
        """Displayed in-vehicle travel time tau_tk = q_hat_tk / vehicle_speed
        (manuscript Eq. 7). Single source of truth for the q -> tau conversion."""
        return float(q_val) / max(float(self.vehicle_speed), 1e-8)

    def _serving_taus(self, qs_ml, pps):
        """Displayed in-vehicle times aligned 1:1 with a_hat ([home, pp_0,
        pp_1, ...] over the full pps list): home carries tau = 0 (no
        in-vehicle ride); each OOH option tau = q_hat/speed. Every pps
        position gets an entry -- the env's choice set indexes the full pps
        list exactly like a_hat, and an inactive pps at price 0 still needs a
        tau when beta_ivt > 0. qs_ml follows the get_prediction layout
        [base, home, pp_0, pp_1, ...]."""
        taus = [0.0]
        for idx, pp in enumerate(pps):
            taus.append(self._tau_from_q(qs_ml[idx + 2]))
        return taus

    def get_prediction_with_q(self, cur_feat, home, pps):
        """Dual-head get_prediction: returns (costs, qs), qs the service-position
        predictions aligned 1:1 with costs ([base, home, pp_0, ...]). Single-head
        CNN yields qs = 0.0 (q never consumed then: beta_ivt=0)."""
        time_int = min(int(home.time / self.interval), self.n_layers - 1)
        new_feat = torch.cat((2 + len(pps)) * [cur_feat])
        new_feat[1][time_int][self.customer_cell[home.id_num][0]][self.customer_cell[home.id_num][1]] += 1
        cap_feat = torch.zeros((2 + len(pps)))
        cap_feat[0] = 1000000
        cap_feat[1] = 1000000
        for idx, p in enumerate(pps):
            new_feat[idx + 2][time_int][self.customer_cell[p.location.id_num][0]][self.customer_cell[p.location.id_num][1]] += 1
            cap_feat[idx + 2] = p.remainingCapacity - 1
        costs = []
        qs = []
        for i, feat in enumerate(new_feat):
            c, q = self._cnn_forward(feat.unsqueeze(0).to(self.device),
                                     cap_feat[i].unsqueeze(0).to(self.device))
            costs.append(c.item())
            qs.append(float(q.item()) if q is not None else 0.0)
        return costs, qs

    def get_prediction(self, cur_feat, home, pps):
        """Dual-head-safe override of DSPO.get_prediction.

        Identical output (per-option cost list, same order as the parent);
        delegates to get_prediction_with_q which also collects the q outputs.
        """
        costs, _ = self.get_prediction_with_q(cur_feat, home, pps)
        return costs

    def _tau_hat_from_q(self, q_slice, K):
        """Differentiable tau_hat for the K service options from one sample's
        batch CNN q outputs (feat order [base, home, pp_0, ...]): home carries
        0 (no in-vehicle ride); OOH option j+1 gets tau = q_{2+j}/speed. Used
        by the SPO+ loss paths so the q head receives the displayed-tau
        decision gradient (profit-regret path)."""
        speed = max(float(self.vehicle_speed), 1e-8)
        dtype = q_slice.dtype if q_slice is not None else torch.float32
        device = q_slice.device if q_slice is not None else self.device
        tau_hat = torch.zeros(K, dtype=dtype, device=device)
        if q_slice is None:
            return tau_hat
        for j in range(K - 1):
            tau_hat[j + 1] = q_slice[2 + j] / speed
        return tau_hat

    def _q_targets_for_memory(self, data, fleet, id_counts):
        """Per-row service-position q labels for the main Huber memory.

        Invoked by DSPO.update at episode end through the _q_targets_for_memory
        hook (absent in the plain DSPO, so that path is unchanged). Rows are
        aligned with the parent's target rows: one per stop occurrence in fleet
        order, sorted by arrival time -- the same key the parent's target uses.
        q for a stop = remaining route distance from its position to the end of
        its vehicle's route, including the final return to the depot.
        """
        if not self.use_q_head:
            return None
        from Src.Utils.counterfactual_labeler import remaining_route_distance
        rows = []
        for v in fleet["fleet"]:
            route = [int(loc.id_num) for loc in v["routePlan"]]
            for loc in v["routePlan"]:
                q = remaining_route_distance(route, int(loc.id_num), self.dist_matrix)
                rows.append([float(loc.time), q])
        rows = sorted(rows, key=itemgetter(0))
        return [r[1] for r in rows]

    @staticmethod
    def _safe_exp(x: float) -> float:
        # Clamp exponent input to avoid math overflow while preserving monotonicity.
        return float(np.exp(np.clip(float(x), -700.0, 700.0)))

    @staticmethod
    def _predicted_option_cost(fixed_term, mltplr, theta, insertion_cost, cnn_out, cnn_base):
        """Blended predicted cost of one option -- SINGLE source of truth for both the
        deployed pricing (get_action_pricing) and the SPO+ loss's c_hat reconstruction.
        Units: insertion_cost is seconds (needs mltplr); CNN outputs are already money
        (trained on mltplr-scaled labels), so the theta term is NOT re-multiplied by mltplr."""
        return fixed_term * mltplr + (1.0 - theta) * mltplr * insertion_cost + theta * (cnn_out - cnn_base)

    @staticmethod
    def lambertw_torch(x: torch.Tensor, max_iter: int = 12) -> torch.Tensor:
        """
        Principal-branch Lambert W approximation for positive tensors.

        PyTorch 2.10 does not expose torch.special.lambertw, while the SPO+
        training oracle calls Lambert-W many times. Keeping this in torch avoids
        repeated GPU->CPU->SciPy round trips inside the SPO+ batch loss.
        """
        x = torch.clamp(x, min=1e-12)
        log_x = torch.log(x)
        log_log_x = torch.log(torch.clamp(log_x, min=1e-6))
        w = torch.where(x < 3.0, torch.log1p(x), log_x - log_log_x)
        w = torch.clamp(w, min=1e-8)

        for _ in range(max_iter):
            ew = torch.exp(torch.clamp(w, max=80.0))
            f = w * ew - x
            wp1 = w + 1.0
            denom = ew * wp1 - (w + 2.0) * f / torch.clamp(2.0 * wp1, min=1e-8)
            step = f / torch.clamp(denom, min=1e-8)
            w_next = torch.clamp(w - step, min=1e-8)
            w = torch.where(torch.isfinite(w_next), w_next, w)
        return w

    @staticmethod
    def _clip_pseudo_costs(pseudo_costs_t, c_true, spreads):
        """Bound the SPO+ pseudo-cost to the true-cost range plus `spreads` spreads.

        Unbounded 2*c_hat - c drove the pricing oracle deep into its price-bound
        saturation region. Two different measurements of that, not to be conflated:
        an offline one-shot check on a single trained checkpoint (the alpha=2
        checkpoint over the RC shared holdout, 1529 samples, measured 2026-07-28)
        found 72.9% of components out of range, excess up to 34.9x the sample's
        spread; the online `spo_probe_pseudo_oor` counter (cumulative over training,
        see docs/spo_stabilization_results.md I8) instead reports 50-60%, because it
        averages over the whole run including warmup/rampup rather than one
        converged snapshot. `spreads <= 0` disables clamping; a zero-width
        true-cost range also disables it, since there is then no scale to bound with.
        """
        if spreads <= 0.0:
            return pseudo_costs_t
        lo = torch.min(c_true)
        hi = torch.max(c_true)
        margin = (hi - lo) * spreads
        if float(margin) <= 0.0:
            return pseudo_costs_t
        return torch.clamp(pseudo_costs_t, min=float(lo - margin), max=float(hi + margin))

    @staticmethod
    def _w0_band(costs, rev, min_p, max_p):
        """Per-sample band for the shared Lambert-W markup W0 (T2, 2026-08-06).

        price_k = c_k - rev - W0 stays inside [min_p, max_p] for EVERY option
        iff  W0 in [max(c) - rev - max_p, min(c) - rev - min_p]. Clamping W0
        into this band makes the box constraint vacuous: prices are decided by
        the markup alone, no joint price saturation, no corner clipping. When
        the band is empty (cost spread wider than the price box) no W0 can put
        all prices in the box and we fall back to the per-option price clip.
        """
        c = np.asarray(costs, dtype=np.float64)
        if c.size == 0 or not np.all(np.isfinite(c)):
            return None
        lo = float(np.max(c)) - rev - max_p
        hi = float(np.min(c)) - rev - min_p
        if hi - lo <= 0.0:
            return None
        return lo, hi

    def _pricing_oracle_lifted_torch(self, costs_t, customer_info, pps_info, taus=None,
                                     include_home=True):
        """
        GPU-friendly lifted local pricing oracle for SPO+ training.

        Returns a detached lifted decision w(a)=[P(a), -R0(a)]. The outside option
        divides the Lambert argument of the price fixed point (division form) and joins
        the choice-probability denominator additively; it is not included in w.

        taus (optional): displayed in-vehicle times aligned with the OOH options;
        when beta_ivt > 0 the OOH utility becomes util - beta_ivt * tau, the same
        utility the deployed passenger model uses (Eq. 7). The oracle output is
        detached either way, so this only changes the decision, never the gradient.

        include_home=False (boundary experiments, offer_mode='ooh_only'): the
        choice set has NO home option -- every cost entry is a meeting point and
        its utility comes from pps_info[j] (option 0 would otherwise be hardcoded
        as home, see the customerchoice_ooh_only path). Default True keeps the
        historical oracle bit-for-bit.
        """
        if costs_t.numel() <= 0:
            return None

        device = costs_t.device
        dtype = costs_t.dtype
        K = int(costs_t.numel())

        base_util = float(customer_info.get('base_util', self.base_util))
        home_util = float(customer_info.get('home_util', 0.0))
        s_val = float(customer_info.get('incentive_sensitivity', 1.0))
        rev_val = float(customer_info.get('revenue', self.revenue))

        if (not np.isfinite(s_val)) or abs(s_val) < 1e-8:
            return None

        pp_utils = []
        for j in range(K - (0 if include_home else 1)):
            if j < len(pps_info):
                pp_utils.append(float(pps_info[j].get('util', 0.0)))
            else:
                pp_utils.append(0.0)
        if self.beta_ivt > 0.0 and taus is not None:
            for j in range(len(pp_utils)):
                if j < len(taus):
                    pp_utils[j] -= self.beta_ivt * float(taus[j])

        outside_util = getattr(self.config, 'outside_option_util', None)
        outside_util = float(outside_util) if outside_util is not None else None

        s = torch.tensor(s_val, dtype=dtype, device=device)
        rev = torch.tensor(rev_val, dtype=dtype, device=device)

        if include_home:
            home_nonprice = base_util + home_util
            util_terms = [
                torch.tensor(home_nonprice, dtype=dtype, device=device) + s * (costs_t[0] - rev)
            ]
            for j in range(K - 1):
                util_terms.append(torch.tensor(pp_utils[j], dtype=dtype, device=device) + s * (costs_t[j + 1] - rev))
        else:
            util_terms = []
            for j in range(K):
                util_terms.append(torch.tensor(pp_utils[j], dtype=dtype, device=device) + s * (costs_t[j] - rev))

        #price fixed point in DIVISION form (spec F2): the outside option does NOT join this sum;
        #it divides the Lambert argument instead. choice_terms below keeps the additive outside
        #term because that is the true MNL choice denominator.
        terms_t = torch.stack(util_terms)
        tmax = torch.max(terms_t)
        sum_mnl = torch.exp(torch.clamp(tmax, -80.0, 80.0)) * torch.sum(torch.exp(terms_t - tmax))
        if (not torch.isfinite(sum_mnl)) or float(sum_mnl.detach().cpu()) <= 0.0:
            return None

        lambert_arg = sum_mnl / torch.tensor(e, dtype=dtype, device=device)
        if outside_util is not None:
            lambert_arg = lambert_arg / torch.exp(torch.clamp(torch.tensor(outside_util, dtype=dtype, device=device), -80.0, 80.0))
        lambertw0 = (self.lambertw_torch(lambert_arg) + 1.0) / s
        if not torch.isfinite(lambertw0):
            return None
        if self.spo_w0_band:
            band = self._w0_band(costs_t.detach().cpu().numpy(), rev_val,
                                 self.min_p, self.max_p)
            if band is not None:
                lambertw0 = torch.clamp(lambertw0, float(band[0]), float(band[1]))

        raw_prices = costs_t - rev - lambertw0
        prices = torch.clamp(raw_prices, min=self.min_p, max=self.max_p)

        if include_home:
            choice_terms = [
                torch.tensor(home_nonprice, dtype=dtype, device=device) + s * prices[0]
            ]
            for j in range(K - 1):
                choice_terms.append(torch.tensor(pp_utils[j], dtype=dtype, device=device) + s * prices[j + 1])
        else:
            choice_terms = []
            for j in range(K):
                choice_terms.append(torch.tensor(pp_utils[j], dtype=dtype, device=device) + s * prices[j])
        if outside_util is not None:
            choice_terms.append(torch.tensor(outside_util, dtype=dtype, device=device))

        choice_t = torch.stack(choice_terms)
        probs_full = torch.softmax(choice_t, dim=0)
        probs = probs_full[:K]
        r0 = torch.dot(probs, rev + prices)
        lifted = torch.cat([probs, -r0.reshape(1)]).detach()
        if not torch.isfinite(lifted).all():
            return None
        return {'prices': prices.detach(), 'probs': probs.detach(), 'r0': r0.detach(), 'lifted': lifted}

    def _exact_pricing_oracle_lifted(self, costs_t, customer_info, pps_info, taus=None):
        """Exact bounded pricing oracle (issue 01).

        Same return contract as _pricing_oracle_lifted_torch, but `prices` is the
        true box optimum from a numerical solver (Lambert-W warm-start + L-BFGS-B
        + restarts), not a clamped unconstrained Lambert-W result. Outputs are
        detached -- the solver is never differentiated through; the SPO+ lifted
        term consumes the resulting probs / r0 / lifted vector unchanged.
        taus (optional): same displayed-tau semantics as _pricing_oracle_lifted_torch.
        """
        from Src.Utils.exact_oracle import exact_pricing_oracle, customer_info_to_nonprice
        costs_np = costs_t.detach().cpu().double().numpy()
        if costs_np.size <= 0:
            return None
        nonprice = customer_info_to_nonprice(customer_info, pps_info)
        if self.beta_ivt > 0.0 and taus is not None:
            nonprice = nonprice.copy()
            for j in range(min(len(taus), nonprice.size - 1)):
                nonprice[j + 1] -= self.beta_ivt * float(taus[j])
        if nonprice.size != costs_np.size:
            return None
        s = float(customer_info.get('incentive_sensitivity', 1.0))
        if (not np.isfinite(s)) or abs(s) < 1e-8:
            return None
        rev = float(customer_info.get('revenue', self.revenue))
        outside_util = getattr(self.config, 'outside_option_util', None)
        outside_util = float(outside_util) if outside_util is not None else None
        try:
            out = exact_pricing_oracle(costs_np, nonprice, s, rev,
                                       float(self.min_p), float(self.max_p),
                                       outside_util=outside_util)
        except Exception:
            return None
        device = costs_t.device
        dtype = torch.float32
        lifted = torch.tensor(out['lifted'], dtype=dtype, device=device)
        if not (torch.isfinite(lifted).all() and float(out['probs'].sum()) > 0.0):
            return None
        return {'prices': torch.tensor(out['prices'], dtype=dtype, device=device),
                'probs': torch.tensor(out['probs'], dtype=dtype, device=device),
                'r0': torch.tensor(out['r0'], dtype=dtype, device=device),
                'lifted': lifted}

    def _resolve_pricing_oracle(self, costs_t, customer_info, pps_info, taus=None,
                                include_home=True):
        """Dispatch to the exact bounded oracle when configured (issue 01), else
        the legacy Lambert-W+clamp path. The flag defaults to absent/False so
        every historical baseline and test reproduces bit-for-bit; ablation rows
        ③+ flip it to use the exact oracle for SPO+ true/pred/pseudo and regret.
        taus (optional): displayed in-vehicle times for the OOH options, passed
        to both implementations when beta_ivt > 0.

        include_home=False (boundary experiments, offer_mode='ooh_only'): the
        exact oracle's external solver hardcodes the home option at index 0, so
        home-less choice sets always use the Lambert-W torch path."""
        if getattr(self.config, 'use_exact_oracle', False) and include_home:
            out = self._exact_pricing_oracle_lifted(costs_t, customer_info, pps_info, taus)
            # diagnostic counters (probe P1): silent fallback would void the probe
            self._eo_calls = getattr(self, '_eo_calls', 0) + 1
            if out is None:
                self._eo_fallbacks = getattr(self, '_eo_fallbacks', 0) + 1
            if self._eo_calls == 1 or self._eo_calls % 1000 == 0:
                _msg = '[DSPO+SPO exact-oracle] calls={} fallbacks={} (probe P1)\n'.format(
                    self._eo_calls, getattr(self, '_eo_fallbacks', 0))
                print(_msg, end='', flush=True)
                try:
                    with open(__import__('os').environ.get('DIAG_FILE', 'eo_probe_diag.log'), 'a') as _f:
                        _f.write(_msg)
                except OSError:
                    pass
            return out
        return self._pricing_oracle_lifted_torch(costs_t, customer_info, pps_info, taus,
                                                  include_home=include_home)

    def get_action_pricing(self, state, training):
        # The exact bounded oracle (issue 01, ablation row ③) is engaged in serving
        # independent of the SPO+ loss -- otherwise rows 2 (Lambert-W) and 3 (exact)
        # collapse to identical prices. Only the plain Huber + Lambert-W case (no SPO+,
        # no exact oracle, no in-vehicle time) delegates to DSPO and stays bit-for-bit
        # with the baseline. beta_ivt > 0 must NOT delegate: the parent's pricing and
        # the env's (prices, taus) action contract need the tau display.
        if not self.spo_enabled and not getattr(self.config, 'use_exact_oracle', False) \
                and self.beta_ivt <= 0.0:
            return super(DSPO_plus_SPO, self).get_action_pricing(state, training)

        """
        重写定价方法，记录预测成本和真实成本用于SPO+损失计算
        
        与DRO类类似，但使用spo_前缀的变量名，避免混淆
        """
        if self.load_data:
            mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])
            pps = mask[mask.mask].data
        else:
            pps = state[2]["parcelpoints"]
        
        if self.initial_phase:
            # 初始阶段，返回零价格 (serving 与 DSPO 臂对称); 但记录真实 CNN 预测
            # + feature_info, 使 spo_training_data 从第 1 集起就有非退化样本
            # (2026-08-05 修复: 此前 predicted=truth 且无 feature_info, 存储门控
            # 跳过全部样本 → SPO+ 直到初始阶段结束 (buffer 10000 需 ~170 集)
            # 才可能激活, 实际只训练了最后 ~25 步)。
            a_hat = np.zeros(len(pps)+1)

            theta = self.init_theta - (state[3] * self.cool_theta)
            mltplr = self.cost_multiplier
            pure_cnn = getattr(self.config, 'spo_pure_cnn_serving', False)

            cur_feat = self.get_feature_rep_infer(state[1]["fleet"])
            predicted_costs_ml, qs_ml = self.get_prediction_with_q(cur_feat, state[0].home, pps)
            taus = self._serving_taus(qs_ml, pps)

            home_insert_cost = self.cheapestInsertionCosts(state[0].home, state[1])
            if pure_cnn:
                homeCosts_pred = predicted_costs_ml[1] - predicted_costs_ml[0]
            else:
                homeCosts_pred = self._predicted_option_cost(
                    self.l0_home + state[0].service_time, mltplr, theta,
                    home_insert_cost, predicted_costs_ml[1], predicted_costs_ml[0])
            homeCosts_true = (self.l0_home + state[0].service_time)*mltplr + mltplr*\
                            home_insert_cost

            true_costs_list = [homeCosts_true]
            predicted_costs_full = [homeCosts_pred]
            pp_insert_costs = {}

            # 计算MNL (boundary experiments: sum covers the offered options only)
            _incl_home_sm = getattr(getattr(self, 'config', None), 'offer_mode', 'all') != 'ooh_only'
            _incl_pps_sm = getattr(getattr(self, 'config', None), 'offer_mode', 'all') != 'home_only'
            home_base_util = self.base_util + state[0].home_util
            sum_mnl_price = 0.0
            if _incl_home_sm:
                sum_mnl_price = self._safe_exp(home_base_util +
                                               (state[0].incentiveSensitivity * a_hat[0]))

            for idx, pp in enumerate(pps):
                if pp.remainingCapacity > 0:
                    util = self.mnl(state[0], pp)
                    pp_insert_cost = self.cheapestInsertionCosts(pp.location, state[1])
                    pp_insert_costs[idx] = float(pp_insert_cost)
                    if pure_cnn:
                        pp_cost = predicted_costs_ml[idx+2] - predicted_costs_ml[0]
                    else:
                        pp_cost = self._predicted_option_cost(
                            self.l_mp, mltplr, theta,
                            pp_insert_cost, predicted_costs_ml[idx+2], predicted_costs_ml[0])
                    pp_cost_true = self.l_mp*mltplr + mltplr * pp_insert_cost
                    true_costs_list.append(pp_cost_true)
                    predicted_costs_full.append(pp_cost)
                    if _incl_pps_sm:
                        sum_mnl_price += self._safe_exp(util - self.beta_ivt * taus[idx+1] +
                                                        (state[0].incentiveSensitivity * a_hat[idx+1]))

            outside_option_util = getattr(self.config, 'outside_option_util', None)
            if outside_option_util is not None:
                sum_mnl_price += self._safe_exp(outside_option_util)

            # 存储决策信息
            pps_info_list = []
            active_pps = []
            for idx, pp in enumerate(pps):
                if pp.remainingCapacity > 0:
                    util = self.mnl(state[0], pp)
                    pps_info_list.append({'util': util})
                    active_pps.append((idx, pp))

            _time_int = min(int(state[0].home.time / self.interval), self.n_layers - 1)
            _feature_info = {
                'base_feat': cur_feat.cpu().clone().detach(),
                'home_cell': tuple(self.customer_cell[state[0].home.id_num]),
                'pp_cells': [tuple(self.customer_cell[pp.location.id_num]) for _, pp in active_pps],
                'time_int': _time_int,
                'theta': float(theta),
                'ins_cost_home': float(home_insert_cost),
                'ins_cost_pps': [float(pp_insert_costs[idx]) for idx, _ in active_pps],
                'service_time': float(state[0].service_time),
                'pp_caps': [float(pp.remainingCapacity - 1) for _, pp in active_pps],
            }

            self.spo_current_episode_data['pricing_decisions'].append({
                'customer_id': state[0].id_num,
                'arrival_time': state[3],
                'prices': a_hat.copy(),
                'taus': taus,
                'predicted_costs': predicted_costs_full,
                'true_costs': true_costs_list,
                'customer': state[0],
                'pps': pps,
                'active_pps': [pp for pp in pps if pp.remainingCapacity > 0],
                'pps_info': pps_info_list,
                'sum_mnl': sum_mnl_price,
                'feature_info': _feature_info,
            })
            self.spo_current_episode_data['predicted_costs'].append(predicted_costs_full)
            self.spo_current_episode_data['true_costs'].append(true_costs_list)
            self.spo_current_episode_data['customer_ids'].append(state[0].id_num)
            self.spo_current_episode_data['arrival_times'].append(state[3])

            if self.beta_ivt > 0.0:
                return np.around(a_hat, decimals=2), taus
            return np.around(a_hat, decimals=2)
        else:
            # 非initial_phase实现（完整实现，参考DRO类但使用spo_前缀）
            if self.load_data:
                mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])
                pps = mask[mask.mask].data
            else:
                pps = state[2]["parcelpoints"]
            pp_costs = np.full(len(pps), 1000000000.0)
            
            cur_feat = self.get_feature_rep_infer(state[1]["fleet"])
            predicted_costs_ml, qs_ml = self.get_prediction_with_q(cur_feat, state[0].home, pps)
            taus = self._serving_taus(qs_ml, pps)

            theta = self.init_theta - (state[3] * self.cool_theta)
            mltplr = self.cost_multiplier
            home_insert_cost = self.cheapestInsertionCosts(state[0].home, state[1])
            # Serving alignment probe (2026-08-05): pure-CNN serving uses the
            # DSPO.py Eq.7 convention (marginal = CNN(candidate) - CNN(base), no
            # theta-blend, no l0/l_mp fixed terms). The SPO+ loss's c_hat is
            # reconstructed on the same convention (_calculate_spo_loss_for_batch),
            # so the decision-oriented objective trains exactly what serving uses.
            pure_cnn = getattr(self.config, 'spo_pure_cnn_serving', False)
            if pure_cnn:
                homeCosts_pred = predicted_costs_ml[1] - predicted_costs_ml[0]
            else:
                homeCosts_pred = self._predicted_option_cost(
                    self.l0_home + state[0].service_time, mltplr, theta,
                    home_insert_cost, predicted_costs_ml[1], predicted_costs_ml[0])
            homeCosts_true = (self.l0_home + state[0].service_time)*mltplr + mltplr*\
                            home_insert_cost

            true_costs_list = [homeCosts_true]
            pp_insert_costs = {}
            pp_utils = {}

            # Boundary experiments (docs-吴工结果0807/boundary_pricing_results.md):
            # with offer_mode home_only / ooh_only the Lambert-W MNL sum covers
            # exactly the offered options plus the outside option -- prices are
            # optimized over the restricted feasible choice set. 'all' keeps the
            # historical sum bit-for-bit.
            offer_mode = getattr(getattr(self, 'config', None), 'offer_mode', 'all')
            include_home = offer_mode != 'ooh_only'
            include_pps = offer_mode != 'home_only'

            home_base_util = self.base_util + state[0].home_util
            sum_mnl_pred = 0.0
            if include_home:
                sum_mnl_pred = self._safe_exp(home_base_util +
                                              (state[0].incentiveSensitivity * (homeCosts_pred - self.revenue)))

            for idx, pp in enumerate(pps):
                if pp.remainingCapacity > 0:
                    util = self.mnl(state[0], pp)
                    pp_utils[idx] = float(util)
                    pp_insert_cost = self.cheapestInsertionCosts(pp.location, state[1])
                    pp_insert_costs[idx] = float(pp_insert_cost)
                    if pure_cnn:
                        pp_costs[idx] = predicted_costs_ml[idx+2] - predicted_costs_ml[0]
                    else:
                        pp_costs[idx] = self._predicted_option_cost(
                            self.l_mp, mltplr, theta,
                            pp_insert_cost, predicted_costs_ml[idx+2], predicted_costs_ml[0])
                    pp_cost_true = self.l_mp*mltplr + mltplr * \
                                   pp_insert_cost
                    true_costs_list.append(pp_cost_true)
                    if include_pps:
                        sum_mnl_pred += self._safe_exp(util - self.beta_ivt * taus[idx+1] +
                                                       (state[0].incentiveSensitivity * (pp_costs[idx] - self.revenue)))
            
            #outside option enters the price fixed point in DIVISION form (spec F2; same as DSPO.py):
            #(m-1)e^m = sum/e^{V0}; V0=None/0 are bit-identical anchors. The probability-side
            #sum_mnl_price bookkeeping below keeps the ADDITIVE outside term (true MNL denominator).
            outside_option_util = getattr(self.config, 'outside_option_util', None)
            if outside_option_util is not None:
                lambertw0_pred = (lambertw(sum_mnl_pred/(e*self._safe_exp(outside_option_util))).real + 1) / state[0].incentiveSensitivity
            else:
                lambertw0_pred = (lambertw(sum_mnl_pred/e).real + 1) / state[0].incentiveSensitivity

            # T2 (2026-08-06): clamp the markup so the unconstrained prices of
            # ALL options lie inside the box -- serving and the SPO+ oracles
            # share the same rule (also applied inside the loss's a_pred).
            if self.spo_w0_band:
                _active_band_costs = [homeCosts_pred] + [
                    pp_costs[i] for i in range(len(pps))
                    if pps[i].remainingCapacity > 0]
                _band = self._w0_band(_active_band_costs, self.revenue,
                                      self.min_p, self.max_p)
                if _band is not None:
                    lambertw0_pred = float(np.clip(lambertw0_pred, _band[0], _band[1]))

            a_hat = np.zeros(len(pps)+1)
            # Exact bounded oracle (issue 01, ablation row ③): when configured, the
            # deployed prices are the true box optimum from the numerical solver, not
            # the clamped unconstrained Lambert-W fixed point. This is independent of
            # the SPO+ loss path, so it is engaged even when spo_loss_weight = 0 -- the
            # row ② vs row ③ contrast isolates the oracle's effect on pricing alone.
            if getattr(self.config, 'use_exact_oracle', False):
                active_idx = [i for i, pp in enumerate(pps) if pp.remainingCapacity > 0]
                active_costs = [homeCosts_pred] + [pp_costs[i] for i in active_idx]
                _ci = {'base_util': self.base_util, 'home_util': state[0].home_util,
                       'incentive_sensitivity': state[0].incentiveSensitivity,
                       'revenue': self.revenue}
                _pi = [{'util': pp_utils[i]} for i in active_idx]
                _tau_list = [taus[i + 1] for i in active_idx]
                _oo = self._resolve_pricing_oracle(
                    torch.tensor(active_costs, dtype=float32), _ci, _pi,
                    taus=_tau_list)
                if _oo is not None:
                    _op = _oo['prices'].detach().cpu().numpy()
                    a_hat[0] = _op[0]
                    for _j, _i in enumerate(active_idx):
                        a_hat[_i + 1] = _op[_j + 1]
                else:
                    a_hat[0] = homeCosts_pred - self.revenue - lambertw0_pred
                    for _i in active_idx:
                        a_hat[_i + 1] = pp_costs[_i] - self.revenue - lambertw0_pred
            else:
                # restricted choice sets: only the offered options receive a price
                # (home_only -> home only; ooh_only -> meeting points only); the
                # non-offered entries stay 0 and are never seen by the env.
                if include_home:
                    a_hat[0] = homeCosts_pred - self.revenue - lambertw0_pred
                if include_pps:
                    for _i, _pp in enumerate(pps):
                        if _pp.remainingCapacity > 0:
                            a_hat[_i + 1] = pp_costs[_i] - self.revenue - lambertw0_pred

            a_hat = np.clip(a_hat, self.min_p, self.max_p)
            
            predicted_costs_full = [homeCosts_pred] + [pp_costs[i] for i in range(len(pps)) if pps[i].remainingCapacity > 0]
            true_costs_full = true_costs_list
            
            # bookkeeping MNL denominator over the offered options only
            # (offer_mode gates defined above: include_home / include_pps)
            home_base_util = self.base_util + state[0].home_util
            sum_mnl_price = 0.0
            if include_home:
                sum_mnl_price = self._safe_exp(home_base_util +
                                               (state[0].incentiveSensitivity * a_hat[0]))
            for idx, pp in enumerate(pps):
                if pp.remainingCapacity > 0:
                    util = pp_utils.get(idx)
                    if util is None:
                        util = self.mnl(state[0], pp)
                        pp_utils[idx] = float(util)
                    if include_pps:
                        sum_mnl_price += self._safe_exp(util - self.beta_ivt * taus[idx+1] +
                                                        (state[0].incentiveSensitivity * a_hat[idx+1]))

            if outside_option_util is not None:
                sum_mnl_price += self._safe_exp(outside_option_util)
            
            # 存储决策信息
            pps_info_list = []
            for idx, pp in enumerate(pps):
                if pp.remainingCapacity > 0:
                    util = pp_utils.get(idx)
                    if util is None:
                        util = self.mnl(state[0], pp)
                        pp_utils[idx] = float(util)
                    pps_info_list.append({'util': util})
            
            # Compute feature metadata for SPO+ gradient connection
            _time_int = min(int(state[0].home.time / self.interval), self.n_layers - 1)
            _active_pps = [(idx, pp) for idx, pp in enumerate(pps) if pp.remainingCapacity > 0]
            _feature_info = {
                'base_feat': cur_feat.cpu().clone().detach(),
                'home_cell': tuple(self.customer_cell[state[0].home.id_num]),
                'pp_cells': [tuple(self.customer_cell[pp.location.id_num]) for _, pp in _active_pps],
                'time_int': _time_int,
                'theta': float(theta),
                'ins_cost_home': float(home_insert_cost),
                'ins_cost_pps': [float(pp_insert_costs[idx]) for idx, _ in _active_pps],
                'service_time': float(state[0].service_time),
                'pp_caps': [float(pp.remainingCapacity - 1) for _, pp in _active_pps],
            }
            self.spo_current_episode_data['pricing_decisions'].append({
                'customer_id': state[0].id_num,
                'arrival_time': state[3],
                'prices': a_hat.copy(),
                'taus': taus,
                'predicted_costs': predicted_costs_full,
                'true_costs': true_costs_full,
                'customer': state[0],
                'pps': pps,
                'active_pps': [pp for _, pp in _active_pps],
                'pps_info': pps_info_list,
                'sum_mnl': sum_mnl_price,
                'feature_info': _feature_info,
            })
            self.spo_current_episode_data['predicted_costs'].append(predicted_costs_full)
            self.spo_current_episode_data['true_costs'].append(true_costs_full)
            self.spo_current_episode_data['customer_ids'].append(state[0].id_num)
            self.spo_current_episode_data['arrival_times'].append(state[3])

            if self.beta_ivt > 0.0:
                return np.around(a_hat, decimals=2), taus
            return np.around(a_hat, decimals=2)

    def _get_terminal_label_route(self, data, route_ids):
        """Terminal label route = the HGS re-optimized end-of-day route -- the same anchor
        the parent's Huber labels use (design decision 2026-07-25, spec F1 / plan A).
        route_ids is validated against data['id'] but takes no part in the solve."""
        assert np.array_equal(np.asarray(route_ids).ravel(), np.asarray(data['id']).ravel()), \
            "route_ids out of sync with data['id']"
        return self.reopt_HGS_final(data)

    def _deployed_regret(self, decision):
        """T1 priority weight: profit regret of the deployed price vector.

        R(a_deployed; c_true) vs the oracle R(a*(c_true)), normalized by
        (|R*| + 1) -- the same quantity the profit-regret loss optimizes,
        evaluated at label time with the prices that were actually served.
        Degeneracies return 0.0 (sampling then falls back to uniform over
        the recent pool). Note: under use_exact_oracle the deployed prices
        ARE the oracle's, so regret ~ 0 and this weight carries no signal --
        it is meant for the Lambert-W serving path."""
        true_costs = decision.get('true_costs', [])
        prices = decision.get('prices', None)
        taus = decision.get('taus', None)
        customer = decision.get('customer', None)
        pps_info = decision.get('pps_info', [])
        if prices is None or customer is None or len(true_costs) < 2:
            return 0.0
        K = len(true_costs)
        if K > len(prices):
            return 0.0
        device = self.device
        dtype = torch.float32
        c_true = torch.tensor(np.asarray(true_costs[:K], dtype=np.float32),
                              dtype=dtype, device=device)
        a = torch.tensor(np.asarray(prices[:K], dtype=np.float32),
                         dtype=dtype, device=device)
        customer_info = {
            'base_util': self.base_util,
            'home_util': customer.home_util,
            'incentive_sensitivity': customer.incentiveSensitivity,
            'revenue': self.revenue,
        }
        s = float(customer_info['incentive_sensitivity'])
        rev = float(customer_info['revenue'])
        if (not np.isfinite(s)) or abs(s) < 1e-8:
            return 0.0
        s_t = torch.tensor(s, dtype=dtype, device=device)
        rev_t = torch.tensor(rev, dtype=dtype, device=device)
        home_nonprice = float(customer_info['base_util']) + float(customer_info['home_util'])
        outside_util = getattr(self.config, 'outside_option_util', None)
        outside_util_f = float(outside_util) if outside_util is not None else None

        choice_terms = [torch.tensor(home_nonprice, dtype=dtype, device=device) + s_t * a[0]]
        for j in range(K - 1):
            u = float(pps_info[j].get('util', 0.0)) if j < len(pps_info) else 0.0
            if self.beta_ivt > 0.0 and taus is not None and (j + 1) < len(taus):
                u -= self.beta_ivt * float(taus[j + 1])
            choice_terms.append(torch.tensor(u, dtype=dtype, device=device) + s_t * a[j + 1])
        if outside_util_f is not None:
            choice_terms.append(torch.tensor(outside_util_f, dtype=dtype, device=device))
        probs = torch.softmax(torch.stack(choice_terms), dim=0)[:K]
        R_deployed = torch.dot(probs, rev_t + a - c_true)

        oracle = self._resolve_pricing_oracle(c_true.detach(), customer_info, pps_info)
        if oracle is None:
            return 0.0
        p_star = oracle['prices'].to(device=device, dtype=dtype)
        probs_star = oracle['probs'].to(device=device, dtype=dtype)
        R_star = torch.dot(probs_star, rev_t + p_star - c_true.detach())

        regret = float((R_star.detach() - R_deployed) / (torch.abs(R_star.detach()) + 1.0))
        if not np.isfinite(regret):
            return 0.0
        return regret

    def update(self, data, state, done=False):
        """
        重写update方法，在episode结束时存储预测成本和真实成本用于SPO+损失训练

        与DRO类类似，但使用spo_前缀的变量名，避免混淆
        """
        cost = super(DSPO_plus_SPO, self).update(data, state, done)
        if not self.spo_enabled:
            # The exact-oracle serving path (use_exact_oracle=True, spo_loss=0) still
            # records pricing decisions; with SPO+ off they are never consumed, so clear
            # them at episode end to avoid unbounded growth across a long run.
            if done:
                self.spo_current_episode_data = {
                    'pricing_decisions': [], 'predicted_costs': [],
                    'true_costs': [], 'customer_ids': [], 'arrival_times': []}
            return cost
        
        if done:
            # 获取全局成本（与DRO类似）
            id_data = data.get('id', [])
            if np.isscalar(id_data):
                route_ids = np.array([int(id_data)])
            else:
                route_ids = np.asarray(id_data)
            positive_customer_count = int(np.sum(route_ids > 0)) if route_ids.size > 0 else 0
            # Tiny routes are not informative for SPO+ and can be unstable for HGS.
            if positive_customer_count <= 1:
                self.spo_current_episode_count += 1
                self.spo_current_episode_data = {
                    'pricing_decisions': [],
                    'predicted_costs': [],
                    'true_costs': [],
                    'customer_ids': [],
                    'arrival_times': []
                }
                return cost
            
            try:
                fleet, final_cost = self._get_terminal_label_route(data, route_ids)
                
                if fleet is None or len(fleet["fleet"]) == 0:
                    self.spo_current_episode_data = {
                        'pricing_decisions': [],
                        'predicted_costs': [],
                        'true_costs': [],
                        'customer_ids': [],
                        'arrival_times': []
                    }
                    return cost

                id_counts = Counter(np.asarray(data['id']).tolist())
                global_costs = self.get_per_customer_costs(fleet, id_counts)
                global_costs_by_time = {
                    float(time): float(global_cost) for time, global_cost in global_costs
                }
                decisions_to_label = self.spo_current_episode_data['pricing_decisions']
                if self.spo_label_sample_size > 0 and len(decisions_to_label) > self.spo_label_sample_size:
                    sampled_idx = np.random.choice(
                        len(decisions_to_label),
                        size=self.spo_label_sample_size,
                        replace=False,
                    )
                    decisions_to_label = [decisions_to_label[int(i)] for i in sampled_idx]

                # 更新每个决策的真实成本
                for decision in decisions_to_label:
                    customer = decision.get('customer', None)
                    arrival_time = float(decision.get('arrival_time', customer.home.time if customer else -1))
                    global_cost_base = global_costs_by_time.get(arrival_time, None)
                    if customer and global_cost_base is not None:
                        pps = decision.get('active_pps', decision.get('pps', []))
                        # T3 (2026-08-06): spo_label_mode overrides the label
                        # source for the SPO+ training-data path only; the
                        # Huber branch keeps --label_mode untouched.
                        _spo_label_mode = ('counterfactual'
                                           if self.spo_label_mode == 'counterfactual'
                                           else getattr(self.config, 'label_mode', 'hep'))
                        if _spo_label_mode == 'counterfactual':
                            labeled = self._calculate_counterfactual_costs_for_all_options(
                                customer, pps, fleet, arrival_time=arrival_time)
                        else:
                            labeled = self._calculate_global_costs_for_all_options_spo(
                                customer, pps, fleet, arrival_time=arrival_time)
                        if labeled is not None and len(labeled[0]) > 1:
                            decision['true_costs'] = labeled[0]
                            # service-position q labels (step 2): per-option
                            # remaining route distance, same order as true_costs
                            decision['q_labels'] = labeled[1]
                            decision['has_terminal_label'] = True
                            # T1 (2026-08-06): priority weight = profit regret
                            # of the DEPLOYED price vector vs the oracle on the
                            # true costs, evaluated once at label time.
                            if self.spo_regret_weighted:
                                decision['regret_w'] = self._deployed_regret(decision)
                
                # 存储训练数据
                if len(decisions_to_label) > 0:
                    for decision in decisions_to_label:
                        if not decision.get('has_terminal_label', False):
                            continue
                        if decision.get('feature_info', None) is None:
                            continue
                        if self.spo_training_data_max_len <= 0:
                            continue
                        memory_idx = self.spo_training_data_next_key
                        self.spo_training_data_next_key += 1
                        if len(self.spo_training_data) >= self.spo_training_data_max_len:
                            oldest_key = min(self.spo_training_data.keys())
                            self.spo_training_data.pop(oldest_key, None)
                        if memory_idx >= 0:
                            customer = decision.get('customer', None)
                            if customer:
                                customer_info = {
                                    'base_util': self.base_util,
                                    'home_util': customer.home_util,
                                    'incentive_sensitivity': customer.incentiveSensitivity,
                                    'revenue': self.revenue
                                }
                            else:
                                customer_info = {
                                    'base_util': self.base_util,
                                    'home_util': 3.75,
                                    'incentive_sensitivity': 0.99,
                                    'revenue': self.revenue
                                }
                            
                            pps_info = decision.get('pps_info', [])
                            predicted_costs_list = decision['predicted_costs']
                            true_costs_list = decision['true_costs']
                            
                            if isinstance(predicted_costs_list, np.ndarray):
                                predicted_costs_list = predicted_costs_list.tolist()
                            if isinstance(true_costs_list, np.ndarray):
                                true_costs_list = true_costs_list.tolist()
                            
                            predicted_costs_list = [float(x) if not isinstance(x, (list, np.ndarray)) else float(x[0]) if len(x) > 0 else 0.0 
                                                    for x in predicted_costs_list]
                            true_costs_list = [float(x) if not isinstance(x, (list, np.ndarray)) else float(x[0]) if len(x) > 0 else 0.0 
                                              for x in true_costs_list]
                            
                            _dec_taus = decision.get('taus', None)
                            self.spo_training_data[memory_idx] = {
                                'predicted_costs': predicted_costs_list,
                                'true_costs': true_costs_list,
                                'q_labels': [float(q) for q in decision.get('q_labels', [])],
                                'prices': np.array(decision['prices'], dtype=np.float32),
                                'taus': ([float(t) for t in _dec_taus] if _dec_taus is not None else None),
                                'customer_info': customer_info,
                                'pps_info': pps_info,
                                'feature_info': decision.get('feature_info', None),
                                'regret_w': float(decision.get('regret_w', 0.0)),
                            }
                            if self.spo_first_data_episode is None and len(self.spo_training_data) > 0:
                                self.spo_first_data_episode = self.spo_current_episode_count

                    if (not self._spo_training_data_logged) and len(self.spo_training_data) > 0:
                        print(f"[SPO+ debug] spo_training_data populated, size={len(self.spo_training_data)}")
                        self._spo_training_data_logged = True
                
                self.spo_current_episode_count += 1
                self.spo_current_episode_data = {
                    'pricing_decisions': [],
                    'predicted_costs': [],
                    'true_costs': [],
                    'customer_ids': [],
                    'arrival_times': []
                }
            except Exception as e:
                self.spo_label_failures += 1
                print(f"  [SPO+错误 #{self.spo_label_failures}] 标签收集失败(共{self.spo_label_failures}次): {type(e).__name__}: {e}")
                self.spo_current_episode_data = {
                    'pricing_decisions': [],
                    'predicted_costs': [],
                    'true_costs': [],
                    'customer_ids': [],
                    'arrival_times': []
                }
        
        return cost
    
    def _route_distance(self, loc_a, loc_b):
        """Distance with implicit depot endpoints for final HGS routes."""
        if self.load_data:
            a_id = 0 if loc_a is None else int(loc_a.id_num)
            b_id = 0 if loc_b is None else int(loc_b.id_num)
            return float(self.dist_matrix[a_id][b_id])

        if loc_a is None:
            loc_a = getattr(getattr(self.config, 'env', None), 'depot', None)
        if loc_b is None:
            loc_b = getattr(getattr(self.config, 'env', None), 'depot', None)
        if loc_a is None or loc_b is None:
            return float("inf")
        return sqrt((loc_a.x - loc_b.x) ** 2 + (loc_a.y - loc_b.y) ** 2)

    def _best_insertion_delta_on_final_routes(self, routes, loc):
        """Cheapest depot-aware insertion delta on routes that omit depot nodes."""
        best_delta = float("inf")
        for route in routes:
            n = len(route)
            for pos in range(n + 1):
                prev_loc = route[pos - 1] if pos > 0 else None
                next_loc = route[pos] if pos < n else None
                delta = (
                    self._route_distance(prev_loc, loc)
                    + self._route_distance(loc, next_loc)
                    - self._route_distance(prev_loc, next_loc)
                )
                if np.isfinite(delta) and delta < best_delta:
                    best_delta = float(delta)
        return best_delta

    def _hep_block_share(self, routes, loc, service_time):
        """This customer's share of the HEP cost of visiting `loc` (spec 2026-07-28 §2).

        A location may appear several times in the final routes -- data['id'] gets one
        entry per accepted customer (Parcelpoint_py.py:222), so a shared OOH point
        becomes N duplicate nodes at zero mutual distance. Charging each duplicate its
        own half-edges splits the block arbitrarily (the middle one gets 0, and which
        node lands in the middle is up to HGS's ordering of identical nodes). We instead
        cost the whole visit block once and divide it by the number of users.

        `routes` must already have this arrival's own stop removed, so the block holds
        N' other users and the divisor is N' + 1.

        Returns (share, n_users); `share` excludes cost_multiplier.

        Note: If `routes == []` (no vehicles), the absent-branch returns (inf, 1) since
        there are no positions to evaluate. In practice, fleets always have ≥1 vehicle.
        """
        target_id = int(loc.id_num)
        block_travel = 0.0
        n_existing = 0
        for route in routes:
            n = len(route)
            pos = 0
            while pos < n:
                if int(route[pos].id_num) != target_id:
                    pos += 1
                    continue
                run_start = pos
                while pos < n and int(route[pos].id_num) == target_id:
                    pos += 1
                run_end = pos - 1
                before = route[run_start - 1] if run_start > 0 else None
                after = route[run_end + 1] if run_end + 1 < n else None
                block_travel += (0.5 * self._route_distance(before, loc)
                                 + 0.5 * self._route_distance(loc, after))
                n_existing += run_end - run_start + 1

        if n_existing == 0:
            # Not visited: pick the position minimising the share we are about to
            # report, then bear it alone.
            best = float('inf')
            for route in routes:
                n = len(route)
                for pos in range(n + 1):
                    before = route[pos - 1] if pos > 0 else None
                    after = route[pos] if pos < n else None
                    cand = (0.5 * self._route_distance(before, loc)
                            + 0.5 * self._route_distance(loc, after))
                    if np.isfinite(cand) and cand < best:
                        best = float(cand)
            if not np.isfinite(best):
                return float('inf'), 1
            return best + float(service_time), 1

        n_users = n_existing + 1
        return (block_travel + float(service_time)) / n_users, n_users

    def _routes_without_arrival(self, fleet, arrival_time):
        """Remove the served location for one arrival from final routes."""
        routes = []
        removed = False
        for vehicle in fleet["fleet"]:
            route = []
            for loc in vehicle["routePlan"]:
                loc_time = float(getattr(loc, 'time', -1))
                if (not removed) and abs(loc_time - float(arrival_time)) < 1e-6:
                    removed = True
                    continue
                route.append(loc)
            routes.append(route)
        return routes if removed else None

    def _calculate_global_costs_for_all_options_spo(self, customer, pps, fleet, arrival_time=None):
        """Per-option HEP block share for one arrival (spec 2026-07-28 §2).

        Previously this used the cheapest *insertion delta* on the final routes, which
        returned exactly 0 for any OOH point another customer had already used -- 92.7%
        of RC's OOH labels were zero (docs/spo_label_audit.md). Block-averaged HEP
        removes that degeneracy, cannot go negative, and matches the caliber DSPO's
        Huber target uses.
        """
        if arrival_time is None:
            arrival_time = customer.home.time

        reduced_routes = self._routes_without_arrival(fleet, arrival_time)
        if reduced_routes is None:
            return None

        mltplr = self.cost_multiplier
        home_share, _ = self._hep_block_share(
            reduced_routes, customer.home, self.service_times[int(customer.home.id_num)])
        if not np.isfinite(home_share):
            return None
        failure_cost = (float(getattr(self.config, 'home_failure', 0.0))
                        * float(getattr(self.config, 'failure_cost', 0.0)))
        option_costs = [home_share * mltplr + failure_cost]
        # q labels (step 2): remaining route distance per option, aligned with
        # option_costs (home first, then pps in order).
        q_labels = [self._q_for_location(reduced_routes, int(customer.home.id_num))]

        for pp in pps:
            share, _ = self._hep_block_share(
                reduced_routes, pp.location, self.service_times[int(pp.location.id_num)])
            if np.isfinite(share):
                option_costs.append(share * mltplr)
                q_labels.append(self._q_for_location(reduced_routes, int(pp.location.id_num)))

        if len(option_costs) <= 1:
            return None
        return option_costs, q_labels

    def _q_for_location(self, routes, loc_id):
        """Service-position q for one location within Location-object routes.

        Last occurrence in any route -> remaining distance to that route's end
        (incl. depot return). If the location is absent (an unchosen option),
        take its cheapest-insertion position across vehicles and the remaining
        distance from there -- the same insertion rule the cost labels use.
        """
        from Src.Utils.counterfactual_labeler import (
            remaining_route_distance, cheapest_insertion)
        routes_ids = [[int(l.id_num) for l in r] for r in routes]
        for route in routes_ids:
            if any(int(s) == int(loc_id) for s in route):
                return remaining_route_distance(route, int(loc_id), self.dist_matrix)
        best = float('inf')
        for route in routes_ids:
            ins_route, _ = cheapest_insertion(route, int(loc_id), self.dist_matrix)
            q = remaining_route_distance(ins_route, int(loc_id), self.dist_matrix)
            if np.isfinite(q) and q < best:
                best = float(q)
        return best

    def _calculate_counterfactual_costs_for_all_options(self, customer, pps, fleet,
                                                         arrival_time=None):
        """Per-option counterfactual marginal routing cost c_tk (issue 02 / D2).

        c_tk = C(B_T^{-t} u {(t,k)}) - C(B_T^{-t}) via insertion + 2-opt, on the
        end-of-day route with passenger t's own stop removed. QUITTERS -- whose
        stop was never inserted, so _routes_without_arrival returns None -- are
        labeled against the FULL realized route (B_T^{-t} = B_T), which removes
        the HEP label's selection bias against high-V0 passengers. This is the
        pure-routing c_routing: no meeting-point capacity penalty, no home-
        failure term, no service-time amortization. Vehicle capacity stays a
        CVRP hard constraint owned by the route solver.
        """
        from Src.Utils.counterfactual_labeler import counterfactual_option_costs_and_q
        if arrival_time is None:
            arrival_time = customer.home.time
        reduced = self._routes_without_arrival(fleet, arrival_time)
        if reduced is None:
            # quitter: nothing to remove -> label against the full realized route
            reduced = [[loc for loc in v["routePlan"]] for v in fleet["fleet"]]
        routes = [[int(loc.id_num) for loc in r] for r in reduced]
        option_locs = [int(customer.home.id_num)] + [int(pp.location.id_num) for pp in pps]
        # q labels (step 2): per-option remaining route distance, taken from
        # the SAME insertion+refine pass that produces the cost labels.
        costs, qs = counterfactual_option_costs_and_q(routes, option_locs, self.dist_matrix)
        if not np.all(np.isfinite(costs)) or len(costs) <= 1:
            return None
        mltplr = self.cost_multiplier
        return [float(c) * mltplr for c in costs], [float(q) for q in qs]

    def optimize(self):
        """优化方法（支持SPO+损失）"""
        feat, cap_feat, target = self.memory.sample(batch_size=self.config.batch_size)
        spo_weight = self._get_current_spo_weight()
        if (not self._spo_positive_weight_logged) and spo_weight > 0.0:
            print(
                f"[SPO+ debug] spo_weight became positive: {spo_weight:.4f}, "
                f"episode_count={self.spo_current_episode_count}, "
                f"training_data_size={len(self.spo_training_data)}"
            )
            self._spo_positive_weight_logged = True
        loss = self.self_supervised_update(feat, cap_feat, target, spo_weight=spo_weight)
        return loss
    
    def initial_phase_training(self, max_epochs=-1):
        """Override: thread the SPO+ weight into the initial-phase epochs so the
        decision-oriented loss participates from the first training batches.
        (2026-08-05 fix: the parent's loop calls self_supervised_update without
        spo_weight, so SPO+ could not act until the phase ended -- with the
        buffer-10000 phase spanning ~170 episodes, that left only the last
        ~25 optimizer steps of training. The loop, epochs, convergence break,
        and save cadence are identical to the parent; only the spo_weight
        kwarg differs. 2026-08-06: the convergence break now judges on the
        PURE Huber per-epoch mean (tracked via self.spo_last_pure_huber), not
        the combined Huber + lam*SPO value -- under gradmatch the combined
        value can be monotonically decreasing (negative regret * lam ~ 1e8),
        which never satisfies the break and runs the phase out to all
        initial_phase_epochs (50). The combined value is still printed for
        diagnostics but no longer drives termination."""
        initial_losses = []      # per-epoch pure-Huber means (convergence judge)
        initial_combined = []    # per-epoch combined values (display only)
        print("Inital training phase started...")
        for counter in range(max_epochs):
            losses = []
            pure_losses = []
            pure_qs = []
            for feat, cap_feat, target in self.memory.batch_sample(
                    batch_size=self.config.batch_size, randomize=True):
                spo_weight = self._get_current_spo_weight()
                loss = self.self_supervised_update(feat, cap_feat, target,
                                                   spo_weight=spo_weight)
                losses.append(loss)
                pure = getattr(self, 'spo_last_pure_huber', None)
                if pure is not None:
                    pure_losses.append(pure)
                pure_q = getattr(self, 'spo_last_pure_q', None)
                if pure_q is not None:
                    pure_qs.append(pure_q)
            initial_combined.append(np.mean(losses))
            epoch_pure = np.mean(pure_losses) if pure_losses else np.mean(losses)
            initial_losses.append(epoch_pure)
            epoch_q = np.mean(pure_qs) if pure_qs else None
            if counter % 1 == 0:
                q_str = " | q-loss:: {:.4f}".format(epoch_q) if epoch_q is not None else ""
                print("Epoch {} Huber loss:: {} (combined {}{})".format(
                    counter, np.mean(initial_losses[-10:]),
                    np.mean(initial_combined[-10:]), q_str))
                if self.config.only_phase_one:
                    self.memory.save(self.config.paths['checkpoint'] + 'initial_')
                    self.save()
                    print("Saved..")
            # Terminate initial phase once it have converged (pure Huber).
            if len(initial_losses) >= 20 and np.mean(initial_losses[-10:]) + 1e-5 >= np.mean(initial_losses[-20:]):
                print("Converged...")
                break
        print('... Initial training phase terminated!')
        self.initial_phase = False
        self.memory.save(self.config.paths['checkpoint'] + 'initial_')
        self.save()
        if self.dspos_stage2_weight > 0.0 and self.dspos_freeze_conv:
            self._freeze_conv_layers()

    # ---- Stage 2: differentiable profit fine-tuning (direction 1) ----------
    def _freeze_conv_layers(self):
        """Freeze conv1 and conv2, leaving only FC layers trainable."""
        for name, param in self.supervised_ml.named_parameters():
            if 'conv1' in name or 'conv2' in name:
                param.requires_grad_(False)
        print('[stage2] conv layers frozen', flush=True)

    def _profit_regret_loss(self):
        """Normalized decision regret: (R(a*;c) - R(a(ĉ);c)) / (|R(a*;c)| + 1).

        Uses differentiable Lambert-W to compute a(ĉ), then MNL expected profit.
        Returns scalar loss (0 with grad if no valid samples)."""
        if len(self.spo_training_data) == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)

        device = self.device
        dtype = torch.float32
        mltplr = self.cost_multiplier

        # Sample batch from spo_training_data
        available = sorted(self.spo_training_data.keys(), reverse=True)
        B = min(max(1, self.spo_batch_size), len(available))
        pool = available[:min(len(available), B * 4)]
        indices = self._sample_spo_batch(pool, B)

        regrets = []
        for idx in indices:
            td = self.spo_training_data.get(idx)
            if td is None or td.get('feature_info') is None:
                continue
            fi = td['feature_info']
            true_costs = td.get('true_costs', [])
            if len(true_costs) < 2:
                continue
            K = 1 + len(fi['pp_cells'])
            if len(true_costs) < K:
                continue

            # Reconstruct predicted costs ĉ via CNN forward (differentiable)
            base_feat = fi['base_feat'].to(device)
            home_cell = fi['home_cell']
            pp_cells = fi['pp_cells']
            time_int = fi['time_int']
            theta_val = fi['theta']
            pp_caps = fi.get('pp_caps', [1000000.0] * len(pp_cells))

            all_feats, all_caps = [], []
            # base
            all_feats.append(base_feat)
            all_caps.append(1000000.0)
            # home
            ha = base_feat.clone()
            ha[0][time_int][home_cell[0]][home_cell[1]] += 1
            all_feats.append(ha)
            all_caps.append(1000000.0)
            # meeting points
            for j in range(len(pp_cells)):
                pa = base_feat.clone()
                pa[0][time_int][pp_cells[j][0]][pp_cells[j][1]] += 1
                all_feats.append(pa)
                all_caps.append(float(pp_caps[j]) if j < len(pp_caps) else 1000000.0)

            batch_f = torch.cat(all_feats, dim=0).to(device)
            batch_c = torch.tensor(all_caps, dtype=dtype, device=device)
            cnn_out, q_out = self._cnn_forward(batch_f, batch_c)
            cnn_out = cnn_out.squeeze(-1)
            q_out = q_out.squeeze(-1) if q_out is not None else None

            # Reconstruct ĉ from CNN outputs (same formula as get_action_pricing)
            base_out = cnn_out[0]
            home_out = cnn_out[1]
            c_hat = torch.zeros(K, dtype=dtype, device=device)
            c_hat[0] = self._predicted_option_cost(
                self.l0_home + fi['service_time'], mltplr, theta_val,
                fi['ins_cost_home'], home_out, base_out)
            for j in range(K - 1):
                c_hat[j + 1] = self._predicted_option_cost(
                    self.l_mp, mltplr, theta_val,
                    fi['ins_cost_pps'][j], cnn_out[2 + j], base_out)

            c_true_t = torch.tensor(
                np.asarray(true_costs[:K], dtype=np.float32),
                dtype=dtype, device=device, requires_grad=False)

            if not torch.isfinite(c_hat).all():
                continue

            # Displayed in-vehicle times from the q head (differentiable): the
            # passenger utility -beta_ivt * tau_hat enters pricing and choice,
            # so the profit-regret gradient reaches the q head.
            tau_hat = self._tau_hat_from_q(q_out, K)

            # Lambert-W pricing from predicted costs
            ci = td.get('customer_info', {})
            s_val = float(ci.get('incentive_sensitivity', -0.25))
            rev_val = float(ci.get('revenue', self.revenue))
            base_util = float(ci.get('base_util', self.base_util))
            home_util = float(ci.get('home_util', 0.0))
            home_nonprice = base_util + home_util
            s_t = torch.tensor(s_val, dtype=dtype, device=device)
            rev_t = torch.tensor(rev_val, dtype=dtype, device=device)
            outside_util = getattr(self.config, 'outside_option_util', None)
            outside_util_f = float(outside_util) if outside_util is not None else None

            # MNL terms for Lambert-W
            terms = [torch.tensor(home_nonprice, dtype=dtype, device=device)
                     + s_t * (c_hat[0] - rev_t)]
            pps_info = td.get('pps_info', [])
            for j in range(K - 1):
                pp_u = float(pps_info[j].get('util', 0.0)) if j < len(pps_info) else 0.0
                terms.append(torch.tensor(pp_u, dtype=dtype, device=device)
                             - self.beta_ivt * tau_hat[j + 1]
                             + s_t * (c_hat[j + 1] - rev_t))
            terms_t = torch.stack(terms)
            tmax = torch.max(terms_t)
            sum_mnl = torch.exp(torch.clamp(tmax, -80.0, 80.0)) * torch.sum(
                torch.exp(terms_t - tmax))
            if float(sum_mnl.detach().cpu()) <= 0.0:
                continue

            lambert_arg = sum_mnl / torch.tensor(np.e, dtype=dtype, device=device)
            if outside_util_f is not None:
                lambert_arg = lambert_arg / torch.exp(
                    torch.clamp(torch.tensor(outside_util_f, dtype=dtype, device=device),
                               -80.0, 80.0))
            lw = (self.lambertw_torch(lambert_arg) + 1.0) / s_t
            if not torch.isfinite(lw):
                continue
            if self.spo_w0_band:
                band = self._w0_band(c_hat.detach().cpu().numpy(), rev_val,
                                     self.min_p, self.max_p)
                if band is not None:
                    lw = torch.clamp(lw, float(band[0]), float(band[1]))

            a_pred = c_hat - rev_t - lw           # unconstrained Lambert-W prices
            # STE with corner masking (same rationale as
            # _calculate_profit_regret_for_batch): forward uses the clipped
            # price; backward only for options inside the box, so saturated
            # decisions (made by the constraint, not the prediction) never
            # push c_hat with a constant one-sided drift.
            lo, hi = float(self.min_p), float(self.max_p)
            a_pred_clipped = torch.clamp(a_pred, lo, hi)
            in_box = ((a_pred >= lo) & (a_pred <= hi)).to(dtype)
            a_pred = (a_pred_clipped.detach()
                      + (a_pred - a_pred.detach()) * in_box)

            # MNL choice probabilities under a_pred
            choice_terms = [torch.tensor(home_nonprice, dtype=dtype, device=device)
                            + s_t * a_pred[0]]
            for j in range(K - 1):
                pp_u = float(pps_info[j].get('util', 0.0)) if j < len(pps_info) else 0.0
                choice_terms.append(torch.tensor(pp_u, dtype=dtype, device=device)
                                    - self.beta_ivt * tau_hat[j + 1]
                                    + s_t * a_pred[j + 1])
            if outside_util_f is not None:
                choice_terms.append(torch.tensor(outside_util_f, dtype=dtype, device=device))
            choice_t = torch.stack(choice_terms)
            probs = torch.softmax(choice_t, dim=0)[:K]

            # Expected profit R(a_pred; c_true)
            R_pred = torch.dot(probs, rev_t + a_pred - c_true_t)

            # Oracle profit R(a*; c_true) from true costs (detached)
            c_true_detached = c_true_t.detach()
            terms_true = [torch.tensor(home_nonprice, dtype=dtype, device=device)
                          + s_t * (c_true_detached[0] - rev_t)]
            for j in range(K - 1):
                pp_u = float(pps_info[j].get('util', 0.0)) if j < len(pps_info) else 0.0
                terms_true.append(torch.tensor(pp_u, dtype=dtype, device=device)
                                  - self.beta_ivt * tau_hat[j + 1]
                                  + s_t * (c_true_detached[j + 1] - rev_t))
            terms_true_t = torch.stack(terms_true)
            tmax_true = torch.max(terms_true_t)
            sum_mnl_true = torch.exp(torch.clamp(tmax_true, -80.0, 80.0)) * torch.sum(
                torch.exp(terms_true_t - tmax_true))
            lambert_arg_true = sum_mnl_true / torch.tensor(np.e, dtype=dtype, device=device)
            if outside_util_f is not None:
                lambert_arg_true = lambert_arg_true / torch.exp(
                    torch.clamp(torch.tensor(outside_util_f, dtype=dtype, device=device),
                               -80.0, 80.0))
            lw_true = (self.lambertw_torch(lambert_arg_true) + 1.0) / s_t
            a_star = c_true_detached - rev_t - lw_true
            a_star = torch.clamp(a_star, self.min_p, self.max_p)

            choice_star = [torch.tensor(home_nonprice, dtype=dtype, device=device)
                           + s_t * a_star[0]]
            for j in range(K - 1):
                pp_u = float(pps_info[j].get('util', 0.0)) if j < len(pps_info) else 0.0
                choice_star.append(torch.tensor(pp_u, dtype=dtype, device=device)
                                   + s_t * a_star[j + 1])
            if outside_util_f is not None:
                choice_star.append(torch.tensor(outside_util_f, dtype=dtype, device=device))
            choice_star_t = torch.stack(choice_star)
            probs_star = torch.softmax(choice_star_t, dim=0)[:K]
            R_star = torch.dot(probs_star, rev_t + a_star - c_true_detached)

            regret = (R_star.detach() - R_pred) / (torch.abs(R_star.detach()) + 1.0)
            if torch.isfinite(regret):
                regrets.append(regret)

        if not regrets:
            return torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True)
        return torch.stack(regrets).mean()

    def self_supervised_update(self, feat, cap_feat, target, spo_weight=0.0, record_loss=True):
        """支持SPO+损失的更新方法 — split backward to reduce peak GPU memory"""
        self.optimizer.zero_grad()

        # Hoisted once, up front, instead of the previous `if 'probe_on' not in
        # locals()` idiom -- that idiom made probe_on's value depend on whether the
        # SPO branch below ran at all, which is fragile and no longer needed now
        # that the probe counters are isolated from the SPO try/except (2026-07-28).
        probe_on = getattr(self, 'spo_probe_interval', 0) > 0

        # Gradient-matched SPO+ path (GradNorm-style). When enabled, bypass the
        # legacy additive path entirely.
        if self.spo_grad_ratio > 0.0:
            return self._self_supervised_update_gradmatch(
                feat, cap_feat, target, spo_weight, record_loss, probe_on)

        # Try SPO+ first. If no finite SPO+ batch is available, keep the
        # supervised update at full weight instead of silently shrinking it.
        spo_loss = None
        effective_spo_weight = 0.0
        if spo_weight > 0.0 and len(self.spo_training_data) > 0:
            if probe_on:
                self.spo_probe_loss_attempts += 1
            try:
                if self.spo_profit_regret:
                    candidate_spo_loss = self._calculate_profit_regret_for_batch(
                        feat, cap_feat, target)
                else:
                    candidate_spo_loss = self._calculate_spo_loss_for_batch(
                        feat, cap_feat, None, target)
                if (
                    candidate_spo_loss is not None
                    and candidate_spo_loss.requires_grad
                    and torch.isfinite(candidate_spo_loss)
                ):
                    spo_loss = candidate_spo_loss
                    effective_spo_weight = spo_weight
                    if probe_on:
                        self.spo_probe_loss_ok += 1
                    (effective_spo_weight * spo_loss).backward()
            except Exception as e:
                self.spo_loss_failures += 1
                print(f"[SPO+ warning #{self.spo_loss_failures}] SPO+ loss computation failed(共{self.spo_loss_failures}次): {type(e).__name__}: {e}")
                spo_loss = None
                effective_spo_weight = 0.0

        # Huber uses a separate forward pass so this branch has its own graph.
        outputs, q_outputs = self._cnn_forward(feat, cap_feat)
        # keep the (N,1) target shape the original criterion call used -- a
        # squeezed (N,) target would change broadcasting semantics
        target_c = target[:, 0:1] if target.dim() > 1 else target
        huber_loss = self.criterion(outputs, target_c)
        # Pure-Huber tracker for the initial-phase convergence break (the
        # phase must converge on the supervised signal, not the combined
        # Huber + alpha*SPO value, which can go monotone when regret < 0).
        self.spo_last_pure_huber = float(huber_loss.item())
        # Service-position head (--use_q_head): same Huber on the q target
        # column, added at q_huber_weight. Active only when the dual head
        # exists AND the memory rows carry a q label (target_dim == 2).
        if q_outputs is not None and self.criterion_q is not None and target.dim() > 1:
            q_huber = self.criterion_q(q_outputs, target[:, 1:2])
            self.spo_last_pure_q = float(q_huber.item())
            huber_loss = huber_loss + self.q_huber_weight * q_huber

        if record_loss:
            self.spo_huber_loss_history.append(huber_loss.item())
            if len(self.spo_huber_loss_history) > 100:
                self.spo_huber_loss_history.pop(0)

        # Huber 权重恒为 1.0。此前是 (1 - effective_spo_weight),在 alpha > 1 时变负,
        # 等于训练网络去最大化预测误差 —— 需求 3/4 中被归因于"SPO+ 高 alpha 不稳定"的
        # 角点崩塌很可能源于此。语义改为加法式:loss = Huber + alpha * SPO。
        spo_gradnorm = self._global_grad_norm() if (probe_on and spo_loss is not None) else None

        huber_loss.backward()
        huber_gradnorm = self._global_grad_norm() if (probe_on and spo_loss is not None) else None

        # Stage 2: differentiable profit fine-tuning
        stage2_loss = None
        if self.dspos_stage2_weight > 0.0 and not self.initial_phase:
            stage2_loss = self._profit_regret_loss()
            if (stage2_loss is not None
                    and stage2_loss.requires_grad
                    and torch.isfinite(stage2_loss)):
                (self.dspos_stage2_weight * stage2_loss).backward()

        if probe_on:
            self.spo_probe_updates += 1
            if spo_loss is not None:
                # The legacy additive path never fed the per-probe lam/spo_gn/
                # huber_gn display (those were gradmatch-only). Populate them
                # here so the probe line shows true magnitudes: spo_gn is
                # captured right after the SPO backward (grad = SPO only);
                # huber_gn after the Huber backward -- combined grad, which
                # equals Huber's to within the (tiny) SPO share.
                self.spo_probe_last_spo_gn = float(spo_gradnorm or 0.0)
                self.spo_probe_last_huber_gn = float(huber_gradnorm or 0.0)
                self.spo_probe_last_lam = float(effective_spo_weight)
            if spo_gradnorm is not None:
                total = self._global_grad_norm()
                if total > 0.0:
                    self.spo_probe_gradnorm_ratios.append(spo_gradnorm / total)
            if self.spo_probe_updates % self.spo_probe_interval == 0:
                self._log_spo_probe()

        self.optimizer.step()

        total = huber_loss.item()
        if spo_loss is not None:
            total = total + effective_spo_weight * spo_loss.item()
        if stage2_loss is not None:
            total = total + self.dspos_stage2_weight * stage2_loss.item()
        return total

    def _self_supervised_update_gradmatch(self, feat, cap_feat, target,
                                          spo_weight, record_loss, probe_on):
        """Gradient-matched SPO+ update (GradNorm-style).

        替代固定 alpha 的 loss = Huber + alpha*SPO。改为：分别反传 Huber 与原始 SPO+，
        再把 SPO+ 梯度乘以 lambda，使 ||lambda * grad_SPO|| = spo_grad_ratio * ||grad_Huber||。
        于是 SPO+ 每步对总梯度贡献一个固定比例，与损失绝对尺度无关 —— 绕开
        "alpha 小则 SPO+ 梯度被 Huber 淹没 (gradnorm_ratio=0.0000)、alpha 大则角崩"的两难。
        """
        ratio = self.spo_grad_ratio

        # ---- Huber branch (own graph) ----
        outputs, q_outputs = self._cnn_forward(feat, cap_feat)
        target_c = target[:, 0:1] if target.dim() > 1 else target
        huber_loss = self.criterion(outputs, target_c)
        self.spo_last_pure_huber = float(huber_loss.item())
        if q_outputs is not None and self.criterion_q is not None and target.dim() > 1:
            q_huber = self.criterion_q(q_outputs, target[:, 1:2])
            self.spo_last_pure_q = float(q_huber.item())
            huber_loss = huber_loss + self.q_huber_weight * q_huber
        if record_loss:
            self.spo_huber_loss_history.append(huber_loss.item())
            if len(self.spo_huber_loss_history) > 100:
                self.spo_huber_loss_history.pop(0)
        huber_loss.backward()
        huber_gradnorm = self._global_grad_norm()

        # snapshot grad_Huber before the SPO+ backward clobbers .grad
        net_params = list(self.supervised_ml.parameters())
        huber_grads = {}
        for p in net_params:
            if p.grad is not None:
                huber_grads[id(p)] = p.grad.detach().clone()

        # ---- SPO+ branch (raw, unscaled by alpha) ----
        spo_loss = None
        spo_gradnorm = 0.0
        lam = 0.0
        spo_active = (spo_weight > 0.0 and len(self.spo_training_data) > 0
                      and huber_gradnorm > 1e-12)
        if probe_on and spo_active:
            self.spo_probe_loss_attempts += 1
        if spo_active:
            self.optimizer.zero_grad()
            try:
                if self.spo_profit_regret:
                    candidate_spo_loss = self._calculate_profit_regret_for_batch(
                        feat, cap_feat, target)
                else:
                    candidate_spo_loss = self._calculate_spo_loss_for_batch(
                        feat, cap_feat, None, target)
                if (candidate_spo_loss is not None
                        and candidate_spo_loss.requires_grad
                        and torch.isfinite(candidate_spo_loss)):
                    candidate_spo_loss.backward()
                    spo_gradnorm = self._global_grad_norm()
                    if probe_on and spo_gradnorm > 1e-12:
                        self.spo_probe_loss_ok += 1
                    spo_loss = candidate_spo_loss
                    lam = (ratio * huber_gradnorm / spo_gradnorm
                           if spo_gradnorm > 1e-12 else 0.0)
            except Exception as e:
                self.spo_loss_failures += 1
                print(f"[SPO+ warning #{self.spo_loss_failures}] SPO+ loss computation failed(共{self.spo_loss_failures}次): {type(e).__name__}: {e}")
                spo_loss = None
                lam = 0.0

        # combine: grad = grad_Huber + lambda * grad_SPO  (per-param, robust to
        # either side being absent -- in practice both forward passes touch the
        # whole CNN, so every param has both)
        for p in net_params:
            hg = huber_grads.get(id(p))
            sg = p.grad  # grad_SPO if SPO touched this param, else None
            if hg is None and sg is None:
                continue
            if lam == 0.0 or sg is None:
                p.grad = hg.clone() if hg is not None else None
            elif hg is None:
                p.grad = lam * sg.detach()
            else:
                p.grad = hg + lam * sg.detach()

        # Realized SPO+ share of the combined gradient, measured BEFORE the
        # clip -- after clip_grad_norm_(1.0) the global norm is ~1.0 by
        # construction, so the pre-2026-08-05 measurement divided an unclipped
        # numerator (lam*spo_gradnorm) by the clipped norm and printed ratios
        # of ~1e4-1e5 that were meaningless.
        total_preclip = self._global_grad_norm() if (
            probe_on and spo_loss is not None and lam > 0.0) else None

        self._clip_grads(1.0)

        if probe_on:
            self.spo_probe_updates += 1
            self.spo_probe_last_lam = float(lam)
            self.spo_probe_last_spo_gn = float(spo_gradnorm)
            self.spo_probe_last_huber_gn = float(huber_gradnorm)
            if total_preclip is not None:
                realized = (lam * spo_gradnorm) / total_preclip if total_preclip > 1e-12 else 0.0
                self.spo_probe_gradnorm_ratios.append(realized)
            if self.spo_probe_updates % self.spo_probe_interval == 0:
                self._log_spo_probe()

        self.optimizer.step()

        if spo_loss is not None:
            return huber_loss.item() + lam * spo_loss.item()
        return huber_loss.item()

    @staticmethod
    def _lifted_spo_term(y_true, y_hat, w_true, w_pseudo):
        """Lifted SPO+ per-sample term.

        <y_true, w_pseudo - w_true> + 2 * <y_hat, w_true - w_pseudo>

        No (K+1) normalisation: dividing here made alpha's scale depend on the
        number of candidate points (~12), so a nominal alpha of 0.7 carried an
        effective gradient weight of only ~6%.
        """
        return (torch.dot(y_true, w_pseudo - w_true)
                + 2.0 * torch.dot(y_hat, w_true - w_pseudo))

    def _sample_spo_batch(self, recent_pool, n):
        """T1 (2026-08-06): draw n sample keys from the recent pool.

        With spo_regret_weighted the draw is proportional to the stored
        deployed-regret weights (floor 1e-6 keeps every sample reachable);
        otherwise uniform over the pool -- bit-for-bit the historical path."""
        if len(recent_pool) <= n:
            return list(recent_pool)
        if self.spo_regret_weighted:
            ws = np.asarray(
                [max(0.0, float(self.spo_training_data[k].get('regret_w', 0.0)))
                 for k in recent_pool], dtype=np.float64)
            if float(ws.sum()) > 0.0 and float(ws.max()) > 1e-6:
                p = ws / float(ws.sum()) + 1e-6
                p = p / p.sum()
                return list(np.random.choice(recent_pool, size=n, replace=False, p=p))
        return list(np.random.choice(recent_pool, size=n, replace=False))

    def _calculate_spo_loss_for_batch(self, feat, cap_feat, predicted_costs, target):
        """
        Compute lifted SPO+ loss with batched CNN forward passes for efficiency.

        Phase 1: Collect all augmented features (base/home/pp) across samples
        Phase 2: Single batched CNN forward pass
        Phase 3: Split results, construct differentiable c_hat, compute lifted SPO+ loss

        Reduces ~384 individual CNN calls to 1 batched call.
        """
        if len(self.spo_training_data) == 0:
            return None

        device = self.device
        dtype = torch.float32

        # ---- Phase 1: Collect all augmented features ----
        available_indices = sorted(self.spo_training_data.keys(), reverse=True)
        max_samples = min(max(1, self.spo_batch_size), len(available_indices))
        recent_pool = available_indices[:min(len(available_indices), max_samples * 4)]
        available_indices = self._sample_spo_batch(recent_pool, max_samples)
        mltplr = self.cost_multiplier

        all_feats = []
        all_caps = []
        sample_meta = []

        for memory_idx in available_indices:
            td = self.spo_training_data.get(memory_idx, None)
            if td is None:
                continue

            feature_info = td.get('feature_info', None)
            if feature_info is None:
                continue

            stored_true_costs = td.get('true_costs', [])
            if len(stored_true_costs) == 0:
                continue

            base_feat = feature_info['base_feat']
            home_cell = feature_info['home_cell']
            pp_cells = feature_info['pp_cells']
            time_int = feature_info['time_int']
            pp_caps = feature_info.get('pp_caps', [1000000.0] * len(pp_cells))

            # Boundary experiments: offer_mode='ooh_only' drops the home option
            # from the SPO+ choice set (the same restricted set serving uses);
            # stored arrays still carry home at index 0, so K excludes it and
            # the c_true slice in Phase 3 shifts by one.
            _incl_home = getattr(getattr(self, 'config', None), 'offer_mode', 'all') != 'ooh_only'
            K = len(pp_cells) + (1 if _incl_home else 0)
            if K < 1 or len(stored_true_costs) < (K if _incl_home else K + 1):
                continue

            # base (no augmentation)
            all_feats.append(base_feat)
            all_caps.append(1000000.0)

            # home augmentation
            if _incl_home:
                home_aug = base_feat.clone()
                home_aug[0][time_int][home_cell[0]][home_cell[1]] += 1
                all_feats.append(home_aug)
                all_caps.append(1000000.0)

            # pp augmentations
            for j in range(len(pp_cells)):
                pc = pp_cells[j]
                pp_aug = base_feat.clone()
                pp_aug[0][time_int][pc[0]][pc[1]] += 1
                pp_cap_val = float(pp_caps[j]) if j < len(pp_caps) else 1000000.0
                all_feats.append(pp_aug)
                all_caps.append(pp_cap_val)

            sample_meta.append({
                'K': K,
                'n_feats': K + 1,  # base + home + (K-1) pps (home absent when ooh_only)
                'stored_true_costs': stored_true_costs,
                'theta_val': feature_info['theta'],
                'ins_cost_home': feature_info['ins_cost_home'],
                'ins_cost_pps': feature_info['ins_cost_pps'],
                'service_time': feature_info['service_time'],
                'customer_info': td.get('customer_info', {
                    'base_util': self.base_util, 'home_util': 3.75,
                    'incentive_sensitivity': 0.99, 'revenue': self.revenue
                }),
                'pps_info': td.get('pps_info', []),
                '_incl_home': _incl_home,
            })

        if len(sample_meta) == 0:
            return None

        # ---- Phase 2: Single batched CNN forward pass ----
        batch_feats = torch.cat(all_feats, dim=0).to(device)
        batch_caps = torch.tensor(all_caps, dtype=dtype, device=device)
        cnn_outputs, q_outputs = self._cnn_forward(batch_feats, batch_caps)
        cnn_outputs = cnn_outputs.squeeze(-1)  # (N_total,)
        if q_outputs is not None:
            q_outputs = q_outputs.squeeze(-1)  # (N_total,) same feat order

        # ---- Phase 3: Split results per sample, compute SPO+ loss ----
        spo_losses = []
        feat_offset = 0

        for meta in sample_meta:
            K = meta['K']
            n_feats = meta['n_feats']

            # Slice CNN outputs for this sample (all differentiable w.r.t. CNN params)
            base_out = cnn_outputs[feat_offset]
            _incl_home = meta['_incl_home']
            # Displayed in-vehicle times from the q head (differentiable). The
            # lifted SPO+ oracles are detached, so tau here changes the oracle
            # DECISION only (behavioral consistency with serving); the q-head
            # decision gradient flows through the profit-regret paths instead.
            # Feat order is [base, home, pp_0, ...] when home is offered and
            # [base, pp_0, ...] when it is not (ooh_only).
            q_slice = q_outputs[feat_offset:feat_offset + n_feats] if q_outputs is not None else None
            if _incl_home:
                tau_hat = self._tau_hat_from_q(q_slice, K)
            else:
                tau_hat = torch.zeros(K, dtype=dtype, device=device)
                if q_slice is not None:
                    _speed = max(float(self.vehicle_speed), 1e-8)
                    for _j in range(K):
                        tau_hat[_j] = q_slice[1 + _j] / _speed

            theta_val = meta['theta_val']
            c_hat = torch.zeros(K, dtype=dtype, device=device)
            # Serving alignment probe (2026-08-05): with spo_pure_cnn_serving the
            # SPO+ objective's c_hat is reconstructed as the pure-CNN marginal
            # (CNN(candidate) - CNN(base)), the same quantity get_action_pricing
            # uses for serving -- decision-oriented training on the deployed cost.
            pure_cnn = getattr(self.config, 'spo_pure_cnn_serving', False)
            if _incl_home:
                home_out = cnn_outputs[feat_offset + 1]
                if pure_cnn:
                    c_hat[0] = home_out - base_out
                else:
                    c_hat[0] = self._predicted_option_cost(
                        self.l0_home + meta['service_time'], mltplr, theta_val,
                        meta['ins_cost_home'], home_out, base_out)

            for j in range(K - (1 if _incl_home else 0)):
                pp_out = cnn_outputs[feat_offset + (2 if _incl_home else 1) + j]
                _dst = j + (1 if _incl_home else 0)
                if pure_cnn:
                    c_hat[_dst] = pp_out - base_out
                else:
                    c_hat[_dst] = self._predicted_option_cost(
                        self.l_mp, mltplr, theta_val,
                        meta['ins_cost_pps'][j], pp_out, base_out)

            # stored arrays carry home at index 0 even for ooh_only; drop it
            _c_true_off = 0 if _incl_home else 1
            c_true = torch.tensor(
                np.asarray(meta['stored_true_costs'][_c_true_off:_c_true_off + K], dtype=np.float32),
                dtype=dtype, device=device, requires_grad=False)

            if not torch.isfinite(c_hat).all() or not torch.isfinite(c_true).all():
                feat_offset += n_feats
                continue
            if c_hat.numel() != c_true.numel() or c_hat.numel() <= 0:
                feat_offset += n_feats
                continue

            customer_info = meta['customer_info']
            pps_info = meta['pps_info']

            taus_list = [float(x) for x in tau_hat.detach().tolist()]
            true_oracle = self._resolve_pricing_oracle(
                c_true.detach(), customer_info, pps_info, taus=taus_list,
                include_home=_incl_home)
            if true_oracle is None:
                feat_offset += n_feats
                continue

            pseudo_costs_t = (2.0 * c_hat - c_true).detach()
            pseudo_costs_t = self._clip_pseudo_costs(pseudo_costs_t, c_true, self.spo_pseudo_clip_spreads)
            pseudo_oracle = self._resolve_pricing_oracle(
                pseudo_costs_t, customer_info, pps_info, taus=taus_list,
                include_home=_incl_home)
            if pseudo_oracle is None:
                feat_offset += n_feats
                continue

            if true_oracle['lifted'].numel() != K + 1 or pseudo_oracle['lifted'].numel() != K + 1:
                feat_offset += n_feats
                continue

            y_true = torch.cat([
                c_true,
                torch.ones(1, dtype=dtype, device=device)
            ])
            y_hat = torch.cat([
                c_hat,
                torch.ones(1, dtype=dtype, device=device)
            ])
            w_true = true_oracle['lifted'].to(device=device, dtype=dtype)
            w_pseudo = pseudo_oracle['lifted'].to(device=device, dtype=dtype)

            # Isolated in its own try/except (2026-07-28): this block sits inside
            # the same call that self_supervised_update wraps in a try/except for
            # the SPO+ loss itself, so a probe-only exception here used to silently
            # zero out SPO+ for the whole update and count as a spo_loss_failure.
            # Instrumentation must never be able to affect training.
            if getattr(self, 'spo_probe_interval', 0) > 0:
                try:
                    lo = torch.min(c_true)
                    hi = torch.max(c_true)
                    oor = ((pseudo_costs_t < lo) | (pseudo_costs_t > hi)).sum()
                    self.spo_probe_pseudo_oor += int(oor)
                    self.spo_probe_pseudo_total += int(pseudo_costs_t.numel())
                    for src, key in ((true_oracle, 'true'), (pseudo_oracle, 'pseudo')):
                        p = src['prices'].detach()
                        at_bound = ((p <= self.min_p + 1e-6) | (p >= self.max_p - 1e-6)).sum()
                        if key == 'true':
                            self.spo_probe_corner_true += int(at_bound)
                            self.spo_probe_price_total += int(p.numel())
                        else:
                            self.spo_probe_corner_pseudo += int(at_bound)
                except Exception:
                    pass

            spo_loss_i = self._lifted_spo_term(y_true, y_hat, w_true, w_pseudo)

            if torch.isfinite(spo_loss_i):
                spo_losses.append(spo_loss_i)

            feat_offset += n_feats

        if len(spo_losses) == 0:
            return None

        return torch.stack(spo_losses).mean()

    def _calculate_profit_regret_for_batch(self, feat, cap_feat, target):
        """Batched profit-regret SPO+ objective (spo_profit_regret=True).

        Same batched feature-collection skeleton as _calculate_spo_loss_for_batch,
        but the Phase-3 objective is the expected-net-profit decision regret

            regret = (R(a*; c_true) - R(a(chat); c_true)) / (|R(a*; c_true)| + 1)

        with R(a) = sum_i p_i(a) * (rev + a_i - c_i) over the K service options
        (the outside option's probability mass is implicitly quit -> 0 profit).
        a(chat) is the differentiable Lambert-W price (STE clip at the box), so
        gradient flows chat -> a -> R_pred; a*(c_true) is the detached oracle
        from the same pricing oracle that serves online.

        Unlike the lifted cost-SPO+ objective (y=[c;1], minimize predicted cost),
        this loss is exactly what the offline decision-regret evaluation measures
        -- SPO+ optimizes the deployed metric instead of cost alone, and quitting
        a customer who would have been profitable is penalized (quit mass lowers
        R_pred, widening the regret).
        """
        if len(self.spo_training_data) == 0:
            return None

        device = self.device
        dtype = torch.float32
        mltplr = self.cost_multiplier
        pure_cnn = getattr(self.config, 'spo_pure_cnn_serving', False)
        outside_util = getattr(self.config, 'outside_option_util', None)
        outside_util_f = float(outside_util) if outside_util is not None else None

        # ---- Phase 1: collect augmented features + per-sample meta ----
        available_indices = sorted(self.spo_training_data.keys(), reverse=True)
        max_samples = min(max(1, self.spo_batch_size), len(available_indices))
        recent_pool = available_indices[:min(len(available_indices), max_samples * 4)]
        available_indices = self._sample_spo_batch(recent_pool, max_samples)

        all_feats = []
        all_caps = []
        sample_meta = []

        for memory_idx in available_indices:
            td = self.spo_training_data.get(memory_idx, None)
            if td is None:
                continue
            feature_info = td.get('feature_info', None)
            if feature_info is None:
                continue
            stored_true_costs = td.get('true_costs', [])
            if len(stored_true_costs) == 0:
                continue

            base_feat = feature_info['base_feat']
            home_cell = feature_info['home_cell']
            pp_cells = feature_info['pp_cells']
            time_int = feature_info['time_int']
            pp_caps = feature_info.get('pp_caps', [1000000.0] * len(pp_cells))
            K = 1 + len(pp_cells)
            if K <= 1 or len(stored_true_costs) < K:
                continue

            all_feats.append(base_feat)
            all_caps.append(1000000.0)
            home_aug = base_feat.clone()
            home_aug[0][time_int][home_cell[0]][home_cell[1]] += 1
            all_feats.append(home_aug)
            all_caps.append(1000000.0)
            for j in range(len(pp_cells)):
                pc = pp_cells[j]
                pp_aug = base_feat.clone()
                pp_aug[0][time_int][pc[0]][pc[1]] += 1
                all_feats.append(pp_aug)
                all_caps.append(float(pp_caps[j]) if j < len(pp_caps) else 1000000.0)

            sample_meta.append({
                'K': K, 'n_feats': K + 1,
                'stored_true_costs': stored_true_costs,
                'theta_val': feature_info['theta'],
                'ins_cost_home': feature_info['ins_cost_home'],
                'ins_cost_pps': feature_info['ins_cost_pps'],
                'service_time': feature_info['service_time'],
                'customer_info': td.get('customer_info', {
                    'base_util': self.base_util, 'home_util': 3.75,
                    'incentive_sensitivity': 0.99, 'revenue': self.revenue}),
                'pps_info': td.get('pps_info', []),
            })

        if len(sample_meta) == 0:
            return None

        # ---- Phase 2: single batched CNN forward ----
        batch_feats = torch.cat(all_feats, dim=0).to(device)
        batch_caps = torch.tensor(all_caps, dtype=dtype, device=device)
        cnn_outputs, q_outputs = self._cnn_forward(batch_feats, batch_caps)
        cnn_outputs = cnn_outputs.squeeze(-1)
        if q_outputs is not None:
            q_outputs = q_outputs.squeeze(-1)  # (N_total,) same feat order

        # ---- Phase 3: per-sample profit regret (differentiable in c_hat) ----
        regrets = []
        feat_offset = 0
        for meta in sample_meta:
            K = meta['K']
            n_feats = meta['n_feats']

            base_out = cnn_outputs[feat_offset]
            home_out = cnn_outputs[feat_offset + 1]
            # Displayed in-vehicle times from the q head (differentiable): the
            # passenger utility -beta_ivt * tau_hat enters the Lambert-W pricing
            # and the choice probabilities, so this path gives the q head its
            # decision gradient (the lifted cost-SPO+ path cannot -- its oracles
            # are detached).
            q_slice = q_outputs[feat_offset:feat_offset + n_feats] if q_outputs is not None else None
            tau_hat = self._tau_hat_from_q(q_slice, K)

            c_hat = torch.zeros(K, dtype=dtype, device=device)
            if pure_cnn:
                c_hat[0] = home_out - base_out
            else:
                c_hat[0] = self._predicted_option_cost(
                    self.l0_home + meta['service_time'], mltplr, meta['theta_val'],
                    meta['ins_cost_home'], home_out, base_out)
            for j in range(K - 1):
                pp_out = cnn_outputs[feat_offset + 2 + j]
                if pure_cnn:
                    c_hat[j + 1] = pp_out - base_out
                else:
                    c_hat[j + 1] = self._predicted_option_cost(
                        self.l_mp, mltplr, meta['theta_val'],
                        meta['ins_cost_pps'][j], pp_out, base_out)

            c_true = torch.tensor(
                np.asarray(meta['stored_true_costs'][:K], dtype=np.float32),
                dtype=dtype, device=device, requires_grad=False)
            feat_offset += n_feats

            if not torch.isfinite(c_hat).all() or not torch.isfinite(c_true).all():
                continue
            if c_hat.numel() != c_true.numel() or c_hat.numel() <= 0:
                continue

            customer_info = meta['customer_info']
            pps_info = meta['pps_info']
            s_val = float(customer_info.get('incentive_sensitivity', 1.0))
            rev_val = float(customer_info.get('revenue', self.revenue))
            if (not np.isfinite(s_val)) or abs(s_val) < 1e-8:
                continue
            base_util = float(customer_info.get('base_util', self.base_util))
            home_util = float(customer_info.get('home_util', 0.0))
            home_nonprice = base_util + home_util
            s_t = torch.tensor(s_val, dtype=dtype, device=device)
            rev_t = torch.tensor(rev_val, dtype=dtype, device=device)
            pp_utils = [float(pps_info[j].get('util', 0.0)) if j < len(pps_info) else 0.0
                        for j in range(K - 1)]

            # ---- differentiable predicted prices a(chat): Lambert-W + STE clip ----
            util_terms = [torch.tensor(home_nonprice, dtype=dtype, device=device)
                          + s_t * (c_hat[0] - rev_t)]
            util_terms += [torch.tensor(pp_utils[j], dtype=dtype, device=device)
                           - self.beta_ivt * tau_hat[j + 1]
                           + s_t * (c_hat[j + 1] - rev_t) for j in range(K - 1)]
            terms_t = torch.stack(util_terms)
            tmax = torch.max(terms_t)
            sum_mnl = torch.exp(torch.clamp(tmax, -80.0, 80.0)) * torch.sum(
                torch.exp(terms_t - tmax))
            if (not torch.isfinite(sum_mnl)) or float(sum_mnl.detach().cpu()) <= 0.0:
                continue
            lambert_arg = sum_mnl / torch.tensor(np.e, dtype=dtype, device=device)
            if outside_util_f is not None:
                lambert_arg = lambert_arg / torch.exp(torch.clamp(
                    torch.tensor(outside_util_f, dtype=dtype, device=device),
                    -80.0, 80.0))
            lw = (self.lambertw_torch(lambert_arg) + 1.0) / s_t
            if not torch.isfinite(lw):
                continue
            if self.spo_w0_band:
                band = self._w0_band(c_hat.detach().cpu().numpy(), rev_val,
                                     self.min_p, self.max_p)
                if band is not None:
                    lw = torch.clamp(lw, float(band[0]), float(band[1]))
            a_raw = c_hat - rev_t - lw
            a_clipped = torch.clamp(a_raw, float(self.min_p), float(self.max_p))
            # STE with corner masking: forward uses the clipped price (box
            # constraints); backward passes through the unclipped price ONLY
            # for options whose unconstrained price is inside the box. At a
            # box boundary the decision is made by the constraint, not by the
            # prediction -- attributing regret to c_hat there yields a
            # constant, c_hat-independent one-sided STE push that drifts the
            # CNN without bound (measured 2026-08-05: c_hat -> 6.8e5, all
            # prices at the box, Huber loss 93k -> 2.75M over epochs 10-19).
            in_box = ((a_raw >= float(self.min_p))
                      & (a_raw <= float(self.max_p))).to(dtype)
            a_pred = a_clipped.detach() + (a_raw - a_raw.detach()) * in_box

            # ---- MNL probs under a_pred (quit mass in the outside option) ----
            choice_terms = [torch.tensor(home_nonprice, dtype=dtype, device=device)
                            + s_t * a_pred[0]]
            choice_terms += [torch.tensor(pp_utils[j], dtype=dtype, device=device)
                             - self.beta_ivt * tau_hat[j + 1]
                             + s_t * a_pred[j + 1] for j in range(K - 1)]
            if outside_util_f is not None:
                choice_terms.append(torch.tensor(
                    outside_util_f, dtype=dtype, device=device))
            probs = torch.softmax(torch.stack(choice_terms), dim=0)[:K]
            R_pred = torch.dot(probs, rev_t + a_pred - c_true)

            # ---- oracle a*(c_true), detached, via the oracle serving uses ----
            taus_list = [float(x) for x in tau_hat.detach().tolist()]
            true_oracle = self._resolve_pricing_oracle(
                c_true.detach(), customer_info, pps_info, taus=taus_list)
            if true_oracle is None:
                continue
            p_star = true_oracle['prices'].to(device=device, dtype=dtype)
            probs_star = true_oracle['probs'].to(device=device, dtype=dtype)
            R_star = torch.dot(probs_star, rev_t + p_star - c_true.detach())

            regret = (R_star.detach() - R_pred) / (torch.abs(R_star.detach()) + 1.0)
            if torch.isfinite(regret):
                regrets.append(regret)

        if len(regrets) == 0:
            return None
        return torch.stack(regrets).mean()

    def _get_current_spo_weight(self):
        """计算当前SPO+损失权重"""
        if len(self.spo_training_data) == 0:
            return 0.0

        # Anchor the warmup/rampup schedule to the first labeled episode, not episode 0 --
        # otherwise the ~112 initial-phase episodes (before any labels exist) silently burn
        # through warmup+rampup and the weight jumps straight to max the moment data appears.
        anchor = self.spo_first_data_episode if self.spo_first_data_episode is not None else self.spo_current_episode_count
        effective_episode = self.spo_current_episode_count - anchor

        if effective_episode < self.spo_warmup_episodes:
            return 0.0

        rampup_progress = (effective_episode - self.spo_warmup_episodes) / self.spo_rampup_episodes if self.spo_rampup_episodes > 0 else 1.0
        rampup_progress = min(max(rampup_progress, 0.0), 1.0)
        return self.max_spo_loss_weight * rampup_progress

    def _global_grad_norm(self):
        total = 0.0
        for p in self.supervised_ml.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum())
        return total ** 0.5

    def _clip_grads(self, max_norm=1.0):
        params = [p for g in self.optimizer.param_groups for p in g['params']]
        torch.nn.utils.clip_grad_norm_(params, max_norm)

    def _log_spo_probe(self):
        def _rate(num, den):
            return (num / den) if den else 0.0
        ratios = self.spo_probe_gradnorm_ratios
        _msg = ('[SPO+ probe] updates={} spo_ok={}/{} pseudo_oor={:.4f} corner_pseudo={:.4f} '
                'corner_true={:.4f} gradnorm_ratio={:.4f} | lam={:.4g} '
                'spo_gn={:.4g} huber_gn={:.4g}\n'.format(
                    self.spo_probe_updates,
                    self.spo_probe_loss_ok,
                    self.spo_probe_loss_attempts,
                    _rate(self.spo_probe_pseudo_oor, self.spo_probe_pseudo_total),
                    _rate(self.spo_probe_corner_pseudo, self.spo_probe_price_total),
                    _rate(self.spo_probe_corner_true, self.spo_probe_price_total),
                    (sum(ratios) / len(ratios)) if ratios else 0.0,
                    self.spo_probe_last_lam,
                    self.spo_probe_last_spo_gn,
                    self.spo_probe_last_huber_gn))
        # reliable side-channel: the runner-side Logger file can lose buffered
        # lines under multiprocessing, so mirror every probe line to eo_probe_diag.log
        print(_msg, end='', flush=True)
        try:
            with open(__import__('os').environ.get('DIAG_FILE', 'eo_probe_diag.log'), 'a') as _f:
                _f.write(_msg)
        except OSError:
            pass
