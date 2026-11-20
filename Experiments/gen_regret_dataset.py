"""Generate the shared held-out regret dataset (spec C1).

StaticPricing rollout (home +2.0 / pp -5.0) with the terminal-route counterfactual
label machinery of DSPO_plus_SPO. No model is trained: optimize() and
self_supervised_update() are stubbed to no-ops (the collector's CNN weights never
matter -- offline evaluation re-predicts from checkpoints). Episodes 1-2 are a
warm-up that only serves to end the initial phase (buffer_size=90; with
~85 non-quitting customers/episode a single episode isn't enough to cross the
buffer threshold, so the initial phase spans two episodes).
"""
import sys
from os import path

import numpy as np

ROOT = path.join(path.dirname(path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)

from Src.parser import Parser                        # noqa: E402
from Src.config import Config                        # noqa: E402
from run import Solver                               # noqa: E402
from Experiments.run_paper_repro import build_argv   # noqa: E402
from Experiments.run_req2_profit import req2_extra   # noqa: E402
from Experiments.run_req3_spo import SCALER, V0      # noqa: E402
from Experiments.regret_lib import save_dataset      # noqa: E402

SEED = 9999
EPISODES = 20
STATIC_HOME = 2.0
STATIC_PP = -5.0
OUT = path.join(ROOT, 'Experiments', 'Parcelpoint_py',
                'regret_dataset_RC_seed{}.pt'.format(SEED))


def main():
    extra = dict(req2_extra(V0))
    extra.update({
        'algo_name': 'DSPO_plus_SPO',
        'spo_loss_weight': '0.7',      # >0 so label collection is enabled
        'spo_buffer_size': '100000',   # keep every labelled sample
        'spo_label_sample_size': '0',  # label every priced customer
        'max_price': '10.0',
        'save_model': 'False',
        'folder_suffix': 'regretds_{}'.format(SEED),
    })
    argv = build_argv('DSPO', SEED, SCALER, EPISODES + 2, 90, extra=extra)
    args = Parser().get_parser().parse_args(argv)
    config = Config(args)
    solver = Solver(config)
    agent = solver.model

    # --- no training (spec C1): stub both training entry points ---
    agent.optimize = lambda *a, **k: 0.0
    agent.self_supervised_update = lambda *a, **k: 0.0

    # --- static pricing: parent call records the decision, then charge fixed prices ---
    orig_pricing = agent.get_action_pricing

    def static_pricing(state, training):
        a = np.asarray(orig_pricing(state, training), dtype=np.float64)
        a_static = np.full_like(a, STATIC_PP)
        a_static[0] = STATIC_HOME
        decs = agent.spo_current_episode_data['pricing_decisions']
        if decs:  # keep the recorded decision consistent with what is charged
            decs[-1]['prices'] = a_static.copy()
        return a_static

    # NOTE: DSPO.__init__ does `self.get_action = self.get_action_pricing` -- a
    # pre-bound method object -- so run.Solver.train()'s `model.get_action(...)`
    # calls would bypass an override of `get_action_pricing` alone. Patch the
    # `get_action` attribute itself so the training loop actually invokes ours.
    agent.get_action = static_pricing

    solver.train()

    entries = list(agent.spo_training_data.values())
    assert len(entries) > 0, 'no labelled samples collected'
    assert agent.spo_label_failures == 0, \
        '{} label failures during collection'.format(agent.spo_label_failures)
    # standing gate: prove the static-pricing override actually charged the
    # fixed prices (home +2.0 / pp -5.0), not the untrained agent's own prices.
    for e in entries:
        prices = np.asarray(e['prices'], dtype=np.float64)
        assert abs(prices[0] - STATIC_HOME) < 1e-6, \
            'home price not static: {}'.format(prices)
        assert np.allclose(prices[1:], STATIC_PP, atol=1e-6), \
            'pp prices not static: {}'.format(prices)
    meta = {'instance': 'RC', 'mnl_dist_scaler': float(SCALER),
            'outside_option_util': float(V0), 'min_price': -10.0, 'max_price': 10.0,
            'home_failure': 0.0, 'seed': SEED, 'episodes': EPISODES,
            'policy': 'static_home{}_pp{}'.format(STATIC_HOME, STATIC_PP)}
    save_dataset(OUT, meta, entries)
    print('saved {} samples -> {}'.format(len(entries), OUT))


if __name__ == '__main__':
    main()
