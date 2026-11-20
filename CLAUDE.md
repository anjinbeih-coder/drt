# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for "Learning Dynamic Selection and Pricing of Out-of-Home Deliveries" (Akkerman, Dieter, Mes — Transportation Science 2024, https://doi.org/10.1287/trsc.2023.0434, preprint: https://arxiv.org/abs/2311.13983). It simulates a parcel carrier that, for each arriving customer, offers a set of out-of-home (OOH) parcel points and/or prices, routes accepted deliveries with a CVRP solver (HGS), and trains a policy (DSPO) to make that offering/pricing decision using a CNN cost predictor. The README states the code was reimplemented from scratch after the original was lost, so bugs/deviations from the paper are expected — see "Known gaps" below before trusting any result.

This working copy is also mid-refactor for a follow-up thesis chapter (DRT/commuting variant with a "lifted SPO+" learning objective and a new Beijing-Yanjiao commuting dataset). `paper/` holds only external reference papers (Akkerman et al., the SPO/SPO+ paper, the Yang et al. benchmark paper). Everything else the project has generated about itself — the cost-alignment audit, the architecture write-up, the in-progress manuscript, and loose Chinese planning/prompt notes — lives under `docs/`. Treat everything in `docs/` and the loose `*.md`/`*.txt` planning notes in the repo root as research notes, not implementation spec — the actual code is often behind or ahead of them. See `docs/架构与算法流程说明.md` for a detailed (Chinese) walkthrough of the algorithm flow and class architecture.

## Commands

```bash
# setup (root folder must literally be named `ooh_code`, see Known gaps)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
python -m pip install -r requirements.txt

# run DSPO or Baseline (only algorithms currently wired end-to-end, see Known gaps)
python run.py

# run PPO — currently non-functional, see Known gaps
python run_ppo.py
```

Hyperparameters and environment settings are CLI args defined in `Src/parser.py` (not a config file) — edit its defaults or pass `--flag value` on the command line, e.g. `python run.py --instance RC --algo_name DSPO --pricing True`.

There is no test suite, linter, or CI config in this repo — don't invent commands for these.

`hygese` (the HGS CVRP solver binding) throws an assertion error on negative coordinates when `--load_data=True`; this requires manually commenting out that check inside the installed `hygese.py` package source (not in this repo) — `Parcelpoint_py.py` prints a reminder of this at env construction time.

## Architecture

**Entry point → Config → dynamic dispatch.** `run.py`/`run_ppo.py` parse args via `Src/parser.py`, then build `Src/config.py::Config`, which: loads instance data (`Utils.load_demand_data`), constructs a train env and a separate `test_env` from the same `env_name` arg, and dynamically imports the algorithm class named by `--algo_name`.

That dynamic import (`Utils.dynamic_load` in `Src/Utils/Utils.py`) recursively searches `Src/Algorithms/` (and `Environments/`) for a **file whose name contains** the requested string, then does `getattr(module, name)` to fetch a same-named class. It also requires the absolute path to contain a segment literally called `ooh_code` (it slices the path at that point to build the dotted import path) — if you check this out under a different folder name, imports break.

**Environment** (`Environments/OOH/Parcelpoint_py.py`, class `Parcelpoint_py`): a Gymnasium-style env (`reset`/`step`) simulating one day's customer arrivals. Each `step` draws/loads the next customer, asks the customer-choice model (`customerchoice.py`, MNL + Gumbel noise) to pick home delivery vs. an offered/priced OOH point given the policy's `action`, inserts the chosen location into an in-memory route via cheapest insertion (`env_utils.py::utils_env`), and periodically re-optimizes with HGS. Episode length is drawn from a negative binomial (`max_steps_r`, `max_steps_p`). `containers.py` holds the plain dataclasses (`Location`, `Customer`, `Vehicle`, `Fleet`, `ParcelPoint(s)`) passed around as state.

**Algorithms** (`Src/Algorithms/`), all subclassing `Agent` (shared save/step/gradient-clearing boilerplate):
- `Baseline.py` — fixed-price/nearest-k heuristic, no learning.
- `DSPO.py` — the paper's contribution: a CNN (`Src/Utils/Predictors.py::CNN_2d`, or `CNN_3d`/`LinReg` variants) predicts marginal routing cost per candidate location from a gridded feature map of the current route state (`grid_dim x grid_dim x n_input_layers`), blended with a cheapest-insertion heuristic via a cooling `theta` weight. Pricing decisions solve a closed-form Lambert-W optimum over the MNL choice model. Training labels come from a post-episode HGS re-solve, half-edge-partitioned across visited stops, stored in `MemoryBuffer` (in `Utils.py`).
- `DSPO_plus_SPO.py` (`DSPO_plus_SPO` class) — subclasses `DSPO` to add a "lifted SPO+" (Smart Predict-then-Optimize) training loss on top of the Huber loss, with warmup/rampup episode scheduling. **Currently broken — see Known gaps.**

`Src/Utils/Actor.py`, `Basis.py`, `Critic.py` implement a Fourier-basis actor/critic for a Gaussian PPO policy but are **not imported anywhere** in the current tree — they're orphaned remnants of the PPO implementation described in the README, which is otherwise absent (see below).

## Known gaps (verify before relying on these)

The repo is actively being edited and several things the README/parser/paper docs describe do not currently exist or do not run:

- **`--algo_name` choices `Heuristic`, `PPO`, `SPO`** (in `Src/parser.py`) have no corresponding source file in `Src/Algorithms/` — `Heuristic.py` and `PPO.py` don't exist at all despite being described in the README; a stray `SPO.cpython-38.pyc` in `__pycache__` shows an `SPO.py` used to exist and was removed. Selecting any of these three currently raises `ValueError: Failed to dynamically load the class` from `dynamic_load`. Only `DSPO` and `Baseline` are known to resolve correctly.
- **`run_ppo.py`** depends on `--algo_name PPO` and therefore fails outright for the same reason — don't assume it runs just because the file exists.
- **`DSPO_plus_SPO`** (the actual "SPO" contribution being developed) is not in the parser's `choices` list at all, so it can't be selected via the normal CLI without editing `parser.py`. Even if selected, it currently: imports `Src.Utils.passenger_utility` which does not exist in the repo, and references `self.l0_home`, `self.l_mp`, `self.use_travel_time_prediction` which are never set in its `__init__` or in `DSPO`'s. Expect `ImportError`/`AttributeError` — this class is mid-refactor, not a working baseline.
- **Cost-accounting is not fully aligned across the three references** (original paper vs. this code vs. the in-progress manuscript), catalogued in `docs/cost_alignment_audit.md`. Several of the concrete bugs it identified have since been fixed (surcharge sign in `Utils.total_costs`, the accepted-price off-by-one in `customerchoice.py`, the `sixhump_func` typo, the missing `mltplr` wrap in `DSPO`'s home-cost branch, and eval methods re-optimizing with the training env instead of `test_env` in `run.py`/`run_ppo.py`/`Experiments/run_baseline1.py`) — but two known-real issues are still **open by design choice, not oversight**: OOH delivery service time is hardcoded to 0 in `Parcelpoint_py.step()` (needs a real per-stop amortization formula, not yet decided), and on Homberger-Gehring instances (C/R/RC) `dist_matrix` stores travel *time* rather than distance, so `fuel_cost` is effectively billed per unit time. Read the audit doc before trusting any cost/revenue/profit output, and don't assume "fixed once" means "fixed everywhere" — always check current code rather than trusting the doc's line numbers, since both the code and the audit have drifted from each other before.
- **`Environments/OOH/yanjiao/`** contains raw Amap-sourced candidate POI/bus-stop CSVs (home clusters, meeting points) for a planned Beijing-Yanjiao commuting scenario. Nothing in the Python code loads or references this data yet — it's staged input, not a wired-up instance.

When asked to run or fix a specific algorithm/instance combination, check that its files and attributes actually exist first rather than trusting `--help`, the README, or the parser's `choices` list.

## Data / instances

`--instance` selects the demand dataset (`Src/Utils/Utils.py::load_demand_data`): `Austin`/`Seattle` load Amazon last-mile data with a precomputed distance matrix and a 20-nearest-parcelpoint adjacency `.npy`; `C`/`R`/`RC` load Homberger-Gehring coordinates and compute Euclidean distances on the fly (see the `README.md` under each `Environments/OOH/*_data/` folder for node-ordering conventions — depot is always node 0, OOH points are always the last N rows). `--data_seed` / `--data_seed_test` pick which of the pre-generated seed files to load for train vs. test.
