# DSPO_plus_SPO Fix and 4-Way Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Src/Algorithms/DSPO_plus_SPO.py` runnable against the current parcel OOH environment (without touching any other file except one `parser.py` line), then run and report a 4-way comparison of NoPricing / StaticPricing / DSPO / DSPO_plus_SPO.

**Architecture:** `DSPO_plus_SPO` currently fails in three progressively-deeper layers (broken import → uninitialized attributes → a customer-choice-model attribute that doesn't exist on this environment's `customerchoicemodel` class). Each task fixes exactly one layer and re-runs the same small smoke script to confirm the *next* layer's failure surfaces (or that everything now passes). Once it runs cleanly, four `run.py` invocations (one per compared method) produce comparable `eval2()` cost summaries, which get compiled into a short results doc.

**Tech Stack:** Python 3.9, PyTorch (CUDA build), conda env `py39`, no test framework in this repo (verification is via a throwaway smoke script + reading real run logs, not pytest).

## Global Constraints

- Run all Python commands via `conda run -n py39 python ...` (the project's local conda env; a plain `python` on PATH is a different, unrelated interpreter on this machine).
- Working directory for all commands: `D:\project\python\ooh_code`.
- **`conda` is only on PATH in PowerShell on this machine, not in the Bash/git-bash tool.** Run every `conda run -n py39 ...` command through the PowerShell tool. `git`/`grep`/`cat`/`rm`/`mkdir`-style commands work fine in either; use whichever tool you already have output from to avoid re-verifying paths twice.
- Part 1 (Tasks 1-4) may modify **only** `Src/Algorithms/DSPO_plus_SPO.py` and `Src/parser.py` (one line). Do not modify `Src/Algorithms/DSPO.py`, `Environments/OOH/customerchoice.py`, `Src/Utils/Utils.py`, or any other file.
- `self.l0_home` and `self.l_mp` must be set to `0.0` — do not invent nonzero values.
- Comparison runs (Tasks 5-8) use current default hyperparameters unchanged (`--instance RC`, `--max_episodes 10`, `--seed 1234` are all already the defaults — don't override them), per the approved spec `docs/superpowers/specs/2026-07-24-dspo-plus-spo-comparison-design.md`.
- The smoke-test script used in Tasks 1-4 is a throwaway diagnostic. Create it outside the repo (e.g. `C:\Users\wwpst\AppData\Local\Temp\dspo_plus_spo_smoke_test.py`) and never `git add` it.
- Commit after each task (Git Safety Protocol from the main agent instructions still applies: review `git status`/`git diff` before committing, never use `git add -A`/`-i`, never force-push).

---

### Task 1: Add `DSPO_plus_SPO` to parser choices and capture the baseline failure

**Files:**
- Modify: `Src/parser.py:25-26`
- Create (outside repo): `C:\Users\wwpst\AppData\Local\Temp\dspo_plus_spo_smoke_test.py`

**Interfaces:**
- Produces: the smoke script (reused unchanged in Tasks 2-4), and `--algo_name DSPO_plus_SPO` becoming a valid CLI choice.

- [ ] **Step 1: Add the new choice to `parser.py`**

Current content at `Src/parser.py:25-26`:
```python
        parser.add_argument("--algo_name", default='DSPO', help="Policy/algorithm used, capital sensitive",
                            choices=['DSPO','Heuristic','Baseline','PPO','SPO'])
```

Change to:
```python
        parser.add_argument("--algo_name", default='DSPO', help="Policy/algorithm used, capital sensitive",
                            choices=['DSPO','Heuristic','Baseline','PPO','SPO','DSPO_plus_SPO'])
```

- [ ] **Step 2: Verify argparse accepts the new choice**

Run:
```bash
conda run -n py39 python -c "from Src.parser import Parser; a = Parser().get_parser().parse_args(['--algo_name','DSPO_plus_SPO']); print(a.algo_name)"
```
Expected: prints `DSPO_plus_SPO` with no error.

- [ ] **Step 3: Write the smoke-test script**

Create `C:\Users\wwpst\AppData\Local\Temp\dspo_plus_spo_smoke_test.py` (working directory must be `D:\project\python\ooh_code` when you run it, since it does repo-relative imports):

```python
import numpy as np
from Src.parser import Parser
from Src.config import Config

args = Parser().get_parser().parse_args([
    '--algo_name', 'DSPO_plus_SPO',
    '--max_episodes', '1',
    '--instance', 'RC',
    '--folder_suffix', 'smoketest',
])
config = Config(args)
model = config.algo(config=config)
print("CONSTRUCTED OK")

env = config.env
state = env.reset()

action = model.get_action(state, training=True)
print("initial_phase get_action OK, action shape:", np.shape(action))

model.initial_phase = False
action2 = model.get_action(state, training=True)
print("non-initial_phase get_action OK, action shape:", np.shape(action2))

new_state, done, stats, route_data = env.step(action=action2)
cost = model.update(route_data, new_state, False)
print("update(done=False) OK, cost:", cost)

print("SMOKE TEST PASSED")
```

- [ ] **Step 4: Run it and confirm the expected baseline failure**

Run:
```bash
conda run -n py39 python "C:\Users\wwpst\AppData\Local\Temp\dspo_plus_spo_smoke_test.py"
```
Expected: it fails before printing `CONSTRUCTED OK`. You should see, near the end of the traceback, something like:
```
ModuleNotFoundError: No module named 'Src.Utils.passenger_utility'
...
During handling of the above exception, another exception occurred:
...
ValueError: Failed to dynamically load the class: DSPO_plus_SPO
```
(`Utils.dynamic_load` catches the real `ModuleNotFoundError` and re-raises it as a generic `ValueError`, but Python's exception chaining still shows the original error above it — that's the one that matters.) If you see this, the baseline failure is confirmed and matches the spec's diagnosis.

- [ ] **Step 5: Commit**

```bash
git add Src/parser.py && git status --short
```
Confirm only `Src/parser.py` is staged, then:
```bash
git commit -m "Add DSPO_plus_SPO to --algo_name choices"
```

---

### Task 2: Fix the broken `passenger_utility` import

**Files:**
- Modify: `Src/Algorithms/DSPO_plus_SPO.py:14` (import line + new local function)
- Modify: `Src/Algorithms/DSPO_plus_SPO.py` (8 call sites, mechanical rename)

**Interfaces:**
- Produces: `_utility_home_nonprice(base_util, home_util, travel_time_weight, home_travel_time) -> float`, a module-level function in `DSPO_plus_SPO.py`, replacing the import of `utility_home_nonprice` from the nonexistent `Src.Utils.passenger_utility`.
- Consumes: nothing new — same call signature as the old (missing) `utility_home_nonprice`.

- [ ] **Step 1: Replace the broken import with a local function definition**

Current (`Src/Algorithms/DSPO_plus_SPO.py:14`):
```python
from Src.Utils.passenger_utility import utility_home_nonprice
```

Replace with:
```python
def _utility_home_nonprice(base_util, home_util, travel_time_weight, home_travel_time):
    if travel_time_weight is not None and home_travel_time is not None:
        return base_util + home_util - travel_time_weight * home_travel_time
    return base_util + home_util
```

(Use the Edit tool with `old_string` set to exactly `from Src.Utils.passenger_utility import utility_home_nonprice` and `new_string` set to the function above — this both removes the bad import and adds the replacement in one edit, since it's a single-occurrence match.)

- [ ] **Step 2: Rename all 8 call sites from `utility_home_nonprice(` to `_utility_home_nonprice(`**

Use the Edit tool with `old_string="utility_home_nonprice("`, `new_string="_utility_home_nonprice("`, `replace_all=true`. This matches only call sites (which all have an opening paren immediately after the name) — it will not touch anything else since the import line no longer exists after Step 1.

Verify no old references remain and all 8 are renamed:
```bash
grep -c "_utility_home_nonprice(" Src/Algorithms/DSPO_plus_SPO.py && grep -c "[^_]utility_home_nonprice(" Src/Algorithms/DSPO_plus_SPO.py
```
Expected: first command prints `8`; second command prints `0` (or errors with no match, meaning grep found nothing — either outcome confirms no un-renamed call sites remain).

- [ ] **Step 3: Re-run the smoke script, confirm it gets further and hits the next known failure**

Run:
```bash
conda run -n py39 python "C:\Users\wwpst\AppData\Local\Temp\dspo_plus_spo_smoke_test.py"
```
Expected: `CONSTRUCTED OK` now prints (the import/construction failure is gone). It then fails during the first `model.get_action(...)` call with:
```
AttributeError: 'DSPO_plus_SPO' object has no attribute 'l0_home'
```
This confirms Step 1-2 fixed exactly the import layer and nothing else — the next layer (Task 3) is now exposed.

- [ ] **Step 4: Commit**

```bash
git add Src/Algorithms/DSPO_plus_SPO.py && git status --short
```
Confirm only `Src/Algorithms/DSPO_plus_SPO.py` is staged, then:
```bash
git commit -m "DSPO_plus_SPO: replace missing passenger_utility import with local helper"
```

---

### Task 3: Initialize `l0_home`, `l_mp`, `use_travel_time_prediction`

**Files:**
- Modify: `Src/Algorithms/DSPO_plus_SPO.py:83` (end of `__init__`)

**Interfaces:**
- Produces: `self.l0_home` (float, `0.0`), `self.l_mp` (float, `0.0`), `self.use_travel_time_prediction` (bool, `False`) — all read (not written) elsewhere in the class.

- [ ] **Step 1: Add the three attributes at the end of `__init__`**

Current end of `__init__` (`Src/Algorithms/DSPO_plus_SPO.py:82-83`):
```python
        self._spo_training_data_logged = False
        self._spo_positive_weight_logged = False
```

Change to:
```python
        self._spo_training_data_logged = False
        self._spo_positive_weight_logged = False
        self.l0_home = 0.0
        self.l_mp = 0.0
        self.use_travel_time_prediction = False
```

- [ ] **Step 2: Re-run the smoke script, confirm it gets further and hits the next known failure**

Run:
```bash
conda run -n py39 python "C:\Users\wwpst\AppData\Local\Temp\dspo_plus_spo_smoke_test.py"
```
Expected: `CONSTRUCTED OK` and `initial_phase get_action OK, ...` do NOT both print yet — it now fails partway through the first `get_action` call with:
```
AttributeError: 'customerchoicemodel' object has no attribute 'travel_time_weight'
```
This confirms the `l0_home`/`l_mp` layer is fixed and the next layer (Task 4, the `customerchoice_model` guard conditions) is now exposed.

- [ ] **Step 3: Commit**

```bash
git add Src/Algorithms/DSPO_plus_SPO.py && git status --short
```
Confirm only `Src/Algorithms/DSPO_plus_SPO.py` is staged, then:
```bash
git commit -m "DSPO_plus_SPO: initialize l0_home, l_mp, use_travel_time_prediction"
```

---

### Task 4: Guard the 8 `customerchoice_model` branches so they fall back correctly

**Files:**
- Modify: `Src/Algorithms/DSPO_plus_SPO.py` (8 occurrences, mechanical condition change)

**Interfaces:**
- Consumes: `getattr(customerchoice_model, 'travel_time_weight', None)` — a defensive read that doesn't require `customerchoicemodel` (in `Environments/OOH/customerchoice.py`) to have this attribute.
- Produces: after this task, `DSPO_plus_SPO.get_action_pricing` always takes its existing, already-correct `else` fallback path (using `self.mnl(...)`, no travel time) when running against the current parcel environment.

- [ ] **Step 1: Verify there are exactly 8 occurrences before editing**

```bash
grep -c "if customerchoice_model is not None:" Src/Algorithms/DSPO_plus_SPO.py
```
Expected: `8`.

- [ ] **Step 2: Change the condition on all 8 occurrences**

Use the Edit tool with `old_string="if customerchoice_model is not None:"`, `new_string="if customerchoice_model is not None and getattr(customerchoice_model, 'travel_time_weight', None) is not None:"`, `replace_all=true`. (The old string has no leading whitespace in the match, so this correctly applies regardless of each occurrence's indentation level.)

- [ ] **Step 3: Verify the replacement landed on all 8 and none of the old bare form remains**

```bash
grep -c "getattr(customerchoice_model, 'travel_time_weight', None) is not None:" Src/Algorithms/DSPO_plus_SPO.py
```
Expected: `8`.

```bash
grep -n "if customerchoice_model is not None:$" Src/Algorithms/DSPO_plus_SPO.py
```
Expected: no output (no bare, un-guarded occurrences left — the `$` anchors to end-of-line so it won't match the new longer condition).

- [ ] **Step 4: Re-run the smoke script, confirm it now passes completely**

Run:
```bash
conda run -n py39 python "C:\Users\wwpst\AppData\Local\Temp\dspo_plus_spo_smoke_test.py"
```
Expected output, in full:
```
CONSTRUCTED OK
initial_phase get_action OK, action shape: (...)
non-initial_phase get_action OK, action shape: (...)
update(done=False) OK, cost: 0.0
SMOKE TEST PASSED
```
(Exact action shape depends on how many parcel points are adjacent for the sampled customer — don't worry about the exact numbers, just that no exception is raised and `SMOKE TEST PASSED` is the last line.)

If it still fails, re-read the traceback: it means either an occurrence was missed in Step 2, or there's a genuinely new issue not covered by this plan — stop and report the exact traceback rather than guessing further fixes.

- [ ] **Step 5: Commit**

```bash
git add Src/Algorithms/DSPO_plus_SPO.py && git status --short
```
Confirm only `Src/Algorithms/DSPO_plus_SPO.py` is staged, then:
```bash
git commit -m "DSPO_plus_SPO: fall back to travel-time-free MNL when travel_time_weight is unavailable"
```

- [ ] **Step 6: Delete the throwaway smoke script**

```bash
rm "C:\Users\wwpst\AppData\Local\Temp\dspo_plus_spo_smoke_test.py"
```
(It's outside the repo and was never committed — nothing to clean up in git.)

---

### Task 5: Run the NoPricing comparison

**Files:** none (this task only runs the existing `run.py`, no code changes).

- [ ] **Step 1: Run NoPricing**

```bash
conda run -n py39 python run.py --algo_name Baseline --price_home 0.0 --price_pp -0.0 --folder_suffix nopricing
```
This takes several minutes (comparable to the smoke tests run earlier in this project — expect roughly 5-10 minutes). Run it in the background if your tooling supports that, and wait for completion rather than polling repeatedly.

- [ ] **Step 2: Verify success and record the summary block**

Read the log:
```bash
cat "D:/project/python/ooh_code/Experiments/Parcelpoint_py/pricing/Baseline/runnopricing/1234/Logs/logfile.log"
```
Expected: it ends with a block containing `percentage home delivery:`, `travel costs:`, `service costs:`, `failure costs:`, `Avg. Charge:`, `Avg. Discount:`, `Charge revenue:`, `Discount costs:`, `total costs:`, `average travelled by customers:`, `total timing:`. Copy this block down — Task 9 needs it. If the log doesn't end this way (e.g., a Python traceback instead), stop and report the error rather than proceeding to Task 6.

No commit for this task — it only produces gitignored output under `Experiments/Parcelpoint_py/`.

---

### Task 6: Run the StaticPricing comparison

**Files:** none.

- [ ] **Step 1: Run StaticPricing**

```bash
conda run -n py39 python run.py --algo_name Baseline --price_home 1.0 --price_pp -2.0 --folder_suffix staticpricing
```

- [ ] **Step 2: Verify success and record the summary block**

```bash
cat "D:/project/python/ooh_code/Experiments/Parcelpoint_py/pricing/Baseline/runstaticpricing/1234/Logs/logfile.log"
```
Same expected shape as Task 5, Step 2. Copy the summary block down for Task 9.

No commit for this task.

---

### Task 7: Run the DSPO comparison

**Files:** none.

- [ ] **Step 1: Run DSPO**

```bash
conda run -n py39 python run.py --algo_name DSPO --folder_suffix dspo
```

- [ ] **Step 2: Verify success and record the summary block**

```bash
cat "D:/project/python/ooh_code/Experiments/Parcelpoint_py/pricing/DSPO/rundspo/1234/Logs/logfile.log"
```
Same expected shape as Task 5, Step 2. Copy the summary block down for Task 9.

No commit for this task.

---

### Task 8: Run the DSPO_plus_SPO comparison

**Files:** none.

- [ ] **Step 1: Run DSPO_plus_SPO**

```bash
conda run -n py39 python run.py --algo_name DSPO_plus_SPO --folder_suffix dspoplusspo
```

- [ ] **Step 2: Verify success, confirm SPO+ actually engaged, and record the summary block**

```bash
cat "D:/project/python/ooh_code/Experiments/Parcelpoint_py/pricing/DSPO_plus_SPO/rundspoplusspo/1234/Logs/logfile.log"
```

Two things to check in this log, per the approved spec's acceptance criteria:
1. It ends with the same kind of summary block as the other three runs (`percentage home delivery:` ... `total timing:`).
2. Somewhere in the log, before that summary, it contains the lines:
   ```
   [SPO+ debug] spo_training_data populated, size=...
   [SPO+ debug] spo_weight became positive: ...
   ```
   Search for them directly if they're not obvious by eye:
   ```bash
   grep -n "\[SPO+ debug\]" "D:/project/python/ooh_code/Experiments/Parcelpoint_py/pricing/DSPO_plus_SPO/rundspoplusspo/1234/Logs/logfile.log"
   ```
   If these lines are **absent**, the SPO+ loss never actually got triggered during this short 10-episode run (it may need `spo_warmup_episodes` + `spo_rampup_episodes` more episodes than this run allows — default warmup is 5, rampup is 10, and we're only running 10 episodes total, plus 20 more during `eval2`'s online-update pass, so there should be enough episodes for it to engage). If they're absent, note this clearly in the Task 9 report as an open finding rather than silently treating the run as fully validated.

Copy the summary block (and note whether the SPO+ debug lines appeared) down for Task 9.

No commit for this task.

---

### Task 9: Compile and commit the comparison report

**Files:**
- Create: `docs/dspo_plus_spo_comparison_results.md`

- [ ] **Step 1: Write the report**

Create `docs/dspo_plus_spo_comparison_results.md` with this structure (fill in the actual numbers you recorded from Tasks 5-8 — every `...` below must be replaced with a real value, there should be no placeholders left in the committed file):

```markdown
# NoPricing / StaticPricing / DSPO / DSPO_plus_SPO 对比结果

跑于 2026-07-24，`--instance RC --max_episodes 10 --seed 1234`（当前默认超参数，未做论文级对齐），四组结果均来自各自 `Logs/logfile.log` 末尾的 `eval2()` 汇总。

## 是否跑通

| 方法 | 是否跑完不崩 | SPO+ 是否实际触发（仅 DSPO_plus_SPO 适用） |
|---|---|---|
| NoPricing | ... | - |
| StaticPricing | ... | - |
| DSPO | ... | - |
| DSPO_plus_SPO | ... | ...(是/否，附 grep 结果) |

## 数值对比

| 指标 | NoPricing | StaticPricing | DSPO | DSPO_plus_SPO |
|---|---|---|---|---|
| percentage home delivery | ... | ... | ... | ... |
| travel costs | ... | ... | ... | ... |
| service costs | ... | ... | ... | ... |
| failure costs | ... | ... | ... | ... |
| Avg. Charge | ... | ... | ... | ... |
| Avg. Discount | ... | ... | ... | ... |
| Charge revenue | ... | ... | ... | ... |
| Discount costs | ... | ... | ... | ... |
| total costs | ... | ... | ... | ... |

## 观察

...(照实描述数字之间的关系，例如 DSPO_plus_SPO 的 total costs 是否低于 DSPO/StaticPricing/NoPricing；如果反常，直接写出来，不要淡化)

## 局限性（据 spec 明确写明，不要省略）

- 单 seed、10 episode，统计上不严谨，不能当作显著性结论。
- `docs/cost_alignment_audit.md` 里仍有两个未修的成本口径问题（OOH 服务时间恒为 0、Homberger 数据集燃油按时间而非距离计费），同等影响四组的绝对数值。
- StaticPricing 的价格（home=1.0, pp=-2.0）是本次示意值，不是论文调优值。
```

- [ ] **Step 2: Commit**

```bash
git add docs/dspo_plus_spo_comparison_results.md && git status --short
```
Confirm only that one new file is staged, then:
```bash
git commit -m "Add NoPricing/StaticPricing/DSPO/DSPO_plus_SPO comparison results"
```
