# polynomial-cl-demo

A small NumPy-only research instrument for one hypothesis:

> Incremental polynomial feature expansion, combined with appropriate knowledge-preservation mechanisms (replay, per-coefficient update rules), can improve a model's ability to learn **new classes** while retaining what it already knows.

The project is a continual-learning study first and a machine-learning library second. It does **not** assume the answer is yes. Every mechanism (zero-initialised expansion, freezing, per-coefficient learning rates, replay, a preserved old function) can be switched on and off independently, and every score, coefficient and prediction change is recorded.

> This project studies _explicit_ polynomial feature expansion, which is related to polynomial kernels. It does not implement an implicit-kernel SVM.

## Research map

| | setting | question | where |
| --- | --- | --- | --- |
| **primary** | **class-incremental learning (CIL)**: ten classes arrive in five two-class experiences, one softmax head | Does expansion (with replay and update rules) let new classes arrive without erasing old ones? | `src/polynomial_cl/class_incremental/`, `experiments/class_incremental.py`, `experiments/rate_selection.py`, notebook 02 Part I |
| CIL protocol check | ten Gaussian classes on a ring | Do replay, class masking, accuracy matrices and forgetting behave as specified? The ring is linearly solvable, so it **cannot** reward extra capacity | same |
| CIL capacity test | Split MNIST under a fixed 16-D random projection | Does polynomial capacity help on image inputs? | same |
| supporting | binary sinusoid, spatially shifting input distribution | Nonlinear representation and capacity under distribution shift. This is **not** class-incremental learning: the label set never changes | `src/polynomial_cl/{data,features,model,training,replay}.py`, `experiments/synthetic_degree_ablation.py`, notebook 02 Part II |

Class-incremental learning and sequential binary classification under a shifting input distribution are different settings, and results from one must not be presented as results for the other. Conventional joint training (notebook 01) is only a reference point.

Observed behaviour, with every number traced to a command, is summarised under [Results](#results). This README is the project's only documentation file.

## Setup

Only Python, NumPy, Matplotlib and Jupyter are used (no PyTorch / TensorFlow / scikit-learn / pretrained models). CPU only.

```bash
cd polynomial-cl-demo
python -m venv .venv && source .venv/bin/activate     # optional
pip install -e ".[dev]"                                # numpy, matplotlib + pytest, jupyter, nbformat, nbconvert
python -m pytest                                       # fast test suite (seconds)
python -m pytest -m slow                               # executes both notebooks in a clean kernel (minutes)
jupyter lab notebooks/                                 # or: jupyter notebook
```

Notebooks are committed without outputs (small diffs); run them to regenerate every figure and table. To re-execute them headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

The Split MNIST cells of notebook 02 are skipped, with a message, when `data/mnist` is absent; the command-line tool downloads the archives (see below). Both notebooks fall back to `../src` if the package is not installed.

## Primary research: class-incremental learning

### Protocol

Ten classes arrive in five experiences of two classes each (`TASK_ORDER`). One ten-output softmax head is trained on the current experience plus a fixed per-class replay memory. During training **and** evaluation only the classes introduced so far compete, so the output weights of future classes are never trained. After each experience the model is evaluated on every experience seen so far, which fills a lower-triangular accuracy matrix; forgetting of an experience is the best accuracy it ever had minus its final accuracy, averaged over experiences 1-4 (the last has no later evaluation). Forgetting is computed by `polynomial_cl.metrics.forgetting`, the same function the binary experiment uses.

Three conditions share task order, replay indices, seed, optimiser and number of epochs, and differ only in polynomial degree:

| condition | degrees over the five experiences | what comparing it tells you |
| --- | --- | --- |
| `fixed_linear` | 1, 1, 1, 1, 1 | baseline: no added capacity |
| `expanding` | 1, 2, 3, 3, 3 (new coefficients start at zero; logit change asserted below 1e-5) | `expanding` vs `fixed_linear`: value of capacity |
| `fixed_capacity` | 3, 3, 3, 3, 3 | `expanding` vs `fixed_capacity`: value of adding capacity gradually |

MNIST images are normalised and mapped to 16 dimensions by a fixed random orthonormal projection (never fitted to data) before the explicit polynomial map (17, 153, 969 features at degrees 1, 2, 3). Each degree group is divided by the square root of its size so that high-degree monomials do not dominate merely by being numerous.

```bash
python experiments/class_incremental.py --dataset synthetic_2d --output results/synthetic_2d.json
python experiments/class_incremental.py --dataset split_mnist  --output results/split_mnist.json   # downloads MNIST to data/mnist
```

### Why old-class accuracy drops: takeover versus flips

Two different mechanisms lower the accuracy on old classes, and the accuracy matrix alone cannot separate them. Every record carries both:

- `old_takeover_fraction_by_task`: after training, **all** classes seen so far compete (as in the accuracy matrix) and a **new** class wins an old-class test input. This can happen although the old logits barely move.
- `old_flip_fraction_by_task`: only old classes compete, and the prediction changes between two old classes. This measures how much the old decision function itself moved.

`old_logit_change_by_task` is the largest change of any old-class logit on old-class test inputs. Large takeover with small flips points to a stability-plasticity imbalance between old and new classes; large flips point to damage to the old function.

### Update rules (`--experiment strategies`, `expanding` schedule)

`ordinary` (replay, everything trains), `ordinary_no_replay`, `frozen_old`, `small_old_lr` (old coefficients get 0.1 x the gradient) and `strategy_b` (every gradient is projected onto the null space of the design matrix at `--n-ref-per-class` stored inputs per old class, so the logits at those inputs cannot change). A coefficient is _old_ if its feature row existed before the current experience **and** its class column was already introduced; new feature rows and the new classes' columns always train.

```bash
python experiments/class_incremental.py --dataset split_mnist --experiment strategies --output results/split_mnist_strategies.json
python experiments/class_incremental.py --dataset split_mnist --experiment strategies --n-ref-per-class 10 --output results/split_mnist_b10.json
```

Every record carries learning indicators that exist for _all_ rules: training loss before and after each experience, L1 size of the update, number of trainable coefficients, and the old-class change, flip and takeover diagnostics above. `null_dim` exists only for Strategy B and is a constraint diagnostic, not a general learning indicator.

### Per-feature learning rates and validated rate selection

`--feature-power p` gives feature i the step size `lr * (max_rms / rms_i) ** p` (p = 0: one shared rate; p = 1: diagonal preconditioning). `--validation-fraction 0.1` evaluates on a held-out part of the training data instead of the test set. `experiments/rate_selection.py` sweeps `(p, lr)` on that split, picks the best cell for each condition (or rule, with `--experiment strategies`) by validation accuracy only, and then reports the test result for the selected cells:

```bash
python -u experiments/rate_selection.py --dataset split_mnist --output results/rate_selection_conditions.json   # minutes
python -u experiments/rate_selection.py --dataset split_mnist --experiment strategies --output results/rate_selection_strategies.json
```

The seeds used for selection (`--select-seeds`, default 0 1) and for the reported test result (`--test-seeds`, default 2-6) must be disjoint: the script refuses overlapping sets, because otherwise the same stochastic runs would both choose the rates and be reported as the test estimate. Check where each selected cell lies in the grid: a selection on the boundary (`--powers`, `--lrs`) means the optimum may be outside it; widen the grid and rerun.

### Sampling policy and per-degree learning rates

Two further mechanisms are available for every update rule (`polynomial_cl.class_incremental`).

**Sampling policy** (`sampling="shuffled"` or `"class_balanced"`). `shuffled` is the original behaviour: current and replay examples are shuffled together, so in the last experience a new class contributes about 5,900 examples per class against 200 stored for an old class. `class_balanced` builds every batch with (almost) equal numbers of every class seen so far; each class is read through its own shuffled cycle, a class with fewer examples than its share is repeated and never padded from another class, and classes with no examples (no replay) do not take part. Both policies make the same number of updates per epoch and use the same task plan and memory; the test set is never used. Every record stores the pool size, the number of examples actually drawn and the mean batch composition per class (`sampling_by_task`).

**Per-degree learning rates** (`learning_rate=` a scalar or a vector). The vector has one entry per polynomial degree group and position k is degree k (0: constant, 1: linear, 2: quadratic, 3: cubic); entries must be finite and non-negative, 0 freezes the group, and an all-zero vector is rejected. All groups train together in one run with their own rate in every update. A group that appears through expansion uses its own entry from the first update. The vector composes with update masks (`small_old_lr`, `frozen_old`) and with Strategy B, whose null-space projection is taken in the rate-scaled coordinates so the reference logits stay fixed; it cannot be combined with `--feature-power`, which is a different mechanism (a step size per _feature_ derived from the measured feature scale). The binary trainer's `_as_vector` is per _coefficient_ and is unrelated. Each record carries `degree_diagnostics`: per experience and degree group the rate, coefficient and trainable counts, raw-gradient and applied-step size, actual coefficient change (L1 and relative), the group's logit contribution and its contribution to old-class logit change.

```bash
python experiments/per_degree_lr.py --dataset split_mnist --experiment sampling --output results/per_degree/sampling_e3_lr0.1.json
# one JSON per (strategy, sampling); notebook 02 merges cell_*.json. The whole study, resumable (skips finished cells):
bash experiments/run_per_degree_study.sh
python experiments/per_degree_lr.py --dataset split_mnist --experiment per-degree --strategies ordinary --samplings class_balanced --output results/per_degree/cell_ordinary_class_balanced.json
```

Protocol of the per-degree study: every candidate is measured on a 10% validation split (seeds 0-1) for final mean accuracy, mean forgetting and newest-experience accuracy, so accuracy alone never decides. The shared rate is chosen from {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30}; one degree group at a time is then made 4x larger or 4x smaller; one follow-up vector takes each group's best factor; and all vectors are run on the test set with disjoint seeds 2-6. The Pareto front on (accuracy, forgetting) is recorded for every cell; where one vector is needed, the front member with the largest accuracy minus forgetting is marked `selected` (optionally with a floor `--min-current-accuracy` on newest-experience accuracy). `selected` is a convenience for reading results, not a claim of optimality. Joint (shuffled, all-class) training on the same features is available as `run_joint_reference` and is shown in notebook 01; it has no forgetting.

### How to read the results

- **Training budget and rates.** The defaults (3 epochs, one shared rate 0.1) are far from convergence, and conclusions drawn at them were reversed once rates were tuned (see [Results](#results)). Do not treat default-rate results as evidence that one condition is better. Compare conditions only at rates selected **per condition on validation data**, over several seeds, and repeat with several `--epochs` values.
- **The ring is linearly solvable.** Equal isotropic Gaussian classes have linear Bayes boundaries, so that dataset checks the machinery and cannot reward extra capacity; Split MNIST and the binary sinusoid are the capacity tests. With the default 160 training samples per class the replay memory (200 per class) holds _all_ earlier data on the ring, whereas it is a genuine 200-example budget on MNIST. A genuinely nonlinear ten-class synthetic dataset is not yet implemented.
- **Seeds.** For the class-incremental runs the data, test sets and projection are fixed; seeds change shuffling and replay selection only, so the spread across seeds understates dataset-level variability.

## Supporting experiment: binary sinusoid

Inputs live in `[-1, 1]^2`. The true rule is `y = 1 iff x2 > 0.5 sin(3 x1)` (a nonlinear boundary). Experience _k_ of 10 samples `x1` from the _k_-th vertical strip and `x2` uniformly, so each experience reveals a different part of the boundary. This is sequential binary learning under a spatially shifting input distribution, **not class-incremental learning**. Each experience has its own training split and its own fixed test split (independent random streams). The model is `f_d(x) = w . phi_d(x)` with the bias as the coefficient of the constant feature. Starting at degree 1, the degree grows by one per experience (configurable via `degrees=`). Expansion appends zero coefficients and the loop **asserts** `f_{d+1}(x) = f_d(x)` for scores and probabilities on training, test, random and out-of-region inputs before any training.

```bash
python experiments/synthetic_degree_ablation.py --seeds 0 1 2 3 4 --output results/synthetic_degree_ablation.json
```

Within each seed the same generated experiences, replay budget, epochs and optimiser are used for `expanding` (degree 1..10), `fixed_initial_degree` (degree 1) and `fixed_capacity` (degree 10 from the start).

### Strategies (`make_config(name)`)

| name | meaning |
| --- | --- |
| `ordinary_no_replay` | one global learning rate, no replay |
| `ordinary` | one global learning rate, balanced replay |
| `frozen_old` | existing coefficients have mask False (never updated), new ones train |
| `small_old_lr` | existing coefficients use `lr_old_factor = 0.1` |
| `strategy_b_all` / `strategy_b_new` | as `ordinary`, plus **preserved old predictions**: `g(x_ref) = 0` enforced exactly on reference inputs |

- There are no separate `strategy_a_*` presets: an unconstrained `f = f_old + g` over all coefficients is the same optimisation problem as `ordinary`, and over only the new coefficients it is `frozen_old`. They are representational aliases, so they are not offered (this is tested).
- Strategy B constrains `g` only at finitely many reference inputs (the first 3 buffered inputs of every old experience, configurable). `g`'s coefficients are restricted to the null space of the reference design matrix (SVD), so `g(x_ref) = 0` holds to rounding. This says **nothing** about inputs elsewhere; the metrics measure how much old scores actually move on the full old test sets. If the constraint uses up every degree of freedom, `g` cannot learn and the run reports that (`null_dim = 0`); constraints are never relaxed automatically.
- Presets cannot be silently overridden: `make_config("ordinary_no_replay", m_replay=100)` raises.

### Replay policy

Each experience stores up to `capacity` training samples (seeded choice without replacement). A batch for experience _t_ contains `m_current` current samples plus `m_replay` samples from **each** earlier experience (int or `{exp_id: m}`), with or without replacement. If fewer than `m` rows are stored and sampling is without replacement, all stored rows are used (never padded, never taken from elsewhere). Actual per-experience counts are stored in `record.batch_counts`. `ReplayBuffer.add_experience` only accepts `Split(kind="train")`. The class-incremental runner uses a different, fixed per-class memory (`make_task_plan`), because its batches are formed from classes rather than experiences; the two replay policies are deliberately not merged.

### Metrics (per experience, identical for every strategy)

Current-experience accuracy; mean accuracy over observed experiences; `max |f_after - f_before|` over old test inputs; fraction of old test predictions that flipped; forgetting (best earlier accuracy minus current accuracy); max change of frozen coefficients (reported as unavailable when none are frozen), change in the preserved old component, maximum combined-coefficient change, and L1 update of trainable coefficients; for Strategy B also the change at reference inputs and the free dimensions left by the constraint.

Hyper-parameters of the binary experiment (`lr=0.5`, 300 epochs per experience, replay `m=100`, capacity 100) were fixed before the continual experiments were run and were not tuned afterwards; all strategies share them. the Results section shows that this shared rate is far from converged.

## Results

All numbers are mean +/- sample standard deviation over seeds, produced by the commands above and the notebooks. Seeds change shuffling and replay selection (and, for the binary task only, the generated data); for Split MNIST the data, test set and projection are fixed, so the spread understates dataset-level variability. Hyper-parameters were not tuned on test data, except where "selected" is stated.

### Class-incremental learning

**Default shared rates (3 epochs, lr 0.1, seeds 0-4): final mean accuracy / mean forgetting.** These runs are far from converged; use them to compare mechanisms, not to rank conditions.

| setting | `fixed_linear` | `expanding` | `fixed_capacity` |
| --- | --- | --- | --- |
| ring, 3 epochs | 0.709 +/- 0.002 / 0.008 | 0.909 +/- 0.001 / 0.023 | 0.908 +/- 0.001 / 0.025 |
| ring, 30 epochs (seeds 0-2) | 0.906 | 0.910 | 0.908 |
| Split MNIST, 3 epochs | 0.176 +/- 0.000 / 0.926 | 0.246 +/- 0.002 / 0.858 | 0.256 +/- 0.001 / 0.846 |
| Split MNIST, 30 epochs (seed 0) | 0.337 | 0.481 | 0.484 |

- **Ring: no capacity effect.** The 3-epoch gap closes with training, as expected for a linearly solvable task: it reflects how fast each model learns a new class from zero-initialised weights.
- **Split MNIST: expansion helps; gradual growth is not shown to matter.** `fixed_linear` < `expanding` ~ `fixed_capacity` from 1 to 30 epochs. The near-equality of the last two shows that capacity matters, not that gradual expansion is better.

**Rates selected on validation data.** For each condition `(p, lr)` was selected on a 10% validation split of the training set (grid p in {0,1,2,3}, lr in {0.01 ... 1}, seeds 0-1) and then evaluated on the test set with disjoint seeds 2-6 (`experiments/rate_selection.py`). A wider grid (p up to 5, lr down to 0.001) selects the same cells for the three conditions, so the optima are not grid-boundary artifacts (that check was run for the conditions only).

| condition / rule | default rates (seeds 0-4) | selected (p, lr) | test accuracy (seeds 2-6) | forgetting | old takeover |
| --- | --- | --- | --- | --- | --- |
| `fixed_linear` | 0.176 | 3, 0.3 | 0.430 +/- 0.016 | 0.602 | 0.569 |
| `expanding` (= `ordinary`) | 0.246 | 3, 0.01 | 0.640 +/- 0.013 | 0.379 | 0.331 |
| `fixed_capacity` | 0.256 | 3, 0.01 | 0.642 +/- 0.012 | 0.381 | 0.287 |
| Strategy B, 3 refs | 0.538 | 2, 0.3 | 0.647 +/- 0.006 | 0.370 | not recorded |
| `small_old_lr` | 0.260 | 2, 1 | 0.627 +/- 0.047 | 0.370 | not recorded |
| `frozen_old` | 0.274 | 3, 0.01 | 0.577 +/- 0.012 | 0.454 | not recorded |
| `ordinary_no_replay` | 0.192 | 1, 0.01 | 0.212 +/- 0.000 | 0.899 | not recorded |

- **Expansion helps.** Without it the model is capacity-limited rather than rate-limited (0.430 with tuned rates versus 0.640).
- **Strategy B has no demonstrated advantage.** Its large advantage at default rates (0.538 versus 0.246) came from an under-trained shared-rate baseline; with rates selected per rule, B (0.647) and `ordinary` (0.640) differ by less than one standard deviation of `ordinary`.
- **`frozen_old` is worse, replay is the dominant ingredient** (0.212 without it), and `small_old_lr` matches `ordinary` with more variance; its selected rate lies on the upper edge of the grid.
- **Strategy B is a stability-plasticity dial** (3 seeds, default rates): with 1, 3, 10, 30, 100, 200 reference inputs per old class, old takeover is 0.648, 0.379, 0.169, 0.055, 0.009, 0.001 and current-experience accuracy is 0.819, 0.669, 0.450, 0.161, 0.005, 0.000.

**Where old-class accuracy goes (default rates, five seeds).** After training, a **new class** can win an old-class input (takeover) or the prediction can move **between old classes** (flip); the accuracy matrix cannot separate the two (see the diagnostics section above).

| rule | final mean acc | forgetting | old flips | old takeover |
| --- | --- | --- | --- | --- |
| `ordinary_no_replay` | 0.192 | 0.925 | 0.139 | 0.996 |
| `ordinary` (= `expanding`) | 0.246 | 0.858 | 0.385 | 0.961 |
| `small_old_lr` | 0.260 | 0.838 | 0.078 | 0.940 |
| `frozen_old` | 0.274 | 0.820 | 0.021 | 0.925 |
| Strategy B, 3 refs | 0.538 | 0.405 | 0.074 | 0.381 |

- **At default rates the dominant failure is takeover, not damage to the old function:** for every rule except Strategy B, 92-100% of old test inputs end up predicted as a new class. `frozen_old` moves old logits among themselves least (flips 0.021) yet loses as much accuracy to takeover as the others, so freezing old coefficients does not stop a new class from outscoring them.
- **Tuned rates reduce takeover but do not remove it** (0.29-0.33 for the polynomial conditions, 0.57 for `fixed_linear`); forgetting and takeover are of the same order, but they are different quantities (best-minus-final per experience versus a per-step fraction), so this is not an identity.
- **Class imbalance** (200 stored examples per old class against about 6,000 per new class) is tested in the sampling-policy results below.

**Sampling policy (Split MNIST, `expanding`, lr 0.1, 3 epochs, seeds 0-4, same updates and memory).** Final mean accuracy / newest-experience accuracy / forgetting / old takeover:

| rule | `shuffled` | `class_balanced` |
| --- | --- | --- |
| `ordinary` | 0.246 / 0.902 / 0.858 / 0.961 | 0.675 / 0.452 / 0.073 / 0.071 |
| `ordinary_no_replay` | 0.192 / 0.902 / 0.925 / 0.996 | 0.192 / 0.902 / 0.925 / 0.996 |
| `small_old_lr` | 0.260 / 0.897 / 0.838 / 0.940 | 0.663 / 0.488 / 0.102 / 0.077 |
| `frozen_old` | 0.274 / 0.893 / 0.820 / 0.925 | 0.667 / 0.493 / 0.102 / 0.077 |
| Strategy B, 3 refs | 0.538 / 0.683 / 0.405 / 0.381 | 0.630 / 0.186 / 0.053 / 0.058 |

- **Budget:** both policies make the same number of updates (315 in the last experience); `shuffled` processes exactly the 40,200 available examples and `class_balanced` 40,320 (+0.3%, because every batch is full). Both counts are stored per experience.
- **The imbalance is a major cause of takeover and forgetting** (the untested explanation above is now tested): balancing the batches removes most takeover. It costs plasticity, so neither policy is best on both axes. Without replay there is nothing to balance.

**Per-degree rate vectors (test seeds 2-6; rates and vectors chosen on validation seeds 0-1).** Accuracy +/- std over seeds / newest-experience accuracy / forgetting; "shared" is the validation-chosen single rate (grid up to 30), "selected" the vector chosen by the rule above:

| strategy | sampling | shared | selected vector `[d0, d1, d2, d3]` |
| --- | --- | --- | --- |
| `ordinary` | balanced | 3: 0.771 +/- 0.003 / 0.708 / 0.098 | `[0.75, 0.75, 0.75, 12]`: 0.776 +/- 0.005 / 0.730 / 0.101 |
| `small_old_lr` | balanced | 3: 0.764 +/- 0.002 / 0.697 / 0.102 | `[3, 3, 3, 12]`: 0.780 +/- 0.003 / 0.729 / 0.102 |
| `frozen_old` | balanced | 3: 0.761 +/- 0.003 / 0.694 / 0.103 | `[3, 3, 3, 12]`: 0.776 +/- 0.005 / 0.726 / 0.104 |
| Strategy B | balanced | 3: 0.765 +/- 0.003 / 0.670 / 0.093 | `[12, 0.75, 0.75, 3]`: 0.759 +/- 0.002 / 0.650 / 0.085 |
| `ordinary` | shuffled | 30: 0.604 +/- 0.082 / 0.881 / 0.372 | `[30, 30, 7.5, 30]`: 0.597 +/- 0.067 / 0.881 / 0.382 |
| `small_old_lr` | shuffled | 10: 0.560 +/- 0.047 / 0.898 / 0.462 | `[2.5, 2.5, 10, 40]`: 0.591 +/- 0.033 / 0.906 / 0.435 |
| `frozen_old` | shuffled | 30: 0.475 +/- 0.084 / 0.862 / 0.541 | `[7.5, 30, 30, 120]`: 0.522 +/- 0.071 / 0.901 / 0.501 |
| Strategy B | shuffled | 30: 0.645 +/- 0.007 / 0.919 / 0.373 | `[120, 120, 120, 120]`: 0.650 +/- 0.016 / 0.896 / 0.334 |

- **Balanced sampling is where rate choices are stable; shuffled results vary by seed.** With balanced sampling every strategy reaches 0.76-0.78 at a shared rate of 3. For `ordinary`, `small_old_lr` and `frozen_old`, a 4x larger cubic rate adds about 0.005-0.016 accuracy and about 0.03 newest-experience accuracy at unchanged forgetting; this agrees across three strategies and five seeds but is small, comes from one dataset and one projection, and for `ordinary` is within one standard deviation. Strategy B gains nothing. Shuffled sampling needs very large rates (10-30), and the seed spread (0.03-0.08) is as large as the differences between vectors, so no shuffled difference between vectors is established; the `[120, 120, 120, 120]` vector for Strategy B is a uniform rate, not a per-degree effect.
- The earlier grid ended at 1 and every cell chose its upper edge; the grid now reaches 30 (a separate check, `results/per_degree/boundary_check_shared_rate*.json`, is kept). Per-degree vectors help much less once the shared rate is tuned, so the large gains seen against the old grid-edge reference were mostly a larger effective step.
- **No strategy is shown optimal.** Under balanced sampling `ordinary`, `small_old_lr` and `frozen_old` differ by about 0.005 after tuning; Strategy B has the least forgetting among them but the weakest newest-experience accuracy.
- **Strategy count.** The class-incremental code has five strategies (`ordinary`, `ordinary_no_replay`, `small_old_lr`, `frozen_old`, `strategy_b`). The eight of older branches were `ordinary_no_replay`, `ordinary`, `frozen_old`, `small_old_lr`, `strategy_a_all`, `strategy_a_new`, `strategy_b_all` and `strategy_b_new`; `main` removed the two Strategy A entries as identical to `ordinary` and `frozen_old`, leaving six distinct rules. Only `strategy_b_new` (constraint on the new coefficients only) has no class-incremental counterpart, and it was not added here.
- Limits: factors 0.25 and 4 only, one-at-a-time perturbations plus one greedy follow-up, 3 epochs, one dataset and projection, selection on two validation seeds, and a seed spread that understates dataset variability.

### Supporting binary sinusoid (five seeds, 300 epochs, shared rate lr 0.5)

| | degree 1 throughout | grow 1..10 | fixed degree 10 |
| --- | --- | --- | --- |
| final mean accuracy / forgetting | 0.910 +/- 0.003 / 0.043 | 0.965 +/- 0.004 / 0.008 | 0.970 +/- 0.005 / 0.011 |

- **Capacity matters; gradual growth does not.** Growing and fixed degree 10 are within one standard deviation at the end; the fixed model is better after experience 1 (0.988 versus 0.959) and they converge by experiences 8-10, because zero-initialised expansion preserves the function and replay covers all experiences.
- **The shared rate is not converged** (`ordinary`: 0.967 at 300, 0.979 at 1,000, 0.991 at 10,000 epochs). A single per-feature setting, `lr_feature_power = 1` with `lr = 4`, which was picked by hand and not selected on held-out data, reaches 0.992 for growing and 0.991 for fixed degree 10 in 300 epochs.
- **Update rules (shared rate):** `ordinary` 0.965, `small_old_lr` 0.916, `frozen_old` 0.842, Strategy B (3 refs per old experience) 0.840, `ordinary_no_replay` 0.866. Strategy B fits the sine curve poorly because of its constraint, not optimisation: pinning `g` at reference inputs leaves little usable freedom in the current strip, and 10x the epochs gives 0.839 +/- 0.031. More references lower accuracy monotonically (0.965 ordinary, 0.912, 0.848, 0.840 for 0, 1, 2, 3 references). `strategy_b_new` has no free dimensions from experience 2 on and does not learn.

### Decision and limits

Keep `ordinary` (replay, all coefficients trainable) with per-feature rates as the default; Strategy B is kept only as a controllable preservation diagnostic. Limits: Split MNIST is one dataset with one random projection and 3 epochs; the selected settings are close to others on the grid (for `expanding`, validation 0.630 at p = 3, lr 0.01 versus 0.625 at p = 2, lr 0.1); Strategy B was selected with 3 references only and the wide-grid check covers only the conditions; per-feature rates were not tested for Strategy B on the binary task, whose trainer requires one shared rate. A genuinely nonlinear ten-class synthetic dataset, where capacity should matter, is not yet implemented.

## Layout

```text
src/polynomial_cl/
  features.py   explicit polynomial map; SHARED by both settings: monomial_exponents (any number of
                variables, prefix-stable order) and append_zero_coefficients (zero-init expansion)
  metrics.py    SHARED: accuracy, forgetting (best minus final), mean/std over seeds, tables;
                binary-only: old-score change, prediction flips, parameter changes
  data.py, model.py, training.py, replay.py, plotting.py     binary sinusoid experiment
  class_incremental/   class-incremental protocol
    data.py         ten-class ring, projected Split MNIST, validation hold-out
    mnist.py        offline-capable MNIST loader
    features.py     degree-balanced scaling of the shared monomials (what is specific here)
    training.py     softmax head with class masking, update masks, null-space projection
    experiment.py   protocol, conditions, update rules, indicators, summaries, joint reference
    degree_rates.py per-degree rate vectors (validation, degree of every feature)
    degree_study.py sampling comparison and the validated per-degree rate study
experiments/    command-line entry points (argument parsing, configuration, writing JSON):
                class_incremental.py, rate_selection.py, per_degree_lr.py, synthetic_degree_ablation.py
notebooks/      01 joint training (reference); 02 continual learning: Part I class-incremental
                (primary), Part II binary sinusoid (supporting)
tests/          see below
```

Reusable experiment logic lives in the package (so a notebook can import `run_comparison()` without a command-line script); `experiments/` holds executable workflows. What is shared is mathematical infrastructure (monomials, zero expansion, forgetting, mean/std aggregation); what is kept separate is the learning protocol and objective (binary logistic regression with per-experience replay versus a masked ten-class softmax with per-class memory) and the feature scaling (raw monomials versus degree-balanced).

## Tests

`python -m pytest` runs the fast suite (a few seconds); `python -m pytest -m slow` additionally executes every notebook top to bottom in a clean kernel (minutes; Split MNIST cells run only if `data/mnist` exists). The same two commands run in CI.

- Shared and binary code: `test_features` (ordering, counts, known values, the shared enumeration and zero-expansion helpers), `test_expansion` (exact copy, zeros, identical scores and probabilities on train/eval/random/outside inputs, invariant checked every step of the loop), `test_training` (finite-difference gradient, loss decrease, reproducibility, frozen coefficients, zero-init learning, per-coefficient learning rates, preservation tests, preset-override guard), `test_replay`, `test_data`, `test_plotting`.
- Class-incremental code: `test_class_incremental` and `test_mnist` (features, expansion invariant and its guard, masked evaluation, forgetting definition, takeover versus flip diagnostics, determinism, offline Split MNIST).
- Per-degree rates and sampling: `test_per_degree` (vector validation, per-group parameter updates, zero rates, masks, Strategy B, expansion, class counts, no future-class draws, reproducibility, accuracy/forgetting).
- Workflows: `test_experiment_scripts` (the command-line tools) and `test_notebooks` (every notebook compiles, is committed without outputs, and executes cleanly under `-m slow`).

## Reproducibility

Everything is driven by explicit seeds (`np.random.default_rng([seed, experience, stream])` for the binary task; `default_rng(seed)` for task plans and `seed + experience` for training shuffles in the class-incremental runner). Re-running a notebook gives the same numbers. Generated experiment files go to `results/` (git-ignored); the tools need the package installed (`pip install -e .`).
