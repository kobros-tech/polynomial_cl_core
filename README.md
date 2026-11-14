# polynomial_cl_core

A small NumPy-only research instrument for one hypothesis:

> Incremental polynomial feature expansion, combined with appropriate knowledge-preservation mechanisms (replay, per-coefficient update rules), can improve a model's ability to learn **new classes** while retaining what it already knows.

The project is a continual-learning study first and a machine-learning library second. It does **not** assume the answer is yes. Every mechanism (zero-initialised expansion, freezing, per-coefficient learning rates, replay, a preserved old function) can be switched on and off independently, and every score, coefficient and prediction change is recorded.

> This project studies _explicit_ polynomial feature expansion, which is related to polynomial kernels. It does not implement an implicit-kernel SVM.

## Research map

| | setting | question | where |
| --- | --- | --- | --- |
| **primary** | **class-incremental learning (CIL)**: ten classes arrive in five two-class experiences, one softmax head | Does expansion (with replay and update rules) let new classes arrive without erasing old ones? | `src/polynomial_cl/continual/`, `experiments/class_incremental.py`, `experiments/rate_selection.py`, notebook 02 Part I |
| CIL protocol check | ten Gaussian classes on a ring | Do replay, class masking, accuracy matrices and forgetting behave as specified? The ring is linearly solvable, so it **cannot** reward extra capacity | same |
| CIL capacity test | Split MNIST under a fixed 16-D random projection | Does polynomial capacity help on image inputs? | same |
| supporting | binary sinusoid, spatially shifting input distribution | Nonlinear representation and capacity under distribution shift. This is **not** class-incremental learning: the label set never changes | `src/polynomial_cl/binary/` (experiences, replay, sequential loop), `models/binary.py`, `training/binary.py`, `experiments/synthetic_degree_ablation.py`, notebook 02 Part II |

Class-incremental learning and sequential binary classification under a shifting input distribution are different settings, and results from one must not be presented as results for the other. Conventional joint training (notebook 01) is only a reference point: it sees all data at once, so it has no sequence of experiences and no forgetting.

This README describes the structure, protocols and commands. It deliberately records no measured results or research conclusions: those are produced by running the notebooks and experiment scripts, and are reviewed in a separate research-results change. This README is the project's only documentation file.

## Architecture

Binary learning and class-incremental learning are different tasks, but they share one polynomial foundation:

```text
                 polynomial_cl.features              monomial_exponents: prefix-stable order of the
                         |                           monomials; append_zero_coefficients: zero-initialised
                         |                           expansion; validate_degree
           +-------------+--------------+
           |                            |
  models/binary.py             models/multiclass.py
  one score, sigmoid, 0/1      softmax head, logits, argmax over the
  labels; raw 2-D monomials    classes in play; scaled 16-D monomials
           |                            |
  training/binary.py           training/multiclass.py
           |                            |
  binary/  sequential binary   continual/  class-incremental learning
  learning under spatial shift (Split MNIST, six update strategies)
```

- **Shared foundation.** Both models enumerate monomials in the same order, so the features of degree _d_ are a prefix of those of degree _d_ + 1. A model grows by appending zero coefficients, which leaves every existing output unchanged. This expansion invariant is tested on actual outputs for both model types (`tests/test_features.py`, `tests/test_multiclass_model.py`, `tests/test_binary_expansion.py`).
- **Binary versus multiclass.** The two models have different outputs, losses and prediction rules and share no trainer or base class. The binary model is a logistic regression on one score; the multiclass model is a softmax head whose prediction only considers the classes introduced so far.
- **Multiclass and class-incremental learning.** `polynomial_cl.continual` uses the multiclass model and its trainer and never imports binary code (enforced by `tests/test_architecture.py`). The experience sequence, replay memory, update strategies and per-degree learning rates are class-incremental concerns and live there, not in the generic data or model modules.
- **Synthetic target interface.** `polynomial_cl.data.make_synthetic_dataset(target_function=..., n_samples=..., seed=..., kind=...)` takes the target geometry as an argument. `sinusoidal_target` (`y = 1 iff x2 > 0.5 sin(3 x1)`) is the default everywhere; `circular_target` (a circle of radius 0.7 at the origin) and `make_circular_target(center, radius)` give a different geometry with the same labels convention. A target only says how labels are generated; how experiences are cut from the data belongs to the experiment (`polynomial_cl.binary` cuts vertical strips).
- **Notebooks.** `notebooks/01_joint_training.ipynb` is the non-continual reference (targets, degree, training budget, Split MNIST joint training). `notebooks/02_continual_learning.ipynb` is the continual-learning study (protocol, expansion invariant, capacity, sampling and replay, the six strategies, per-degree learning rates, Split MNIST across seeds). Each section starts from a research question and states what the executed cells do and do not show.

## Setup

Only Python, NumPy, Matplotlib and Jupyter are used (no PyTorch / TensorFlow / scikit-learn / pretrained models). CPU only.

```bash
cd polynomial_cl_core
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

The notebooks call package code only and run their own experiments (reduced where the full study is expensive); they never read result files from `results/`. Notebook 01 is the joint-training reference (synthetic sinusoid and circle targets, degree and training budget, Split MNIST joint training). Notebook 02 is the continual-learning notebook (its configuration cell sets `sampling` explicitly to `class_balanced`, the library default; `shuffled`, the protocol of the earlier experiments, appears only as the explicit contrast in I.4 and in the I.7 transfer check): Part I is class-incremental learning (protocol, expansion invariant, capacity, sampling and replay, the six update strategies, per-degree learning rates, degree control, stability across seeds) and Part II is the separate binary sequential experiment. The Split MNIST cells of both notebooks are skipped, with a message, when `data/mnist` is absent; the command-line tool downloads the archives (see below). Both notebooks fall back to `../src` if the package is not installed.

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
python experiments/class_incremental.py --dataset split_mnist --sampling shuffled --output results/split_mnist_shuffled.json   # the earlier (shuffled) protocol
```

### Cost of a run: diagnostics are opt-in, and timings

A class-incremental run does two kinds of work. The _learning_ work is the task plan, replay, mini-batch updates and the accuracy matrix after every experience. The _observing_ work only measures what happened: the two full training-set loss passes per experience, the per-degree-group statistics (`degree_diagnostics`), the old-class logit, flip and takeover indicators, and the null-space rank bookkeeping. On Split MNIST the observing work is roughly half of a run's time, so it is **off by default** everywhere and has to be asked for:

```python
run = run_condition(
    "expanding", data, tasks, memory, seed=0
)  # accuracy and forgetting only
run = run_condition(
    "expanding", data, tasks, memory, seed=0, diagnostics=True
)  # also the observation
run = run_comparison(data, seeds, diagnostics=True)  # the same for all three conditions
```

```bash
python experiments/class_incremental.py --dataset split_mnist --output results/split_mnist.json
python experiments/class_incremental.py --dataset split_mnist --diagnostics --output results/split_mnist_diagnostics.json
```

The same `diagnostics=` argument exists on `run_strategy_comparison`, `run_sampling_comparison`, `run_per_degree_study` (passed on to every selection and test run), `select_rate_vectors`, `evaluate_vectors` and `run_epoch_robustness`, and the same `--diagnostics` flag on `class_incremental.py` and `per_degree_lr.py`. `rate_selection.py` reads only accuracy and forgetting, so it has no such flag. The flag is stored in each run (`"diagnostics"`) and in the protocol of a result file.

`diagnostics=True` only observes. The task plan, replay, sampling, update rule, number of updates and evaluation are the same either way, so the accuracy matrix, per-class matrix, forgetting and `sampling_by_task` are bit-for-bit identical; `tests/test_lean_mode.py` checks this for every condition and update rule. A run without diagnostics also does none of the observation work, not even building the stacked training set that the loss passes read. The observation fields (`degree_diagnostics`, `reference_diagnostics_by_task`, `train_loss_by_task`, `l1_update_by_task`, `n_trainable_by_task`, `null_dim_*`, `old_*_by_task`, `ref_logit_change_by_task`, `mean_old_logit_change`, `mean_old_flip_fraction`, `mean_old_takeover_fraction` and `mean_loss_reduction`) are **absent** from such a record rather than `None`, so reading one fails loudly instead of feeding a `NaN` into a summary. Summaries (`run_strategy_comparison`, `run_sampling_comparison`, `vector_summary`, `study_rows`) cover the diagnostic metrics only for runs that recorded them. The helpers that need diagnostics (`old_class_rows`, `strategy_diagnostic_rows`, `degree_group_rows`, `plot_cil_takeover_flips`, `mean_degree_diagnostic`) raise a clear `ValueError` for a run made without them. Records written before the flag existed count as full records.

To see where time goes, pass a dict as `timings`; it is filled in place (the record itself stays deterministic):

```python
timings = {}
run_condition("expanding", data, tasks, memory, seed=0, timings=timings)
# {'training_seconds': ..., 'evaluation_seconds': ..., 'diagnostics_seconds': ...,
#  'other_seconds': ..., 'total_seconds': ...}
```

`total_seconds` covers the whole call from just after argument validation until the record is complete, and is the sum of the other four. `diagnostics_seconds` is all observation work (including the stacked training set when only the loss passes need it) and is exactly `0.0` without diagnostics; `other_seconds` is everything else, such as allocation, forgetting and building the record. Timings are never part of the record.

The polynomial design matrix (969 columns for 16 variables at degree 3) is built once per dataset by `make_feature_data`. `scaled_polynomial_features` computes each variable's powers once per chunk and forms every monomial as a product of gathered columns, instead of materialising a `(rows, 969, 16)` array. The result is bit-for-bit the previous one (it still uses the same `np.power` float32 loop, which is why even `x ** 1` is not replaced by `x`), and on the 60 000-row Split MNIST training set construction dropped from about 17 s to well under a second in the reference environment. The work is memory-bound (`np.power` itself is a small part), so the default chunk of 256 rows keeps the temporaries in cache; the result does not depend on the chunk size.

### Why old-class accuracy drops: takeover versus flips

Two different mechanisms lower the accuracy on old classes, and the accuracy matrix alone cannot separate them. A record made with diagnostics (`diagnostics=True`, `--diagnostics`) carries both:

- `old_takeover_fraction_by_task`: after training, **all** classes seen so far compete (as in the accuracy matrix) and a **new** class wins an old-class test input. This can happen although the old logits barely move.
- `old_flip_fraction_by_task`: only old classes compete, and the prediction changes between two old classes. This measures how much the old decision function itself moved.

`old_logit_change_by_task` is the largest change of any old-class logit on old-class test inputs. Large takeover with small flips points to a stability-plasticity imbalance between old and new classes; large flips point to damage to the old function.

### Update rules (`--experiment strategies`, `expanding` schedule)

`ordinary` (replay, everything trains), `ordinary_no_replay`, `frozen_old`, `small_old_lr` (old coefficients get 0.1 x the gradient) and `strategy_b` (every gradient is projected onto the null space of the design matrix at `--n-ref-per-class` stored inputs per old class, so the logits at those inputs cannot change). `strategy_b_new` is the binary `strategy_b_new`: old coefficients are frozen as in `frozen_old`, and the old classes' outputs may move only through newly added feature rows, inside the same null space; the new classes' columns use the full null space. A coefficient is _old_ if its feature row existed before the current experience **and** its class column was already introduced; new feature rows and the new classes' columns always train.

```bash
python experiments/class_incremental.py --dataset split_mnist --experiment strategies --diagnostics --output results/split_mnist_strategies.json
python experiments/class_incremental.py --dataset split_mnist --experiment strategies --n-ref-per-class 10 --diagnostics --output results/split_mnist_b10.json
```

With diagnostics on, every record carries learning indicators that exist for _all_ rules: training loss before and after each experience, L1 size of the update, number of trainable coefficients, and the old-class change, flip and takeover diagnostics above. `null_dim` exists only for the two Strategy B variants and is a constraint diagnostic, not a general learning indicator. It is the free dimension for the old-class output columns (`strategy_b`: one basis shared by every column; `strategy_b_new`: the dimension left in the newly added feature rows, which is 0 in an experience that adds none), and `null_dim_new_class_columns` is the free dimension for the new classes' columns. `n_trainable` counts the directions that can actually move after the learning-rate vector is applied (zero-rate groups excluded).

### Per-feature learning rates and validated rate selection

`--feature-power p` gives feature i the step size `lr * (max_rms / rms_i) ** p` (p = 0: one shared rate; p = 1: diagonal preconditioning). `--validation-fraction 0.1` evaluates on a held-out part of the training data instead of the test set. `experiments/rate_selection.py` sweeps `(p, lr)` on that split, picks the best cell for each condition (or rule, with `--experiment strategies`) by validation accuracy only, and then reports the test result for the selected cells:

```bash
python -u experiments/rate_selection.py --dataset split_mnist --output results/rate_selection_conditions.json   # minutes
python -u experiments/rate_selection.py --dataset split_mnist --experiment strategies --output results/rate_selection_strategies.json
```

The seeds used for selection (`--select-seeds`, default 0 1) and for the reported test result (`--test-seeds`, default 2-6) must be disjoint: the script refuses overlapping sets, because otherwise the same stochastic runs would both choose the rates and be reported as the test estimate. Check where each selected cell lies in the grid: a selection on the boundary (`--powers`, `--lrs`) means the optimum may be outside it; widen the grid and rerun.

### Sampling policy and per-degree learning rates

Two further mechanisms are available for every update rule (`polynomial_cl.continual`).

**Sampling policy** (`sampling="class_balanced"` or `"shuffled"`; the default is `polynomial_cl.continual.sampling.DEFAULT_SAMPLING`, `class_balanced`, and the command-line tools take `--sampling`). `shuffled` is the behaviour of the earlier experiments: current and replay examples are shuffled together, so in the last experience a new class contributes about 5,900 examples per class against 200 stored for an old class. Pass `sampling="shuffled"` or `--sampling shuffled` to reproduce those runs; notebook 02 runs class-balanced throughout and uses `shuffled` only where it contrasts the two policies. The default was set from the paired comparison of `--experiment sampling` below (same seeds, strategies, rates and replay memory), and the joint-training reference is unaffected (it has no replay). `class_balanced` builds every batch with (almost) equal numbers of every class seen so far; each class is read through its own shuffled cycle, a class with fewer examples than its share is repeated and never padded from another class, and classes with no examples (no replay) do not take part. Both policies make the same number of updates per epoch and use the same task plan and memory; the test set is never used. Every record stores the pool size, the number of examples actually drawn and the mean batch composition per class (`sampling_by_task`).

**Per-degree learning rates** (`learning_rate=` a scalar or a vector). This is a different mechanism from `feature_power`, which scales the step of every _individual feature_ by its RMS value; the per-degree vector sets one step size per degree _group_. Neither decides which polynomial dimensions exist (that is the degree schedule), and the two cannot be combined: setting both is rejected. The vector has one entry per polynomial degree group and position k is degree k (0: constant, 1: linear, 2: quadratic, 3: cubic); entries must be finite and non-negative, 0 freezes the group, and an all-zero vector is rejected. All groups train together in one run with their own rate in every update. A group that appears through expansion uses its own entry from the first update. The vector composes with update masks (`small_old_lr`, `frozen_old`) and with Strategy B, whose null-space projection is taken in the rate-scaled coordinates so the reference logits stay fixed; it cannot be combined with `--feature-power`, which is a different mechanism (a step size per _feature_ derived from the measured feature scale). The binary trainer's `_as_vector` is per _coefficient_ and is unrelated. Each record carries `degree_diagnostics`: per experience and degree group the rate, coefficient and trainable counts, raw-gradient and applied-step size, actual coefficient change (L1 and relative), the group's logit contribution and its contribution to old-class logit change.

```bash
python experiments/per_degree_lr.py --dataset split_mnist --experiment sampling --output results/per_degree/sampling_e3_lr0.1.json
# one JSON per (strategy, sampling), the full study (notebook 02 runs a reduced version itself and does not read these files). The whole study, resumable (skips finished cells):
bash experiments/run_per_degree_study.sh
python experiments/per_degree_lr.py --dataset split_mnist --experiment per-degree --strategies ordinary --samplings class_balanced --output results/per_degree/cell_ordinary_class_balanced.json
```

Protocol of the per-degree study: every candidate is measured on a 10% validation split (seeds 0-2) for final mean accuracy, mean forgetting and newest-experience accuracy, so accuracy alone never decides. The shared rate is chosen from {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30}; one degree group at a time is then made 0.25x, 0.5x, 2x or 4x of it; one follow-up vector takes each group's best factor; and all vectors are run on the test set with disjoint seeds 3-9. The Pareto front on (accuracy, forgetting) is recorded for every cell; where one vector is needed, the front member with the largest accuracy minus forgetting is marked `selected` (optionally with a floor `--min-current-accuracy` on newest-experience accuracy). `selected` is a convenience for reading results, not a claim of optimality. Joint (shuffled, all-class) training on the same features is available as `run_joint_reference` and is shown in notebook 01; it has no forgetting.

### How to read the results

- **Training budget and rates.** The defaults (3 epochs, one shared rate 0.1) are a fixed historical protocol, not a tuned or converged setting. Do not treat default-rate results as evidence that one condition is better. Compare conditions only at rates selected **per condition on validation data**, over several seeds, and repeat with several `--epochs` values.
- **The ring is linearly solvable.** Equal isotropic Gaussian classes have linear Bayes boundaries, so that dataset checks the machinery and cannot reward extra capacity; Split MNIST and the binary sinusoid are the capacity tests. With the default 160 training samples per class the replay memory (200 per class) holds _all_ earlier data on the ring, whereas it is a genuine 200-example budget on MNIST. A genuinely nonlinear ten-class synthetic dataset is not yet implemented.
- **Seeds.** For the class-incremental runs the data, test sets and projection are fixed; seeds change shuffling and replay selection only, so the spread across seeds understates dataset-level variability.

## Supporting experiment: binary sinusoid

Inputs live in `[-1, 1]^2`. The true rule is `y = 1 iff x2 > 0.5 sin(3 x1)` (a nonlinear boundary). Experience _k_ of 10 samples `x1` from the _k_-th vertical strip and `x2` uniformly, so each experience reveals a different part of the boundary. This is sequential binary learning under a spatially shifting input distribution, **not class-incremental learning**. Each experience has its own training split and its own fixed test split (independent random streams). The model is `f_d(x) = w . phi_d(x)` with the bias as the coefficient of the constant feature. Starting at degree 1, the degree grows by one per experience (configurable via `degrees=`). Expansion appends zero coefficients and the loop **asserts** `f_{d+1}(x) = f_d(x)` for scores and probabilities on training, test, random and out-of-region inputs before any training.

```bash
python experiments/synthetic_degree_ablation.py --seeds 0 1 2 3 4 --output results/synthetic_degree_ablation.json
```

Within each seed the same generated experiences, replay budget, epochs and optimiser are used for `expanding` (degree 1..10), `fixed_initial_degree` (degree 1) and `fixed_capacity` (degree 10 from the start).

### Synthetic target functions and the joint-training reference

The target function is a parameter of the synthetic generator (`polynomial_cl.data.synthetic`). A target maps inputs of shape `(n, 2)` to a signed score and the label is `score > 0`. `sinusoidal_target` is the rule above and the default everywhere; `circular_target` (class 1 inside a circle of radius 0.7 around the origin; `make_circular_target(center, radius)` for others) is a geometry demonstration, not a benchmark. `make_synthetic_dataset(target_function, n_samples, seed, kind="train" | "test")` draws one reproducible dataset (the two kinds use independent random streams), and `make_experiences(..., target_function=...)` cuts the same target into vertical-strip experiences. How experiences are built is separate from the target, so the same target serves joint and sequential binary training. Joint training (all data at once, no forgetting) is run by `experiments/joint_training.py`:

```bash
python experiments/joint_training.py --target sinusoidal --output results/joint_sinusoidal.json
python experiments/joint_training.py --target circular --binary-degrees 1 2 3 --output results/joint_circular.json
python experiments/joint_training.py --experiment ten_class --dataset split_mnist --output results/joint_split_mnist.json
```

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

Hyper-parameters of the binary experiment (`lr=0.5`, 300 epochs per experience, replay `m=100`, capacity 100) were fixed before the continual experiments were run and were not tuned afterwards; all strategies share them.

## Layout

```text
src/polynomial_cl/
  features.py       SHARED polynomial primitives: monomial_exponents (any number of variables,
                    prefix-stable order), append_zero_coefficients (zero-init expansion), validate_degree
  metrics.py        SHARED: forgetting (best minus final), mean/std over seeds, tables, plus
                    array-level measures of the binary experiment (accuracy, old-score change, flips)
  models/
    binary.py       binary logistic model on the raw float64 2-D monomial map
    multiclass.py   softmax-head semantics: logits, class-eligible prediction, zero-init expansion
  training/
    binary.py       gradient descent for the binary model (masks, per-coefficient rates, null space)
    multiclass.py   softmax head trainer: class masking, update masks, null-space projection;
                    mini-batch sampling is injected (never imports continual/)
  data/
    synthetic.py    target functions (sinusoid, circle), sampling/labelling, the ten-class ring
    mnist.py        offline-capable MNIST loader and the fixed random projection
  binary/           sequential BINARY learning under spatial shift (not class-incremental)
    data.py         vertical-strip experiences built on a target function
    replay.py       per-experience replay buffer
    experiment.py   the expansion-aware sequential loop, its strategies, joint study, summaries
  continual/        CLASS-INCREMENTAL learning (never imports the binary code)
    features.py     degree-balanced float32 scaling of the shared monomials
    experiment.py   protocol, conditions, indicators, summaries, joint reference
    strategies.py   update restrictions of the six strategies (masks and null spaces)
    degree_rates.py per-degree rate vectors (validation, degree of every feature)
    replay.py       replay memory: per-class draw and the stored indices every use reads
    sampling.py     shuffled / class-balanced mini-batch policies handed to the trainer
    degree_study.py sampling comparison and the validated per-degree rate study
  plotting/         synthetic.py (binary plots), continual.py (CIL plots); imported on demand
experiments/        command-line entry points (argument parsing, configuration, writing JSON);
                    common.py holds the argument/dataset helpers they share:
                    class_incremental.py, rate_selection.py, per_degree_lr.py,
                    synthetic_degree_ablation.py, joint_training.py, run_per_degree_study.sh
notebooks/          01 joint training (reference); 02 continual learning: Part I class-incremental
                    (primary), Part II binary sequential learning (supporting, not CIL)
tests/              see below
```

Run the scripts as `python experiments/<name>.py` from the repository root. Reusable experiment logic lives in the package (so a notebook can import `run_comparison()` without a command-line script); `experiments/` holds executable workflows and no entry point imports another. What is shared is mathematical infrastructure (monomials, zero expansion, forgetting, mean/std aggregation); what is kept separate is the learning protocol and objective (binary logistic regression with per-experience replay versus a masked ten-class softmax with per-class memory), the model (a binary model class versus a plain coefficient array) and the feature scaling (raw monomials versus degree-balanced). The class-incremental code has no dependency on the binary model or trainer.

## Tests

`python -m pytest` runs the fast suite (a few seconds); `python -m pytest -m slow` additionally executes every notebook top to bottom in a clean kernel (minutes; Split MNIST cells run only if `data/mnist` exists). The same two commands run in CI.

- Shared and binary code: `test_features` (ordering, counts, the shared enumeration and zero-expansion helpers, output-level expansion invariance for binary and multiclass), `test_binary_features`, `test_binary_expansion` (exact copy, zeros, identical scores and probabilities on train/eval/random/outside inputs, invariant checked every step of the loop), `test_binary_training` (finite-difference gradient, loss decrease, reproducibility, frozen coefficients, zero-init learning, per-coefficient learning rates, preservation tests, preset-override guard), `test_binary_replay`, `test_binary_data`, `test_binary_plotting`.
- Synthetic data: `test_synthetic_data` (sinusoidal experiences bit-identical to the pre-refactor generator, circular target geometry, reproducibility, independent train/test streams, label noise), `test_joint_studies`.
- Class-incremental code: `test_class_incremental` and `test_mnist` (features, expansion invariant and its guard, masked evaluation, forgetting definition, per-class accuracy, takeover versus flip diagnostics, determinism, offline Split MNIST), `test_multiclass_model`, `test_multiclass_training`, `test_strategies` (the update restrictions of the six strategies), `test_class_incremental_plotting`, `test_reporting_rows`.
- Execution efficiency: `test_lean_mode` (feature construction bit-identical to the previous formula, matching an independent float64 evaluation, correct at chunk boundaries and free of the large temporary array; diagnostics off by default in every function that takes them and opt-in on every command line; `diagnostics=True` learns exactly what the default run learns for every condition and update rule; a default run skips every observing helper and records none of its fields; summaries, study tables and the degree studies for both kinds of run; timings only on request and attributed to the operation they name; the helpers refusing a run without diagnostics) and `test_checkout` (the tests and the experiment scripts import this checkout, and a failing script shows what it printed).
- Per-degree rates and sampling: `test_per_degree` (vector validation, per-group parameter updates, zero rates, masks, Strategy B, expansion, class counts, no future-class draws, reproducibility, accuracy/forgetting).
- Replay and sampling: `test_replay_sampling` (memory draw and stored-index order, policy-to-sampler mapping, the trainer's injected sampler).
- Structure: `test_architecture` (import graph without cycles, class-incremental code independent of the binary code, binary/model/trainer code independent of the class-incremental code, entry points not importing each other, no stale imports in source, tests, experiments and notebooks).
- Workflows: `test_experiment_scripts` (the command-line tools) and `test_notebooks` (every notebook compiles, has unique valid cell ids, is committed without outputs, reads no result file, and executes cleanly under `-m slow`).

## Reproducibility

Everything is driven by explicit seeds (`np.random.default_rng([seed, experience, stream])` for the binary task; `default_rng(seed)` for task plans and `seed + experience` for training shuffles in the class-incremental runner). Re-running a notebook gives the same numbers. Generated experiment files go to `results/` (git-ignored); the tools need the package installed (`pip install -e .`).

## Degree-control experiment

### Mechanism

`run_condition(..., degree_control_factors=[m0, m1, m2, m3])` multiplies the per-degree learning rates by factors in [0, 1]: the effective rate of degree `d` is `eta_d * m_d`, and 0 freezes that degree. The result is one rate per feature, `d_j = eta_deg(j) * m_deg(j)` (`degree_rates.diagonal_learning_rates`), applied to the diagonal step every strategy already uses; it is never materialised as a matrix. A feature that appears through expansion takes the rate of its own degree from its first update, and all factors equal to 1 reproduce the uncontrolled run bit for bit.

Compatibility with `feature_power`: a control that changes anything (some factor below 1) cannot be combined with `feature_power > 0` and raises `ValueError`. An all-ones control is the identity, is treated as no control, and is accepted with `feature_power`. A per-degree rate _vector_ (`learning_rate=[...]`) is rejected together with `feature_power` whether or not a control is given.

Strategy B computes its null-space projection for the controlled step, so what it preserves is the logits on its _selected reference examples_. That is not the same as retaining the old-data distribution: the old-class accuracy can still fall, and the experiment below measures it.

Degree control is an experimental mechanism for controlling degree-wise plasticity. Nothing here shows that it improves retention; see the results and their limits below.

### Candidate selection (`experiments/degree_control.py`, `polynomial_cl/continual/degree_control.py`)

Per strategy (`ordinary`, `frozen_old`, `strategy_b`, each with its fixed rate vector, class-balanced sampling), on **validation data only**:

- **A, single degrees.** The all-ones baseline, then one degree at a time with each factor in `0, 0.05, 0.1, 0.25, 0.5, 0.75` and the other degrees at 1.
- **B, a limited set, no grid search.** (i) Combinations of two or more degrees whose best single-degree candidate beat the baseline's accuracy minus forgetting _and_ is feasible. (ii) Four fixed example vectors, which are **always evaluated**, independently of how their single-degree parts scored or whether those were feasible. Each is then selected or rejected only on its own measured metrics.
- **Selection rule.** (1) The acquisition constraint below. (2) Among the feasible candidates, the Pareto front of final accuracy and forgetting. (3) The largest accuracy minus forgetting on that front. It is applied to all of A and B ("best overall"), to A alone ("best individual") and to B alone ("best multiple").
- **C, final comparison.** The baseline and the selected configurations are run once on the test set with the test seeds, paired by seed with the baseline. The selection seeds (`--select-seeds`) and test seeds (`--test-seeds`) must be non-empty, free of duplicates, and disjoint; the script and `run_degree_control_study` reject anything else before training. Test data never enters selection.

### Acquisition constraint

A candidate is feasible only if, for **every** experience, its seed-mean accuracy right after that experience was learned (`accuracy_matrix[t][t]`) is at least `--min-acquisition` (default `ACQUISITION_FLOOR = 0.5`). Without it, accuracy minus forgetting can reward a configuration that does not learn an experience on arrival and recovers it later through replay.

- The constraint is applied to the **seed mean** on validation data; an individual seed can be below the floor.
- 0.5 is an experimental constraint, not an independently established threshold: it was set in the code before the first run but with earlier poor first-experience results in view. Do not tune it against test results; if you change it, fix the new value before running.
- There is **no fallback**. If no candidate is feasible, the choice is `null` and the result lists it under `no_feasible_candidate`. The baseline is always run and reported, feasible or not.

### Reported quantities (distinct, never under one name)

- `final_mean_accuracy`, `mean_forgetting` (best accuracy an experience ever had minus its final accuracy, averaged over the experiences that have a later evaluation) and `final_current_accuracy`.
- `immediate_accuracy_by_experience`: the seed mean of the accuracy on each experience right after it was learned.
- `min_mean_immediate_accuracy`: the minimum of that list. **This is what the floor is applied to** (on validation data); it is also reported for the test runs, where it is informative only.
- `worst_immediate_accuracy`: a _per-run_ value, the worst experience within one run. Its `summary` entry is the mean over seeds of those per-run minima, which is never larger than `min_mean_immediate_accuracy` and smaller whenever the seeds disagree about which experience is worst. The floor is never applied to it.

### Result of one saved run (Split MNIST)

`python experiments/degree_control.py --dataset split_mnist --output results/degree_control.json`: 10% of the training set held out for validation, selection seeds 0-2, test seeds 3-9, 3 epochs, batch size 128, 200 replay examples per class, 3 reference examples per class, floor 0.5. About four minutes on the CPU environment used here. The run is deterministic (a rerun after the final code changes reproduced the same selections and numbers); the generated JSON is not part of the repository. The table shows the baseline and the configuration selected as best overall (test seeds, means over 7 seeds).

| Strategy | Selected factors | Accuracy (baseline → selected) | Forgetting (baseline → selected) | Second experience on arrival (baseline → selected) |
| --- | --- | --- | --- | --- |
| `ordinary` | `[0.05, 0.05, 0, 1]` | 0.7689 → 0.7595 | 0.0968 → 0.0739 | 0.867 → 0.786 |
| `frozen_old` | `[1, 0.05, 0, 1]` | 0.7782 → 0.7684 | 0.1046 → 0.0832 | 0.883 → 0.769 |
| `strategy_b` | `[1, 0.25, 0, 1]` | 0.7585 → 0.7538 | 0.0852 → 0.0669 | 0.839 → 0.716 |

For each strategy the selected configuration had lower final accuracy (by 0.0048 to 0.0099) and lower forgetting (by 0.0183 to 0.0229) than its baseline in all 7 test seeds. The worst-case `min_mean_immediate_accuracy` on the test runs was 0.706 → 0.680 (`ordinary`), 0.722 → 0.704 (`frozen_old`) and 0.656 → 0.642 (`strategy_b`); in every strategy it is the last experience. On validation data 3 of 33 (`ordinary`), 3 of 30 (`frozen_old`) and 6 of 33 (`strategy_b`) candidates were infeasible. For example `m[1,0,0.05,1]` reaches only 0.485 on the first experience under `ordinary` and `frozen_old`, and an unconstrained accuracy-minus-forgetting rule would have chosen it.

Notebook 02 (I.7) re-runs the three selected vectors on the test seeds under both sampling policies, paired by seed with their baselines. It does not re-select: `experiments/degree_control.py` fixes `class_balanced`, so the `shuffled` rows are a transfer check of vectors chosen under the other policy.

### What this does and does not show

- The results are evidence from this experiment of a **plasticity-retention trade-off**: under the acquisition constraint, the selected configurations have lower measured forgetting and slightly lower final accuracy than their baselines. They do not show that degree control improves retention, and lower forgetting is not by itself better continual learning.
- The floor is permissive. Every selected configuration passes it while learning the second experience 8 to 12 points worse on arrival than its baseline (table above), and all three set the degree-2 factor to 0. This experiment cannot separate "retains more" from "learns less strongly"; per-experience arrival accuracy has to be read together with forgetting.
- Strategy B preserves its reference logits; the table does not measure old-data retention separately, and Strategy B's forgetting is still non-zero.
- Selection ran under class-balanced sampling only; whether the trade-off holds under `shuffled` is what the notebook's I.7 transfer rows check, and they are not recorded here.
- One dataset (Split MNIST with a fixed random projection, `--projection-seed 0`), one set of base rates per strategy, one floor, selection on three seeds of a 10% validation split. The seven test seeds vary the task plan and shuffling, not the data or the projection. The consistent direction over seeds is not a significance test and not evidence of generalisation. How the result changes with the floor was not studied.
- No configuration is claimed to be optimal. The selected vectors are what this protocol chose on this validation split.
