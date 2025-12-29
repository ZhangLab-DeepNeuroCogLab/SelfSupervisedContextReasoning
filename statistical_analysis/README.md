## Statistical Analysis

## 1. Overview

This directory contains the **statistical hypothesis tests** used in the
paper to support claims about:

- Humans being **above chance** in the fribble object classification task,
- Differences between **supervised vs. self-supervised** human training conditions,
- How **SeCo** and baseline models compare to humans,
- How **context manipulations** (blur, jigsaw, contextual occlusion amount)
  affect humans and models,
- Additional analyses on **global vs. local** context and **crowding**.

The notebooks here do **not** train models or re-compute accuracies from
scratch. Instead, they:

1. Load pre-computed accuracies and confusion matrices for humans and models
   (downloaded via `download_data.sh` and/or produced by the notebooks in
   `../analysis_and_comparative_study/`),  
2. Use **bootstrap resampling** to reconstruct sampling distributions of
   accuracy from those summary statistics,  
3. Apply classical frequentist tests (t‑tests, one-way ANOVA, variance tests)
   to answer the questions above.

This directory shows the **effect sizes and p‑values**.


## 2. Files in this directory

### 2.1 `download_data.sh`

Shell script that downloads all **large data dependencies** (pre-computed
confusion matrices, rule/dummy dictionaries, model evaluation logs, etc.)
from our Google Drive archive.

- Prefer using `rclone` over `gdrive`.    
- Usage (from this directory):

  ```bash
  # Download into the current directory (default)
  ./download_data.sh
  ```

  Or directly download from these two links:
  [link1](https://drive.google.com/drive/folders/1AMGiKz185ZDMoilXVPLgohjzzhusJW__?usp=sharing)
  [link2](https://drive.google.com/drive/folders/1pe9HB7sfowRps9wvFjL7JEldkTVrO2Hq?usp=sharing)

After the download, the notebooks expect to find:

- Human confusion matrices and rule/dummy dictionaries under paths such as  
  `../fribble_v2/rules/official_rule_*.pkl` and related files,
- Model evaluation logs for fribble experiments (the multirun `.pkl` files)
  under a directory you will point to via a `root = ...` variable inside the
  notebooks.

If you place the downloaded data in a different location, **update the
relevant path variables (e.g., `root`, or base data folders) near the top of
each notebook**.


### 2.2 `Statistical Test.ipynb`

A **small, self-contained demo notebook** that introduces the bootstrap-based
testing approach on a simple case.

What it does:

- Hard-codes coarse human accuracies and trial counts for the fribble task in
  two conditions (self-supervised vs. supervised), aggregated over:
  - `normal`,  
  - `jigsaw`,  
  - `blur`,  
  - `amount` (context-occlusion) conditions.
- Reconstructs **pseudo trial-level data** from each accuracy / trial-count
  pair:

  ```python
  num_correct = round(accuracy * frequency)
  data = [1] * num_correct + [0] * (frequency - num_correct)
  ```

- Draws many **bootstrap samples** (subsamples of half the trials, without
  replacement) and computes their mean accuracy,
- Compares two bootstrap distributions using two-sample t‑tests, and
  visualizes them as histograms.

This notebook is mainly for **illustration and sanity checking**:
  it shows how we move from summary accuracy + N to a resampled
  distribution and a t‑test.


### 2.3 `statistical_test_models.ipynb`

Notebook for **model-only statistical tests** (no human data here).

It:

1. Loads **multi-run evaluation logs** for all models from a directory such as

   ```python
   root = '/path/to/fribble_exp_0shot_logs/multiruns'
   model_names = ['vicreg', 'seco', 'dino', 'orl',
                  'supervised', 'simclr', 'simsiam', 'context_encoder']
   ```

   where each file `'{root}/{model}/{rule_no}_{run_no}.pkl'` contains:
   - a summary string `"cond: accuracy"` for each context condition, and
   - a dict mapping stimulus paths → predicted labels.

2. Reconstructs **trial-level correctness arrays** (`0` = incorrect,
   `1` = correct) for each `(model, condition)` combination by comparing each
   prediction to the ground-truth label derived from the stimulus path.

3. For each model and condition:

   - Builds a **bootstrap distribution** of mean accuracy by repeatedly
     sampling half of the trials without replacement,
   - Constructs a synthetic **chance-level observer** with 25% accuracy
     (25% ones, 75% zeros) and bootstraps from that as well,
   - Uses `statsmodels.stats.weightstats.ttest_ind` to test whether each
     model’s bootstrap distribution is **significantly above chance**
     for each condition.

4. Computes simple **standard errors** as

   ```python
   se = np.std(trial_level_correctness) / np.sqrt(N)
   ```

   and stores them in `model_error_bars_all[model][condition]`. These are the
   same standard errors used for model error bars in the fribble plots in
   `analysis_and_comparative_study/model_results_plot*.ipynb`.

This notebook supports claims such as “all models are significantly above
chance under condition X” and helps quantify **how robust** each model is to
different context manipulations.


### 2.4 `statistical_test_model_vs_human_new.ipynb`

This is the **main statistical analysis notebook** comparing humans and models
on the fribble experiments (and related global/local/crowding analyses).

At a high level, it:

1. **Loads and filters human data**

   - Connects to `expFribble.db` (the Psiturk SQLite database in
     `../analysis_and_comparative_study/`),
   - Iterates over participants, parses the `datastring` JSON, and reconstructs
     test trials for each:
     - rule index (0, 2, 4),
     - training mode (`"self-supervised"` vs `"supervised"`),
     - context manipulation (fine-grained: `normal1`, `jigsaw1–3`,
       `blur1–5`, `amount1–5`),
   - Applies the **subject inclusion criteria** (based on training
     performance, attention checks, and time stamps) and aggregates:
     - total correct/total trials per condition = `(nCorrect, N)`,
     - confusion matrices per rule and condition
       (4×4 over the fribble classes).

   The same `expFribble.db` and filtering logic underpin the processed human
   summaries used in `../analysis_and_comparative_study/`.

2. **Loads model confusion matrices**

   - Reads multirun model evaluation logs, as in `statistical_test_models`,
   - Reconstructs 4×4 confusion matrices for each `(model, rule, condition)`,
   - Stores them in dicts like `model_confmat_by_rule[model][rule][cond]`.

3. **Bootstraps from confusion matrices**

   To compare humans and models properly, this notebook works at the level of
   **confusion matrices**:

   - For a given confusion matrix, it bootstraps each class separately:
     - builds a population of `1`s and `0`s from the diagonal count
       (`nCorrect`) and row sum (`N`),
     - samples half the trials, computes class-wise accuracies, and
       averages across classes,
   - Repeats this for many iterations (e.g. 10,000) and, when appropriate,
     averages across rules to yield a **bootstrap distribution of overall
     accuracy** per `(group, condition)`.

   For extreme conditions where some classes **never appear** (e.g. certain
   `amount5` or `jigsaw3` cases), those classes are **explicitly skipped** in
   the bootstrap, matching how those conditions are interpreted in the paper.

4. **Human vs. chance**

   For each human training mode and condition, it:

   - Uses the aggregated `(nCorrect, N)` to build a bootstrap distribution of
     accuracy,
   - Constructs a chance-level distribution (again at 25% accuracy),
   - Uses one-sided independent-samples t‑tests
     (`statsmodels.stats.weightstats.ttest_ind(..., alternative='larger')`)
     to assess whether humans are **significantly above chance** in each
     context condition.

5. **Supervised vs. self-supervised humans**

   Compares supervised vs. self-supervised conditions using:

   - Two-sample t‑tests on the bootstrap distributions for:
     - `normal` context,
     - individual blur levels (`blur1–blur5`),
     - various amount-of-occlusion conditions,
   - In some sections, direction-specific tests (e.g. testing whether
     supervised > self-supervised under a given manipulation).

   These tests address questions like whether **supervised training helps or
   hurts** robustness to each type of context manipulation.

6. **Models vs. chance and models vs. humans**

   - For each model and condition, compares the bootstrap distribution to the
     same 25%-accuracy chance observer (as in `statistical_test_models`).
   - Compares SeCo and individual baselines to:
     - **self-supervised humans**, and
     - **supervised humans**,

     using independent-samples t‑tests on the paired bootstrap distributions.
   - Includes targeted sections such as:
     - “compare SeCo, ORL, SimSiam, VICReg to self-supervised humans”,
     - “compare SeCo to self-supervised/supervised humans under specific
       manipulations (blur, amount, jigsaw)”,
     - “extreme cases” comparisons for baseline methods.

7. **ANOVAs over context conditions**

   For blur, amount, and jigsaw manipulations, the notebook runs one-way
   ANOVAs (via `statsmodels.stats.oneway.anova_oneway`) on the bootstrap
   distributions:

   - For humans (separately for supervised and self-supervised),
   - For each model.

   This tests whether **mean accuracy differs across conditions** within a
   given group (e.g. between `normal` and `blur1–5`). The notebook also
   inspects variance assumptions (e.g. using Levene’s test) and, where
   appropriate, comments on violations for particular models/conditions.

8. **Global vs. local vs. crowding analysis**

   In the later cells:

   - Defines sets of object–context pairs corresponding to **global context
     cues**, **local context cues**, and **crowding** configurations,
   - Extracts the relevant confusion-matrix entries for those subsets,
   - Bootstraps accuracies for:
     - humans (supervised / self-supervised),
     - SeCo and the other models,
   - Runs t‑tests to compare:
     - global vs. local performance,
     - local vs. crowding conditions,
     - humans vs. models under these subsets.

   These analyses support the claims about **what type of contextual
   information** SeCo and the baseline models are leveraging (or failing to
   leverage), relative to humans.

9. **Rule-wise comparisons**

   Finally, the notebook groups results by rule index (0, 2, 4) and performs:

   - ANOVAs across rules for humans and models,
   - Pairwise t‑tests between SeCo and humans per rule.

   This addresses whether the main effects hold **consistently across
   different context–object rules**, rather than being driven by a single
   rule.


---

## 3. How to run the statistical analyses

1. **Install dependencies**

   Use a Python 3.8+ environment with:

   - `numpy`, `scipy`, `matplotlib`,
   - `statsmodels`,
   - `mpmath` (used for numerical stability in a few places),
   - `sqlite3` and `json` (standard library).

2. **Download data**

   From this directory:

   ```bash
   ./download_data.sh
   ```

   This will populate the required folders (fribble rule dictionaries,
   confusion matrices, model evaluation logs, etc.). You may also need to
   place `expFribble.db` from `analysis_and_comparative_study/` into this
   directory (or adjust the path in `statistical_test_model_vs_human_new.ipynb`
   so that `sqlite3.connect(...)` points to it).

3. **Update any hard-coded paths**

   - In both `statistical_test_models.ipynb` and
     `statistical_test_model_vs_human_new.ipynb` there is a `root = ...`
     variable that points to the folder containing model multirun logs.
   - If you downloaded the logs into a different location, update `root`
     accordingly.
   - If your fribble dataset or rule/dummy dictionaries live in a different
     location than `../fribble_v2/rules/...`, adjust those paths near the
     top of the notebook.

4. **Run notebooks**

   - Open each notebook in Jupyter or VS Code,
   - Run all cells from top to bottom.
   - The notebooks will print **t‑statistics, p‑values, ANOVA results, and
     summary dictionaries** corresponding to the comparisons described above.

used throughout this directory and in the main paper.
