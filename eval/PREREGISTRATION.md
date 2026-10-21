# DS Cortex evaluation: pre-registration

Committed before any evaluation result was produced (21 October 2026). Changes after this commit are
listed under "Amendments" with their date and reason; the metrics, comparisons and tests below are not
changed after seeing results.

## Research questions

| RQ | Question | Evidence |
| --- | --- | --- |
| RQ1 | Does deliberation improve held-out performance and flaw catch rate? | Config D against C, and against A and B |
| RQ2 | Does evidence verification cut hallucinated claims? | D against F |
| RQ3 | Is the computed confidence calibrated? | Reliability diagram and ECE over D and E runs |
| RQ4 | Does adaptive memory help, and when does it mislead? | E against D; memory counts; the mismatched-fingerprint test |
| RQ5 | What does deliberation cost? | Time, tokens and API calls per config |

## Primary metric per task family (fixed in code before any run: `tools/router_tools.py`)

| Task family | Primary | Secondary |
| --- | --- | --- |
| Classification (balanced) | F1 macro | ROC-AUC |
| Classification (imbalanced: minority share < 0.2 or ratio > 5) | PR-AUC | Recall at precision 0.8 |
| Regression | RMSE | R2, MAE |
| Clustering | Silhouette | Davies-Bouldin |
| Anomaly detection | PR-AUC with labels, else score stability | none |
| Time series | sMAPE | MAE |
| Offline RL / bandit | Estimated policy value (IPW) | Doubly robust value |

## Datasets (`eval/benchmark_datasets.py`)

Nine public datasets: three classification (Titanic, Telco churn, oil spill: imbalanced), two regression
(California housing, diabetes), two clustering (Palmer penguins, wine; labels removed), one time series
(airline passengers) and one anomaly task (mammography, labelled). Rows are capped at 3,000 per dataset
with a fixed stratified sample (seed 0), so the full ablation fits on a laptop.

## Configurations

| Config | What it is |
| --- | --- |
| A | Single-LLM baseline: one Gemini call writes a scikit-learn pipeline from the prompt; run in a separate process. Requires a Gemini key; classification and regression only. |
| B | AutoML baseline: FLAML with default settings, 60-second budget; classification and regression only. |
| C | Sequential pipeline: agents run once in order; no critic, no debate, no memory. |
| D | Full deliberative: critic, debate and verification; no memory. |
| E | Full with memory: all three pillars; memory suggestions only from other datasets. |
| F | No verification: D with the evidence verifier switched off during the run. |

All internal configs use the same seeds, the same held-out split procedure (20 percent, stratified, or the
last 20 percent by time) and the same settings: 5-fold CV with 1 repeat and 8 tuning trials ("fast
settings"; `--full-settings` gives 5x3 CV and 20 trials).

## Runs and seeds

Seeds 42, 43, 44, 45, 46 (five runs per dataset and config). Flaw-injection runs use seeds 42 and 43.

## Metrics logged per run

Primary metric on the held-out test set (read once); baseline score; relative gain over the baseline;
flaw catch rate; hallucinated-claim rate (claims rejected by the verifier / all claims, measured after
the run for every config, including F); time; tokens; API calls; stated confidence and its outcome
(the chosen pipeline beat the baseline and matched the runner-up within one CV standard deviation on
the held-out data).

## Flaw injection and catch criteria (`eval/inject_flaws.py`)

| Flaw | Injection | Counted as caught when |
| --- | --- | --- |
| Target leakage | column = target + N(0, 0.02 sd) | auditor target-leakage finding on that column, or a critic leakage objection |
| Proxy feature | column = target + N(0, 0.35 sd) | auditor leakage, proxy or suspicious-name finding on that column, or a critic leakage objection |
| Label noise | 10 percent of labels flipped (classification) | auditor label-noise finding |
| Missing not at random | 80 percent of the values above the 70th percentile of the key feature removed | missing-value finding on that column not labelled MCAR-like |
| Duplicates across the split | 10 percent of rows duplicated and shuffled in | auditor exact-duplicate finding |
| Covariate shift | key feature shifted by one sd in a new batch | drift check lists that feature as drifted |

The same detectors run on the clean dataset to give the false-alarm rate.

## Statistical tests

Paired Wilcoxon signed-rank tests over datasets (scores averaged over seeds first), on relative gain over
each dataset's baseline so different metrics share a scale: D vs C, D vs B, D vs A, E vs D, D vs F.
Effect size: rank-biserial correlation. Uncertainty: bootstrap 95 percent interval of the median
difference (5,000 resamples). Calibration: expected calibration error with ten 0.1-wide bins. With nine
datasets every result is reported as indicative, and negative results are reported where they occur.

## Amendments

None yet.
