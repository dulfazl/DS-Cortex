# DS Cortex

Upload a CSV, describe the problem in a sentence, and a team of agents audits the data, picks the task,
builds and debates candidate pipelines, and returns the best model with plots, a Decision Ledger and a
script that reproduces the score.

Three ideas are what the project defends; everything else is infrastructure:

1. **Deliberative data science.** Statistician, ML Architect, Business Insight and Research agents state
   positions on the candidate pipelines, a Pipeline Critic attacks each one, and a resolution table
   written as code decides. Every debate claim must cite a tool run; an evidence verifier rejects the rest.
2. **Evidence and confidence.** Every recommendation carries the executed tool results and a confidence
   computed from experiments (CV stability, bootstrap separation, effect over the baseline), never stated
   by the language model. The evaluation suite checks whether that confidence is calibrated.
3. **Adaptive memory.** Dataset fingerprints suggest pipelines that worked on similar past data. Suggestions
   are advice: the Critic can reject them, and nothing is applied silently.

The language model (Gemini Flash) only plans, critiques and narrates. Every number comes from a Python
tool. Without a key the system runs fully offline with template narration.

## Quick start

```bash
docker compose up --build
```

Then open http://localhost:3000. The API image is about 3 GB, mostly Chromium for PNG copies of the plots;
`WITH_CHROMIUM=0 docker compose up --build` builds a much smaller one without them. To enable Gemini, copy `.env.example` to `.env` and set `GOOGLE_API_KEY`
(create it in Google AI Studio in a Cloud project with billing disabled; enabling billing removes the free tier).

### Local development

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --port 8000 --workers 1
```

```bash
cd frontend && npm install && npm run dev
```

Run the whole chain from the command line on any CSV:

```bash
python -m scripts.run_chain tests/fixtures/classification.csv "Predict whether a customer will churn"
```

Explore single agents interactively with the ADK web UI (needs a key): `PYTHONPATH=. adk web agents`.

## Live demo

**https://ds-cortex.vercel.app** is a free, server-less demo: the website replays two real runs recorded with the full
system (a telco churn file with real signal, and an e-commerce file where the system reports no signal beyond
chance), with every results tab. Uploading a new file needs the analysis server, which the demo does not host.

The demo is the `frontend` folder built with `NEXT_PUBLIC_DEMO=1` (on Vercel: Root Directory `frontend`, that
environment variable, nothing else). Its recordings are static files made from completed runs:

```bash
python -m scripts.export_demo frontend/public/demo SESSION_ID:telco-churn SESSION_ID:ecommerce-orders
```

## Hosting the full system

The API needs a long-running server: runs execute in background threads and stream events, and the
machine-learning libraries are far too large for serverless functions. Any Docker host works. Build the root
`Dockerfile`, set `GOOGLE_API_KEY` (optional) and `PORT`, allow the website's address with `ALLOWED_ORIGINS` or
`ALLOWED_ORIGIN_REGEX`, and build the website with `NEXT_PUBLIC_API_URL` pointing at the API. While a sleeping
API wakes up, the website shows a short notice instead of an error.

### Small servers

Hosts with little memory or a fraction of a CPU need a lighter configuration; these settings were measured on a
container limited to 512 MB and 0.1 CPU:

- **Rows.** `MAX_ROWS=2000` samples larger files down (stratified for classification); the sampling is logged in
  the Decision Ledger and repeated by the exported script. A 20,640-row file peaked at 507 MB without a cap and
  295 MB with a 5,000-row cap.
- **Threads.** Numerical libraries start one thread per host core, and in a container with little CPU the
  threads stall each other: a logistic regression that fits in 1.4 seconds on a laptop took 25 minutes. The app
  sizes its thread pools to the container's CPU quota; set `N_JOBS=1` if the quota is not visible (`/readyz`
  reports the value in use).
- **A lighter search.** `CRITIC_TOP_K=2`, `CRITIC_ROUNDS=1` and `EXCLUDE_MODELS=gradient_boosting` (scikit-learn's
  gradient boosting costs about 13 times as much as XGBoost, which still covers boosted trees with LightGBM).
  The candidate factory lists the exclusion among its reasons, and the Critic's opening event states its budget.
  On a 10,000-row e-commerce file these settings cut the work from 247 to 29 CPU-seconds.
- **Restarts.** Without a persistent disk, finished runs disappear when the server restarts (the memory store is
  re-seeded from `memory/seed_fingerprints.json`), and the run page says so instead of waiting. Keep the run page
  open: it checks the run once a minute, which also keeps a sleeping free instance awake until the run ends.

The free Gemini tier allows about 20 requests per model per day and one analysis makes about 8, so the model
chain (`GEMINI_FALLBACK_MODELS`) spreads calls across Flash models and falls back to template narration when
all are used up.

## How a run works

| Stage | Agent | What happens |
| --- | --- | --- |
| Quick profile, routing | Router | Types, missing values, candidate targets; task family, target and metric (fixed before any model runs). Asks one clarifying question when unsure. |
| Audit | Data Auditor | Missing-value mechanism, types, outliers, duplicates, imbalance, leakage, inconsistencies, label noise. Proposes a cleaning plan; a leakage blocker pauses the run. |
| Statistics | Statistician | Normality, VIF, heteroscedasticity, autocorrelation, effect sizes, distribution fits, fingerprint. |
| Candidates | ML Architect | 4 to 6 pipelines conditioned on the findings, each a scikit-learn Pipeline evaluated with repeated CV against a baseline. |
| Critique | Pipeline Critic | Overfitting, leakage probes, subgroups, seed stability, covariate shift, unnecessary steps; up to 3 revision rounds. |
| Debate | Supervisor | Positions, evidence verification, resolution table, at most 2 rebuttal rounds, computed confidence. |
| Tuning, test | ML Architect, Supervisor | Optuna on the two finalists; the held-out test set is read once. |
| Report | Explainability, Business, Research, Report | SHAP (with a logged fallback), error analysis, ranked plots, narratives, model card, export bundle. Drivers are named only when the held-out gain over the baseline is more than chance (paired bootstrap over the test rows); otherwise the summary says plainly that the file shows no signal. |

**Run modes:** Autopilot stops only for blockers; Review pauses at the preprocessing, feature-engineering,
modelling and tuning bundles for approval. Overriding the recommendation warns once, with numbers, and
both choices are logged and evaluated on the held-out rows.

## Configuration

All settings are environment variables (see `.env.example`): `GOOGLE_API_KEY`, `GEMINI_MODEL`, `LLM_MODE`
(`auto`, `gemini`, `offline`), `ALLOWED_ORIGINS`, `MAX_UPLOAD_MB`, `MAX_CONCURRENT_RUNS`,
`DATA_RETENTION_HOURS`, `DAILY_CALL_BUDGET`, `BLOCKER_POLICY`, `FAST_MODE`, `STATIC_PLOTS`, `MAX_ROWS`,
`N_JOBS`, `CRITIC_TOP_K`, `CRITIC_ROUNDS`, `EXCLUDE_MODELS`, `NEXT_PUBLIC_API_URL`. Run the API with **one worker**: run threads and their pauses belong to one process.

## Evaluation

```bash
python -m eval.run_all --quick        # smoke run: 1 seed, 4 datasets, configs B C D F
python -m eval.run_all --seeds 5      # the pre-registered evaluation (several hours)
```

Tables and figures go to `reports/eval/` (ablation, paired Wilcoxon tests, flaw catch and false-alarm
rates, hallucinated-claim rate, reliability diagram, memory and cost). The metrics, configurations and
tests were committed in `eval/PREREGISTRATION.md` before any result. Config A (single-LLM baseline) needs a
Gemini key and runs model-written code in a separate process.

## Reproducibility

`pipeline.py` in every export bundle reproduces the reported score exactly on the same platform and package
versions (recorded in `manifest.json`). Across operating systems, boosted-tree libraries can differ slightly:
an XGBoost model scored on Linux reproduced exactly in the Linux image but differed by 0.6% on macOS.

## Tests

```bash
pytest -q                              # offline; the live Gemini smoke test skips without a key
cd frontend && npm run e2e             # Playwright, against a running stack
```

The suite includes known-answer tests for the statistics tools, an evidence-trace test (every number in
the report must come from a logged tool run or the ledger), determinism, a test-set guard, the export
reproduction check, leakage and edge-case files, API pause/resume, and security checks.

## Limits

- The drift check is a two-sample test against the training data. It does not retrain, monitor live
  traffic or track performance.
- Confidence comes from a hand-designed formula; check `reports/eval/reliability.png` before trusting it.
- Memory suggestions come from a small store of past runs and are advisory only.
- Research notes are short summaries written for this project: context, not a literature review.
- On the free Gemini tier, Google may use submitted content to improve its products. Only profiles and
  tool outputs are sent, but column names and some category values go with them: do not upload
  confidential or personal data.

## Layout

```
app/        FastAPI app, sessions, ledger, events, LLM gateway, chat, replay
agents/     one folder per agent (core.py: the logic; agent.py: ADK definition)
tools/      pandas, statistics, sklearn, critic, SHAP, plotting, drift, verification, export
workflows/  pipeline graph, revision loop, debate graph, resolution table, report graph
memory/     fingerprint store, research notes, vector store
eval/       benchmark suite, flaw injection, baselines, ablation runner, calibration, statistics
frontend/   Next.js dashboard
tests/      unit, contract, agent, workflow, API and live tests
paper/      outline for the write-up
```
