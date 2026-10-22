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

## Deploy on Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/dulfazl/DS-Cortex)

1. Click the button (or in Render choose **New > Blueprint** and pick this repository). `render.yaml` creates two services:
   `ds-cortex-api` (Docker) and `ds-cortex-web` (Node).
2. Enter `GOOGLE_API_KEY` when Render asks (leave it empty to run offline).
3. If Render gives the API a URL other than `https://ds-cortex-api.onrender.com`, set `NEXT_PUBLIC_API_URL` on
   `ds-cortex-web` to that URL and redeploy the web service (the value is built into the page).

Free instances have 512 MB of RAM, no persistent disk and sleep after 15 idle minutes, so the first request
after a pause is slow and finished runs disappear on restart (the memory store is re-seeded from
`memory/seed_fingerprints.json`). The Blueprint uses fast settings, one run at a time and `MAX_ROWS=5000` to fit:
files with more rows are sampled down (stratified for classification), and the sampling is logged in the
Decision Ledger and repeated by the exported script. In a test with a 512 MB limit, a 20,640-row file peaked
at 507 MB without the cap and 295 MB with it. For larger CSVs choose a bigger plan, raise `MAX_ROWS` and add a
disk mounted at `/data`.

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
| Report | Explainability, Business, Research, Report | SHAP (with a logged fallback), error analysis, ranked plots, narratives, model card, export bundle. |

**Run modes:** Autopilot stops only for blockers; Review pauses at the preprocessing, feature-engineering,
modelling and tuning bundles for approval. Overriding the recommendation warns once, with numbers, and
both choices are logged and evaluated on the held-out rows.

## Configuration

All settings are environment variables (see `.env.example`): `GOOGLE_API_KEY`, `GEMINI_MODEL`, `LLM_MODE`
(`auto`, `gemini`, `offline`), `ALLOWED_ORIGINS`, `MAX_UPLOAD_MB`, `MAX_CONCURRENT_RUNS`,
`DATA_RETENTION_HOURS`, `DAILY_CALL_BUDGET`, `BLOCKER_POLICY`, `FAST_MODE`, `STATIC_PLOTS`,
`NEXT_PUBLIC_API_URL`. Run the API with **one worker**: run threads and their pauses belong to one process.

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
