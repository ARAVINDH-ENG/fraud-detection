<div align="center">

# 🛡️ FraudShield — Real-Time Fraud Detection System

**Production-grade ML system for real-time transaction fraud detection**  
*XGBoost · SHAP · Gemini LLM · FastAPI · Docker · GitHub Actions · Render*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docker.com)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?logo=render)](https://fraud-detection-system-6bq4.onrender.com/dashboard/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

<br/>

**[🚀 Live Demo](https://fraud-detection-system-6bq4.onrender.com/dashboard/) · [📖 API Docs](https://fraud-detection-system-6bq4.onrender.com/docs) · [📊 Dashboard](https://fraud-detection-system-6bq4.onrender.com/dashboard/)**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Design Decisions](#-key-design-decisions)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [ML Model & Performance](#-ml-model--performance)
- [API Reference](#-api-reference)
- [Quick Start](#-quick-start)
- [Docker Deployment](#-docker-deployment)
- [Cloud Deployment (Render)](#-cloud-deployment-render)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Running Tests](#-running-tests)
- [Model Versioning & Rollback](#-model-versioning--rollback)
- [Drift Detection](#-drift-detection)
- [Simulation Engine](#-simulation-engine)
- [Cost Matrix & Business Logic](#-cost-matrix--business-logic)
- [Dashboard](#-dashboard)
- [Environment Variables](#-environment-variables)

---

## 🎯 Overview

FraudShield is a **production-grade, real-time fraud detection system** built on the Kaggle Credit Card Fraud dataset (284,807 transactions, 0.17% fraud rate). It goes well beyond a typical ML project — every layer is designed with production thinking: sub-100ms API responses, async LLM explanations, human-in-the-loop review, cost-driven decision thresholds, and live drift monitoring.

### What makes this different from a typical ML project

| Typical ML Project | FraudShield |
|---|---|
| Jupyter notebook with accuracy score | Production FastAPI service with <100ms latency |
| Single threshold at 0.5 | Cost-matrix-tuned thresholds (FN=₹500 / FP=₹10) |
| Feature importance charts | Per-prediction SHAP values via TreeExplainer |
| No explanations | Async Gemini LLM explanations grounded in SHAP |
| No human oversight | Human-in-the-loop review queue for uncertain cases |
| No monitoring | KS-test drift detection against training baseline |
| Manual runs | GitHub Actions CI/CD + Docker + Render deployment |

---

## 🏗️ System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              REQUEST LIFECYCLE           │
                        │                 (~80ms total)            │
                        └─────────────────────────────────────────┘

 Transaction Input
        │
        ▼
 ┌──────────────┐    422     ┌─────────────────────────────────────┐
 │   Pydantic   │──────────▶│  Validation Error (bad input)        │
 │  Validation  │           └─────────────────────────────────────┘
 └──────┬───────┘
        │ valid
        ▼
 ┌──────────────┐  <50ms
 │  XGBoost     │──── predict_proba() ────▶ fraud_probability (0.0–1.0)
 │  Model       │──── SHAP values    ────▶ top 5 feature attributions
 └──────┬───────┘
        │
        ▼
 ┌──────────────────┐
 │  Decision Engine │
 │  prob < 0.30  ──▶│ APPROVE (LOW risk)    auto-resolved
 │  0.30–0.70    ──▶│ REVIEW  (MEDIUM risk) → human queue
 │  prob ≥ 0.70  ──▶│ REJECT  (HIGH risk)   auto-resolved
 └──────┬───────────┘
        │
        ▼
 ┌──────────────┐
 │ Cost Analysis│  expected_fraud_loss = P(fraud) × max(amount, ₹500)
 │  Evaluator   │  expected_friction   = P(legit) × ₹10
 └──────┬───────┘
        │
        ▼
 ┌──────────────┐
 │  Database    │──── Transaction record persisted (SQLite/PostgreSQL)
 │  (SQLAlchemy)│──── ReviewQueue entry created (if REVIEW)
 └──────┬───────┘
        │
        ├──── ⚡ Response returned immediately (<100ms)
        │
        ▼ (background, non-blocking)
 ┌──────────────┐  1–3s
 │  Gemini LLM  │──── 4 models tried in sequence (2.0-flash-lite → fallback)
 │  Explainer   │──── SHAP-grounded 2-3 sentence explanation
 │              │──── Rule-based fallback if all LLM quota exceeded
 └──────┬───────┘
        │
        ▼
 ┌──────────────┐
 │  DB Update   │──── explanation stored, analyst can poll GET /explanation/{id}
 └──────────────┘
```

### Service Layers

```
fraud_detection/
├── api/                    ← FastAPI routes + Pydantic schemas
├── model_service/          ← XGBoost inference + SHAP (loaded once at startup)
├── decision_engine/        ← Threshold logic, cost-matrix-driven
├── business_layer/         ← Financial cost quantification per decision
├── llm_service/            ← Gemini async explainer + rule-based fallback
├── drift_detection/        ← KS-test monitoring against training baseline
├── database/               ← SQLAlchemy models, session management
├── training/               ← Model training + stat baseline generation
├── model_service/          ← Version registry, rollback support
├── simulation/             ← Synthetic transaction generator for testing
├── dashboard/              ← Vanilla JS single-page dashboard (served by FastAPI)
├── tests/                  ← pytest suite (API, cost, decision engine)
├── scripts/                ← Rollback, watch_metrics, test_decisions utilities
├── docker/                 ← Dockerfile + docker-compose
└── .github/workflows/      ← CI/CD pipeline
```

---

## 🧠 Key Design Decisions

### 1. Async LLM — Respond Fast, Enrich Later
```
Model inference:  <50ms  → synchronous  (caller needs decision NOW)
LLM explanation: 1–3s   → background   (caller doesn't need to wait)
```
The `/predict` endpoint returns in <100ms. The explanation generates in the background and is stored in the DB. The client polls `GET /explanation/{id}` when ready. Without this pattern, every prediction would take 2–3 seconds.

### 2. Cost-Matrix-Driven Thresholds (not accuracy)
The model is scored on **business cost**, not F1. With FN cost = ₹500 and FP cost = ₹10 (50:1 ratio), the approve threshold is set at 0.30 (not 0.50). This means the system aggressively catches fraud at the expense of some false positives — the financially correct decision given the asymmetry.

### 3. SHAP for Grounded Explainability
Raw feature importances are global averages — useless for explaining individual decisions. `shap.TreeExplainer` produces per-prediction attributions using Shapley values (cooperative game theory), which the LLM then converts to natural language. The LLM is **never** used for prediction, only narration.

### 4. Human-in-the-Loop (HITL) for Uncertain Cases
Transactions with probability 0.30–0.70 are neither auto-approved nor auto-rejected — they enter a human review queue. Analysts see the SHAP explanation and make the final call. These human decisions are stored as ground-truth labels for future retraining, **closing the ML feedback loop**.

### 5. Model Versioning with Rollback
Every trained model is registered in `models/version_registry.json` with its metrics and path. If a new model degrades in production (fraud rate spikes), rollback is a single command:
```bash
python scripts/rollback.py v1
```

### 6. Non-Auto-Increment UUIDs
Transaction IDs are UUIDs, not auto-increment integers. This is the correct choice for distributed systems — no collision risk across multiple instances.

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **API** | FastAPI 0.104 | Async, type-safe, automatic OpenAPI docs |
| **ML Model** | XGBoost 2.0 | Best-in-class for tabular fraud data, fast inference |
| **Explainability** | SHAP 0.43 | Per-prediction Shapley values, not global importance |
| **LLM** | Google Gemini Flash | Async explanation generation, 4-model fallback chain |
| **Schema Validation** | Pydantic v2 | Zero-trust input validation, 422 on bad data |
| **ORM / DB** | SQLAlchemy 2.0 + SQLite/PostgreSQL | Env-switched, indexed on decision/status/timestamp |
| **Drift Detection** | SciPy KS test | Non-parametric, no distribution assumptions |
| **Containerization** | Docker + Compose | Reproducible, non-root user, health checks |
| **CI/CD** | GitHub Actions | Test → lint → Docker build → smoke test |
| **Cloud** | Render | Persistent disk, env secrets, auto-deploy |
| **Data Processing** | pandas + numpy | Feature engineering and drift analysis |
| **Testing** | pytest + httpx | API integration tests + unit tests |

---

## 📁 Project Structure

```
fraud_detection/
│
├── api/
│   ├── main.py                 # FastAPI app, CORS, static dashboard mount
│   ├── routes/
│   │   ├── predict.py          # POST /predict, GET /explanation/{id}
│   │   ├── review.py           # GET /review, POST /review/{id}
│   │   └── metrics.py          # GET /metrics, GET /drift
│   └── schemas/
│       └── transaction.py      # Pydantic TransactionRequest / TransactionResponse
│
├── model_service/
│   ├── predict.py              # XGBoost inference + SHAP (loaded once at module import)
│   └── version_manager.py      # Registry CRUD, get_active_version(), rollback()
│
├── decision_engine/
│   └── engine.py               # make_decision(probability) → APPROVE/REVIEW/REJECT
│
├── business_layer/
│   └── cost_evaluator.py       # evaluate_cost(decision, probability, amount)
│
├── llm_service/
│   └── explainer.py            # Gemini chain (4 models) + rule-based fallback
│
├── drift_detection/
│   └── detector.py             # KS test, get_recent_features(), detect_drift()
│
├── database/
│   ├── db.py                   # SQLAlchemy engine, SessionLocal, get_db()
│   └── models.py               # Transaction + ReviewQueue ORM models w/ indexes
│
├── training/
│   └── train.py                # XGBoost training, stat baseline, model registration
│
├── simulation/
│   └── data_simulator.py       # Synthetic fraud/legit transaction generator
│
├── dashboard/
│   └── index.html              # Single-file SPA dashboard (served at /dashboard)
│
├── tests/
│   ├── test_api.py             # Integration tests (health, predict, review, metrics)
│   ├── test_cost_evaluator.py  # Unit tests for cost math
│   └── test_decision_engine.py # Boundary tests for thresholds
│
├── scripts/
│   ├── rollback.py             # Switch active model version
│   ├── watch_metrics.py        # Terminal live metrics poller
│   └── test_decisions.py       # Bulk decision distribution checker
│
├── models/
│   ├── fraud_model.pkl         # Trained XGBoost model
│   ├── training_stats.json     # Feature distribution baseline for drift detection
│   └── version_registry.json  # Model version history + active pointer
│
├── docker/
│   ├── Dockerfile              # python:3.10-slim, non-root user, health check
│   └── docker-compose.yml      # API + nginx dashboard services
│
├── .github/workflows/
│   └── ci.yml                  # Test → lint → Docker build → smoke test
│
├── render.yaml                 # Render cloud deployment config
├── config.py                   # Thresholds, cost matrix, paths (env-aware)
└── requirements.txt
```

---

## 📊 ML Model & Performance

### Dataset
- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud rate:** 492 fraud (0.17%) — extreme class imbalance
- **Features:** Time, Amount + V1–V28 (PCA-anonymized)

### Imbalance Handling
SMOTE was rejected in favor of `scale_pos_weight`:
```python
# ratio ≈ 577 — tells XGBoost: misclassifying fraud is 577x more costly
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
```
SMOTE generates synthetic fraud samples, which teaches the model fake patterns. `scale_pos_weight` preserves the real data distribution while accounting for cost asymmetry.

### Why `eval_metric="aucpr"` not `auc-roc`
With 0.17% fraud rate, AUC-ROC can hit 0.99 even with poor fraud detection because the huge legitimate class dominates the curve. Precision-Recall AUC focuses on the minority class performance — the metric that actually matters.

### Model Performance (v1)

| Metric | Score |
|---|---|
| **Precision** | 0.8817 |
| **Recall** | 0.8367 |
| **F1 Score** | 0.8586 |
| **AUC-ROC** | 0.9684 |

> **Note:** Recall is prioritized over precision. Missing fraud (FN) costs ₹500; false alarm (FP) costs ₹10. The model is tuned to catch as much fraud as possible even at the cost of some false positives.

### Training

```bash
# Download creditcard.csv to data/ first
python training/train.py
```

This automatically:
1. Trains XGBoost with stratified split and imbalance weighting
2. Saves `models/fraud_model.pkl`
3. Computes and saves `models/training_stats.json` (feature distribution baseline)
4. Registers the model in `models/version_registry.json`

---

## 📡 API Reference

Base URL: `https://fraud-detection-system-6bq4.onrender.com` (or `http://localhost:8000`)

Interactive docs available at `/docs` (Swagger UI) and `/redoc`.

---

### `POST /predict`
Score a transaction for fraud.

**Request body:**
```json
{
  "Time": 1000.0,
  "Amount": 149.62,
  "V1": -1.3598,
  "V2": -0.0728,
  "V3": 2.5364,
  ...
  "V28": 0.0103
}
```

**Response** (< 100ms):
```json
{
  "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
  "decision": "APPROVE",
  "risk_level": "LOW",
  "probability": 0.0021,
  "explanation": null,
  "cost_analysis": {
    "decision": "APPROVE",
    "expected_fraud_loss": 1.05,
    "expected_friction_cost": 9.98,
    "transaction_amount": 149.62,
    "dominant_risk": "FRICTION"
  },
  "model_version": "v1",
  "shap_features": {
    "V14": -0.3821,
    "V4":  0.1204,
    "V12": -0.0987,
    "V10": -0.0812,
    "V17": -0.0634
  }
}
```
> `explanation` is `null` in the immediate response — it generates async and is fetchable via `GET /explanation/{id}`.

---

### `GET /explanation/{transaction_id}`
Fetch the LLM explanation for a scored transaction.

```json
{
  "transaction_id": "550e8400-...",
  "decision": "APPROVE",
  "risk_level": "LOW",
  "probability": 0.0021,
  "explanation": "This transaction scored 0.2% fraud probability and was classified as LOW risk. V14 contributed strongly toward legitimacy (SHAP: -0.3821), which is a primary fraud discriminator in this dataset. All top SHAP features pushed toward legitimate patterns, consistent with normal transaction behavior.",
  "explanation_ready": true,
  "model_version": "v1",
  "timestamp": "2026-04-19T10:23:45"
}
```

---

### `GET /metrics`
System health dashboard metrics.

```json
{
  "total_transactions": 1284,
  "approved": 1198,
  "rejected": 24,
  "under_review": 62,
  "fraud_rate_percent": 1.87,
  "review_rate_percent": 4.83,
  "approval_rate_percent": 93.30,
  "avg_fraud_probability": 0.0412
}
```

---

### `GET /review`
Fetch all transactions pending human analyst review.

```json
{
  "pending_reviews": [
    {
      "transaction_id": "uuid...",
      "review_status": "PENDING",
      "reason": "V4 (+0.8821) and V11 (+0.6342) strongly pushed toward fraud...",
      "queued_at": "2026-04-19T10:15:00"
    }
  ],
  "count": 3
}
```

---

### `POST /review/{transaction_id}?human_decision=APPROVED`
Submit analyst decision on a queued transaction.

| Parameter | Values |
|---|---|
| `human_decision` | `APPROVED` or `REJECTED` |

Human decisions are stored as gold-standard labels for future retraining — this closes the ML feedback loop.

---

### `GET /drift`
Run Kolmogorov-Smirnov test against training distribution baseline.

```json
{
  "total_features_checked": 30,
  "drifted_features": ["V14", "Amount"],
  "drift_detected": true,
  "drift_severity_percent": 6.7,
  "recommendation": "MONITOR",
  "details": {
    "V14": { "ks_statistic": 0.1821, "p_value": 0.0023, "drifted": true },
    "V1":  { "ks_statistic": 0.0412, "p_value": 0.3412, "drifted": false }
  }
}
```

---

### `GET /health`
Liveness probe for load balancers and Docker health checks.

```json
{ "status": "ok", "version": "2.0.0" }
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- `data/creditcard.csv` (download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))
- Gemini API key (free at [Google AI Studio](https://aistudio.google.com))

### 1. Clone and install
```bash
git clone https://github.com/ARAVINDH-ENG/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```

### 3. Train the model
```bash
python training/train.py
```

### 4. Start the API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access the system
| URL | Description |
|---|---|
| `http://localhost:8000/dashboard/` | Live dashboard |
| `http://localhost:8000/docs` | Swagger UI / API playground |
| `http://localhost:8000/health` | Health check |

### 6. (Optional) Run the simulator to populate data
```bash
python simulation/data_simulator.py
# Sends transactions at 0.3s intervals, 5% fraud rate
# Stop when dashboard shows enough data for drift analysis
```

---

## 🐳 Docker Deployment

```bash
# Build and start all services
cd docker
docker-compose up --build

# API available at http://localhost:8000
# Dashboard at http://localhost:3000 (nginx)
```

The compose setup runs:
- **fraud-api** — FastAPI + XGBoost, port 8000, health check every 30s
- **dashboard** — nginx serving the static dashboard, port 3000

```bash
# Check health
curl http://localhost:8000/health

# Stop
docker-compose down
```

### Environment variables for Docker
```bash
# Create .env in project root (docker-compose reads it automatically)
GEMINI_API_KEY=your_key_here
DATABASE_URL=sqlite:////app/data/fraud.db
```

---

## ☁️ Cloud Deployment (Render)

The project includes a `render.yaml` for one-click deployment.

### Steps
1. Push to GitHub
2. Go to [render.com](https://render.com) → New → Blueprint
3. Connect your GitHub repository
4. Set the `GEMINI_API_KEY` environment variable in the Render dashboard
5. Deploy — Render will run `training/train.py` as the build command, then start uvicorn

**Live deployment:** https://fraud-detection-system-6bq4.onrender.com/dashboard/

> Note: Render free tier spins down after inactivity. First request may take ~30 seconds to cold-start.

---

## 🔄 CI/CD Pipeline

The GitHub Actions pipeline (`.github/workflows/ci.yml`) runs on every push to `main` or `develop`:

```
push/PR
    │
    ▼
┌─────────────────── job: test ───────────────────┐
│  1. Checkout                                     │
│  2. Setup Python 3.10                            │
│  3. Cache pip dependencies (keyed by requirements.txt) │
│  4. pip install -r requirements.txt              │
│  5. pytest tests/ -v --tb=short                  │
│  6. flake8 linting (max-line-length=120)         │
└────────────────────┬────────────────────────────┘
                     │ (only if tests pass)
                     ▼
┌─────────────────── job: build ──────────────────┐
│  1. docker build -f docker/Dockerfile            │
│  2. docker run + sleep 8                         │
│  3. curl -f http://localhost:8000/health  ◄── smoke test │
│  4. docker stop                                  │
└─────────────────────────────────────────────────┘
```

---

## 🧪 Running Tests

```bash
# Full test suite
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=. --cov-report=term-missing

# Specific test file
pytest tests/test_decision_engine.py -v
```

### Test coverage

| Test File | What it covers |
|---|---|
| `test_api.py` | Health, predict, metrics, review endpoints + Pydantic validation (422 on negative amounts) |
| `test_decision_engine.py` | Boundary tests at 0.0, 0.299, 0.3, 0.7, 1.0 — threshold edge cases |
| `test_cost_evaluator.py` | Cost math correctness, dominant risk classification, asymmetry check |

---

## 🔀 Model Versioning & Rollback

Every training run registers a new version:

```json
// models/version_registry.json
{
  "versions": [
    {
      "version": "v1",
      "path": "models/fraud_model.pkl",
      "trained_at": "2026-04-19T06:35:54",
      "metrics": {
        "precision": 0.8817,
        "recall": 0.8367,
        "f1": 0.8586,
        "auc_roc": 0.9684
      }
    }
  ],
  "active": "v1"
}
```

```bash
# Rollback to a previous version (if new model degrades in production)
python scripts/rollback.py v1

# Watch live metrics in terminal
python scripts/watch_metrics.py

# Test decision distribution against sample data
python scripts/test_decisions.py
```

---

## 📉 Drift Detection

The drift monitor detects when live transaction distribution diverges from the training baseline using the **Kolmogorov-Smirnov test**.

**Why KS test?**
- Non-parametric — no assumption about distribution shape
- Works on any continuous feature
- Computationally cheap
- p-value < 0.05 → distributions differ significantly → drift detected

**How it works:**
1. `training/train.py` computes and saves per-feature mean/std/percentiles to `models/training_stats.json`
2. `GET /drift` pulls the last 500 transactions from DB, runs KS test for each feature
3. Returns per-feature results + severity percentage + recommendation

| Severity | Recommendation |
|---|---|
| > 20% features drifted | `RETRAIN` — run training/train.py |
| 10–20% features drifted | `MONITOR` — watch closely |
| < 10% features drifted | `STABLE` — no action needed |

```bash
# Check drift status
curl http://localhost:8000/drift
```

---

## 🎲 Simulation Engine

Generates synthetic transactions to populate the database for testing and demo purposes.

```bash
# Default: 5% fraud rate, 0.3s interval
python simulation/data_simulator.py

# Watch the dashboard update in real time as transactions are scored
```

The simulator generates statistically realistic fraud patterns based on the actual feature distributions from `creditcard.csv` — V14, V12, and V4 are the strongest fraud discriminators and are shifted accordingly.

---

## 💰 Cost Matrix & Business Logic

The decision engine is driven by **financial cost**, not model accuracy.

```python
COST_MATRIX = {
    "false_positive": 10,   # Cost of blocking a legitimate transaction (friction)
    "false_negative": 500   # Base cost of missing fraud (loss)
}

# Actual fraud loss accounts for transaction amount:
fraud_loss_basis = max(transaction_amount, 500)
expected_fraud_loss    = probability × fraud_loss_basis
expected_friction_cost = (1 - probability) × 10
```

**Example at probability = 0.6, amount = ₹2000:**
```
Expected fraud loss    = 0.6 × ₹2000 = ₹1200  ← dominant
Expected friction cost = 0.4 × ₹10   = ₹4
→ Decision leans toward REJECT
```

**Threshold derivation:**
The 50:1 cost asymmetry (FN/FP) justifies placing the approve threshold at 0.30 instead of 0.50. At 0.30, the system already treats 30% probability of fraud as too risky to approve — because the cost of missing it (₹500+) vastly outweighs the friction cost (₹10).

---

## 📊 Dashboard

A single-page application served directly by FastAPI at `/dashboard/`. No React, no build step — pure HTML/CSS/JS.

**Pages:**
- **Dashboard** — Live metrics (total, approved, rejected, review), distribution bars, system health
- **Score Transaction** — Split-panel form with 30 input fields, instant result display, async explanation polling
- **Explanations** — Filterable table with full LLM explanations, individual transaction lookup
- **Review Queue** — HITL analyst interface with Approve/Reject buttons
- **Drift Monitor** — KS test results table per feature with drift status badges
- **Model Registry** — Version history with metrics and rollback command

---

## 🔐 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key for LLM explanations |
| `DATABASE_URL` | No | SQLAlchemy DB URL. Defaults to `sqlite:///fraud.db` |

```bash
# .env file
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=sqlite:///fraud.db   # or postgresql://user:pass@host/db
```

**Get a free Gemini API key:** https://aistudio.google.com/apikey

> ⚠️ **Never commit your `.env` file.** It is gitignored by default. For production, use Render's environment variable settings or GitHub Actions secrets.

---

## 🗺️ Roadmap

- [ ] Redis caching for repeated SHAP signature explanations
- [ ] Celery + Redis for durable async tasks (survive server restarts)
- [ ] Prometheus metrics endpoint for Grafana integration
- [ ] Parallel LLM fallback with `asyncio.gather` instead of sequential
- [ ] PostgreSQL in production (replace SQLite)
- [ ] Rate limiting with SlowAPI
- [ ] API key authentication middleware
- [ ] Automated retraining trigger when drift severity > 20%
- [ ] `/simulate` endpoint to trigger data generation from dashboard UI

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with production ML engineering principles**

*If this project helped you, consider giving it a ⭐*

</div>
