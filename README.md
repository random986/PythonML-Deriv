# PythonML-Deriv

A real-time, self-learning algorithmic trading bot for Deriv synthetic markets (Volatility and Jump indices). Built entirely in Python with `asyncio` for non-blocking execution and `scikit-learn`'s `SGDClassifier` for continuous online machine learning.

---

## Architecture

```
Python Deriv pure/
├── config.py          ← All tunable constants and API credentials
├── market_brain.py    ← RealTimeMarketBrain (ML engine, one per market)
├── decision_engine.py ← GlobalDecisionEngine (scanner / router)
├── trading_bot.py     ← DerivTradingBot (WebSocket + Martingale execution)
├── main.py            ← Entry point
├── requirements.txt   ← Python dependencies
└── README.md          ← This file
```

```
              Deriv WSS API
                    │
             ┌──────▼──────┐
             │ trading_bot │  ←── demo-to-real token swap after 10 min
             └──────┬──────┘
    tick per symbol │
        ┌───────────┼──────────┐
        ▼           ▼          ▼
     Brain_R10   Brain_R50  Brain_R100  ...  (one per symbol)
        │           │          │
        └───────────┴──────────┘
                    │  predict_proba(side)
             ┌──────▼──────────┐
             │ decision_engine │  ←── balance guard + recovery scanner
             └──────┬──────────┘
                    │  best_market / get_recovery_target
             ┌──────▼──────┐
             │ trading_bot │  ←── fires direct buy payload
             └─────────────┘
```

---

## Prerequisites

- Python 3.11 or newer
- A Deriv account with two API tokens: one for Demo, one for Real

---

## Setup

### 1. Clone / download the project

```bash
cd "Python Deriv pure"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your tokens

Open `config.py` and replace the placeholder values:

```python
DEMO_TOKEN = "YOUR_DEMO_TOKEN_HERE"   # ← paste your Virtual account token
REAL_TOKEN = "YOUR_REAL_TOKEN_HERE"   # ← paste your Real account token
```

Get tokens from: https://app.deriv.com/account/api-token

Required token permissions: **Read**, **Trade**

---

## Running the Bot

```bash
python main.py
```

The bot will:

1. **Phase 1 — Demo (10 minutes):** Connect with your demo token and stream ticks from all configured markets. Shadow-predict every tick, compare predictions to real outcomes, and continuously train the `SGDClassifier` via `partial_fit`. No real trades are fired.

2. **Phase 2 — Live:** Gracefully disconnect the demo session, reconnect with your real token, and begin executing trades. The trained ML models are retained in memory — no retraining from scratch.

---

## How It Works

### Online Learning (Log-Loss Feedback Loop)

Each market has its own `RealTimeMarketBrain`. On every tick:

- Features are extracted: last 5 digits, inter-tick velocity, virtual error rate, current loss streak.
- The model predicts `P(Over5 wins)` and `P(Under5 wins)`.
- The actual outcome is compared against the previous prediction.
- If the model was **wrong** (especially confidently wrong), `partial_fit()` is called with a large gradient update.
- `log_loss` ensures the model is penalised proportionally to its confidence — a confident wrong prediction causes a much larger weight adjustment than an uncertain wrong prediction.

### Cross-Market Recovery (Martingale)

The bot maintains **two independent Martingale sequences** — one for Over 5, one for Under 5.

When a trade **wins**: stake resets to base.

When a trade **loses**:
- Stake doubles.
- The `GlobalDecisionEngine` **re-scans all markets** at that exact moment.
- The recovery trade is placed on whichever market currently shows the **highest predicted probability** of a win for that side — not necessarily the same market as the previous trade.

### Balance Guard

If one side (e.g. Over 5) is winning significantly more than the other, the engine temporarily suppresses new trades on that side. This prevents one side from running far ahead and exposing the other side to a long, unchecked loss streak.

---

## Frontend (JavaScript)

If `ENABLE_LOCAL_WS_BROADCAST = True` in `config.py`, the bot broadcasts a JSON state snapshot to `ws://localhost:8765` after every tick. You can build a lightweight JavaScript dashboard that connects to this and renders live charts, confidence meters, and Martingale status without modifying the Python code.

Snapshot format:
```json
{
  "type": "state_update",
  "is_demo": true,
  "elapsed_seconds": 42.3,
  "markets": [
    { "symbol": "R_50", "tick_count": 210, "is_ready": true,
      "p_over": 0.623, "p_under": 0.441, "virtual_error_rate": 0.487 }
  ],
  "engine": { "wins": {"over": 5, "under": 3}, "losses": {"over": 2, "under": 4} },
  "martingale": {
    "over":  { "stake": 0.35, "step": 0, "active": false, "current_market": "R_50" },
    "under": { "stake": 0.70, "step": 1, "active": true,  "current_market": "R_75" }
  }
}
```

---

## Configuration Reference (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `DEMO_DURATION_SECONDS` | `600` | Demo/shadow phase length before going live |
| `BASE_STAKE` | `0.35` | Starting stake per trade (USD) |
| `MAX_MARTINGALE_STEPS` | `6` | Max doublings before aborting and resetting |
| `MIN_CONFIDENCE_THRESHOLD` | `0.58` | Minimum P(win) before firing a real trade |
| `MIN_TICKS_BEFORE_TRADE` | `50` | Ticks required per market before trusting predictions |
| `BALANCE_GUARD_THRESHOLD` | `3` | Suppress a side if it leads by more than N wins |
| `DIGIT_WINDOW` | `5` | Number of past last-digits used as ML features |
| `SGD_ALPHA` | `1e-4` | L2 regularisation strength |
| `ENABLE_LOCAL_WS_BROADCAST` | `True` | Enable/disable the frontend broadcast server |
| `LOCAL_WS_PORT` | `8765` | Port for the local WebSocket broadcast server |

---

## Disclaimer

**This software is provided for educational and research purposes only.**
Algorithmic trading involves substantial financial risk. Synthetic index markets on Deriv are random-walk instruments — no strategy, including machine learning, can guarantee profitability. You may lose all capital you deploy. Use this system at your own risk and only with funds you can afford to lose.

The authors assume no liability for any financial losses incurred through the use of this software.
