"""
config.py
=========
Central configuration for the Deriv Algorithmic Trading System.

NEW API SYSTEM (2025+):
  Authentication uses PAT (Personal Access Token) — your pat_... token — as a
  Bearer token in REST HTTP headers, plus a registered Deriv-App-ID.
  The old wss://ws.derivws.com WebSocket + authorize flow is REMOVED.

  How to get your App ID:
    1. Go to https://api.deriv.com/app-registration/
    2. Register a new app (name it e.g. "AlgoBot")
    3. Copy the numeric App ID (e.g. 12345)
    4. Paste it below as APP_ID

  Your PAT token (pat_...) is already correct and goes into BEARER_TOKEN.
"""

# ---------------------------------------------------------------------------
# API Credentials — NEW system
# ---------------------------------------------------------------------------
# Your PAT token — used as "Authorization: Bearer <token>" in REST calls
BEARER_TOKEN = "pat_f5b03a248d156f94cf70e568ccd29b3d753d0d829a9dd24f53ded17d4d74b257"

# Your registered Deriv App ID — hardcoded for local desktop use (no redirects needed)
APP_ID = "33t649hQZFdG2406vCqdN"


# ---------------------------------------------------------------------------
# Deriv API v2 — New endpoint URLs (2025+)
# ---------------------------------------------------------------------------
# REST base — all account/trading REST calls go here
DERIV_REST_BASE = "https://api.derivws.com"

# Public WebSocket — tick streaming, NO auth required
# Format for subscribing: same JSON protocol as before {"ticks": "R_50", "subscribe": 1}
DERIV_PUBLIC_WS = "wss://api.derivws.com/trading/v1/options/ws/public"

# REST endpoints (relative to DERIV_REST_BASE)
ACCOUNTS_ENDPOINT   = "/trading/v1/options/accounts"           # GET — list accounts
OTP_ENDPOINT_TMPL   = "/trading/v1/options/accounts/{}/otp"   # POST — get WS URL


# ---------------------------------------------------------------------------
# Synthetic markets to monitor simultaneously
# ---------------------------------------------------------------------------
SYMBOLS = [
    "R_10",    # Volatility 10 Index
    "R_25",    # Volatility 25 Index
    "R_50",    # Volatility 50 Index
    "R_75",    # Volatility 75 Index
    "R_100",   # Volatility 100 Index
    "1HZ10V",  # Volatility 10 (1s) Index
    "1HZ25V",  # Volatility 25 (1s) Index
    "1HZ50V",  # Volatility 50 (1s) Index
    "1HZ75V",  # Volatility 75 (1s) Index
    "1HZ100V", # Volatility 100 (1s) Index
    "JD10",    # Jump 10 Index
    "JD25",    # Jump 25 Index
    "JD50",    # Jump 50 Index
    "JD75",    # Jump 75 Index
    "JD100",   # Jump 100 Index
]

# ---------------------------------------------------------------------------
# Demo warm-up period before switching to real money
# ---------------------------------------------------------------------------
DEMO_DURATION_SECONDS = 600  # 10 minutes of shadow trading + learning

# ---------------------------------------------------------------------------
# Martingale configuration
# ---------------------------------------------------------------------------
BASE_STAKE = 0.35          # Starting stake (USD) for each side
MAX_MARTINGALE_STEPS = 6   # Maximum doublings before aborting the sequence
MARTINGALE_MULTIPLIER = 2  # Stake multiplier on each loss

# ---------------------------------------------------------------------------
# ML / Prediction engine
# ---------------------------------------------------------------------------
# Number of warm-up ticks required per market before predictions are trusted.
MIN_TICKS_BEFORE_TRADE = 15

# SGDClassifier hyper-parameters
SGD_LOSS = "log_loss"       # Logarithmic loss → calibrated probabilities
SGD_LEARNING_RATE = "optimal"
SGD_ALPHA = 1e-4            # L2 regularisation strength
SGD_MAX_ITER = 1            # One pass per partial_fit call (online learning)

# Feature window — how many past last-digits to include
DIGIT_WINDOW = 5

# ---------------------------------------------------------------------------
# Balancing guard — suppress a side if it is winning too much relative to
# the other side. E.g. 3 means "suppress if wins lead by more than 3".
# ---------------------------------------------------------------------------
BALANCE_GUARD_THRESHOLD = 3

# ---------------------------------------------------------------------------
# Optional local WebSocket broadcast server for a JS frontend
# ---------------------------------------------------------------------------
ENABLE_LOCAL_WS_BROADCAST = True   # Set False to disable
LOCAL_WS_HOST = "0.0.0.0"         # Bind on all interfaces (fixes IPv4/IPv6 mismatch on Windows)
LOCAL_WS_PORT = 8765
