"""
trading_bot.py
==============
DerivTradingBot — owns Martingale state and trade execution.

Now natively powered by deriv_base_brain.py instead of scikit-learn.
"""

import asyncio
import json
import logging
import time
import random
import collections
from dataclasses import dataclass, field
from typing import Optional

from deriv_client import DerivClient
from continuous_brain import GatekeeperBrain
import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MartingaleTracker
# ---------------------------------------------------------------------------

@dataclass
class MartingaleTracker:
    """Tracks the current Martingale sequence for one trade side."""
    side: str
    stake: float = field(default_factory=lambda: config.BASE_STAKE)
    step: int = 0
    current_market: str = ""
    active: bool = False
    last_contract_id: Optional[str] = None
    features: list[float] = field(default_factory=list)
    global_ctx: float = 0.5

    def reset_stake(self) -> None:
        logger.info("[Martingale/%s] Resetting stake to base (%.2f) and resetting step.",
                    self.side, config.BASE_STAKE)
        self.stake = config.BASE_STAKE
        self.step = 0
        self.active = False
        self.last_contract_id = None
        self.features = []

    def record_win(self) -> None:
        logger.info("[Martingale/%s] WIN at step %d (stake=%.2f). Resetting.",
                    self.side, self.step, self.stake)
        self.stake = config.BASE_STAKE
        self.step = 0
        self.active = False
        self.last_contract_id = None
        self.features = []

    def record_loss(self) -> None:
        logger.info("[Martingale/%s] LOSS at step %d (stake=%.2f).",
                    self.side, self.step, self.stake)
        self.step += 1
        self.stake = round(
            config.BASE_STAKE * (config.MARTINGALE_MULTIPLIER ** self.step), 2
        )
        self.active = False
        self.last_contract_id = None
        self.features = []

    @property
    def loss_streak(self) -> int:
        return self.step


# ---------------------------------------------------------------------------
# DerivTradingBot
# ---------------------------------------------------------------------------

class DerivTradingBot:
    """
    Main bot — wires DerivClient → Native Python Brain → trades.
    """

    def __init__(self):
        self._client = DerivClient()

        # Wire callbacks
        self._client.on_tick             = self._on_tick
        self._client.on_buy_resp         = self._on_buy_response
        self._client.on_contract_update  = self._on_contract_update

        # Martingale sequences GLOBAL
        self._global_tracker_over  = MartingaleTracker(side="over")
        self._global_tracker_under = MartingaleTracker(side="under")
        
        # Virtual Martingale sequences PER SYMBOL
        self._virtual_trackers_over  = {s: MartingaleTracker(side="over") for s in config.SYMBOLS}
        self._virtual_trackers_under = {s: MartingaleTracker(side="under") for s in config.SYMBOLS}

        # Pending buys: req_id → (side, symbol, stake, state_key)
        self._pending_buys: dict[int, tuple] = {}
        self._req_id_counter: int = 1000

        # Timing
        self._start_time: float = 0.0
        self._is_paused: bool = True  # Starts paused; user clicks Start

        # Local broadcast
        self._broadcast_clients: set = set()
        self._broadcast_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        
        # UI State tracking — NO TRUNCATION
        self._trade_history: list[dict] = []
        self._learning_logs: list[str] = []
        self._MAX_LOGS = 200
        
        # Feature tracking (Ticks per symbol)
        self._tick_history = {s: [] for s in config.SYMBOLS}
        
        # Virtual trade queue: maps symbol to list of pending virtual trades
        self._pending_virtuals: dict[str, list] = {}
        
        # Stats tracking for UI — VIRTUAL stats (training only)
        self._market_stats = {
            s: {"p_over": 0.0, "p_under": 0.0, "err": 0.5} for s in config.SYMBOLS
        }
        self._engine_stats = {
            "wins": {"over": 0, "under": 0},
            "losses": {"over": 0, "under": 0}
        }
        # REAL trade stats — completely separate from virtual
        self._real_stats = {
            "wins": {"over": 0, "under": 0},
            "losses": {"over": 0, "under": 0}
        }
        self._session_pnl: float = 0.0
        
        # Max loss / max win tracking with timestamps
        self._max_loss = {"amount": 0.0, "timestamp": "", "symbol": "", "side": ""}
        self._max_win  = {"amount": 0.0, "timestamp": "", "symbol": "", "side": ""}
        self._max_stake = 0.0
        
        self._accumulated_uptime = 0.0
        self._session_start_time = None

        # Consecutive loss prevention for real trades
        self._last_real_result = {s: {"over": None, "under": None} for s in config.SYMBOLS}
        # Per-symbol virtual recovery
        self._last_virtual_result = {s: {"over": None, "under": None} for s in config.SYMBOLS}
        self._over_brain = GatekeeperBrain("over")
        self._under_brain = GatekeeperBrain("under")

        self._global_trade_results = collections.deque(maxlen=100)
        
        # TRUE prediction accuracy counters
        # A "prediction" = whichever side had higher confidence from the model.
        # Correct = that side actually won the digit outcome.
        self._ml_predictions_total: int = 0
        self._ml_predictions_correct: int = 0
        
        # Toast notification queue
        self._toast_queue: list[dict] = []

    def _log_ai(self, msg: str) -> None:
        """Helper to append a message to the AI logs sent to the frontend."""
        logger.info(msg)
        time_str = time.strftime("%H:%M:%S")
        self._learning_logs.append(f"[{time_str}] {msg}")
        if len(self._learning_logs) > self._MAX_LOGS:
            self._learning_logs.pop(0)

    # ── Entry point ───────────────────────────────────────────────────────────

    async def start(self, force_real: bool = False) -> None:
        """Boot the bot: start DerivClient, run demo phase then live phase (unless force_real)."""
        self._start_time = time.monotonic()

        tasks = [self._run_bot_lifecycle(force_real)]
        if config.ENABLE_LOCAL_WS_BROADCAST:
            tasks.append(self._run_broadcast_server())
            tasks.append(self._broadcast_loop())

        await asyncio.gather(*tasks)

    async def _run_bot_lifecycle(self, force_real: bool) -> None:
        use_demo = not force_real
        if force_real:
            self._is_paused = False  # Auto-start when forced real

        self._log_ai("=" * 60)
        self._log_ai("Bot connected to Deriv. Streaming tick data.")
        self._log_ai("Press START to begin firing trades.")
        self._log_ai("=" * 60)

        await self._client.start(use_demo=use_demo)

        # Keep running indefinitely — _is_paused controls trade execution
        while True:
            await asyncio.sleep(60)

    # ── Bar Color Simulation ──────────────────────────────────────────────────
    
    def _get_bar_color(self, symbol: str) -> str:
        prices = self._tick_history[symbol][-10:]
        if len(prices) < 10:
            return "YELLOW"
        
        delta_p = prices[-1] - prices[0]
        avg_p = sum(prices) / 10
        if avg_p == 0:
            return "YELLOW"
            
        pct_change = (delta_p / avg_p) * 10000  # Basis points

        if pct_change > 10: return "DBL_GREEN"
        if pct_change > 2:  return "GREEN"
        if pct_change < -5: return "RED"
        if abs(pct_change) <= 2: return "YELLOW"
        return "BLUE"

    # ── Tick processing ───────────────────────────────────────────────────────

    def _extract_continuous_tensor(self, history: list[float]) -> list[float]:
        """Extract rich probabilistic features the AI uses to learn Over/Under patterns.
        
        Features:
        - Digit frequency distribution (10 features: count of each digit 0-9)
        - Most appearing digit position (normalized 0-1)
        - Least appearing digit position (normalized 0-1)
        - Most appearing digit frequency (how dominant it is)
        - Least appearing digit frequency
        - Presence of consecutive doubles of the most appearing digit (0 or 1)
        - Presence of consecutive doubles of the least appearing digit (0 or 1)
        - Over 5 ratio in recent window
        - Under 5 ratio in recent window
        - Mean reversion signal: how far the over5 ratio deviates from 0.5
        - Current digit streak length (same digit appearing consecutively)
        - Last digit (normalized 0-1)
        - Inter-tick velocity (momentum)
        - Digit entropy (how spread/concentrated the digit distribution is)
        - Consecutive over-5 streak
        - Consecutive under-5 streak
        - digit delta
        - is_last_digit_5 (1.0 if last digit was 5, else 0.0)
        - ticks_since_5_norm (distance since last digit 5, normalized)
        - freq_5_short (frequency of digit 5 in last 15 ticks to detect clustering)
        """
        from collections import Counter
        import math
        
        if len(history) < 50:
            return [0.0] * 29  # Return zeros if not enough data
        
        # Extract last digits from recent price history
        recent = history[-50:]
        digits = [int(str(p).replace('.', '')[-1]) for p in recent]
        
        # ── DIGIT FREQUENCY DISTRIBUTION (10 features) ──
        digit_counts = Counter(digits)
        total = len(digits)
        freq = [digit_counts.get(d, 0) / total for d in range(10)]  # normalized [0,1]
        
        # ── MOST and LEAST appearing digit ──
        most_common_digit = max(range(10), key=lambda d: digit_counts.get(d, 0))
        least_common_digit = min(range(10), key=lambda d: digit_counts.get(d, 0))
        most_freq = digit_counts.get(most_common_digit, 0) / total
        least_freq = digit_counts.get(least_common_digit, 0) / total
        
        # ── PRESENCE OF DOUBLES ──
        # Check if the most/least appearing digit appeared consecutively (back-to-back)
        has_double_most = 0.0
        has_double_least = 0.0
        for i in range(1, len(digits)):
            if digits[i] == digits[i-1]:
                if digits[i] == most_common_digit:
                    has_double_most = 1.0
                if digits[i] == least_common_digit:
                    has_double_least = 1.0
        
        # ── OVER/UNDER 5 RATIOS ──
        over5_count = sum(1 for d in digits if d > 5)
        under5_count = sum(1 for d in digits if d < 5)  # digit 5 itself is neutral
        over5_ratio = over5_count / total
        under5_ratio = under5_count / total
        
        # ── MEAN REVERSION SIGNAL ──
        # How far the over5 ratio deviates from the expected 0.4 (digits 6,7,8,9 = 4/10)
        # Positive = over5 has been appearing too much (expect under5 next)
        # Negative = under5 has been dominating (expect over5 next)
        mean_reversion = over5_ratio - 0.4
        
        # ── CURRENT DIGIT STREAK ──
        # How many consecutive ticks had the same last digit
        streak = 1
        for i in range(len(digits) - 2, -1, -1):
            if digits[i] == digits[-1]:
                streak += 1
            else:
                break
        streak_norm = min(streak / 5.0, 1.0)  # normalize, cap at 5
        
        # ── LAST DIGIT (normalized) ──
        last_digit = digits[-1] / 9.0
        
        # ── INTER-TICK VELOCITY (momentum) ──
        momentum = sum(abs(recent[i] - recent[i-1]) for i in range(1, len(recent)))
        momentum_norm = momentum / len(recent)
        
        # ── DIGIT ENTROPY ──
        # High entropy = digits are spread evenly (hard to predict)
        # Low entropy = one digit dominates (easier to predict)
        entropy = 0.0
        for d in range(10):
            p = digit_counts.get(d, 0) / total
            if p > 0:
                entropy -= p * math.log2(p)
        entropy_norm = entropy / math.log2(10)  # normalize to [0,1]
        
        # ── CONSECUTIVE OVER-5 and UNDER-5 STREAKS ──
        over5_streak = 0
        for d in reversed(digits):
            if d > 5:
                over5_streak += 1
            else:
                break
        under5_streak = 0
        for d in reversed(digits):
            if d < 5:
                under5_streak += 1
            else:
                break

        # ── DIGIT 5 SPECIFIC FEATURES ──
        is_last_digit_5 = 1.0 if digits[-1] == 5 else 0.0
        
        # Ticks since last digit 5
        ticks_since_5 = 0
        for i in range(len(digits) - 1, -1, -1):
            if digits[i] == 5:
                break
            ticks_since_5 += 1
        ticks_since_5_norm = min(ticks_since_5 / 20.0, 1.0)
        
        # Short-term digit 5 frequency (last 15 ticks)
        short_recent = digits[-15:]
        freq_5_short = sum(1 for d in short_recent if d == 5) / len(short_recent)
        
        # Combine all features into a single vector (29 features total)
        features = (
            freq +  # 10 features: digit frequency distribution
            [
                most_common_digit / 9.0,    # most appearing digit position (normalized)
                least_common_digit / 9.0,   # least appearing digit position (normalized)
                most_freq,                   # how dominant the most appearing digit is
                least_freq,                  # how rare the least appearing digit is
                has_double_most,             # 1.0 if doubles of most appearing digit present
                has_double_least,            # 1.0 if doubles of least appearing digit present
                over5_ratio,                 # proportion of digits > 5
                under5_ratio,                # proportion of digits < 5
                mean_reversion,              # deviation from expected over5 ratio
                streak_norm,                 # current same-digit streak length
                last_digit,                  # the last digit itself (normalized)
                momentum_norm,               # inter-tick velocity
                entropy_norm,                # digit distribution entropy
                over5_streak / 5.0,          # recent consecutive over-5 streak
                under5_streak / 5.0,         # recent consecutive under-5 streak
                (digits[-1] - digits[-2]) / 9.0 if len(digits) >= 2 else 0.0,  # digit delta
                is_last_digit_5,             # 1.0 if last digit was 5
                ticks_since_5_norm,          # distance since last digit 5
                freq_5_short,                # short term digit 5 clustering
            ]
        )
        return features

    def _get_global_shadow_context(self) -> float:
        if not self._global_trade_results:
            return 0.5
        return sum(self._global_trade_results) / len(self._global_trade_results)

    async def _on_tick(self, symbol: str, price: float) -> None:
        """
        Hot path: called for every incoming tick from the public WebSocket.
        """
        quote = price
        digit = int(str(quote).replace('.', '')[-1])

        # Resolve any pending virtual trades from the previous tick
        if self._pending_virtuals.get(symbol):
            for vtrade in self._pending_virtuals[symbol]:
                self._resolve_virtual_trade(vtrade, digit)
            self._pending_virtuals[symbol] = []

        self._tick_history[symbol].append(quote)
        if len(self._tick_history[symbol]) > 100:
            self._tick_history[symbol].pop(0)

        # Always broadcast state snapshot so UI can track market readiness
        await self._enqueue_broadcast()

        # Only evaluate if we have enough history to form features
        if len(self._tick_history[symbol]) < 60:
            return

        # Extract continuous features
        features = self._extract_continuous_tensor(self._tick_history[symbol])
        global_ctx = self._get_global_shadow_context()

        # Get the brain's entry score for this market tick
        # (warm-up returns 0.5 neutral — no hardcoded gates)
        over_step_now = self._global_tracker_over.step
        under_step_now = self._global_tracker_under.step
        recovery_context = [
            over_step_now / 10.0,    # How deep is the over recovery? (normalized)
            under_step_now / 10.0,   # How deep is the under recovery? (normalized)
            global_ctx,              # Global win/loss ratio across all recent trades
        ]
        score_over = self._over_brain.get_entry_score(features, recovery_context)
        score_under = self._under_brain.get_entry_score(features, recovery_context)
        
        # Decide the unified entry score for this market based on who needs to recover most
        if over_step_now > under_step_now:
            entry_score = score_over
        elif under_step_now > over_step_now:
            entry_score = score_under
        else:
            entry_score = max(score_over, score_under)

        predicted_side = "over" if score_over >= score_under else "under"
        state_key = "CONTINUOUS_ML"

        # Update UI stats
        self._market_stats[symbol]["p_over"] = entry_score
        self._market_stats[symbol]["p_under"] = 1.0 - entry_score
        self._market_stats[symbol]["entry_score"] = entry_score

        # ── VIRTUAL TRADES — always fire both sides, all markets, no rules ──
        v_tracker_over = self._virtual_trackers_over[symbol]
        v_tracker_under = self._virtual_trackers_under[symbol]

        predicted_side = "over" if entry_score >= 0.5 else "under"

        if not v_tracker_over.active:
            v_tracker_over.features = features
            v_tracker_over.global_ctx = global_ctx
            self._queue_virtual_trade(symbol, "over", state_key, entry_score * 100, predicted_side=predicted_side)
        if not v_tracker_under.active:
            v_tracker_under.features = features
            v_tracker_under.global_ctx = global_ctx
            self._queue_virtual_trade(symbol, "under", state_key, (1.0 - entry_score) * 100)

        # ── REAL TRADES ────────────────────────────────────────────────
        if self._is_paused:
            return

        import time
        now = time.time()

        # ── CALIBRATION PAUSE (after 4+ consecutive losses) ──
        if hasattr(self, "_pause_until") and now < self._pause_until:
            remaining = self._pause_until - now
            if int(remaining) % 10 == 0:
                self._log_ai(f"⏳ [CALIBRATION] Paused {remaining:.0f}s — ML absorbing market shift.")
            return

        # ── WARM-UP CHECK ──
        in_warmup = (self._over_brain.update_count < self._over_brain.WARMUP_UPDATES or 
                     self._under_brain.update_count < self._under_brain.WARMUP_UPDATES)

        # ── WATCHDOG: release stale lock ──
        if getattr(self, "_global_real_trade_active", False):
            if hasattr(self, "_last_real_trade_time") and now - self._last_real_trade_time > 15.0:
                self._log_ai("⚠️ [WATCHDOG] Lock timeout — forcing release.")
                self._global_tracker_over.active = False
                self._global_tracker_under.active = False
                self._global_real_trade_active = False
                for t in self._trade_history:
                    if t.get("status") == "open":
                        t["status"] = "failed"
                        t["profit"] = 0.0
                        t["digit"] = "Err"
            else:
                return  # Trade in-flight, skip tick

        # ── ML THINKING: Is this market a good entry right now? ──
        # The brain returns an entry_score already computed above for this tick.
        # We compare this market's score against all others the ML has scored
        # this cycle. Fire only on the market the ML ranks highest.
        # No hardcoded gaps, no step comparisons — this IS the ML's decision.
        if not in_warmup:
            # Avoid placing trades if the ML predicts a digit 5 (both sides under 0.5 confidence)
            if score_over < 0.5 and score_under < 0.5:
                self._log_ai(
                    f"🛑 [ML-AVOID] {symbol} | score_over={score_over:.3f} | "
                    f"score_under={score_under:.3f} | Predicted Digit 5 (both sides < 0.5). Skipping trade."
                )
                return

            best_score = max(
                (m.get("entry_score", 0.0) for m in self._market_stats.values()),
                default=0.0
            )
            self._log_ai(
                f"🧠 [ML-THINK] {symbol} | entry_score={entry_score:.3f} | "
                f"best_in_market={best_score:.3f} | "
                f"recovery=[over_step={over_step_now} under_step={under_step_now}]"
            )
            # Fire if this market is among the best options currently available
            # (within a tiny margin of the max) to prevent stale-max race conditions.
            # No hardcoded minimum confidence! The ML freely chooses the best relative market.
            if entry_score < best_score - 0.02:
                return  # Another market is relatively better

        # ── FIRE ──────────────────────────────────────────────────────────────
        self._global_real_trade_active = True
        self._last_real_trade_time = now

        # Snapshot the exact features + recovery context this ML decision was based on
        self._global_tracker_over.features = features
        self._global_tracker_over.global_ctx = global_ctx
        self._global_tracker_over.recovery_context = recovery_context
        self._global_tracker_under.features = features
        self._global_tracker_under.global_ctx = global_ctx
        self._global_tracker_under.recovery_context = recovery_context

        self._log_ai(
            f"🚀 [TRADE] {symbol} | ML_score={entry_score:.3f} | "
            f"step=[O:{over_step_now} U:{under_step_now}] | warmup={'YES' if in_warmup else 'NO'}"
        )

        import asyncio
        asyncio.create_task(
            self._dual_execute(symbol, state_key, entry_score * 100, (1.0 - entry_score) * 100)
        )

    def _get_current_balance(self) -> float:
        current_id = self._client._current_account_id
        if self._client.accounts:
            for acc in self._client.accounts:
                if acc.get("account_id") == current_id:
                    return float(acc.get("balance", 0.0))
        return 0.0

    async def _dual_execute(self, symbol: str, state_key: str, p_over: float, p_under: float) -> None:
        """Fires both sides at the exact same instance."""
        import asyncio
        import time

        if not self._client.is_auth_ready:
            self._log_ai("⚠️ [TRADE ERROR] Auth WS is not ready. Skipping trade.")
            self._global_real_trade_active = False
            return

        # Ensure trackers are not active
        if self._global_tracker_over.active or self._global_tracker_under.active:
            self._log_ai("⚠️ [TRADE ERROR] One of the trackers is already active. Skipping trade.")
            self._global_real_trade_active = False
            return

        # Balance check
        needed_balance = self._global_tracker_over.stake + self._global_tracker_under.stake
        current_balance = self._get_current_balance()
        if current_balance < needed_balance:
            self._log_ai(f"⚠️ [BALANCE CHECK] Insufficient balance for dual trade: need ${needed_balance:.2f}, have ${current_balance:.2f}. Resetting Martingale stakes to base!")
            self._global_tracker_over.reset_stake()
            self._global_tracker_under.reset_stake()
            needed_balance = self._global_tracker_over.stake + self._global_tracker_under.stake
            if current_balance < needed_balance:
                self._log_ai("⚠️ [BALANCE CHECK] Still insufficient balance even for base stake! Skipping trade.")
                self._global_real_trade_active = False
                return

        # Prepare Over trade
        stake_over = self._global_tracker_over.stake
        if stake_over > self._max_stake:
            self._max_stake = stake_over
        req_id_over = self._next_req_id()
        payload_over = {
            "buy": "1",
            "subscribe": 1,
            "price": stake_over,
            "parameters": {
                "amount": stake_over,
                "basis": "stake",
                "contract_type": "DIGITOVER",
                "currency": "USD",
                "duration": 1,
                "duration_unit": "t",
                "underlying_symbol": symbol,
                "barrier": "5"
            },
            "req_id": req_id_over
        }

        # Prepare Under trade
        stake_under = self._global_tracker_under.stake
        if stake_under > self._max_stake:
            self._max_stake = stake_under
        req_id_under = self._next_req_id()
        payload_under = {
            "buy": "1",
            "subscribe": 1,
            "price": stake_under,
            "parameters": {
                "amount": stake_under,
                "basis": "stake",
                "contract_type": "DIGITUNDER",
                "currency": "USD",
                "duration": 1,
                "duration_unit": "t",
                "underlying_symbol": symbol,
                "barrier": "5"
            },
            "req_id": req_id_under
        }

        # Synchronously setup trackers
        self._global_tracker_over.active = True
        self._global_tracker_over.current_market = symbol
        self._global_tracker_under.active = True
        self._global_tracker_under.current_market = symbol

        self._pending_buys[req_id_over] = ("over", symbol, stake_over, state_key)
        self._pending_buys[req_id_under] = ("under", symbol, stake_under, state_key)

        # Record in trade history
        now_ts = time.time()
        trade_record_over = {
            "req_id": req_id_over,
            "timestamp": now_ts,
            "symbol": symbol,
            "side": "over",
            "stake": stake_over,
            "status": "open",
            "profit": 0.0,
            "digit": "-"
        }
        trade_record_under = {
            "req_id": req_id_under,
            "timestamp": now_ts,
            "symbol": symbol,
            "side": "under",
            "stake": stake_under,
            "status": "open",
            "profit": 0.0,
            "digit": "-"
        }
        self._trade_history.insert(0, trade_record_over)
        self._trade_history.insert(0, trade_record_under)

        self._log_ai(f"[DECISION] {symbol} | OVER 5 | conf={p_over:.1f}% | stake=${stake_over:.2f} | req_id={req_id_over}")
        self._log_ai(f"[DECISION] {symbol} | UNDER 5 | conf={p_under:.1f}% | stake=${stake_under:.2f} | req_id={req_id_under}")

        # Send both WebSocket payloads back-to-back using asyncio.gather
        try:
            results = await asyncio.gather(
                self._client.send_trade(payload_over),
                self._client.send_trade(payload_under)
            )
            if not all(results):
                logger.error(f"[DUAL EXECUTE ERROR] One or both sends failed: {results}")
                if not results[0]:
                    self._global_tracker_over.active = False
                    self._pending_buys.pop(req_id_over, None)
                    trade_record_over["status"] = "failed"
                if not results[1]:
                    self._global_tracker_under.active = False
                    self._pending_buys.pop(req_id_under, None)
                    trade_record_under["status"] = "failed"
                if not self._global_tracker_over.active and not self._global_tracker_under.active:
                    self._global_real_trade_active = False
        except Exception as e:
            logger.error(f"[DUAL EXECUTE ERROR] send failed: {e}")
            self._global_tracker_over.active = False
            self._global_tracker_under.active = False
            self._global_real_trade_active = False
            self._pending_buys.pop(req_id_over, None)
            self._pending_buys.pop(req_id_under, None)
            trade_record_over["status"] = "failed"
            trade_record_under["status"] = "failed"

    def _queue_virtual_trade(self, symbol: str, side: str, state_key: str, conf: float, predicted_side: str = None) -> None:
        tracker = self._virtual_trackers_over[symbol] if side == "over" else self._virtual_trackers_under[symbol]
        if tracker.active:
            return
        
        tracker.active = True
        tracker.current_market = symbol
        # Note: state_key no longer stored on tracker
        
        self._log_ai(f"[V-DECISION] {symbol} | {side.upper()} 5 | conf={conf:.1f}% | stake=${tracker.stake:.2f}")
        
        vtrade = {
            "symbol": symbol,
            "side": side,
            "stake": tracker.stake,
            "features": tracker.features,
            "global_ctx": tracker.global_ctx,
            "predicted_side": predicted_side  # Only set on OVER-side to count prediction once per cycle
        }
        self._pending_virtuals.setdefault(symbol, []).append(vtrade)

    def _resolve_virtual_trade(self, vtrade: dict, digit: int) -> None:
        symbol = vtrade["symbol"]
        side = vtrade["side"]
        stake = vtrade["stake"]
        features = vtrade["features"]
        global_ctx = vtrade["global_ctx"]
        predicted_side = vtrade.get("predicted_side")  # Only present on OVER-side vtrades
        tracker = self._virtual_trackers_over[symbol] if side == "over" else self._virtual_trackers_under[symbol]
        
        tracker.active = False
        won = (digit > 5) if side == "over" else (digit < 5)
        profit = round(stake * 0.90 if won else -stake, 2)
        
        self._log_ai(f"[V-SETTLE] side={side} | profit=${profit:.2f} | won={won}")
        
        if won:
            tracker.record_win()
            self._engine_stats["wins"][side] += 1
            self._last_virtual_result[symbol][side] = True
            self._log_ai(f"[Martingale/{symbol}/{side}] V-WIN. Resetting stake.")
        else:
            tracker.record_loss()
            self._engine_stats["losses"][side] += 1
            self._last_virtual_result[symbol][side] = False
            self._log_ai(f"[Martingale/{symbol}/{side}] V-LOSS. Escalating stake to ${tracker.stake:.2f}.")

        # ── TRUE PREDICTION ACCURACY ──────────────────────────────────────────
        # Only evaluated on the OVER-side vtrade (once per tick cycle, not twice).
        # Digit == 5 is excluded: it's the house-edge outcome where nobody wins.
        if predicted_side is not None and digit != 5:
            actual_winner = "over" if digit > 5 else "under"
            self._ml_predictions_total += 1
            if predicted_side == actual_winner:
                self._ml_predictions_correct += 1

        # Reset the OTHER side's gate so it can fire again on the next tick
        other_side = "under" if side == "over" else "over"
        self._last_virtual_result[symbol][other_side] = None

        # Update the brain: was this a recovery success?
        # The virtual trade helps the brain learn faster (many more samples per second).
        v_recovery_ctx = [
            self._global_tracker_over.step / 10.0,
            self._global_tracker_under.step / 10.0,
            global_ctx
        ]
        # Route the virtual learning to the correct side-specific brain
        # If the digit is 5, apply a 3.0x weight penalty to make the AI learn to avoid it
        v_weight = tracker.stake * 3.0 if digit == 5 else tracker.stake
        try:
            if side == "over":
                self._over_brain.update(
                    market_features=features,
                    recovery_context=v_recovery_ctx,
                    recovery_succeeded=won,
                    weight=v_weight
                )
            else:
                self._under_brain.update(
                    market_features=features,
                    recovery_context=v_recovery_ctx,
                    recovery_succeeded=won,
                    weight=v_weight
                )
        except Exception as e:
            logger.error(f"[V-BRAIN UPDATE ERROR] {e} — skipping, continuing.")

        tracker.current_market = ""

    # ── Trade execution ───────────────────────────────────────────────────────

    async def _fire_trade(self, symbol: str, side: str, stake: float, state_key: str) -> None:
        if stake > self._max_stake:
            self._max_stake = stake
        contract_type = "DIGITOVER" if side == "over" else "DIGITUNDER"
        req_id = self._next_req_id()

        payload = {
            "buy": "1",
            "subscribe": 1,
            "price": stake,
            "parameters": {
                "amount": stake,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": 1,
                "duration_unit": "t",
                "underlying_symbol": symbol,
                "barrier": "5"
            },
            "req_id": req_id
        }

        self._pending_buys[req_id] = (side, symbol, stake, state_key)
        self._log_ai(f"[TRADE] {contract_type} on {symbol} | stake=${stake:.2f} | req_id={req_id}")
        
        trade_record = {
            "req_id": req_id,
            "timestamp": time.time(),
            "symbol": symbol,
            "side": side,
            "stake": stake,
            "status": "open",
            "profit": 0.0,
            "digit": "-"
        }
        self._trade_history.insert(0, trade_record)

        await self._client.send_trade(payload)

    async def _on_buy_response(self, msg: dict) -> None:
        req_id = msg.get("req_id")
        entry  = self._pending_buys.pop(req_id, None)

        if "error" in msg:
            error_code = msg["error"].get("code", "")
            logger.error("[TRADE] Buy failed (req_id=%d): %s", req_id, msg["error"])
            if entry:
                side, symbol, stake, state_key = entry
                tracker = self._global_tracker_over if side == "over" else self._global_tracker_under
                tracker.active = False
                
                # If we hit API limits or ran out of money, we MUST reset the Martingale
                # otherwise the bot will try to place the exact same impossible stake forever.
                if error_code in ("PayoutLimits", "InsufficientBalance", "InvalidStake"):
                    logger.warning(f"[{error_code}] API rejected stake of ${stake}. Resetting Martingale for {side} side.")
                    tracker.reset_stake()
                    self._real_stats["losses"][side] += 1 # Optionally tally as a definitive loss

            for t in self._trade_history:
                if t.get("req_id") == req_id:
                    t["status"] = "failed"
                    break
            
            # Release the global lock only if both trackers are now inactive
            if not self._global_tracker_over.active and not self._global_tracker_under.active:
                self._global_real_trade_active = False
            return

        buy_data    = msg.get("buy", {})
        contract_id = str(buy_data.get("contract_id", ""))

        # Link the contract_id to the exact trade in history
        for t in self._trade_history:
            if t.get("req_id") == req_id:
                t["contract_id"] = contract_id
                break

        if entry:
            side, symbol, stake, state_key = entry
            tracker = self._global_tracker_over if side == "over" else self._global_tracker_under
            tracker.last_contract_id = contract_id
            logger.info("[TRADE] Confirmed contract_id=%s side=%s market=%s",
                        contract_id, side, symbol)

    async def _on_contract_update(self, msg: dict) -> None:
        poc    = msg.get("proposal_open_contract", {})
        status = poc.get("status")
        contract_id = str(poc.get("contract_id", ""))

        tracker = None
        if self._global_tracker_over.last_contract_id == contract_id:
            tracker = self._global_tracker_over
        elif self._global_tracker_under.last_contract_id == contract_id:
            tracker = self._global_tracker_under

        if tracker:
            self._log_ai(f"[POC DEBUG] side={tracker.side} contract_id={contract_id} status={status}")

        if status not in ("sold", "won", "lost", "cancelled"):
            current_spot = poc.get("current_spot_display_value", "")
            if current_spot and tracker is not None:
                digit = current_spot[-1]
                for t in self._trade_history:
                    if t.get("contract_id") == contract_id and t.get("status") == "open":
                        t["digit"] = digit
                        break
            return

        # Update balance locally so UI reflects PnL
        profit = float(poc.get("profit", 0.0))
        for acc in self._client._accounts:
            if acc["account_id"] == self._client._current_account_id:
                acc["balance"] = round(float(acc["balance"]) + profit, 2)
                break

        won = (status == "won")
        
        exit_val = poc.get("exit_tick_display_value", "")
        if not exit_val:
            exit_val = poc.get("current_spot_display_value", "")
        final_digit = exit_val[-1] if exit_val else "-"

        if tracker is None:
            return

        side = tracker.side
        symbol = tracker.current_market
        tracker.active = False
        
        # CORE FIX: Only release the global lock when BOTH Over and Under have settled!
        # This prevents the async offset that was causing single trades to fire.
        if not self._global_tracker_over.active and not self._global_tracker_under.active:
            self._global_real_trade_active = False
            
        if final_digit == "5":
            self._log_ai(f"[DIGIT 5 PENALTY] {symbol} landed on 5! AI applying massive double-penalty to avoid this pattern.")
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        self._log_ai(f"[SETTLE] side={side} | contract={contract_id} | profit=${profit:.2f} | won={won}")

        # Find the exact matching trade entry by contract_id
        matched_symbol = ""
        for t in self._trade_history:
            if t.get("contract_id") == contract_id and t.get("status") == "open":
                t["status"] = "sold"
                t["profit"] = profit
                t["digit"] = final_digit
                matched_symbol = t.get("symbol", "")
                break

        step_at_win = tracker.step  # Must capture before record_win / record_loss modifies it
        
        # CRITICAL FIX: Capture features BEFORE record_win/loss clears them
        features_to_learn = tracker.features
        ctx_to_learn = tracker.global_ctx
        stake_at_trade = tracker.stake
        
        # Update REAL stats (completely separate from virtual)
        self._session_pnl += profit
        if won:
            tracker.record_win()
            self._real_stats["wins"][side] += 1
            self._last_real_result[symbol][side] = True
            self._log_ai(f"[Martingale/{symbol}/{side}] WIN. Resetting stake.")
        else:
            tracker.record_loss()
            self._real_stats["losses"][side] += 1
            self._last_real_result[symbol][side] = False
            self._log_ai(f"[Martingale/{symbol}/{side}] LOSS. Escalating stake to ${tracker.stake:.2f}.")

            if tracker.step >= 4:
                import random
                pause_dur = random.randint(20, 60)
                self._pause_until = time.time() + pause_dur
                self._log_ai(f"⚠️ [CALIBRATION PAUSE] {tracker.step} consecutive losses on {side.upper()}! Pausing real trades for {pause_dur}s to recalibrate AI.")

        # Track max loss and max win with timestamps
        if profit < 0 and abs(profit) > abs(self._max_loss["amount"]):
            self._max_loss = {"amount": profit, "timestamp": now_str, "symbol": matched_symbol, "side": side}
            self._log_ai(f"[MAX-LOSS] New max loss: ${profit:.2f} at {now_str} on {matched_symbol} ({side})")
        if profit > 0 and profit > self._max_win["amount"]:
            self._max_win = {"amount": profit, "timestamp": now_str, "symbol": matched_symbol, "side": side}
            self._log_ai(f"[MAX-WIN] New max win: ${profit:.2f} at {now_str} on {matched_symbol} ({side})")

        # PERFECT TRADE toast: won on step 0 or 1 (within 2 or fewer losses)
        if won and step_at_win <= 1:  # Won first try or after exactly 1 loss
            toast_msg = f"🎯 PERFECT TRADE! {side.upper()} 5 on {matched_symbol} | +${profit:.2f} | {now_str}"
            self._toast_queue.append({"message": toast_msg, "timestamp": now_str, "type": "success"})
            self._log_ai(f"[PERFECT] {toast_msg}")

        # Determine if this is a demo/virtual trade for learning weight scaling
        is_demo = any(
            acc["account_id"] == self._client._current_account_id and acc.get("account_type") == "demo"
            for acc in self._client._accounts
        )
        
        # Add to global shadow context
        self._global_trade_results.append(won)

        # ── ML BRAIN UPDATE ──
        # features_to_learn and ctx_to_learn were captured at line 739,
        # BEFORE record_win/loss cleared them on the tracker.
        recovery_ctx = getattr(tracker, "recovery_context", [
            self._global_tracker_over.step / 10.0,
            self._global_tracker_under.step / 10.0,
            ctx_to_learn
        ])
        recovery_succeeded = won

        try:
            if features_to_learn is not None:
                real_weight = stake_at_trade * 3.0 if final_digit == "5" else stake_at_trade
                if side == "over":
                    self._over_brain.update(
                        market_features=features_to_learn,
                        recovery_context=recovery_ctx,
                        recovery_succeeded=recovery_succeeded,
                        weight=real_weight
                    )
                else:
                    self._under_brain.update(
                        market_features=features_to_learn,
                        recovery_context=recovery_ctx,
                        recovery_succeeded=recovery_succeeded,
                        weight=real_weight
                    )
                
                # Check for periodic save on the over brain (as a proxy for both)
                if len(self._global_trade_results) % 5 == 0:
                    self._over_brain.save()
                    self._under_brain.save()
        except Exception as e:
            logger.error(f"[BRAIN UPDATE ERROR] {e} — skipping this update, continuing trading.")

    # ── Local WebSocket broadcast server ──────────────────────────────────────

    async def _run_broadcast_server(self) -> None:
        import websockets as ws_lib

        async def handler(websocket):
            logger.info("[Broadcast] Frontend connected: %s", websocket.remote_address)
            self._broadcast_clients.add(websocket)
            try:
                async for message in websocket:
                    await self._handle_ui_command(message)
            except ws_lib.exceptions.ConnectionClosed:
                pass
            finally:
                self._broadcast_clients.discard(websocket)

        server = await ws_lib.serve(
            handler,
            config.LOCAL_WS_HOST,
            config.LOCAL_WS_PORT,
        )
        logger.info("[Broadcast] Listening on ws://%s:%d",
                    config.LOCAL_WS_HOST, config.LOCAL_WS_PORT)
        await server.wait_closed()

    async def _broadcast_loop(self) -> None:
        while True:
            snapshot = await self._broadcast_queue.get()
            if not self._broadcast_clients:
                continue
            msg  = json.dumps(snapshot)
            dead = set()
            for ws in list(self._broadcast_clients):
                try:
                    await ws.send(msg)
                except Exception:
                    dead.add(ws)
            self._broadcast_clients -= dead

    async def _handle_ui_command(self, raw_msg: str) -> None:
        try:
            self._log_ai(f"[UI DEBUG] Received raw message from frontend: {raw_msg}")
            data = json.loads(raw_msg)
            cmd = data.get("cmd")
            if cmd == "set_account":
                acc = data.get("account_id")
                if acc:
                    await self._client.switch_to_account(acc)
                    self._log_ai(f"[UI] Account switched to {acc}")
            elif cmd == "set_stake":
                config.BASE_STAKE = float(data.get("stake"))
                self._log_ai(f"[UI] Base stake updated to ${config.BASE_STAKE:.2f}")
            elif cmd == "set_martingale":
                config.MARTINGALE_MULTIPLIER = float(data.get("multiplier"))
                self._log_ai(f"[UI] Martingale multiplier updated to {config.MARTINGALE_MULTIPLIER:.1f}x")
            elif cmd == "start_trades":
                self._is_paused = False
                self._session_start_time = time.time()
                # Reset global lock in case it got stuck from a previous session
                self._global_real_trade_active = False
                self._global_tracker_over.active = False
                self._global_tracker_under.active = False
                self._log_ai("[UI] ▶ Bot STARTED — real trades active.")
            elif cmd == "stop_trades":
                self._is_paused = True
                if self._session_start_time:
                    self._accumulated_uptime += time.time() - self._session_start_time
                    self._session_start_time = None
                self._log_ai("[UI] Bot paused. Awaiting start command.")
            elif cmd == "clear_history":
                self._trade_history.clear()
                self._session_pnl = 0.0
                self._max_loss = {"amount": 0.0, "timestamp": "", "symbol": "", "side": ""}
                self._max_win  = {"amount": 0.0, "timestamp": "", "symbol": "", "side": ""}
                self._max_stake = 0.0
                self._real_stats = {
                    "wins": {"over": 0, "under": 0},
                    "losses": {"over": 0, "under": 0}
                }
                # User requested: Delete button resets martingale
                self._global_tracker_over.record_win()
                self._global_tracker_under.record_win()
                self._accumulated_uptime = 0.0
                if not self._is_paused:
                    self._session_start_time = time.time()
                self._log_ai("[UI] Trade history cleared and Martingale reset to base stake.")
        except Exception as e:
            logger.error("Error parsing UI command: %s", e)

    async def _enqueue_broadcast(self) -> None:
        markets = []
        for sym in config.SYMBOLS:
            hist_len = len(self._tick_history[sym])
            markets.append({
                "symbol": sym,
                "is_ready": hist_len >= 50,
                "tick_count": hist_len,
                "p_over": self._market_stats[sym]["p_over"],
                "p_under": self._market_stats[sym]["p_under"],
                "virtual_err": self._market_stats[sym]["err"]
            })
            
        # REAL trade stats for UI
        r_wins_o = self._real_stats["wins"]["over"]
        r_loss_o = self._real_stats["losses"]["over"]
        r_wins_u = self._real_stats["wins"]["under"]
        r_loss_u = self._real_stats["losses"]["under"]
        
        r_total_o = r_wins_o + r_loss_o
        r_total_u = r_wins_u + r_loss_u
        
        # VIRTUAL stats for virtual trade count display
        v_wins_o = self._engine_stats["wins"]["over"]
        v_wins_u = self._engine_stats["wins"]["under"]
        v_loss_o = self._engine_stats["losses"]["over"]
        v_loss_u = self._engine_stats["losses"]["under"]
        
        # REAL accuracy: actual win rate from real/demo trades placed
        r_total = r_total_o + r_total_u
        r_total_wins = r_wins_o + r_wins_u
        if r_total > 0:
            real_win_rate = r_total_wins / r_total  # e.g. 0.44 = 44%
            avg_err = 1.0 - real_win_rate           # pass as error so frontend shows accuracy = win rate
        else:
            avg_err = 0.5  # No real trades yet — show 50% placeholder
        
        # Determine current account type
        acct_type = "demo"
        if self._client._accounts:
            for acc in self._client._accounts:
                if acc["account_id"] == self._client._current_account_id:
                    acct_type = acc.get("account_type", "demo")
                    break

        # Drain toast queue
        toasts = list(self._toast_queue)
        self._toast_queue.clear()

        # Find the tracker with the highest step across all markets
        dom_over = self._global_tracker_over
        dom_under = self._global_tracker_under

        if self._is_paused:
            status_text = "Paused"
        else:
            status_text = f"Live"

        session_uptime = self._accumulated_uptime
        if self._session_start_time and not self._is_paused:
            session_uptime += time.time() - self._session_start_time

        snapshot = {
            "type": "state_update",
            "is_paused": self._is_paused,
            "status": status_text,
            "elapsed_sec": int(session_uptime),
            "account_type": acct_type,
            "elapsed_seconds": round(time.monotonic() - self._start_time, 1),
            "markets": markets,
            "engine": {
                "wins": {"over": r_wins_o, "under": r_wins_u},
                "losses": {"over": r_loss_o, "under": r_loss_u},
                "win_rate_over": r_wins_o / r_total_o if r_total_o > 0 else 0,
                "win_rate_under": r_wins_u / r_total_u if r_total_u > 0 else 0,
                "avg_error_rate": avg_err,
                "markets_ready": sum(1 for m in markets if m["is_ready"]),
                "balance_guard_over_suppressed": False,
                "balance_guard_under_suppressed": False,
                "session_pnl": round(self._session_pnl, 2),
                "max_loss": self._max_loss,
                "max_win": self._max_win,
                "max_stake": self._max_stake,
                "virtual_wins": {"over": v_wins_o, "under": v_wins_u},
                "virtual_losses": {"over": self._engine_stats["losses"]["over"], "under": self._engine_stats["losses"]["under"]},
                # TRUE prediction accuracy: how often the model's higher-confidence side won
                "ml_prediction_accuracy": round(
                    self._ml_predictions_correct / self._ml_predictions_total, 4
                ) if self._ml_predictions_total > 0 else None,
                "ml_predictions_total": self._ml_predictions_total,
            },
            "toasts": toasts,
            "martingale": {
                "over": {
                    "stake":          dom_over.stake,
                    "step":           dom_over.step,
                    "active":         dom_over.active,
                    "current_market": dom_over.current_market,
                },
                "under": {
                    "stake":          dom_under.stake,
                    "step":           dom_under.step,
                    "active":         dom_under.active,
                    "current_market": dom_under.current_market,
                },
            },
            "accounts": self._client.accounts,
            "current_account_id": self._client._current_account_id,
            "trade_history": self._trade_history,
            "learning_logs": self._learning_logs
        }
        try:
            self._broadcast_queue.put_nowait(snapshot)
        except asyncio.QueueFull:
            pass

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _next_req_id(self) -> int:
        self._req_id_counter += 1
        return self._req_id_counter

