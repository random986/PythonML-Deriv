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
        
        self._over_brain = GatekeeperBrain("OVER")
        self._under_brain = GatekeeperBrain("UNDER")
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
        if len(history) < 30:
            return [0.0, 0.0, 0.0]
        recent = history[-30:]
        over5 = sum(1 for x in recent if int(str(x).replace('.', '')[-1]) > 5)
        evens = sum(1 for x in recent if int(str(x).replace('.', '')[-1]) % 2 == 0)
        momentum = sum(abs(recent[i] - recent[i-1]) for i in range(1, len(recent)))
        return [momentum / 30.0, over5 / 30.0, evens / 30.0]

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
        if len(self._tick_history[symbol]) > 50:
            self._tick_history[symbol].pop(0)

        # Always broadcast state snapshot so UI can track market readiness
        await self._enqueue_broadcast()

        # Only evaluate if we have enough history to form features
        if len(self._tick_history[symbol]) < 50:
            return

        # Extract continuous features
        features = self._extract_continuous_tensor(self._tick_history[symbol])
        global_ctx = self._get_global_shadow_context()

        # Get continuous ML verdicts
        over_v, over_c = self._over_brain.get_verdict(features, global_ctx)
        under_v, under_c = self._under_brain.get_verdict(features, global_ctx)
        
        # State key is no longer a string, just a placeholder for the virtual queues
        state_key = "CONTINUOUS_ML"
        
        # Update UI stats
        self._market_stats[symbol]["p_over"] = over_c
        self._market_stats[symbol]["p_under"] = under_c

        # ── VIRTUAL TRADES — always fire both sides, all markets, no rules ──
        # The brain learns purely from outcomes. No hardcoded gating.
        v_tracker_over = self._virtual_trackers_over[symbol]
        v_tracker_under = self._virtual_trackers_under[symbol]
        
        # Determine the model's actual prediction: whichever side it's MORE confident about.
        # Only the OVER-side vtrade carries this flag so we count once per cycle.
        predicted_side = "over" if over_c >= under_c else "under"
        
        if not v_tracker_over.active:
            v_tracker_over.features = features
            v_tracker_over.global_ctx = global_ctx
            self._queue_virtual_trade(symbol, "over", state_key, over_c * 100, predicted_side=predicted_side)
        if not v_tracker_under.active:
            v_tracker_under.features = features
            v_tracker_under.global_ctx = global_ctx
            self._queue_virtual_trade(symbol, "under", state_key, under_c * 100)

        # ── REAL TRADES ────────────────────────────────────────────────
        if self._is_paused:
            return
            
        # During warm-up, bypass the AI gate so the model can gather real data.
        in_warmup = (
            self._over_brain.update_count < self._over_brain.WARMUP_UPDATES or
            self._under_brain.update_count < self._under_brain.WARMUP_UPDATES
        )
        # The continuous_brain no longer returns BLOCK. It always returns EVALUATE and the true Quality Score.
        # ── DYNAMIC CROSS-MARKET PROBABILITY SCALING ──
        # The user explicitly requested to avoid hardcoded confidence levels.
        # Instead, we look at which side needs recovery (Martingale step > 0),
        # and we scan all 15 markets to ensure THIS market is currently the BEST market for that side.
        over_step = self._global_tracker_over.step
        under_step = self._global_tracker_under.step
        
        priority_side = "none"
        if over_step > under_step:
            priority_side = "over"
        elif under_step > over_step:
            priority_side = "under"
            
        best_score = 0.0
        for m_stat in self._market_stats.values():
            if priority_side == "over":
                score = m_stat.get("p_over", 0.0)
            elif priority_side == "under":
                score = m_stat.get("p_under", 0.0)
            else:
                score = max(m_stat.get("p_over", 0.0), m_stat.get("p_under", 0.0))
            if score > best_score:
                best_score = score
                
        if priority_side == "over":
            current_score = over_c
        elif priority_side == "under":
            current_score = under_c
        else:
            current_score = max(over_c, under_c)
            
        # We execute if this market is the best market available right now.
        # We also enforce a bare minimum logical edge (>50%) so it doesn't fire on inherently bad trades.
        if not in_warmup:
            if current_score < best_score or current_score <= 0.50:
                return
            
        # GLOBAL LOCK: Only one trade at a time across ALL 15 markets.
        if getattr(self, "_global_real_trade_active", False):
            import time
            if hasattr(self, "_last_real_trade_time") and time.time() - self._last_real_trade_time > 15.0:
                self._log_ai("⚠️ [WATCHDOG] Lock timeout — forcing release.")
                self._global_tracker_over.active = False
                self._global_tracker_under.active = False
                self._global_real_trade_active = False
            else:
                return  # Trade already in-flight, skip this tick

        # COOLDOWN: only one trade every 2 seconds to avoid tick-flood
        import time
        now = time.time()
        if now - getattr(self, "_last_real_trade_time", 0) < 2.0:
            return

        # ── FIRE ────────────────────────────────────────────────────────
        self._global_real_trade_active = True
        self._last_real_trade_time = now
        
        # Save features for learning when the trade resolves
        self._global_tracker_over.features = features
        self._global_tracker_over.global_ctx = global_ctx
        self._global_tracker_under.features = features
        self._global_tracker_under.global_ctx = global_ctx

        self._log_ai(f"🚀 [TRADE] Firing on {symbol} | OVER={over_c*100:.0f}% UNDER={under_c*100:.0f}% | warmup={'YES' if in_warmup else 'NO'}")

        import asyncio
        asyncio.create_task(
            self._dual_execute(symbol, state_key, over_c * 100, under_c * 100)
        )

    async def _dual_execute(self, symbol: str, state_key: str, p_over: float, p_under: float) -> None:
        """Fires both sides at the exact same instance."""
        import asyncio
        try:
            await asyncio.gather(
                self._execute_side(symbol, "over", state_key, p_over),
                self._execute_side(symbol, "under", state_key, p_under)
            )
        except Exception as e:
            logger.error(f"[DUAL EXECUTE ERROR] {e}")
            self._global_real_trade_active = False

    async def _execute_side(self, symbol: str, side: str, state_key: str, conf: float) -> None:
        if not self._client.is_auth_ready:
            return
            
        tracker = self._global_tracker_over if side == "over" else self._global_tracker_under
        
        if tracker.active:
            return
            
        tracker.active = True
        tracker.current_market = symbol
        # Note: state_key is no longer stored on tracker (replaced by features/global_ctx)
        
        self._log_ai(f"[DECISION] {symbol} | {side.upper()} 5 | conf={conf:.1f}%")
        await self._fire_trade(symbol, side, tracker.stake, state_key)

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

        if side == "over":
            self._over_brain.update(features, global_ctx, won)
        else:
            self._under_brain.update(features, global_ctx, won)
            
        tracker.current_market = ""

    # ── Trade execution ───────────────────────────────────────────────────────

    async def _fire_trade(self, symbol: str, side: str, stake: float, state_key: str) -> None:
        if stake > self._max_stake:
            self._max_stake = stake
        contract_type = "DIGITOVER" if side == "over" else "DIGITUNDER"
        req_id = self._next_req_id()

        payload = {
            "buy": "1",
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
            self._global_real_trade_active = False
            logger.error("[TRADE] Buy failed (req_id=%d): %s", req_id, msg["error"])
            if entry:
                side, symbol, stake, state_key = entry
                tracker = self._global_tracker_over if side == "over" else self._global_tracker_under
                tracker.active = False
            for t in self._trade_history:
                if t.get("req_id") == req_id:
                    t["status"] = "failed"
                    break
            return

        buy_data    = msg.get("buy", {})
        contract_id = str(buy_data.get("contract_id", ""))

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
                    if t.get("side") == tracker.side and t.get("status") == "open":
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

        # Find the matching trade entry
        matched_symbol = ""
        for t in self._trade_history:
            if t.get("side") == side and t.get("status") == "open":
                t["status"] = "sold"
                t["profit"] = profit
                t["digit"] = final_digit
                matched_symbol = t.get("symbol", "")
                break

        step_at_win = tracker.step  # Must capture before record_win / record_loss modifies it
        
        # CRITICAL FIX: Capture features BEFORE record_win/loss clears them
        features_to_learn = tracker.features
        ctx_to_learn = tracker.global_ctx
        
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

        if side == "over":
            self._over_brain.update(features_to_learn, ctx_to_learn, won)
            # We also save to disk occasionally
            if len(self._global_trade_results) % 5 == 0:
                self._over_brain.save()
        else:
            self._under_brain.update(features_to_learn, ctx_to_learn, won)
            if len(self._global_trade_results) % 5 == 0:
                self._under_brain.save()

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

