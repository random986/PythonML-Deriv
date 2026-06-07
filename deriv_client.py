"""
deriv_client.py
===============
DerivClient — handles the new Deriv API v2 authentication and connection flow.

New API flow (no OAuth, no redirects — pure PAT for local desktop apps):

  Step 1  REST GET  /trading/v1/options/accounts
          → returns list of account IDs (e.g. "DOT90004580")

  Step 2  REST POST /trading/v1/options/accounts/{accountId}/otp
          → returns a short-lived authenticated WebSocket URL
            e.g. "wss://api.derivws.com/trading/v1/options/ws/demo?otp=abc123"

  Step 3  WebSocket connect to that URL
          → authenticated session, can send buy/sell/portfolio/balance messages

  Public ticks (no auth):
          WebSocket connect to wss://api.derivws.com/trading/v1/options/ws/public
          → subscribe with {"ticks": "R_50", "subscribe": 1}

Every REST call sends:
  Authorization: Bearer pat_...
  Deriv-App-ID:  <app_id>
  Content-Type:  application/json

No browser redirects. No OAuth flow. Token is hardcoded for desktop use.
"""

import asyncio
import json
import logging
from typing import Optional

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP headers used for every REST call
# ---------------------------------------------------------------------------
def _auth_headers() -> dict:
    return {
        "Authorization": f"Bearer {config.BEARER_TOKEN}",
        "Deriv-App-ID":  config.APP_ID,
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }


# ---------------------------------------------------------------------------
# REST helpers
# ---------------------------------------------------------------------------

async def fetch_accounts(session: aiohttp.ClientSession) -> list[dict]:
    """
    GET /trading/v1/options/accounts
    Returns a list of account dicts:
      [{"account_id": "DOT90004580", "balance": 10000, "currency": "USD",
        "account_type": "demo", "status": "active"}, ...]
    """
    url = config.DERIV_REST_BASE + config.ACCOUNTS_ENDPOINT
    logger.info("[REST] GET %s", url)
    async with session.get(url, headers=_auth_headers()) as resp:
        body = await resp.json()
        if resp.status != 200:
            raise RuntimeError(
                f"GET accounts failed ({resp.status}): {body}"
            )
        accounts = body.get("data", [])
        logger.info("[REST] Got %d account(s).", len(accounts))
        for acc in accounts:
            logger.info(
                "  Account: %s | type=%s | balance=%s %s | status=%s",
                acc.get("account_id"), acc.get("account_type"),
                acc.get("balance"), acc.get("currency"),
                acc.get("status"),
            )
        return accounts


async def fetch_otp_ws_url(
    session: aiohttp.ClientSession,
    account_id: str,
) -> str:
    """
    POST /trading/v1/options/accounts/{accountId}/otp
    Returns the ready-to-connect authenticated WebSocket URL.
    The OTP is embedded in the URL — connect immediately before it expires.
    """
    endpoint = config.OTP_ENDPOINT_TMPL.format(account_id)
    url = config.DERIV_REST_BASE + endpoint
    logger.info("[REST] POST %s (requesting OTP WS URL)...", url)
    async with session.post(url, headers=_auth_headers()) as resp:
        body = await resp.json()
        if resp.status != 200:
            raise RuntimeError(
                f"OTP request failed ({resp.status}): {body}"
            )
        ws_url = body["data"]["url"]
        logger.info("[REST] OTP WS URL obtained.")
        return ws_url


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------

class DerivClient:
    """
    Manages the full lifecycle of Deriv API connections.

    - Public WebSocket for tick streaming (no auth, always open)
    - Authenticated WebSocket for trading (OTP-based, refreshed as needed)

    Usage:
        client = DerivClient()
        await client.connect()            # opens public WS + auth WS
        await client.send_trade(payload)  # sends to auth WS
        # client.on_tick is set by DerivTradingBot
    """

    def __init__(self):
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._public_ws = None          # WebSocket for market data (ticks)
        self._auth_ws   = None          # WebSocket for trading (authenticated)
        self._accounts:  list[dict] = []
        self._demo_account_id:  Optional[str] = None
        self._real_account_id:  Optional[str] = None
        self._current_account_id: Optional[str] = None  # active trading account

        # Callbacks — set by DerivTradingBot
        self.on_tick            = None   # async fn(symbol, price)
        self.on_proposal_resp   = None   # async fn(msg)
        self.on_buy_resp        = None   # async fn(msg)
        self.on_contract_update = None   # async fn(msg)

        self._running = False
        self._tick_sub_task  = None
        self._trade_sub_task = None

    # ── Public interface ──────────────────────────────────────────────────────

    async def start(self, use_demo: bool = True) -> None:
        """
        Open HTTP session, fetch accounts, open both WebSockets.
        Call this once at startup; call switch_to_real() later if needed.
        """
        self._running = True
        self._http_session = aiohttp.ClientSession()

        # Fetch accounts list (REST) with infinite retry on connection drop
        while True:
            try:
                self._accounts = await fetch_accounts(self._http_session)
                break
            except Exception as e:
                logger.error(f"[REST] Error fetching accounts: {e}. Retrying in 5s...")
                await asyncio.sleep(5)
                
        self._classify_accounts()

        target = self._demo_account_id if use_demo else self._real_account_id
        if not target:
            # Fall back to first available account
            target = self._accounts[0]["account_id"] if self._accounts else None
        self._current_account_id = target

        # Launch both WS loops concurrently
        self._tick_sub_task  = asyncio.create_task(self._public_ws_loop())
        self._trade_sub_task = asyncio.create_task(self._auth_ws_loop())

    async def switch_to_account(self, account_id: str) -> None:
        """
        Called when user switches account via the UI.
        Reconnects the auth WS using the specified account.
        The public tick WS is unaffected.
        """
        if account_id == self._current_account_id:
            return
            
        logger.info("[Client] Switching auth WS to account: %s", account_id)
        self._current_account_id = account_id
        # Cancel and restart auth WS loop
        if self._trade_sub_task:
            self._trade_sub_task.cancel()
        self._trade_sub_task = asyncio.create_task(self._auth_ws_loop())

    async def stop(self) -> None:
        self._running = False
        for task in [self._tick_sub_task, self._trade_sub_task]:
            if task:
                task.cancel()
        if self._public_ws:
            await self._public_ws.close()
        if self._auth_ws:
            await self._auth_ws.close()
        if self._http_session:
            await self._http_session.close()

    async def send_trade(self, payload: dict) -> bool:
        """Send a trading payload to the authenticated WebSocket."""
        if self._auth_ws is None:
            logger.warning("[Client] Auth WS not ready — cannot send trade payload.")
            return False
        try:
            await self._auth_ws.send(json.dumps(payload))
            return True
        except Exception as exc:
            logger.error("[Client] send_trade error: %s", exc)
            return False

    async def send_public(self, payload: dict) -> None:
        """Send a payload on the public WebSocket (tick subscriptions)."""
        if self._public_ws is None:
            logger.warning("[Client] Public WS not ready.")
            return
        try:
            await self._public_ws.send(json.dumps(payload))
        except Exception as exc:
            logger.error("[Client] send_public error: %s", exc)

    @property
    def accounts(self) -> list[dict]:
        return self._accounts

    @property
    def is_auth_ready(self) -> bool:
        return self._auth_ws is not None

    # ── Internal loops ────────────────────────────────────────────────────────

    async def _public_ws_loop(self) -> None:
        """
        Maintains a persistent connection to the PUBLIC tick WebSocket.
        Reconnects automatically on any error.
        Subscribes to all configured symbols on connect.
        """
        while self._running:
            try:
                logger.info("[PublicWS] Connecting to %s", config.DERIV_PUBLIC_WS)
                async with websockets.connect(
                    config.DERIV_PUBLIC_WS,
                    ping_interval=20,
                    ping_timeout=30,
                ) as ws:
                    self._public_ws = ws
                    logger.info("[PublicWS] Connected. Subscribing to %d symbols...",
                                len(config.SYMBOLS))
                    # Subscribe to all tick streams
                    for sym in config.SYMBOLS:
                        await ws.send(json.dumps({
                            "ticks": sym,
                            "subscribe": 1,
                        }))

                    async for raw in ws:
                        if not self._running:
                            break
                        msg = json.loads(raw)
                        await self._dispatch_public(msg)

            except ConnectionClosed as e:
                logger.warning("[PublicWS] Disconnected: %s. Reconnecting in 3s...", e)
            except asyncio.CancelledError:
                logger.info("[PublicWS] Loop cancelled.")
                return
            except Exception as e:
                logger.error("[PublicWS] Error: %s. Reconnecting in 5s...", e)
                await asyncio.sleep(2)

            self._public_ws = None
            if self._running:
                await asyncio.sleep(3)

    async def _auth_ws_loop(self) -> None:
        """
        Maintains an authenticated trading WebSocket via OTP flow:
          1. POST OTP endpoint → get WS URL with embedded token
          2. Connect immediately before OTP expires
          3. Process trading messages (buy responses, contract updates)
        Reconnects + refreshes OTP automatically on any error.
        """
        while self._running:
            try:
                if not self._current_account_id:
                    logger.warning("[AuthWS] No account ID — waiting 5s...")
                    await asyncio.sleep(5)
                    continue

                # Step 1: Get fresh OTP URL from REST
                ws_url = await fetch_otp_ws_url(
                    self._http_session,
                    self._current_account_id,
                )

                # Step 2: Connect immediately using OTP URL
                logger.info("[AuthWS] Connecting to authenticated WS...")
                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=30,
                ) as ws:
                    self._auth_ws = ws
                    logger.info("[AuthWS] Authenticated session open for account %s.",
                                self._current_account_id)
                    
                    # Globally subscribe to all open contracts for this session
                    await ws.send(json.dumps({"proposal_open_contract": 1, "subscribe": 1}))

                    async for raw in ws:
                        if not self._running:
                            break
                        msg = json.loads(raw)
                        await self._dispatch_auth(msg)

            except ConnectionClosed as e:
                logger.warning("[AuthWS] Disconnected: %s. Re-fetching OTP in 3s...", e)
            except asyncio.CancelledError:
                logger.info("[AuthWS] Loop cancelled.")
                return
            except Exception as e:
                logger.error("[AuthWS] Error: %s. Retrying in 5s...", e, exc_info=True)
                await asyncio.sleep(2)

            self._auth_ws = None
            if self._running:
                await asyncio.sleep(3)

    # ── Message dispatch ──────────────────────────────────────────────────────

    async def _dispatch_public(self, msg: dict) -> None:
        """Route messages from the public (tick) WebSocket."""
        msg_type = msg.get("msg_type")
        if msg_type == "tick" and self.on_tick:
            tick = msg.get("tick", {})
            symbol = tick.get("symbol")
            quote  = tick.get("quote")
            if symbol and quote is not None:
                await self.on_tick(symbol, float(quote))
        elif msg_type == "proposal":
            if self.on_proposal_resp:
                await self.on_proposal_resp(msg)
        elif msg_type == "error":
            logger.error("[PublicWS] API error: %s", msg.get("error"))

    async def _dispatch_auth(self, msg: dict) -> None:
        """Route messages from the authenticated trading WebSocket."""
        msg_type = msg.get("msg_type")
        if msg_type == "proposal":
            if self.on_proposal_resp:
                await self.on_proposal_resp(msg)
        elif msg_type == "buy":
            if self.on_buy_resp:
                await self.on_buy_resp(msg)
        elif msg_type == "proposal_open_contract":
            if self.on_contract_update:
                await self.on_contract_update(msg)
        elif msg_type == "balance":
            bal = msg.get("balance", {})
            logger.info("[AuthWS] Balance update: %s %s",
                        bal.get("balance"), bal.get("currency"))
        elif msg_type == "error":
            logger.error("[AuthWS] API error: %s", msg.get("error"))
        else:
            logger.debug("[AuthWS] Unhandled msg_type: %s", msg_type)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _classify_accounts(self) -> None:
        """Identify demo vs real account IDs from the accounts list."""
        for acc in self._accounts:
            acc_type = acc.get("account_type", "")
            acc_id   = acc.get("account_id", "")
            if acc_type == "demo" and not self._demo_account_id:
                self._demo_account_id = acc_id
                logger.info("[Client] Demo account: %s (balance=%s %s)",
                            acc_id, acc.get("balance"), acc.get("currency"))
            elif acc_type != "demo" and not self._real_account_id:
                self._real_account_id = acc_id
                logger.info("[Client] Real account: %s (balance=%s %s)",
                            acc_id, acc.get("balance"), acc.get("currency"))

        if not self._demo_account_id and not self._real_account_id:
            logger.warning(
                "[Client] Could not classify accounts. Using first available: %s",
                self._accounts[0]["account_id"] if self._accounts else "NONE"
            )
