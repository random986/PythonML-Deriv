"""
token_test.py
=============
Validates the new Deriv API v2 credentials (PAT Bearer token + App ID).

Tests performed:
  1. REST GET /trading/v1/options/accounts
     - Confirms Bearer token + App-ID are accepted
     - Prints account list (demo / real)

  2. REST POST /trading/v1/options/accounts/{id}/otp
     - Confirms the OTP endpoint works and returns a WS URL

  3. WebSocket connect using OTP URL
     - Sends {"balance": 1, "subscribe": 1} and reads first response
     - Confirms the authenticated WS works end-to-end

  4. Public WebSocket (no auth)
     - Subscribes to R_50 ticks for 3 seconds
     - Confirms market data is streaming

Run:
    python -X utf8 token_test.py
"""

import asyncio
import json
import sys
import time

import aiohttp
import websockets

import config

# Force UTF-8 output on Windows
if sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

def _headers():
    return {
        "Authorization": f"Bearer {config.BEARER_TOKEN}",
        "Deriv-App-ID":  config.APP_ID,
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }

async def test_rest_accounts(session) -> list[dict]:
    """Test 1: list accounts via REST."""
    print(f"\n{INFO} Test 1 — REST GET /trading/v1/options/accounts")
    url = config.DERIV_REST_BASE + config.ACCOUNTS_ENDPOINT
    print(f"  URL: {url}")
    print(f"  PAT: {config.BEARER_TOKEN[:20]}...")
    print(f"  App: {config.APP_ID}")

    try:
        async with session.get(url, headers=_headers(), timeout=aiohttp.ClientTimeout(total=15)) as resp:
            body = await resp.json()
            if resp.status == 200:
                accounts = body.get("data", [])
                print(f"{PASS} Authenticated! Found {len(accounts)} account(s).")
                for acc in accounts:
                    print(f"     Account: {acc.get('account_id')} | "
                          f"type={acc.get('account_type')} | "
                          f"balance={acc.get('balance')} {acc.get('currency')} | "
                          f"status={acc.get('status')}")
                return accounts
            else:
                print(f"{FAIL} HTTP {resp.status}")
                print(f"       Response: {json.dumps(body, indent=2)}")
                return []
    except aiohttp.ClientError as e:
        print(f"{FAIL} Network error: {e}")
        return []


async def test_otp_endpoint(session, account_id: str) -> str | None:
    """Test 2: get OTP WebSocket URL."""
    print(f"\n{INFO} Test 2 — REST POST OTP for account {account_id}")
    endpoint = config.OTP_ENDPOINT_TMPL.format(account_id)
    url = config.DERIV_REST_BASE + endpoint
    print(f"  URL: {url}")

    try:
        async with session.post(url, headers=_headers(), timeout=aiohttp.ClientTimeout(total=15)) as resp:
            body = await resp.json()
            if resp.status == 200:
                ws_url = body["data"]["url"]
                print(f"{PASS} OTP URL obtained.")
                print(f"       WS URL: {ws_url[:60]}...")
                return ws_url
            else:
                print(f"{FAIL} HTTP {resp.status}")
                print(f"       Response: {json.dumps(body, indent=2)}")
                return None
    except aiohttp.ClientError as e:
        print(f"{FAIL} Network error: {e}")
        return None


async def test_auth_websocket(ws_url: str) -> None:
    """Test 3: connect using OTP URL and request balance."""
    print(f"\n{INFO} Test 3 — Authenticated WebSocket (balance)")
    print(f"  Connecting...")

    try:
        async with websockets.connect(ws_url, open_timeout=10) as ws:
            # Request balance
            await ws.send(json.dumps({"balance": 1, "req_id": 1}))
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            msg = json.loads(raw)
            if msg.get("msg_type") == "balance":
                bal = msg.get("balance", {})
                print(f"{PASS} Balance received!")
                print(f"       Balance: {bal.get('balance')} {bal.get('currency')}")
                print(f"       LoginID: {bal.get('loginid')}")
            elif "error" in msg:
                print(f"{FAIL} API error: {msg['error']}")
            else:
                print(f"{INFO} Got msg_type={msg.get('msg_type')} (OK — WS works)")
    except websockets.exceptions.WebSocketException as e:
        print(f"{FAIL} WebSocket error: {e}")
    except asyncio.TimeoutError:
        print(f"{FAIL} Timed out waiting for balance response.")
    except Exception as e:
        print(f"{FAIL} Unexpected error: {e}")


async def test_public_ticks() -> None:
    """Test 4: public WebSocket tick streaming (no auth)."""
    print(f"\n{INFO} Test 4 — Public WebSocket tick stream (R_50, 5 seconds)")
    print(f"  URL: {config.DERIV_PUBLIC_WS}")

    tick_count = 0
    deadline = time.monotonic() + 5

    try:
        async with websockets.connect(config.DERIV_PUBLIC_WS, open_timeout=10) as ws:
            await ws.send(json.dumps({"ticks": "R_50", "subscribe": 1}))
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                msg = json.loads(raw)
                if msg.get("msg_type") == "tick":
                    tick = msg.get("tick", {})
                    tick_count += 1
                    if tick_count <= 5:
                        print(f"     Tick {tick_count}: R_50 = {tick.get('quote')}")

        if tick_count > 0:
            print(f"{PASS} Received {tick_count} tick(s) in 5 seconds.")
        else:
            print(f"{FAIL} No ticks received in 5 seconds.")
    except Exception as e:
        print(f"{FAIL} Public WS error: {e}")


async def main():
    print("=" * 60)
    print("  Deriv API v2 — Connection Validation")
    print("=" * 60)
    print(f"\n  PAT Token : {config.BEARER_TOKEN[:20]}...")
    print(f"  App ID    : {config.APP_ID}")
    print(f"  REST Base : {config.DERIV_REST_BASE}")

    async with aiohttp.ClientSession() as session:
        # Test 1: Accounts
        accounts = await test_rest_accounts(session)

        # Test 2 + 3: OTP + authenticated WS
        if accounts:
            # Prefer demo account for testing
            demo = next((a for a in accounts if a.get("account_type") == "demo"), None)
            test_account = demo or accounts[0]
            account_id = test_account["account_id"]

            ws_url = await test_otp_endpoint(session, account_id)
            if ws_url:
                await test_auth_websocket(ws_url)
            else:
                print(f"\n{FAIL} Cannot test authenticated WS — no OTP URL.")
        else:
            print(f"\n{FAIL} Cannot test OTP — no accounts returned.")

    # Test 4: Public WS (no account/token needed)
    await test_public_ticks()

    print("\n" + "=" * 60)
    print("  Validation complete.")
    print("  If all tests PASS, run:  python main.py")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
