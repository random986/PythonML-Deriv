"""
main.py
=======
Entry point for the Deriv Algorithmic Trading System.

Run with:
    python main.py

Before running, make sure you have edited config.py to add your
DEMO_TOKEN and REAL_TOKEN.
"""

import asyncio
import logging
import sys
import os

# Force UTF-8 output on Windows to prevent UnicodeEncodeError with print()
if sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from deriv_client import DerivClient
from trading_bot import DerivTradingBot
import config


# ---------------------------------------------------------------------------
# Logging setup — writes to both console and a rotating log file
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]

    try:
        from logging.handlers import RotatingFileHandler
        handlers.append(
            RotatingFileHandler(
                "deriv_bot.log",
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=3,
                encoding="utf-8",
            )
        )
    except Exception:
        pass  # Log to console only if file handler fails

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )

    # Suppress noisy websockets library debug output
    logging.getLogger("websockets").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# System startup banner
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║        DERIV ALGORITHMIC TRADING SYSTEM — STARTED           ║
╠══════════════════════════════════════════════════════════════╣
║  Markets   : {len(config.SYMBOLS)} symbols monitored                          ║
║  Demo time : {config.DEMO_DURATION_SECONDS // 60} minutes ({config.DEMO_DURATION_SECONDS}s) of shadow learning           ║
║  Base stake: ${config.BASE_STAKE:<10.2f}                                    ║
║  Recovery   : Infinite (No limit), 2x multiplier                  ║
║  Broadcast : {"ws://" + config.LOCAL_WS_HOST + ":" + str(config.LOCAL_WS_PORT) if config.ENABLE_LOCAL_WS_BROADCAST else "disabled":<35}   ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


# ---------------------------------------------------------------------------
# Credential validation
# ---------------------------------------------------------------------------

def _validate_config() -> None:
    if not hasattr(config, "BEARER_TOKEN") or config.BEARER_TOKEN.startswith("YOUR_"):
        print(
            "ERROR: Please set your BEARER_TOKEN (pat_...) in config.py before running.\n"
            "       Get an API token from: https://app.deriv.com/account/api-token\n"
        )
        sys.exit(1)

    if not hasattr(config, "APP_ID") or config.APP_ID.startswith("YOUR_"):
        print(
            "ERROR: Please set your APP_ID in config.py before running.\n"
            "       Get an App ID from: https://api.deriv.com/app-registration/\n"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main coroutine
# ---------------------------------------------------------------------------

async def main(force_real: bool = False) -> None:
    _setup_logging()
    _print_banner()
    _validate_config()

    logger = logging.getLogger("main")

    # ── 1. The Continuous ML Engine handles its own memory state ──────────────
    logger.info("Starting Shadow Prophet ML Engine...")

    # ── 2. Create and start the trading bot ───────────────────────────────────
    bot = DerivTradingBot()

    logger.info("Starting trading bot...")
    try:
        await bot.start(force_real=force_real)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down cleanly.")
    except Exception as exc:
        logger.critical("Fatal error in trading loop: %s", exc, exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Deriv Algorithmic Trading Bot")
    parser.add_argument("--real", action="store_true", help="Start directly in real trading mode, skipping demo phase")
    args = parser.parse_args()

    try:
        asyncio.run(main(force_real=args.real))
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
