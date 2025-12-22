import os
import asyncio
from typing import Optional
import logging

try:
    from telegram import Bot
    from telegram.error import TelegramError
except ImportError:
    Bot = None
    TelegramError = None

# Config – Render env vars (add these in dashboard)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Lazy bot init – resilient even if creds missing
_bot: Optional[Bot] = None

logger = logging.getLogger(__name__)

def _get_bot() -> Optional[Bot]:
    global _bot
    if _bot is None and Bot and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            _bot = Bot(token=TELEGRAM_BOT_TOKEN)
            logger.info("Telegram bot initialized – alerts armed")
        except Exception as e:
            logger.warning(f"Telegram bot init failed: {e}")
            _bot = None
    return _bot

async def send_alert(message: str) -> bool:
    """
    Send alert to Telegram chat. Returns True if successful.
    """
    bot = _get_bot()
    if not bot:
        logger.debug("Telegram alert skipped: bot not available")
        return False
    
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode="HTML")
        logger.info(f"Telegram alert sent: {message[:50]}...")
        return True
    except TelegramError as e:
        logger.error(f"Telegram send failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected Telegram error: {e}")
        return False

async def send_startup_alert(message: str = "🤖 AI Crypto Oracle Awakened\nAutonomous Swarm Active • Empire Compounding 🔥") -> bool:
    """
    Special startup broadcast – loud and proud.
    """
    return await send_alert(f"<b>{message}</b>")

# Sync wrappers for optional sync calls (rare)
def send_alert_sync(message: str):
    if asyncio.get_event_loop().is_running():
        asyncio.create_task(send_alert(message))
    else:
        asyncio.run(send_alert(message))

def send_startup_alert_sync(message: str = "🤖 AI Crypto Oracle Awakened\nAutonomous Swarm Active • Empire Compounding 🔥"):
    asyncio.create_task(send_startup_alert(message))
