
import asyncio
from telegram import Bot
from telegram.error import TelegramError
from config import Config
from analytics.logger import logger

bot = Bot(token=Config.TELEGRAM_TOKEN) if Config.TELEGRAM_TOKEN else None
chat_id = Config.TELEGRAM_CHAT_ID

async def send_alert(message: str):
    if not bot or not chat_id:
        logger.warning("Telegram not configured – alert skipped")
        return
    
    try:
        await bot.send_message(chat_id=chat_id, text=message, parse_mode='HTML')
        logger.info(f"Telegram alert sent: {message}")
    except TelegramError as e:
        logger.error(f"Telegram send failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected Telegram error: {e}")

async def send_startup_alert():
    await send_alert("<b>🤖 AI Crypto Oracle Awakened</b>\n"
                     "ML models loaded • Swarm deployed • Autonomous trading live\n"
                     "The empire begins stacking. 🔥")
