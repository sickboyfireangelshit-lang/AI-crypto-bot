"""Telegram Alerts Module"""
import logging
logger = logging.getLogger(__name__)

class TelegramAlerts:
    def __init__(self, token=None, chat_id=None):
        self.enabled = False
        logger.warning("Telegram alerts disabled - no token provided")
    
    def send_alert(self, message):
        logger.debug(f"Alert (not sent): {message}")
        return False

telegram_alerts = TelegramAlerts()
