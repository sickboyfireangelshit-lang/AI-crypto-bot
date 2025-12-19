import os
from dotenv import load_dotenv

load_dotenv()  # Loads .env locally only

class Config:
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
    SECRET_KEY = os.getenv('SECRET_KEY', 'change-me-to-strong-secret')
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
    RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.02'))
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '10000'))
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')  # For alerts
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
