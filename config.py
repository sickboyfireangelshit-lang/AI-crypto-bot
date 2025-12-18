import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
    SECRET_KEY = os.getenv('SECRET_KEY', 'change-me-to-strong-secret')
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
    RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.02))  # 2%
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 10000))
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
