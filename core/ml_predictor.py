import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from data.exchange import ExchangeAdapter
from core.signals import get_signal  # Fuse with TA
from analytics.logger import logger
import pandas as pd
import time

class CryptoDataset(Dataset):
    def __init__(self, data, seq_len=60):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return (self.data[idx:idx+self.seq_len], self.data[idx+self.seq_len][3])  # Predict close price direction (up=1/down=0)

class LSTMOracle(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.sigmoid(self.fc(h_n[-1]))

class MLPredictor:
    def __init__(self):
        self.exchange = ExchangeAdapter('binance')
        self.scaler = MinMaxScaler()
        self.seq_len = 60
        self.lstm = LSTMOracle()
        self.rf = RandomForestClassifier(n_estimators=100)
        self.xgb = XGBClassifier(n_estimators=100)
        self.last_train_time = 0

    def fetch_and_prepare(self, symbol='BTC/USDT', limit=1000):
        df = self.exchange.fetch_ohlcv(symbol, '1h', limit)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        scaled = self.scaler.fit_transform(df)
        labels = (df['close'].shift(-1) > df['close']).astype(int)[:-1]  # 1=up, 0=down
        return scaled, labels

    def train(self):
        if time.time() - self.last_train_time < 86400:  # Retrain daily
            return
        data, labels = self.fetch_and_prepare()
        
        # LSTM train
        dataset = CryptoDataset(data[:-1], self.seq_len)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.lstm.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        for epoch in range(10):  # Quick epochs
            for x, y in loader:
                optimizer.zero_grad()
                out = self.lstm(x.float())
                loss = criterion(out.squeeze(), y.float())
                loss.backward()
                optimizer.step()
        logger.info("LSTM Oracle retrained – foresight sharpened")
        
        # RF & XGB train
        features = data[:-1, :]
        self.rf.fit(features, labels)
        self.xgb.fit(features, labels)
        logger.info("Ensemble Oracles awakened")
        
        self.last_train_time = time.time()

    def predict_direction(self, data):
        self.train()  # Check if retrain needed
        
        scaled = self.scaler.transform(data[-self.seq_len:])  # Last seq
        lstm_in = torch.tensor(scaled).unsqueeze(0).float()
        lstm_pred = self.lstm(lstm_in).item() > 0.5  # 1=up
        
        rf_pred = self.rf.predict(scaled[-1].reshape(1, -1))[0]
        xgb_pred = self.xgb.predict(scaled[-1].reshape(1, -1))[0]
        
        votes = [lstm_pred, rf_pred, xgb_pred]
        return 'buy' if sum(votes) > 1.5 else 'sell' if sum(votes) < 1.5 else None

def get_ml_signal(df):
    predictor = MLPredictor()
    ml_dir = predictor.predict_direction(df.values)
    ta_sig = get_signal(df)  # Fuse with existing TA
    return ml_dir if ml_dir else ta_sig  # Prioritize ML
"""
ML Predictor Module with XGBoost
"""
import logging
from typing import Optional, Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available - predictions will be disabled")


class MLPredictor:
    """Machine Learning Predictor for trading signals"""
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize the ML Predictor
        
        Args:
            model_type: Type of model to use ('xgboost', 'random_forest', etc.)
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_names = []
        
        if not XGBOOST_AVAILABLE and model_type == 'xgboost':
            logger.warning("XGBoost requested but not available")
        
        logger.info(f"MLPredictor initialized with model_type={model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Train the model
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            feature_names: Optional feature names
        """
        if not XGBOOST_AVAILABLE:
            logger.error("Cannot train - XGBoost not available")
            return False
        
        try:
            self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
            
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names)
            
            # XGBoost parameters
            params = {
                'max_depth': 6,
                'eta': 0.3,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss'
            }
            
            # Train model
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                verbose_eval=False
            )
            
            self.is_trained = True
            logger.info(f"Model trained successfully on {X.shape[0]} samples")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Make predictions on input data
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predictions array or None if failed
        """
        if not XGBOOST_AVAILABLE:
            logger.warning("Cannot predict - XGBoost not available")
            return None
        
        if not self.is_trained:
            logger.warning("Cannot predict - model not trained")
            return None
        
        try:
            dtest = xgb.DMatrix(X, feature_names=self.feature_names)
            predictions = self.model.predict(dtest)
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict probabilities
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions
        """
        return self.predict(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores
        
        Returns:
            Dictionary of feature names to importance scores
        """
        if not self.is_trained:
            logger.warning("Model not trained")
            return None
        
        try:
            importance = self.model.get_score(importance_type='weight')
            return importance
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return None
    
    def save_model(self, filepath: str):
        """
        Save model to file
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            logger.warning("Cannot save - model not trained")
            return False
        
        try:
            self.model.save_model(filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str):
        """
        Load model from file
        
        Args:
            filepath: Path to load the model from
        """
        if not XGBOOST_AVAILABLE:
            logger.error("Cannot load - XGBoost not available")
            return False
        
        try:
            self.model = xgb.Booster()
            self.model.load_model(filepath)
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Evaluate model performance
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        if predictions is None:
            return None
        
        try:
            # Convert probabilities to binary predictions
            binary_preds = (predictions > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = np.mean(binary_preds == y)
            
            # True positives, false positives, etc.
            tp = np.sum((binary_preds == 1) & (y == 1))
            fp = np.sum((binary_preds == 1) & (y == 0))
            tn = np.sum((binary_preds == 0) & (y == 0))
            fn = np.sum((binary_preds == 0) & (y == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
            
            logger.info(f"Model evaluation: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None
# In _safe_import() section – fix logger ordering + default
logger = None  # Define early
try:
    from analytics.logger import logger as _l
    logger = _l
except Exception as e:
    print(f"[WARN] analytics.logger unavailable: {e}")

# Update ML warning path if needed
try:
    from core.ml_predictor import get_ml_signal as _gms  # Correct path
    get_ml_signal = _gms
except Exception as e:
    print(f"[WARN] ML predictor not available: {e}")

# Telegram → move to analytics/telegram.py if using previous version
try:
    from analytics.telegram import send_alert as _sa, send_startup_alert as _ssa
    send_alert, send_startup_alert = _sa, _ssa
except Exception as e:
    print(f"[WARN] Telegram alerts not available: {e}")
