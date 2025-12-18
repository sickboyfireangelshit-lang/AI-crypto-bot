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
