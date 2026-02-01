import yfinance as yf
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

class MarketDataLoader:
    def __init__(self, ticker, lookback=90):
        self.ticker = ticker
        self.lookback = lookback
        self.scaler = MinMaxScaler()

    def download_data(self, period_days=None):
        """
        Downloads raw data from Yahoo Finance.
        Used by the Scanner and Backtester.
        """
        # If no specific period provided, use slightly more than the lookback
        days = period_days if period_days else self.lookback * 2

        # Calculate start date
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')

        try:
            # Download data
            df = yf.download(self.ticker, start=start_date, progress=False)

            # Fix for recent yfinance versions returning MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Ensure we have data
            if df.empty:
                raise ValueError(f"No data found for {self.ticker}")

            return df

        except Exception as e:
            print(f"Error downloading {self.ticker}: {e}")
            return pd.DataFrame()

    def fetch_realtime_data(self):
        """
        Prepares normalized Tensors for the Neural SDE model.
        Used by the Forecast Engine.
        """
        # 1. Get Data
        df = self.download_data()
        if df.empty:
            raise ValueError("Empty dataset")

        # 2. Preprocess (Log Returns & Normalization)
        closes = df['Close'].values.reshape(-1, 1)
        normalized_data = self.scaler.fit_transform(closes)

        # 3. Create Sequences for LSTM/SDE
        X, y = [], []
        # We need at least 'lookback' data points
        if len(normalized_data) <= self.lookback:
            # Fallback if not enough data: duplicate the data we have
            normalized_data = np.pad(normalized_data, ((self.lookback, 0), (0,0)), mode='edge')

        for i in range(len(normalized_data) - self.lookback):
            X.append(normalized_data[i:i+self.lookback])
            y.append(normalized_data[i+1:i+self.lookback+1]) # Next step prediction target

        # 4. Convert to Tensors
        X = np.array(X)
        y = np.array(y)

        # Returns: Inputs, Targets, Current Spot Price, Scaler
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(closes[-1], dtype=torch.float32),
            self.scaler
        )