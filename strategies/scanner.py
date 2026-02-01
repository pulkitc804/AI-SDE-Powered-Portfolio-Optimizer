import pandas as pd
import numpy as np
import torch
from data.loader import MarketDataLoader
from models.neural_sde import NeuralSDE
from engine.trainer import SDETrainer
from engine.lsm import LongstaffSchwartzPricer
from config.settings import DEVICE

class MarketScanner:
    def __init__(self, tickers):
        self.tickers = tickers
        self.results = []

    def scan_market(self):
        print(f"\n[SCANNER] Starting AI Analysis on {len(self.tickers)} assets...")

        for ticker in self.tickers:
            try:
                print(f" > Analyzing {ticker}...", end=" ")

                # 1. Fetch Data
                loader = MarketDataLoader(ticker=ticker, lookback=90)
                X, y, current_spot, scaler = loader.fetch_realtime_data()
                spot = current_spot.item()

                # 2. Train Model (Fast Mode for Scanning)
                model = NeuralSDE(input_dim=3, hidden_dim=128, num_layers=2)
                trainer = SDETrainer(model)
                trainer.train(X, y, epochs=50) # Fewer epochs for speed

                # 3. Predict AI Volatility
                last_seq = torch.tensor(X[-1], dtype=torch.float32).to(DEVICE)
                ai_paths = trainer.predict_future(last_seq, spot, n_paths=1000)
                ai_vol = np.std(ai_paths) * np.sqrt(12) # Annualized AI Vol

                # 4. Calculate Historical Vol (The "Market" View)
                hist_returns = y
                mkt_vol = np.std(hist_returns) * np.sqrt(252)

                # 5. The "Edge"
                vol_edge = ai_vol - mkt_vol

                # 6. Recommendation
                if vol_edge > 0.05:
                    signal = "BUY VOL (Options Cheap)"
                elif vol_edge < -0.05:
                    signal = "SELL VOL (Options Expensive)"
                else:
                    signal = "FAIR VALUE"

                print(f"Done. Edge: {vol_edge:.2%}")

                self.results.append({
                    'Ticker': ticker,
                    'Price': spot,
                    'Market_Vol': mkt_vol,
                    'AI_Vol': ai_vol,
                    'Edge': vol_edge,
                    'Signal': signal
                })

            except Exception as e:
                print(f"Error: {e}")

        return pd.DataFrame(self.results)

if __name__ == "__main__":
    # Top Tech & Volatility Names
    tickers = ['AAPL', 'NVDA', 'TSLA', 'AMD', 'AMZN', 'GOOGL', 'MSFT', 'META']
    scanner = MarketScanner(tickers)
    df = scanner.scan_market()

    print("\n" + "="*50)
    print("      NEURAL MARKET SCANNER RESULTS      ")
    print("="*50)
    print(df.sort_values(by='Edge', ascending=False).to_string(index=False))