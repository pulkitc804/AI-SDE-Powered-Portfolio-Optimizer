import torch
import numpy as np
import matplotlib.pyplot as plt
from data.loader import MarketDataLoader
from models.neural_sde import NeuralSDE
from models.classical import GeometricBrownianMotion
from engine.trainer import SDETrainer
from engine.lsm import LongstaffSchwartzPricer
from strategies.backtest import AIAlphaStrategy
from analytics.greeks import NeuralGreeks
from config.settings import DEVICE

def main():
    print(f"\n{'='*60}")
    print(f"   NEURAL QUANTITATIVE RESEARCH PLATFORM (NQRP) v3.0")
    print(f"   Architecture: LSTM-SDE | American LSM | Greeks Engine")
    print(f"{'='*60}\n")

    # ---------------------------------------------------------
    # 1. MARKET DATA & REGIME DETECTION
    # ---------------------------------------------------------
    # We use 90 days of history so the LSTM can see the full quarter
    loader = MarketDataLoader(ticker="^GSPC", lookback=90)
    X, y, current_spot, scaler = loader.fetch_realtime_data()
    spot = current_spot.item()

    print(f" [MARKET] Asset: S&P 500 | Price: ${spot:.2f}")

    # ---------------------------------------------------------
    # 2. AI CORE TRAINING
    # ---------------------------------------------------------
    # Initialize the LSTM-SDE (The "Brain")
    model = NeuralSDE(input_dim=3, hidden_dim=256, num_layers=3)
    trainer = SDETrainer(model)

    # Train using Maximum Likelihood Estimation (MLE)
    # The trainer handles the Numpy -> Tensor conversion internally
    trainer.train(X, y, epochs=150)

    # ---------------------------------------------------------
    # 3. COMPARATIVE SIMULATION (AI vs. Classical)
    # ---------------------------------------------------------
    print(f"\n [SIMULATION] Running Parallel Universe Simulations...")

    # A. Neural SDE Paths (The AI's view of the future)
    # FIX: Convert the last sequence to a Tensor explicitly for the prediction engine
    last_seq = torch.tensor(X[-1], dtype=torch.float32).to(DEVICE)
    ai_paths = trainer.predict_future(last_seq, spot, n_paths=5000)

    # B. Classical GBM Paths (The Standard Model view)
    # FIX: 'y' is already a numpy array, so we don't need .cpu().numpy()
    hist_returns = y
    mu_hist = np.mean(hist_returns) * 252
    sigma_hist = np.std(hist_returns) * np.sqrt(252)

    gbm = GeometricBrownianMotion(mu=mu_hist, sigma=sigma_hist)
    gbm_paths = gbm.simulate(spot, T=21/252, dt=1/252, n_paths=5000)

    print(f"   > Classical GBM: mu={mu_hist:.4f}, sigma={sigma_hist:.4f}")

    # ---------------------------------------------------------
    # 4. VALUATION & GREEKS (The "Mega" Part)
    # ---------------------------------------------------------
    # Strike Price is 5% below current price (Out of the Money Put)
    K = spot * 0.95
    lsm = LongstaffSchwartzPricer(r=0.045, dt=1/252)

    # Calculate Prices
    ai_price = lsm.price_option(ai_paths, K, option_type='put')
    gbm_price = lsm.price_option(gbm_paths, K, option_type='put')

    # Calculate Greeks (Delta & Gamma)
    print(f"\n [DERIVATIVES] Calculating High-Order Sensitivities (Greeks)...")
    greeks_engine = NeuralGreeks(trainer, lsm)
    # FIX: Ensure last_seq is passed correctly (it was converted to tensor above)
    delta, gamma = greeks_engine.calculate_delta_gamma(last_seq, spot, K, 21/252)

    # ---------------------------------------------------------
    # 5. STRATEGY BACKTEST
    # ---------------------------------------------------------
    print(f"\n [STRATEGY] Backtesting AI-Driven Trend Strategy...")

    # Get the AI's drift signals from the historical training data
    model.eval()
    with torch.no_grad():
        # FIX: Explicitly convert X (numpy) to Tensor for the model
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        mus, _ = model(X_tensor)
        drift_signals = mus.detach().cpu().numpy().flatten()

    # Run the backtest (Mock simulation using the signals)
    backtester = AIAlphaStrategy()
    dummy_prices = [100 + i for i in range(len(drift_signals))]
    equity_curve, signals = backtester.run_backtest(dummy_prices, drift_signals)

    # ---------------------------------------------------------
    # 6. REPORT & VISUALIZATION
    # ---------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"      FINAL EXECUTIVE SUMMARY      ")
    print(f"{'='*60}")
    print(f" 1. VALUATION (Put Option Strike ${K:.2f})")
    print(f"    > AI Fair Value:       ${ai_price:.4f}")
    print(f"    > Classical GBM Value: ${gbm_price:.4f}")
    print(f"    > Model Premium:       ${(ai_price - gbm_price):.4f}")

    print(f"\n 2. HEDGING PARAMETERS (Greeks)")
    print(f"    > Delta: {delta:.4f} (Hedge Ratio)")
    print(f"    > Gamma: {gamma:.4f} (Convexity)")

    print(f"\n 3. FORECAST METRICS")
    print(f"    > AI Predicted Volatility: {np.std(ai_paths)*np.sqrt(12):.4f}")
    print(f"    > Historical Volatility:   {sigma_hist:.4f}")
    print(f"{'='*60}")

    # PLOTTING DASHBOARD
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.style.use('dark_background')

    # Plot 1: AI Paths
    axes[0, 0].plot(ai_paths[:, :500], color='cyan', alpha=0.05)
    axes[0, 0].plot(ai_paths.mean(axis=1), color='white', linestyle='--', label='AI Mean')
    axes[0, 0].set_title("Neural SDE Simulations (AI)")
    axes[0, 0].legend()

    # Plot 2: GBM Paths
    axes[0, 1].plot(gbm_paths[:, :500], color='orange', alpha=0.05)
    axes[0, 1].plot(gbm_paths.mean(axis=1), color='white', linestyle='--', label='GBM Mean')
    axes[0, 1].set_title("Classical GBM Simulations (Standard)")
    axes[0, 1].legend()

    # Plot 3: Option Payoff Heatmap
    final_prices = ai_paths[-1]
    axes[1, 0].hist(final_prices, bins=50, color='cyan', alpha=0.7, label='Price Dist')
    axes[1, 0].axvline(K, color='magenta', linestyle='--', label=f'Strike ${K:.0f}')
    axes[1, 0].set_title(f"Price Distribution at Maturity")
    axes[1, 0].legend()

    # Plot 4: Drift Signals
    axes[1, 1].plot(drift_signals, color='lime', label='AI Drift Signal')
    axes[1, 1].axhline(0.045, color='white', linestyle=':', label='Risk Free Rate')
    axes[1, 1].set_title("AI Drift Signals (Trend Detection)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()