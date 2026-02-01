import numpy as np

class AIAlphaStrategy:
    """
    Backtests a trading strategy based on Neural SDE Drift signals.
    Signal: If AI Drift > Risk-Free Rate -> BUY. Else -> SELL.
    """
    def __init__(self, initial_capital=10000, risk_free_rate=0.045):
        self.capital = initial_capital
        self.r = risk_free_rate

    def run_backtest(self, prices, drift_signals):
        cash = self.capital
        position = 0
        portfolio_value = [self.capital]
        signals = [] # 1 for Buy, -1 for Sell

        for i in range(len(prices) - 1):
            current_price = prices[i]
            predicted_drift = drift_signals[i]

            # LOGIC:
            # If AI expects the stock to grow faster than the bank (Risk Free Rate), BUY.
            if predicted_drift > self.r and position == 0:
                position = cash / current_price
                cash = 0
                signals.append(1) # BUY

            # If AI expects growth to slow down, SELL.
            elif predicted_drift < self.r and position > 0:
                cash = position * current_price
                position = 0
                signals.append(-1) # SELL
            else:
                signals.append(0) # HOLD

            # Mark to Market
            val = cash + (position * current_price)
            portfolio_value.append(val)

        return portfolio_value, signals