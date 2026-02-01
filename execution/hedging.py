import time

class HedgingEngine:
    def execute_hedge(self, ticker, delta, spot_price):
        start_time = time.time()

        # Delta Neutral Strategy: If Delta is 0.5, sell 50 shares
        hedge_quantity = -1 * delta * 100
        action = "SELL" if hedge_quantity < 0 else "BUY"

        # Simulate Exchange Latency (5ms)
        time.sleep(0.005)

        latency_ms = (time.time() - start_time) * 1000
        print(f" [EXECUTION] {ticker} | Action: {action} {abs(hedge_quantity):.1f} Shares | Filled @ ${spot_price:.2f}")
        return latency_ms