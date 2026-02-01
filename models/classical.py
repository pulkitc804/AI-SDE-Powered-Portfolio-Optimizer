import numpy as np
from scipy.stats import norm

class GeometricBrownianMotion:
    """
    Standard GBM for benchmarking simulations.
    """
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def simulate(self, S0, T, dt, n_paths):
        n_steps = int(T / dt)
        paths = np.zeros((n_steps + 1, n_paths))
        paths[0] = S0

        for t in range(1, n_steps + 1):
            Z = np.random.normal(0, 1, n_paths)
            paths[t] = paths[t-1] * np.exp((self.mu - 0.5 * self.sigma**2)*dt +
                                           self.sigma * np.sqrt(dt) * Z)
        return paths[1:]

class BlackScholes:
    """
    Fast Analytic Solver for standard option pricing.
    Used as the 'Market Benchmark' to compare against AI.
    """
    @staticmethod
    def price(S, K, T, r, sigma, option_type='call'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price