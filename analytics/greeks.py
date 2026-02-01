import numpy as np

class NeuralGreeks:
    def __init__(self, simulator, pricer):
        self.simulator = simulator # The Neural Trainer/Predictor
        self.pricer = pricer       # The LSM Pricer

    def calculate_delta_gamma(self, sequence, S0, K, T):
        """
        Calculates Delta and Gamma using Finite Differences on the Neural SDE.
        We bump the price up and down to see how the option value changes.
        """
        bump = S0 * 0.01 # 1% price bump

        # Scenario 1: Base Price (S0)
        paths_base = self.simulator.predict_future(sequence, S0, n_paths=2000, n_steps=21)
        price_base = self.pricer.price_option(paths_base, K)

        # Scenario 2: Price Up (S0 + bump)
        paths_up = self.simulator.predict_future(sequence, S0 + bump, n_paths=2000, n_steps=21)
        price_up = self.pricer.price_option(paths_up, K)

        # Scenario 3: Price Down (S0 - bump)
        paths_down = self.simulator.predict_future(sequence, S0 - bump, n_paths=2000, n_steps=21)
        price_down = self.pricer.price_option(paths_down, K)

        # Delta: (Price_Up - Price_Down) / (2 * Bump)
        delta = (price_up - price_down) / (2 * bump)

        # Gamma: (Price_Up - 2*Base + Price_Down) / (Bump^2)
        gamma = (price_up - 2*price_base + price_down) / (bump**2)

        return delta, gamma