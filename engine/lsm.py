import numpy as np

class LongstaffSchwartzPricer:
    """
    Prices American Options using Least Squares Monte Carlo (LSM).
    Integrates with Neural SDE paths.
    """
    def __init__(self, r=0.045, dt=1/252):
        self.r = r
        self.dt = dt

    def price_option(self, paths, K, option_type='put'):
        # paths shape: [num_steps, num_paths] -> Transpose to [num_paths, num_steps]
        # We need rows=paths for easier matrix math
        S = paths.T
        num_paths, num_steps = S.shape

        # 1. Initialize Cashflows at Maturity (T)
        if option_type == 'put':
            payoff = np.maximum(K - S[:, -1], 0)
        else:
            payoff = np.maximum(S[:, -1] - K, 0)

        cashflows = payoff.copy()

        print(f" [LSM] Starting Backward Induction on {num_paths} AI-generated paths...")

        # 2. Backward Induction Loop
        # We step back from T-1 down to 1
        for t in range(num_steps - 2, 0, -1):
            S_t = S[:, t] # Prices at time t

            # Determine which paths are "In-The-Money" (ITM)
            # We only care about paths where exercising is even an option
            if option_type == 'put':
                itm_mask = S_t < K
            else:
                itm_mask = S_t > K

            # If no paths are ITM, just discount and continue
            if np.sum(itm_mask) == 0:
                cashflows = cashflows * np.exp(-self.r * self.dt)
                continue

            # X = Current Price (Features)
            # Y = Discounted Future Cashflow (Target)
            X = S_t[itm_mask]
            Y = cashflows[itm_mask] * np.exp(-self.r * self.dt)

            # 3. Polynomial Regression (The "Least Squares" part)
            # We fit a curve: E[Y | X] = a + bX + cX^2
            # This estimates "Holding Value"
            A = np.vstack([np.ones_like(X), X, X**2]).T
            try:
                # Solve linear equation (A * coeffs = Y)
                coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
                continuation_value = A @ coeffs
            except:
                continue

            # 4. Decision: Exercise or Hold?
            if option_type == 'put':
                exercise_value = K - X
            else:
                exercise_value = X - K

            # Identify where exercising yields more money than holding
            should_exercise = exercise_value > continuation_value

            # Update Cashflows:
            # If we exercise: cashflow = exercise_value
            # If we hold: cashflow stays as future cashflow (discounted later)

            # We must be careful to only update the ITM paths
            # First, discount ALL cashflows one step back
            cashflows = cashflows * np.exp(-self.r * self.dt)

            # Now overwrite the specific paths where we chose to exercise
            # We map the subset 'should_exercise' back to the full 'cashflows' array
            # This logic is tricky in numpy, so we use indices
            itm_indices = np.where(itm_mask)[0]
            exercise_indices = itm_indices[should_exercise]

            cashflows[exercise_indices] = exercise_value[should_exercise]

        # 3. Final Discounting to Present Value
        price = np.mean(cashflows * np.exp(-self.r * self.dt))
        return price