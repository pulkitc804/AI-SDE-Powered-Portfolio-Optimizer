import torch
import torch.optim as optim
import numpy as np
from config.settings import DEVICE

class SDETrainer:
    def __init__(self, model, lr=0.01):
        self.model = model.to(DEVICE)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

    def train(self, X_train, y_train, epochs=100):
        self.model.train()

        # Ensure data is on the correct device
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output, _ = self.model(X_train)
            loss = self.criterion(output, y_train)
            loss.backward()
            self.optimizer.step()

    def predict_future(self, last_sequence, current_price, num_simulations, days_ahead):
        self.model.eval()
        with torch.no_grad():
            # ---------------------------------------------------------
            # 1. SETUP INPUT SHAPE (Batch, Seq_Len, Features=3)
            # ---------------------------------------------------------
            if last_sequence.dim() == 2:
                last_sequence = last_sequence.unsqueeze(0)

            # Replicate the history for every simulation path
            # Shape: [num_simulations, 30, 3]
            current_input = last_sequence.repeat(num_simulations, 1, 1)

            # Store paths (Price only)
            paths = np.zeros((num_simulations, days_ahead + 1))
            paths[:, 0] = current_price

            dt = 1/252
            sqrt_dt = np.sqrt(dt)

            # ---------------------------------------------------------
            # 2. SIMULATION LOOP
            # ---------------------------------------------------------
            for day in range(1, days_ahead + 1):
                # A. MODEL PREDICTION
                # The model sees 3 features and outputs Drift & Diffusion
                drift, diffusion = self.model(current_input)

                # B. SDE MATH (Brownian Motion)
                z = torch.randn(num_simulations, 1).to(DEVICE)

                # Calculate the shock (percentage change)
                shock = (drift * dt) + (diffusion * sqrt_dt * z)

                # C. UPDATE PRICE
                # Get previous day's price from our storage array
                prev_price = torch.tensor(paths[:, day-1], dtype=torch.float32).to(DEVICE).unsqueeze(1)
                new_price = prev_price * (1 + shock)

                # Store the new price in our results
                paths[:, day] = new_price.squeeze().cpu().numpy()

                # D. UPDATE INPUT TENSOR FOR NEXT STEP
                # We need to feed the new state back into the model.
                # The model expects [Price, Returns, Volatility].

                # 1. Clone the last timestep to use as a template
                new_timestep = current_input[:, -1:, :].clone()

                # 2. Update Feature 0: Price (Normalized relative to start)
                # (Simple relative scaling for stability)
                new_timestep[:, 0, 0] = (new_price.squeeze() / current_price)

                # 3. Update Feature 1: Returns (The shock we just calculated)
                new_timestep[:, 0, 1] = shock.squeeze()

                # 4. Update Feature 2: Volatility
                # (We keep the previous volatility estimate to avoid noise)
                # new_timestep[:, 0, 2] = ... (Left as is from previous step)

                # 5. Shift the sequence window (Drop oldest day, add new day)
                # [Batch, 29, 3] + [Batch, 1, 3] -> [Batch, 30, 3]
                current_input = torch.cat((current_input[:, 1:, :], new_timestep), dim=1)

        return paths