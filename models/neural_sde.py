import torch
import torch.nn as nn
from config.settings import DEVICE

class NeuralSDE(nn.Module):
    """
    Advanced Architecture: LSTM-driven Stochastic Differential Equation.
    Now supports deep stacked LSTMs via 'num_layers'.
    """
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2):
        super(NeuralSDE, self).__init__()

        # 1. Feature Extractor (LSTM)
        # Captures long-term dependencies (e.g., momentum, regimes)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # 2. Drift Network (Mu) - The 'Trend'
        # Maps LSTM Hidden State -> Expected Return
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

        # 3. Diffusion Network (Sigma) - The 'Risk'
        # Maps LSTM Hidden State -> Volatility
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softplus() # Enforces positivity
        )

        self.to(DEVICE)

    def forward(self, x):
        # x shape: [Batch, Sequence_Length, Features]

        # Pass through LSTM
        # out shape: [Batch, Seq_Len, Hidden_Dim]
        lstm_out, _ = self.lstm(x)

        # We only care about the final state (the accumulation of all history)
        final_state = lstm_out[:, -1, :]

        # Decode SDE Parameters
        mu = self.mu_head(final_state)
        sigma = self.sigma_head(final_state)

        return mu, sigma