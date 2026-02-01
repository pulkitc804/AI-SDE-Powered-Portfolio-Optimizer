import torch

# Check for Mac 'MPS' (Metal Performance Shaders) acceleration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(" [SYSTEM] Apple Metal (MPS) Acceleration ENABLED")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Market Simulation Constants
RISK_FREE_RATE = 0.045
DT = 1/252  # Daily time step