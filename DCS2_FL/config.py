import os
import torch

# Training parameters
NUM_CLIENTS = 100
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.01
NUM_ROUNDS = 300

# DCS2 parameters
DCS2_LAMBDA_G = 0.7
DCS2_LAMBDA_X = 1.0
DCS2_LAMBDA_Z = 1.0
DCS2_EPSILON = 0.1
DCS2_SYNTH_STEPS = 15
DCS2_SYNTH_LR = 0.1
DCS2_INIT_MODE = "random"

# Results directory
RESULTS_DIR = os.path.join("results", "dcs2_fl")
os.makedirs(RESULTS_DIR, exist_ok=True)
