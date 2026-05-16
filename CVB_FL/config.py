import os
import torch

# Training parameters
NUM_CLIENTS = 100
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.01
NUM_ROUNDS = 300

# CVB parameters (paper defaults for CNN)
CVB_POSITION = 1
CVB_KERNEL_SIZE = 5
CVB_SCALE = 0.5
CVB_BETA = 0.1

# Results directory
RESULTS_DIR = os.path.join("results", "cvb_fl")
os.makedirs(RESULTS_DIR, exist_ok=True)
