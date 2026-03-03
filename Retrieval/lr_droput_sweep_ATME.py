import subprocess
from itertools import product
from tqdm import tqdm

# 🎯 Hyperparameter sweep (phase 1) for ATM_E retrieval (ATME_retrieval.py)
# This mirrors sweep_phase1_one_subject.py but targets the ATM_E model

# Learning rates and dropouts to try
target_lrs      = [1e-3, 3e-4, 1e-4]
target_dropouts = [0.1, 0.25, 0.5]

# Fixed settings for phase 1
fixed_batch_size = 64
fixed_epochs     = 40

# Script and encoder configuration
script  = "ATME_retrieval.py"
encoder = "ATM_E"
subject = "sub-08"

# Build and launch sweep combinations
for lr, dropout in tqdm(product(target_lrs, target_dropouts), desc="Launching ATME phase1 sweep"):
    run_name = f"ATME_lr{lr:.0e}_do{int(dropout*100)}"
    cmd = [
        "python", script,
        "--lr",         str(lr),
        "--dropout",    str(dropout),
        "--batch_size", str(fixed_batch_size),
        "--epochs",     str(fixed_epochs),
        "--encoder_type", encoder,
        "--insubject",  
        "--subjects",   subject,
        "--logger",  
        "--name",       run_name
    ]
    print(f"Launching {run_name}...")
    subprocess.run(cmd, check=True)