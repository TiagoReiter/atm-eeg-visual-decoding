import subprocess
from itertools import product
from tqdm import tqdm

# Possible values to try
batch_sizes = [32, 64, 128]
epochs_list = [20, 40, 80]

# Fixed settings (you already know best lr/dropout)
fixed_lr = 3e-4
fixed_dropout = 0.25

# Build sweep combinations
sweep_combinations = list(product(batch_sizes, epochs_list))

for idx, (batch_size, epochs) in enumerate(tqdm(sweep_combinations, desc="Launching sweep")):
    run_name = f"bs{batch_size}_ep{epochs}"

    cmd = [
        "python", "ATME_retrieval.py",
        "--batch_size", str(batch_size),
        "--epochs",     str(epochs),
        "--lr",         str(fixed_lr),
        "--dropout",    str(fixed_dropout),
        "--encoder_type", "ATM_E",
        "--insubject",  
        "--subjects",   "sub-08",
        "--logger",    
        "--name",       run_name
    ]

    print(f"Launching {run_name}...")
    subprocess.run(cmd, check=True)