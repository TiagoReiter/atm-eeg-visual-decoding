import itertools
import subprocess
from tqdm import tqdm

# Replace these with the best values you found in Phase 1
best_lr = 1e-4
best_do = 0.25

# Phase 2: capacity sweep
#d_models  = [128, 256, 512]
#heads     = [4, 8]
#layers    = [1, 2]
d_models = [250]
heads = [4]
layers = [1]

# build list so we know total
param_list = list(itertools.product(d_models, heads, layers))
total = len(param_list)

for idx, (dm, h, L) in enumerate(tqdm(param_list, desc="Launching capacity jobs"), 1):
    # format d_model, heads, layers into a name
    run_name = f"dm{dm}_h{h}_L{L}"

    print(f"[{idx}/{total}] → Launching {run_name}")

    cmd = [
        "python", "ATMS_retrieval_hyperparameter_robustness.py",
        "--lr",        str(best_lr),
        "--dropout",   str(best_do),
        "--d_model",   str(dm),
        "--n_heads",   str(h),
        "--e_layers",  str(L),
        "--epochs",    "40",
        "--batch_size","64",
        "--project",   "Forschungspraxis",
        "--entity",    "t-reiter-technical-university-of-munich",
        "--name",      run_name,
        "--encoder_type","ATMS",
        "--insubject",   "True",
        "--subjects",    "sub-08",     # single‐subject run
        "--logger",      "True"
    ]

    subprocess.run(cmd, check=True)