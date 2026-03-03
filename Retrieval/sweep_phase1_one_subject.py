import itertools
import subprocess
from tqdm import tqdm

# Phase 1: optimizer / regularization sweep
lrs      = [1e-4, 3e-4, 1e-3]
dropouts = [0.1, 0.25, 0.5]

# build a list so we know total
param_list = list(itertools.product(lrs, dropouts))
total = len(param_list)

for idx, (lr, do) in enumerate(tqdm(param_list, desc="Launching jobs"), 1):
    # format lr as "1e-4" instead of "0.0001"
    lr_str = f"{lr:.0e}"               # e.g. "1e-04"
    lr_str = lr_str.replace("e-0", "e-")  # => "1e-4"
    lr_str = lr_str.replace("e+0", "e+")  # => "3e-4" stays correct

    run_name = f"lr{lr_str}_do{do}"

    # progress statement
    print(f"[{idx}/{total}] → Launching {run_name}")

    cmd = [
        "python", "ATMS_retrieval_alljoined2.py",
        "--lr",        str(lr),
        "--dropout",   str(do),
        "--d_model",   "250",    # keep capacity fixed
        "--n_heads",   "4",
        "--e_layers",  "1",
        "--epochs",    "40",
        "--batch_size","64",
        "--project",   "Forschungspraxis",
        "--entity",    "t-reiter-technical-university-of-munich",
        "--name",      run_name,
        "--encoder_type","ATMS",
        "--insubject",   "True",
        "--subjects",    "sub-08",     # note the comma here
        "--logger",      "True"
    ]

    subprocess.run(cmd, check=True)
