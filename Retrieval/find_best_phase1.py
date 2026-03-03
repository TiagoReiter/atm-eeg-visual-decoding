import wandb
import os
from dateutil.parser import parse as parse_date

os.environ["WANDB_MODE"]     = 'online'

api = wandb.Api()

# 1) Grab all runs in your project
all_runs = api.runs("t-reiter-technical-university-of-munich/Forschungspraxis")

# 2) Filter locally for in‐subject sub-08
filtered = []
for r in all_runs:
    cfg = r.config
    # sometimes wandb stores lists as strings, so handle both
    subjects = cfg.get("subjects") or cfg.get("config.subjects") or []
    # ensure it's a Python list
    if isinstance(subjects, str):
        # e.g. "[\"sub-01\", \"sub-08\", ...]" → eval it
        try:
            subjects = eval(subjects)
        except:
            subjects = [subjects]
    if cfg.get("insubject") and "sub-08" in subjects:
        filtered.append(r)

if not filtered:
    print("❌ No in-subject sub-08 runs found!")
    exit()

# 3) Sort by creation time
runs_sorted = sorted(filtered, key=lambda r: parse_date(r.created_at))

# 4) Take the last nine
last9 = runs_sorted[-9:]

print(f"Most recent {len(last9)} in-subject sub-08 runs:")
for r in last9:
    acc = r.summary.get("Test Accuracy") or r.summary.get("test_accuracy") or float("nan")
    lr  = r.config.get("lr")
    do  = r.config.get("dropout")
    print(f"→ {r.name:20s}  created_at={r.created_at}  Test Acc={acc:.4f}  lr={lr}  dropout={do}")