from huggingface_hub import snapshot_download
from pathlib import Path
import shutil

# 1. Download the whole repo into your local cache (no subfolder arg)
cache_dir = snapshot_download(
    repo_id="LidongYang/EEG_Image_decode",
    repo_type="dataset",
    # you can still pass cache_dir=..., use_auth_token=..., etc.
)

# 2. Point to the Preprocessed_data_250Hz sub-folder inside that cache
src_root = Path(cache_dir) / "Preprocessed_data_250Hz"

dst_root = Path("./Data/Preprocessed_data_250Hz/ThingsEEG")
dst_root.mkdir(parents=True, exist_ok=True)

# 3. Copy everything except sub-08, preserving names
for item in src_root.iterdir():
    if item.name == "sub-08":
        continue

    dest = dst_root / item.name
    if item.is_dir():
        shutil.copytree(item, dest)  # folder → folder
    else:
        shutil.copy2(item, dest)      # file → file

    print(f"Copied {item.name}")

print("Done!")