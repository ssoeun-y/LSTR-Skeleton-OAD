import os
import shutil

src_dir = "data/oad/targets"
dst_dir = "data/oad/targets_thumos"

os.makedirs(dst_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    if not fname.endswith(".npy"):
        continue

    stem = os.path.splitext(fname)[0]
    if not stem.isdigit():
        continue

    idx = int(stem)

    for prefix in ["video_train", "video_validation", "video_test"]:
        new_name = f"{prefix}_{idx:07d}.npy"
        shutil.copy(
            os.path.join(src_dir, fname),
            os.path.join(dst_dir, new_name)
        )

print("train / validation / test targets generated.")
