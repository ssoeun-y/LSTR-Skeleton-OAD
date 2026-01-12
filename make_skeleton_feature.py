import numpy as np
import os

# PKU skeleton txt 원본 경로 (레포 기준)
src_dir = "data/PKUMMD/Data"

# OAD용 skeleton feature 저장 경로
dst_dir = "data/pku_mmd/features"
os.makedirs(dst_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    if not fname.endswith(".txt"):
        continue

    seq_id = os.path.splitext(fname)[0]
    if not seq_id.isdigit():
        continue

    data = []
    with open(os.path.join(src_dir, fname), "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            data.append(vals)

    data = np.array(data, dtype=np.float32)
    np.save(os.path.join(dst_dir, f"{seq_id}.npy"), data)

    print(f"saved {seq_id}.npy")
