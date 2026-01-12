import numpy as np
import os

label_dir = "/Users/cheonso-eun/Desktop/OAD/Label"
feat_dir = "data/oad/skeleton_oad"
target_dir = "data/oad/targets"

os.makedirs(target_dir, exist_ok=True)

# 1. feature 기준 seq_id 목록
feat_ids = set(
    os.path.splitext(f)[0]
    for f in os.listdir(feat_dir)
    if f.endswith(".npy")
)

# 2. label 중 feature가 있는 것만 처리
for fname in os.listdir(label_dir):
    if not fname.endswith(".txt"):
        continue

    seq_id = os.path.splitext(fname)[0]

    if seq_id not in feat_ids:
        print(f"[SKIP] no feature for seq {seq_id}")
        continue

    feat = np.load(os.path.join(feat_dir, f"{seq_id}.npy"))
    T = feat.shape[0]

    target = np.zeros(T, dtype=np.int64)

    with open(os.path.join(label_dir, fname), "r") as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            start = int(parts[1]) - 1
            end = int(parts[2]) - 1
            target[start:end+1] = cls

    np.save(os.path.join(target_dir, f"{seq_id}.npy"), target)
