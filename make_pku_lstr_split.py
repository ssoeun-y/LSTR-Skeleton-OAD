
import os
import random

FEATURE_DIR = "data/pku_mmd/features"
OUT_DIR = "data/pku_mmd"

TRAIN_RATIO = 0.7
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

ids = sorted([
    os.path.splitext(f)[0]
    for f in os.listdir(FEATURE_DIR)
    if f.endswith(".npy")
])

print("Total sequences:", len(ids))

random.shuffle(ids)
n_train = int(len(ids) * TRAIN_RATIO)

train_ids = ids[:n_train]
test_ids = ids[n_train:]

with open(os.path.join(OUT_DIR, "split_train.txt"), "w") as f:
    for vid in train_ids:
        f.write(vid + "\n")

with open(os.path.join(OUT_DIR, "split_test.txt"), "w") as f:
    for vid in test_ids:
        f.write(vid + "\n")

print("Train:", len(train_ids))
print("Test :", len(test_ids))
print("Split files generated.")
