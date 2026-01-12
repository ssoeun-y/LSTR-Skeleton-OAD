import os

feat_dir = "data/oad/skeleton_oad"
out_train = "data/oad/split_train.txt"
out_test = "data/oad/split_test.txt"

ids = sorted(
    os.path.splitext(f)[0]
    for f in os.listdir(feat_dir)
    if f.endswith(".npy") and f.replace(".npy", "").isdigit()
)

n = len(ids)
n_train = int(0.8 * n)

train_ids = ids[:n_train]
test_ids = ids[n_train:]

with open(out_train, "w") as f:
    for i in train_ids:
        f.write(i + "\n")

with open(out_test, "w") as f:
    for i in test_ids:
        f.write(i + "\n")

print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")
