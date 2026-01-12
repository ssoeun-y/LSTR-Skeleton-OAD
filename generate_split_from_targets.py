import os
import random

DATA_ROOT = "data/pku_mmd"
TARGET_DIR = os.path.join(DATA_ROOT, "targets")

def generate_split_from_targets():
    sessions = [
        os.path.splitext(f)[0]
        for f in os.listdir(TARGET_DIR)
        if f.endswith(".npy")
    ]

    sessions.sort()
    random.seed(42)
    random.shuffle(sessions)

    split_idx = int(len(sessions) * 0.8)
    train_set = sessions[:split_idx]
    test_set = sessions[split_idx:]

    with open(os.path.join(DATA_ROOT, "split_train.txt"), "w") as f:
        f.write("\n".join(train_set))

    with open(os.path.join(DATA_ROOT, "split_test.txt"), "w") as f:
        f.write("\n".join(test_set))

    print(f"Train: {len(train_set)}, Test: {len(test_set)}")

if __name__ == "__main__":
    generate_split_from_targets()
