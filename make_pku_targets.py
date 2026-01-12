import os
import csv
import numpy as np
import glob

# ===============================
# PATHS (필요하면 여기만 수정)
# ===============================
PKU_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "PKUMMD")
)

FEATURE_DIR = "long-short-term-transformer/data/pku_mmd/features"
TARGET_DIR = "long-short-term-transformer/data/pku_mmd/targets"

ACTIONS_CSV = os.path.join(PKU_ROOT, "Actions.csv")
LABEL_DIR = os.path.join(PKU_ROOT, "Label")

os.makedirs(TARGET_DIR, exist_ok=True)

# ===============================
# Load action mapping
# ===============================
def load_action_mapping(csv_path):
    action_map = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row) < 2:
                continue
            action_id = int(row[0])
            action_name = row[1]
            action_map[action_id] = action_name
    return action_map


# ===============================
# Read PKU label (segment-level)
# ===============================
def read_label(label_path):
    segments = []
    if not os.path.exists(label_path):
        return segments
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                segments.append(
                    (int(parts[0]), int(parts[1]), int(parts[2]))
                )
    return segments

# ===============================
# Main conversion
# ===============================
def make_targets():
    action_map = load_action_mapping(ACTIONS_CSV)

    NUM_CLASSES = 22   # 🔴 LSTR 기준으로 고정
    print(f"NUM_CLASSES = {NUM_CLASSES}")

    feature_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.npy")))
    print(f"Found {len(feature_files)} feature files")

    for feat_path in feature_files:
        video_id = os.path.splitext(os.path.basename(feat_path))[0]

        feat = np.load(feat_path)
        T = feat.shape[0]

        # (T, 22)
        target = np.zeros((T, NUM_CLASSES), dtype=np.float32)
        target[:, 0] = 1.0  # background

        label_path = os.path.join(LABEL_DIR, f"{video_id}.txt")
        segments = read_label(label_path)

        for action_id, start, end in segments:
            # 🔴 LSTR가 처리 못 하는 action은 버림
            if action_id >= NUM_CLASSES:
                continue

            start = max(0, start)
            end = min(T - 1, end)

            target[start:end + 1, :] = 0.0
            target[start:end + 1, action_id] = 1.0

        save_path = os.path.join(TARGET_DIR, f"{video_id}.npy")
        np.save(save_path, target)

        print(f"[OK] {video_id}: T={T}, segments={len(segments)}")

    print("\n✅ PKU-MMD targets generation DONE")


# ===============================
if __name__ == "__main__":
    make_targets()
