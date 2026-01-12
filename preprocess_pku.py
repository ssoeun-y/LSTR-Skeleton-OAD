import os
import glob
import numpy as np

RAW_DATA_PATH = "./data/PKUMMD/Data"
OUTPUT_PATH = "./data/pku_mmd/features"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def parse_pku_skeleton(file_path):
    frames = []
    with open(file_path, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            if len(values) != 150:
                continue
            frames.append(values)
    return np.array(frames, dtype=np.float32)  # (T, 150)

def main():
    txt_files = glob.glob(os.path.join(RAW_DATA_PATH, "*.txt"))
    print(f"{len(txt_files)} files found")

    for file_path in txt_files:
        video_id = os.path.basename(file_path).replace(".txt", "")
        features = parse_pku_skeleton(file_path)

        if features.shape[0] == 0:
            continue

        save_path = os.path.join(OUTPUT_PATH, f"{video_id}.npy")
        np.save(save_path, features)
        print(f"Saved {video_id}: {features.shape}")

if __name__ == "__main__":
    main()
