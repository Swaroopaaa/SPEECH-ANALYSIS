import os
import numpy as np
from config import Config

def merge_all_features():
    features = []
    labels = []

    # loop through each language directory inside results/
    for lang in os.listdir(Config.RESULTS_FOLDER):
        lang_path = os.path.join(Config.RESULTS_FOLDER, lang)
        if not os.path.isdir(lang_path) or lang == "plots":
            continue

        print(f"🔹 Merging features from {lang}")
        for f in os.listdir(lang_path):
            if f.endswith(".npy"):
                fpath = os.path.join(lang_path, f)
                data = np.load(fpath)
                features.append(data)
                labels.append(lang)  # language label

    # Stack all features
    features = np.vstack(features)
    labels = np.array(labels)

    np.save(os.path.join(Config.RESULTS_FOLDER, "features.npy"), features)
    np.save(os.path.join(Config.RESULTS_FOLDER, "labels.npy"), labels)
    print(f"✅ Merged features saved in {Config.RESULTS_FOLDER}")

if __name__ == "__main__":
    merge_all_features()
