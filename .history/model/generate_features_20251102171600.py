# create features for all wav files in data/ into numpy arrays + csv
import os
import numpy as np
import joblib
from config import Config
from model.feature_extractor import extract_mfcc
from model.hubert_model import HuBERTExtractor
from tqdm import tqdm

OUT_DIR = os.path.join(Config.BASE_DIR, "model", "features")
os.makedirs(OUT_DIR, exist_ok=True)

def generate_all():
    extractor = HuBERTExtractor()
    rows = []
    for lang in sorted(os.listdir(Config.DATA_FOLDER)):
        lang_path = os.path.join(Config.DATA_FOLDER, lang)
        if not os.path.isdir(lang_path): 
            continue
        for fname in os.listdir(lang_path):
            if not fname.lower().endswith(".wav"):
                continue
            fpath = os.path.join(lang_path, fname)
            try:
                mfcc = extract_mfcc(fpath)
                hubert = extractor.extract_path(fpath) if hasattr(extractor, 'extract_path') else extractor.extract(__import__('librosa').load(fpath, sr=16000)[0])
            except Exception as e:
                print("skip", fpath, e)
                continue
            # save each as npz
            base = os.path.splitext(fname)[0]
            np.savez_compressed(os.path.join(OUT_DIR, f"{lang}__{base}.npz"), mfcc=mfcc, hubert=hubert)
            rows.append((lang, fpath))
    print("features saved to", OUT_DIR)

if __name__ == "__main__":
    generate_all()
