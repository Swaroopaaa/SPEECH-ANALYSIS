import os
import numpy as np
import torch
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from config import Config

# Safety defaults (keep as before)
if not hasattr(Config, "DATA_FOLDER") or not Config.DATA_FOLDER:
    Config.DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")
if not hasattr(Config, "RESULTS_FOLDER") or not Config.RESULTS_FOLDER:
    Config.RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "results")
if not hasattr(Config, "HUBERT_MODEL") or not Config.HUBERT_MODEL:
    Config.HUBERT_MODEL = "facebook/hubert-base-ls960"

DATA_DIR = os.path.abspath(Config.DATA_FOLDER)
OUT_DIR = os.path.abspath(Config.RESULTS_FOLDER)
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔹 Loading HuBERT model: {Config.HUBERT_MODEL}")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(Config.HUBERT_MODEL)
model = HubertModel.from_pretrained(Config.HUBERT_MODEL).to(device)
model.eval()

def extract_features_from_file(wav_path):
    try:
        y, sr = librosa.load(wav_path, sr=Config.SAMPLE_RATE)
        inputs = feature_extractor(y, sampling_rate=Config.SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            features = model(**inputs.to(device)).last_hidden_state
        feat_mean = torch.mean(features, dim=1).cpu().numpy().squeeze()
        return feat_mean
    except Exception as e:
        print(f" Error processing {wav_path}: {e}")
        return None

def generate_all():
    print("🔹 Starting feature extraction from all datasets...")
    langs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if not langs:
        print(f"❌ No language folders found in: {DATA_DIR}")
        return

    for lang in langs:
        lang_dir = os.path.join(DATA_DIR, lang)
        lang_out = os.path.join(OUT_DIR, lang)
        os.makedirs(lang_out, exist_ok=True)

        wav_files = [f for f in os.listdir(lang_dir) if f.lower().endswith(".wav")]
        if not wav_files:
            print(f" No .wav files in {lang_dir}")
            continue

        for fname in tqdm(wav_files, desc=f"Processing {lang}"):
            fpath = os.path.join(lang_dir, fname)
            feat = extract_features_from_file(fpath)
            if feat is not None:
                np.save(os.path.join(lang_out, fname.replace(".wav", ".npy")), feat)

    print(f" Feature extraction complete!\n Features saved to: {OUT_DIR}")

if __name__ == "__main__":
    generate_all()
