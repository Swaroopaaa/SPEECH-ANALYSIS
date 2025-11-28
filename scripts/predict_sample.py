import os
import sys
import librosa
import numpy as np
import joblib
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
from config import Config

# Path to trained model
MODEL_PATH = os.path.join(Config.RESULTS_FOLDER, "language_model.pkl")
HUBERT_MODEL = "facebook/hubert-base-ls960"

def extract_features(filepath):
    """Extract HuBERT embeddings for a single .wav file"""
    processor = Wav2Vec2Processor.from_pretrained(HUBERT_MODEL)
    model = Wav2Vec2Model.from_pretrained(HUBERT_MODEL)
    
    y, sr = librosa.load(filepath, sr=16000)
    inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def predict_language(filepath):
    """Predict the native language for the given audio file"""
    if not os.path.exists(MODEL_PATH):
        print(f" Model not found: {MODEL_PATH}")
        print(" Run `python -m scripts.train_model` first.")
        return

    clf = joblib.load(MODEL_PATH)
    features = extract_features(filepath).reshape(1, -1)
    pred = clf.predict(features)[0]
    print(f"🎧 Predicted Language: {pred}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.predict_sample path/to/audio.wav")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f" File not found: {filepath}")
    else:
        predict_language(filepath)
