# Example preprocessing: resample, normalize, trim silence (very simple)
import os
import librosa
import soundfile as sf
from config import Config

def preprocess_all(target_sr=16000):
    for lang in os.listdir(Config.DATA_FOLDER):
        lang_path = os.path.join(Config.DATA_FOLDER, lang)
        if not os.path.isdir(lang_path): continue
        for fname in os.listdir(lang_path):
            if not fname.lower().endswith(".wav"): continue
            p = os.path.join(lang_path, fname)
            try:
                y, sr = librosa.load(p, sr=target_sr)
                # normalize
                y = y / max(1e-9, max(abs(y)))
                sf.write(p, y, target_sr)
            except Exception as e:
                print("preprocess error", p, e)

if __name__ == "__main__":
    preprocess_all()
