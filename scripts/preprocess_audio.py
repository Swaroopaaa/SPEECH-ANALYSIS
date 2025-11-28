import os
import librosa
import soundfile as sf
from config import Config

def preprocess_all(target_sr=16000):
    print("🔹 Starting preprocessing...")
    count = 0

    for lang in os.listdir(Config.DATA_FOLDER):
        lang_path = os.path.join(Config.DATA_FOLDER, lang)
        if not os.path.isdir(lang_path):
            continue

        for fname in os.listdir(lang_path):
            if not fname.lower().endswith(".wav"):
                continue

            p = os.path.join(lang_path, fname)
            try:
                print(f"Processing: {p}")
                # Load only first 5 seconds, convert to mono
                y, sr = librosa.load(p, sr=target_sr, mono=True, duration=5.0)

                # Normalize audio
                if len(y) > 0:
                    y = y / max(1e-9, max(abs(y)))
                    sf.write(p, y, target_sr)
                    count += 1
                else:
                    print(f" Empty file skipped: {p}")

            except Exception as e:
                print(f" Error processing {p}: {e}")
                continue

    print(f" Preprocessing done! Total files processed: {count}")

if __name__ == "__main__":
    preprocess_all()

