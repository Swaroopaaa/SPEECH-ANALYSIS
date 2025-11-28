# Simple dataset downloader (without torchcodec)
from datasets import load_dataset
import os
import soundfile as sf
from config import Config

def download(limit_per_lang=10):
    print("Downloading a small subset of IndicAccentDb for testing...")
    ds = load_dataset("DarshanaS/IndicAccentDb", split="train[:1%]")  # only 1% for quick test
    os.makedirs(Config.DATA_FOLDER, exist_ok=True)
    for example in ds:
        lang = example.get("language", "unknown")
        audio = example["audio"]
        arr = audio["array"]
        sr = audio.get("sampling_rate", 16000)
        lang_dir = os.path.join(Config.DATA_FOLDER, lang)
        os.makedirs(lang_dir, exist_ok=True)
        idx = len(os.listdir(lang_dir))
        if idx >= limit_per_lang:
            continue
        fname = f"{idx+1}.wav"
        sf.write(os.path.join(lang_dir, fname), arr, sr)
    print("✅ Dataset downloaded to:", Config.DATA_FOLDER)

if __name__ == "__main__":
    download()
