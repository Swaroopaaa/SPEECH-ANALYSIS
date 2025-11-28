# Downloads IndicAccentDb from Hugging Face and organizes into data/<language>/*.wav
from datasets import load_dataset
import os
import soundfile as sf
from config import Config

def download():
    print("Downloading dataset (this may take a while)...")
    ds = load_dataset("DarshanaS/IndicAccentDb")
    os.makedirs(Config.DATA_FOLDER, exist_ok=True)
    for split in ds.keys():
        for i, example in enumerate(ds[split]):
            lang = example.get("language", "unknown")
            audio = example["audio"]
            arr = audio["array"]
            sr = audio.get("sampling_rate", 16000)
            lang_dir = os.path.join(Config.DATA_FOLDER, lang)
            os.makedirs(lang_dir, exist_ok=True)
            fname = f"{split}_{i}.wav"
            path = os.path.join(lang_dir, fname)
            try:
                sf.write(path, arr, sr)
            except Exception as e:
                print("failed write", path, e)
    print("Dataset downloaded to", Config.DATA_FOLDER)

if __name__ == "__main__":
    download()
