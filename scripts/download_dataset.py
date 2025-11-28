# scripts/download_dataset.py
import os
import requests
from tqdm import tqdm
from config import Config

# --- IndicAccentDb on HuggingFace ---
# We'll directly download audio files using public URLs (no torchcodec)

LANGUAGES = ["hindi", "tamil", "telugu", "malayalam", "kannada"]

BASE_URL = "https://huggingface.co/datasets/DarshanaS/IndicAccentDb/resolve/main/data/"
# ^ This path serves audio files directly from the HuggingFace repo

def download_dataset(limit_per_lang=5):
    os.makedirs(Config.DATA_FOLDER, exist_ok=True)

    for lang in LANGUAGES:
        lang_folder = os.path.join(Config.DATA_FOLDER, lang)
        os.makedirs(lang_folder, exist_ok=True)
        print(f"\n📥 Downloading {lang} samples...")

        for i in tqdm(range(1, limit_per_lang + 1)):
            file_url = f"{BASE_URL}{lang}/{i}.wav"
            dest_path = os.path.join(lang_folder, f"{i}.wav")
            try:
                response = requests.get(file_url, timeout=20)
                if response.status_code == 200:
                    with open(dest_path, "wb") as f:
                        f.write(response.content)
                else:
                    print(f"Skipped: {file_url} not found.")
            except Exception as e:
                print(f" Error downloading {file_url}: {e}")

    print("\n Dataset downloaded successfully to:", Config.DATA_FOLDER)

if __name__ == "__main__":
    download_dataset()
