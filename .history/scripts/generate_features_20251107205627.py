# Improved wrapper that calls model/generate_features.py
import os
from model.generate_features import generate_all
from config import Config

if __name__ == "__main__":
    print("🔹 Starting feature extraction...")
    
    # Check if dataset exists
    if not os.path.exists(Config.DATA_FOLDER):
        print(f"❌ Dataset folder not found: {Config.DATA_FOLDER}")
        print("➡️ Please ensure your .wav files are in the data/ folder.")
    else:
        try:
            generate_all()
            print("✅ Feature extraction completed successfully!")
            print(f"📂 Features saved under: {os.path.abspath('results/')}")
        except Exception as e:
            print(f"❌ Error during feature extraction: {e}")
