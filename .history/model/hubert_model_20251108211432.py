import torch
import numpy as np
import soundfile as sf
import librosa
from transformers import Wav2Vec2FeatureExtractor, HubertModel


class HuBERTExtractor:
    def __init__(self, model_name="facebook/hubert-base-ls960"):
        print("🔹 Loading HuBERT base model and feature extractor...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print(f"✅ HuBERT model loaded successfully on {self.device}!")

    def extract_features(self, audio_path):
        print(f"🎧 Extracting features from: {audio_path}")

        # Load audio
        data, sr = sf.read(audio_path)

        # Convert stereo → mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Resample to 16k if needed
        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Convert to tensor
        waveform = torch.tensor(data, dtype=torch.float32)

        # Prepare input for HuBERT model
        inputs = self.feature_extractor(
            waveform.cpu().numpy(),
            sampling_rate=sr,
            return_tensors="pt"
        ).to(self.device)

        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooled embedding (final 768-dim vector)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        return embeddings
