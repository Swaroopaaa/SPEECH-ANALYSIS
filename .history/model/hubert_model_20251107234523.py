import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel

# ✅ Try to switch to "soundfile" if available (new torchaudio syntax)
try:
    torchaudio.use_audio_backend("soundfile")
    print("🔊 Using 'soundfile' backend for torchaudio")
except Exception:
    print("⚠️ 'soundfile' backend not available, using default backend")

class HuBERTExtractor:
    def __init__(self):
        print("🔹 Loading HuBERT base model and feature extractor...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/hubert-base-ls960"
        )
        self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.model.eval()
        print("✅ HuBERT model loaded successfully!")

    def extract_features(self, audio_path):
        print(f"🎧 Extracting features from: {audio_path}")
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            sr = 16000
        inputs = self.feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=sr,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings
