import torch
import numpy as np
import soundfile as sf
import librosa
from transformers import Wav2Vec2FeatureExtractor, HubertModel


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

        # 🔸 Load the audio file using soundfile (no backend issues)
        data, sr = sf.read(audio_path)

        # 🔸 Convert stereo to mono if necessary
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # 🔸 Resample to 16 kHz if needed
        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            sr = 16000

        # 🔸 Convert to tensor
        waveform = torch.tensor(data).unsqueeze(0)

        # 🔸 Extract HuBERT features
        inputs = self.feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=sr,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 🔸 Mean-pooling for utterance-level embedding
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        return embeddings
