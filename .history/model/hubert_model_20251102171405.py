# utilities to extract HuBERT embeddings
import torch
from transformers import HubertModel, Wav2Vec2Processor
import numpy as np

class HuBERTExtractor:
    def __init__(self, device="cpu"):
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960")
        self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
        self.model.eval()

    def extract(self, waveform, sr=16000):
        # waveform: numpy array float32
        inputs = self.processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            # mean pooling over time dimension
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return emb.squeeze()
