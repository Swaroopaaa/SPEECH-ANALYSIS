# NativeLanguageIdentification - Flask Demo

Structure: see project. This repo provides:
- Flask web app: `webapp/` (templates + static)
- Model code: `model/` for HuBERT embeddings, MFCCs, training & evaluation
- Dataset scripts: `scripts/download_dataset.py` (Hugging Face)
- Placeholder `data/`, `results/`, `docs/`

Quick start:
1. Create and activate a Python venv (recommended).
2. `pip install -r requirements.txt`
3. (Optional) `python scripts/download_dataset.py`
4. (Optional) `python model/generate_features.py`
5. (Optional) `python model/train_model.py`
6. Run app: `python app.py`
7. Open http://127.0.0.1:5000

Note: Training may require GPU and takes time. The web app falls back to dummy prediction if no classifier is present.
