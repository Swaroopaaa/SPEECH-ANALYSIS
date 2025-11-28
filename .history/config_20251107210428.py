import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    SECRET_KEY = os.environ.get("SECRET_KEY", "change-me")
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "webapp", "static", "uploads")

    MODEL_FOLDER = os.path.join(BASE_DIR, "model")
    DATA_FOLDER = os.path.join(BASE_DIR, "data")
    RESULTS_FOLDER = os.path.join(BASE_DIR, "results")

    ALLOWED_EXTENSIONS = {"wav"}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    SAMPLE_RATE = 16000

    # ✅ Add this
    HUBERT_MODEL = "facebook/hubert-base-ls960"

