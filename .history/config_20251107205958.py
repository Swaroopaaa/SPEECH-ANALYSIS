import os

class Config:
    # Base directory of the whole project
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Web app settings
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-me")
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "webapp", "static", "uploads")

    # Project folders
    MODEL_FOLDER = os.path.join(BASE_DIR, "model")
    DATA_FOLDER = os.path.join(BASE_DIR, "data")
    RESULTS_FOLDER = os.path.join(BASE_DIR, "results")

    # Audio settings
    ALLOWED_EXTENSIONS = {"wav"}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    SAMPLE_RATE = 16000

