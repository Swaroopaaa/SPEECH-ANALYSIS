
    from flask import Flask
from webapp.routes.main_routes import main_bp
from config import Config
import os
import joblib
import json

def create_app():
    app = Flask(
        __name__,
        template_folder="webapp/templates",
        static_folder="webapp/static"
    )

    app.config.from_object(Config)

    for folder in [
        app.config['UPLOAD_FOLDER'],
        app.config['MODEL_FOLDER'],
        app.config['DATA_FOLDER'],
        app.config['RESULTS_FOLDER']
    ]:
        os.makedirs(folder, exist_ok=True)

    hubert_path = os.path.join(app.config['MODEL_FOLDER'], "classifier.pkl")
    mfcc_path = os.path.join(app.config['RESULTS_FOLDER'], "language_model.pkl")

    app.hubert_model = None
    app.mfcc_model = None

    if os.path.exists(hubert_path):
        try:
            app.hubert_model = joblib.load(hubert_path)
            print(f"Loaded HuBERT model from: {hubert_path}")
        except Exception as e:
            print(f"Failed to load HuBERT model: {e}")

    if os.path.exists(mfcc_path):
        try:
            app.mfcc_model = joblib.load(mfcc_path)
            print(f"Loaded MFCC model from: {mfcc_path}")
        except Exception as e:
            print(f"Failed to load MFCC model: {e}")

    if not app.hubert_model and not app.mfcc_model:
        print("No models loaded. Train at least one model first.")

    metrics_path = os.path.join(app.config['RESULTS_FOLDER'], "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                app.metrics = json.load(f)
            print("Loaded model performance metrics")
        except Exception as e:
            app.metrics = None
            print(f"Could not read metrics.json: {e}")
    else:
        app.metrics = None
        print("metrics.json not found. Train the model to generate it.")

    app.register_blueprint(main_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
