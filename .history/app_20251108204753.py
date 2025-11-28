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

    # Ensure required folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

    # ✅ Load trained model
    model_path = os.path.join(app.config['RESULTS_FOLDER'], "language_model.pkl")
    if os.path.exists(model_path):
        try:
            app.model = joblib.load(model_path)
            print(f"✅ Loaded trained model from: {model_path}")
        except Exception as e:
            app.model = None
            print(f"⚠️ Could not load model: {e}")
    else:
        app.model = None
        print("❌ Model not found. Train it using: python -m scripts.train_model")


    # ✅ Load metrics.json (for performance display)
    metrics_path = os.path.join(app.config['RESULTS_FOLDER'], "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                app.metrics = json.load(f)
            print("📊 Loaded model performance metrics")
        except:
            app.metrics = None
            print("⚠️ Could not read metrics.json")
    else:
        app.metrics = None
        print("❌ metrics.json not found. Train the model to generate it.")

    # Register routes
    app.register_blueprint(main_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
