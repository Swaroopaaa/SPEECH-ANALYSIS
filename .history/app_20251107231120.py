from flask import Flask
from webapp.routes.main_routes import main_bp
from config import Config
import os
import joblib

def create_app():
    app = Flask(
        __name__,
        template_folder="webapp/templates",
        static_folder="webapp/static"
    )

    app.config.from_object(Config)

    # Ensure necessary folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

    # 🔹 Load trained model automatically
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
        print("❌ No trained model found. Please run: python -m scripts.train_model")

    # Register routes
    app.register_blueprint(main_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
