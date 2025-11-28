from flask import Flask
from webapp.routes.main_routes import main_bp
from config import Config
import os

def create_app():
    app = Flask(__name__, template_folder="webapp/templates", static_folder="webapp/static")
    app.config.from_object(Config)
    # ensure uploads dir exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    app.register_blueprint(main_bp)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
