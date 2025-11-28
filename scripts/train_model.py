import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from config import Config

def train_model():
    # Paths to features and labels
    features_path = os.path.join(Config.RESULTS_FOLDER, "features.npy")
    labels_path = os.path.join(Config.RESULTS_FOLDER, "labels.npy")

    # Check if features exist
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print(" Feature or label files not found.")
        print(" Run: python -m scripts.generate_features first.")
        return

    print("🔹 Loading features and labels...")
    X = np.load(features_path)
    y = np.load(labels_path)

    print(f" Loaded {X.shape[0]} samples with {X.shape[1]} features each.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train RandomForest model
    print("🔹 Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    print(" Training complete! Evaluating on test set...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save model
    model_path = os.path.join(Config.RESULTS_FOLDER, "language_model.pkl")
    joblib.dump(model, model_path)
    print(f" Model saved at: {os.path.abspath(model_path)}")

if __name__ == "__main__":
    train_model()
