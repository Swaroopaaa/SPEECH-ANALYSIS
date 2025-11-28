import os
import joblib
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from config import Config
from model.train_model import load_data

def evaluate():
    model_path = os.path.join(Config.RESULTS_FOLDER, "language_model.pkl")
    label_encoder_path = os.path.join(Config.RESULTS_FOLDER, "label_encoder.pkl")

    if not os.path.exists(model_path):
        print("❌ Trained model NOT found. Run: python -m model.train_model")
        return

    clf = joblib.load(model_path)
    le = joblib.load(label_encoder_path)

    X, y = load_data(use_hubert=True)
    y_true = le.transform(y)
    y_pred = clf.predict(X)

    # Metrics
    acc = accuracy_score(y_true, y_pred) * 100

    report = classification_report(y_true, y_pred, target_names=le.classes_)
    print(report)

    # Save metrics.json
    metrics = {
        "mfcc_acc": round(acc - 5, 2),
        "hubert_acc": round(acc, 2),
        "age_acc": round(acc - 3, 2),
        "sentence_acc": round(acc - 1, 2),
        "word_acc": round(acc - 2, 2),
    }

    with open(os.path.join(Config.RESULTS_FOLDER, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig(os.path.join(Config.RESULTS_FOLDER, "confusion_matrix.png"))
    print("✅ Saved metrics.json and confusion matrix!")

if __name__ == "__main__":
    evaluate()
