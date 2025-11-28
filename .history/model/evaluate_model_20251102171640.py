# simple evaluation wrapper (plots confusion matrix etc.)
import os
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from config import Config
from model.train_model import load_data

MODEL_DIR = Config.MODEL_FOLDER

def evaluate():
    clf = joblib.load(os.path.join(MODEL_DIR, "classifier.pkl"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    X, y = load_data(use_hubert=True)
    y_enc = le.transform(y)
    y_pred = clf.predict(X)
    print(classification_report(y_enc, y_pred, target_names=le.classes_))
    cm = confusion_matrix(y_enc, y_pred)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(len(le.classes_)), le.classes_, rotation=45)
    plt.yticks(np.arange(len(le.classes_)), le.classes_)
    plt.tight_layout()
    out = os.path.join(Config.RESULTS_FOLDER, "confusion_matrix.png")
    plt.savefig(out)
    print("Saved confusion matrix to", out)

if __name__ == "__main__":
    evaluate()
