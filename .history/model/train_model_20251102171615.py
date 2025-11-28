# trains a simple classifier using HuBERT embeddings (preferred) or MFCCs
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from config import Config
from model.feature_extractor import extract_mfcc
from model.hubert_model import HuBERTExtractor
import soundfile as sf

MODEL_DIR = Config.MODEL_FOLDER
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(use_hubert=True):
    X = []
    y = []
    extractor = HuBERTExtractor()
    for lang in sorted(os.listdir(Config.DATA_FOLDER)):
        lang_path = os.path.join(Config.DATA_FOLDER, lang)
        if not os.path.isdir(lang_path):
            continue
        for fname in os.listdir(lang_path):
            if not fname.lower().endswith(".wav"):
                continue
            path = os.path.join(lang_path, fname)
            try:
                if use_hubert:
                    # load waveform and run hubert
                    waveform, sr = sf.read(path)
                    if waveform.ndim > 1:
                        waveform = waveform.mean(axis=1)
                    emb = extractor.extract(waveform, sr=sr)
                    feat = emb
                else:
                    feat = extract_mfcc(path)
            except Exception as e:
                print("skip", path, e)
                continue
            X.append(feat)
            y.append(lang)
    X = np.stack(X)
    return X, y

def train(use_hubert=True):
    print("Loading data...")
    X, y = load_data(use_hubert=use_hubert)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    print("Training classifier...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print("Validation accuracy:", acc)
    print(classification_report(y_val, y_pred, target_names=le.classes_))
    # save
    joblib.dump(clf, os.path.join(MODEL_DIR, "classifier.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print("Saved classifier and label encoder in", MODEL_DIR)

if __name__ == "__main__":
    # default: train using hubert features
    train(use_hubert=True)
