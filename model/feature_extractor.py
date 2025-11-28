# MFCC extractor and helpers
import librosa
import numpy as np

def extract_mfcc(path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # use mean and std to get fixed-length vector
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    feat = np.concatenate([mfcc_mean, mfcc_std])
    return feat
