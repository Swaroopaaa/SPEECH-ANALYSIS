#TEAM - SPARKMATES

# NativeLanguageIdentification - SPEECH-ANALYSIS 
NLP Mini Project || MFCC + HuBERT || Accent Classification System

#1. Project Overview

This project implements a complete system for native language (accent) identification from speech using two approaches:

*MFCC-based acoustic model

*HuBERT-embedding-based deep learning model

The system classifies accents from short words or full sentences and includes:

*Adult + child speech dataset

*Age generalization study

*Word-level vs Sentence-level detection experiments

*Layer-wise HuBERT transformer analysis

*Accent-aware cuisine recommendation demo

*A simple Flask web application for testing audio files

This repository includes code, documentation, models, and notebooks.

#2. Folder Structure

NativeLanguageIdentification/
│
├── Include/
├── Lib/
├── Scripts/
├── share/
├── pyvenv.cfg
│
├── models/
│   ├── __init__.py
│   ├── classifier.pkl                # Trained MFCC/HuBERT classifier
│   ├── evaluate_model.py             # Evaluation and accuracy metrics
│   ├── feature_extractor.py          # MFCC + HuBERT feature extraction
│   ├── generate_features.py          # Converts audio → feature vectors
│   ├── hubert_model.py               # HuBERT embedding model wrapper
│   ├── label_encoder.pkl             # Label mapping file
│   └── train_model.py                # Complete training pipeline
│
|
│
├── results/
│   ├── classifier.pkl                # Final model for web app
│   ├── language_model.pkl            # MFCC model file
│   ├── metrics.json                  # Accuracy, layer analysis, A→C generalization
│   ├── confusion_matrix.png          # Visualization of model performance
│   └── sample_outputs/               # Any saved predictions or reports
│
├── scripts/
│   ├── __pycache__/                  # Auto-generated Python cache
│   │
│   ├── download_dataset.py           # Downloads the speech dataset
│   ├── generate_features.py          # Feature extraction script
│   ├── merge_features.py             # Merges all feature files
│   ├── predict_sample.py             # Predicts accent for a single audio file
│   ├── preprocess_audio.py           # Audio cleaning & normalization
│   └── train_model.py                # Model training entry script
│
├── webapp/
│   ├── routes/
│   │    └── main_routes.py           # Flask routing logic
│   │
│   ├── static/
│   │    ├── css/
│   │    │    └── style.css
│   │    ├── img/
│   │    │    ├── image.png
│   │    │    └── mic.svg
│   │    └── js/
│   │         └── record.js           # Browser mic recording script
│   │
│   ├── uploads/
│   │    └── User audio files
│   │
│   └── templates/
│        ├── index.html               # Home page (record / upload)
│        ├── layout.html              # Base HTML structure
│        └── result.html              # Accent prediction output
│
├── data/
│   └── (Downloaded datasets, processed audio)
│
├── docs/
│   └── Project documentation files
│
├── app.py                             # Flask application entry point
├── config.py                          # App configuration and folder paths
├── .gitignore
├── README.md
├── requirements.txt


#3. Completed Tasks 
#3.1 Dataset Preprocessing

Explore: Cleaned, normalized, and organized both adult and child speech samples.
Includes silence removal, resampling, metadata mapping, and label encoding.

#3.2 MFCC-based Accent Classification Model

Explore: Built a traditional ML model using MFCC features (20–40 coefficients) with SVM/RandomForest for baseline accent identification.

#3.3 HuBERT-based Embedding Model

Explore: Extracted deep speech embeddings using HuBERT Base and trained a classifier for accent prediction with higher robustness.

#3.4 Layer-wise Transformer Analysis

Explore: Inspected transformer layers to understand which layers capture accent cues versus phonetic/semantic information.

#3.5 Word-level Accent Detection

Explore: Split audio into words, extracted embeddings, and predicted the accent of each word-level segment.

#3.6 Sentence-level Accent Detection

Explore: Performed full-sentence inference for accent classification using average-pooled HuBERT embeddings.

#3.7 Age-Generalization (Adult → Child)

Explore: Evaluated how an accent model trained on adult speech performs on child speech, observing generalization gaps.

#3.8 Flask Web Application

Explore: Built a complete UI + backend:

Upload audio

Run MFCC/HuBERT model

Show predictions with confidence

Visualize waveform, spectrogram, and embeddings

#3.9 Complete Documentation & Notebooks

Explore: Added clean Jupyter notebooks for:

Preprocessing

Feature extraction

Model training

Evaluation

Visualization

Also created full documentation including README and file structure.


#4. Installation
Step 1 — Clone the repository
git clone https://github.com/Swaroopaaa/SPEECH-ANALYSIS.git
cd SPEECH-ANALYSIS

Step 2 — Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate   # Linux / Mac

Step 3 — Install dependencies
pip install -r requirements.txt  

#5. How to Run the Application (Explore)
#5.1 Ensure Required Files Are Present

Before running the app, confirm that the following files exist:

model/classifier.pkl – MFCC or HuBERT classifier

model/language_model.pkl – Language model for embeddings

scripts/metrics.json – Evaluation metrics for reference

#5.2 Run the Flask App

Execute the Flask application from the project root directory:

python app.py

#5.3 Open in Browser

Once the server is running, access the application at:

http://127.0.0.1:5000/

#5.4 Upload Audio for Accent Prediction

Upload a .wav or supported audio file through the web interface

View predicted accent, confidence scores, and optional visualizations such as waveform and spectrogram

#6. Output Description (Explore)

The application provides the following outputs:

| **Output**                         | **Description**                                                                                     |
| ---------------------------------- | --------------------------------------------------------------------------------------------------- |
| Predicted Accent / Native Language | Shows the predicted accent or native language for the uploaded audio.                               |
| Confidence Score                   | Displays the model’s confidence in its prediction (0–100%).                                         |
| Model Used                         | Indicates whether the prediction was made using MFCC or HuBERT.                                     |
| Side-by-Side Comparison            | If both MFCC and HuBERT models are available, shows predictions and confidence scores side by side. |



| **Metric**               | **Description**                                                  |
| ------------------------ | ---------------------------------------------------------------- |
| MFCC Model Accuracy      | Overall accuracy of the MFCC-based classifier on test data.      |
| HuBERT Model Accuracy    | Overall accuracy of the HuBERT-based model on test data.         |
| Word-level Accuracy      | Accuracy of accent detection at the word level.                  |
| Sentence-level Accuracy  | Accuracy of accent detection at the sentence level.              |
| Age-Generalization Score | Performance of adult-trained models when tested on child speech. |


#8. How to Cite This Work

If you use this project, please cite it as:

SPARKMATES (2025). Native Language Identification — SPEECH-ANALYSIS.
GitHub Repository: https://github.com/Swaroopaaa/SPEECH-ANALYSIS

For the HuBERT model, cite:

Hsu, W.-N. et al. (2021). HuBERT: Self-Supervised Speech Representation Learning.
Meta AI Research

#9. Explore

This project allows you to explore the following aspects of accent and language identification:

#Accent Patterns Across Age Groups
Analyze differences in accent between adult and child speech.

#MFCC vs Embedding-Based Models
Compare traditional MFCC-based classifiers with HuBERT embedding-based models.

#Transformer Layer-Level Contributions
Investigate which transformer layers capture accent-specific information.

#Word-Level vs Sentence-Level Accent Cues
Study short-segment (word-level) versus long-utterance (sentence-level) accent patterns.

#Age Transfer and Generalization
Explore how models trained on adult speech generalize to child speech.

#Practical Accent-Based Applications
Consider real-world scenarios where accent detection can inform recommendations.

#End-to-End Web Interface
Upload audio and get predictions with visualizations through the Flask application.

#10. Team Members

Bolisetti Jyothi Swarupa

Pydikodanala Devi

#11. Project Repository

GitHub: https://github.com/Swaroopaaa/SPEECH-ANALYSIS
