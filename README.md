
# Native Language Identification – SPEECH-ANALYSIS  
### NLP Mini Project | MFCC + HuBERT | Accent Classification System  
### TEAM – SPARKMATES

---

##  1. Project Overview

This project implements a complete system for **Native Language (Accent) Identification** from speech using:

- **MFCC-based acoustic model**
- **HuBERT-embedding-based deep learning model**

The system includes:
- Adult + Child speech dataset  
- Word-level & Sentence-level accent detection  
- Transformer layer analysis  
- Age generalization (Adult → Child)
- Flask web app for audio upload & prediction  
- Full documentation + notebooks  

# Native Language Identification

This project identifies the speaker's native language/accent using MFCC features and HuBERT embeddings. The repository includes complete training scripts, preprocessing pipeline, evaluation tools, and a Flask-based web interface.

---

## Project Directory Structure

<pre>
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
├── results/
│   ├── classifier.pkl                # Final model for web app
│   ├── language_model.pkl            # MFCC model file
│   ├── metrics.json                  # Accuracy, layer analysis
│   ├── confusion_matrix.png          # Visualization of model performance
│   └── sample_outputs/               # Saved predictions or reports
│
├── scripts/
│   ├── __pycache__/                  # Python cache files
│   ├── download_dataset.py           # Downloads speech dataset
│   ├── generate_features.py          # Feature extraction script
│   ├── merge_features.py             # Merges MFCC/HuBERT features
│   ├── predict_sample.py             # Predicts accent for audio file
│   ├── preprocess_audio.py           # Audio cleaning and normalization
│   └── train_model.py                # Model training entry script
│
├── webapp/
│   ├── routes/
│   │   └── main_routes.py            # Flask routing logic
│   │
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   ├── img/
│   │   │   ├── image.png
│   │   │   └── mic.svg
│   │   └── js/
│   │       └── record.js             # Mic recording script
│   │
│   ├── uploads/                      # User audio files
│   │
│   └── templates/
│       ├── index.html                # Home page (record/upload)
│       ├── layout.html               # Base HTML structure
│       └── result.html               # Prediction output page
│
├── data/                             # Downloaded datasets, processed audio
├── docs/                             # Documentation files
│
├── app.py                            # Flask application entry point
├── config.py                         # App configuration
├── .gitignore
├── README.md
└── requirements.txt

</pre>


##  3. Completed Tasks

###  3.1 Dataset Preprocessing  
- Cleaned, normalized, resampled audio  
- Silence removal  
- Metadata mapping  
- Label encoding  

###  3.2 MFCC-Based Model  
- Extracted 20–40 MFCC  
- Trained SVM/RandomForest classifier  

###  3.3 HuBERT-Based Model  
- Extracted HuBERT Base embeddings  
- Layer-wise embedding comparison  
- High accuracy & robustness  

###  3.4 Layer-wise Transformer Analysis  
- Studied which HuBERT layers capture accent cues  

###  3.5 Word-Level Accent Detection  
- Word segmentation  
- Per-word prediction  

###  3.6 Sentence-Level Accent Detection  
- Full-sentence embeddings  
- Stable prediction  

###  3.7 Age Generalization  
- Adult-trained → Child speech evaluation  

###  3.8 Flask Web App  
- Audio upload  
- MFCC + HuBERT prediction  
- Waveform & spectrogram  

###  3.9 Documentation  
- Preprocessing notebook  
- Training notebook  
- Evaluation notebook  

---

##  4. Installation

### Step 1 — Clone the Repository  
git clone https://github.com/Swaroopaaa/SPEECH-ANALYSIS.git
cd SPEECH-ANALYSIS

###Step 2 — Create Virtual Environment
python -m venv venv
venv\Scripts\activate      # Windows
# or
source venv/bin/activate   # Linux/Mac

###Step 3 — Install Requirements
pip install -r requirements.txt

## 5. How to Run the Application
###5.1 Required Files

Ensure the following files exist:

models/classifier.pkl

models/language_model.pkl

results/metrics.json

###5.2 Run the Flask App
python app.py

###5.3 Open Browser
http://127.0.0.1:5000/

##6. Output Description
###Prediction Outputs
| Output           | Description                       |
| ---------------- | --------------------------------- |
| Predicted Accent | Accent/native language            |
| Confidence Score | Model confidence                  |
| Model Used       | MFCC or HuBERT                    |
| Comparison       | Both model outputs (if available) |

###Performance Metrics
| Metric                  | Meaning                    |
| ----------------------- | -------------------------- |
| MFCC Accuracy           | Baseline model accuracy    |
| HuBERT Accuracy         | Deep model performance     |
| Word-Level Accuracy     | Word-segment predictions   |
| Sentence-Level Accuracy | Full-utterance predictions |
| Age-Generalization      | Adult → Child accuracy     |

##8. How to Cite
SPARKMATES (2025). Native Language Identification — SPEECH-ANALYSIS.
GitHub: https://github.com/Swaroopaaa/SPEECH-ANALYSIS


###HuBERT:

Hsu, W.-N. et al. (2021). HuBERT: Self-Supervised Speech Representation Learning.
Meta AI Research.

##9. Explore
###9.1 Accent Patterns Across Age Groups

Different age groups have unique pronunciation and intonation patterns. Studying this improves accent detection accuracy.

###9.2 MFCC vs HuBERT

MFCC captures basic speech features, while HuBERT extracts deeper contextual patterns. Comparing them shows efficiency vs accuracy trade-offs.

###9.3 Word vs Sentence Accent Cues

Word-level cues focus on pronunciation; sentence-level cues capture rhythm and stress. Both improve model performance.

###9.4 Transformer Layer Analysis

Early layers detect phonetics; deeper layers capture semantic and accent features. Layer-wise study helps understand model behavior.

###9.5 Real-World Applications

Accent detection can enhance:

Voice assistants

Language learning tools

Regional content or cuisine recommendations

Speech-to-text for non-native speakers

###9.6 Web Interface

Allows users to record/upload audio, see accent predictions, get cuisine recommendations, and view interactive visualizations.

##10. Team Members

Bolisetti Jyothi Swarupa

Pydikodanala Devi

##11. Project Repository

GitHub: https://github.com/Swaroopaaa/SPEECH-ANALYSIS
