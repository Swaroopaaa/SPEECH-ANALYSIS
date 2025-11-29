
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


<pre> Team_AOT/ │ ├── code/ │ ├── app_cuisine_recommender.py │ ├── datasets.py │ ├── evaluate.py │ ├── inference_child_audio.py │ ├── layerwise_analysis.py │ ├── models.py │ ├── preprocess.py │ ├── random_split_baseline.py │ ├── train_hubert.py │ ├── train_mfcc.py │ └── word_sentence_analysis.py │ ├── models/ │ ├── best_cnn_mfcc.pth │ ├── best_hubert_clf.pth │ ├── hubert_classifier.joblib │ └── mfcc_cnn.pth │ ├── outputs/ │ ├── predictions.csv │ ├── confusion_matrix.png │ ├── classification_report.txt │ ├── layerwise_results.json │ └── mfcc_vs_hubert_results.csv │ ├── Team_AOT_project.ipynb ├── README.md └── requirements.txt </pre>



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

## ⚙️ 4. Installation

### Step 1 — Clone the Repository  
```bash
git clone https://github.com/Swaroopaaa/SPEECH-ANALYSIS.git
cd SPEECH-ANALYSIS

Step 2 — Create Virtual Environment
python -m venv venv
venv\Scripts\activate      # Windows
# or
source venv/bin/activate   # Linux/Mac

Step 3 — Install Requirements
pip install -r requirements.txt

 5. How to Run the Application
5.1 Required Files

Ensure the following files exist:

models/classifier.pkl

models/language_model.pkl

results/metrics.json

5.2 Run the Flask App
python app.py

5.3 Open Browser
http://127.0.0.1:5000/

 6. Output Description
 Prediction Outputs
Output	Description
Predicted Accent	Accent/native language
Confidence Score	Model confidence
Model Used	MFCC or HuBERT
Comparison	Both model outputs (if available)
✔ Performance Metrics
Metric	Meaning
MFCC Accuracy	Baseline model accuracy
HuBERT Accuracy	Deep model performance
Word-Level Accuracy	Word-segment predictions
Sentence-Level Accuracy	Full-utterance predictions
Age-Generalization	Adult → Child accuracy
 8. How to Cite
SPARKMATES (2025). Native Language Identification — SPEECH-ANALYSIS.
GitHub: https://github.com/Swaroopaaa/SPEECH-ANALYSIS


HuBERT:

Hsu, W.-N. et al. (2021). HuBERT: Self-Supervised Speech Representation Learning.
Meta AI Research.

9. Explore

Accent patterns across age groups

MFCC vs HuBERT comparison

Word vs Sentence accent cues

Transformer layer-level analysis

Real-world accent-based applications

End-to-end web interface

 10. Team Members

Bolisetti Jyothi Swarupa

Pydikodanala Devi

🔗 11. Project Repository

GitHub: https://github.com/Swaroopaaa/SPEECH-ANALYSIS
