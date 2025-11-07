"""
Fake Job Detection - Training Script
Trains a RandomForest classifier using DistilBERT embeddings + custom features
"""

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import os
import sys
import time
import warnings
import argparse
warnings.filterwarnings('ignore')

# Create models directory
os.makedirs('models/tokenizer', exist_ok=True)

parser = argparse.ArgumentParser(description="Train fake job detection model")
parser.add_argument("--verbose", action="store_true", help="Print detailed progress logs")
args = parser.parse_args()
VERBOSE = args.verbose

def log(msg, force=False):
    """Conditional logger. Use force=True for essential messages."""
    if VERBOSE or force:
        print(msg)

log("Starting training (Fake Job Detection)", force=True)

# ============================================
# 1. LOAD DATASET
# ============================================
log("\nLoading dataset...", force=True)

# Try different encodings to handle the file
encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
df = None

for encoding in encodings:
    try:
        if VERBOSE:
            print(f"  Trying encoding: {encoding}")
        df = pd.read_csv('merged_job_postings.csv', encoding=encoding, on_bad_lines='skip')
        break
    except (UnicodeDecodeError, Exception):
        continue

if df is None:
    log("Primary encodings failed; falling back with ignore errors.", force=True)
    df = pd.read_csv('merged_job_postings.csv', encoding='utf-8', errors='ignore', on_bad_lines='skip')

log(f"Dataset records: {len(df)}", force=True)
if VERBOSE:
    print(f"Columns: {list(df.columns)}")

# Check class distribution
log("\nClass distribution:", force=True)
log(f"  Legitimate (0): {len(df[df['fraudulent'] == 0])}", force=True)
log(f"  Fake      (1): {len(df[df['fraudulent'] == 1])}", force=True)

# Use balanced dataset: 4000 fake + 4000 legitimate = 8000 total
log("\nBalancing dataset to 4000 fake / 4000 legit (or max available)...", force=True)
fake_jobs = df[df['fraudulent'] == 1]
legit_jobs = df[df['fraudulent'] == 0]

# Sample 4000 from each class
SAMPLES_PER_CLASS = 4000
fake_sample = fake_jobs.sample(n=min(SAMPLES_PER_CLASS, len(fake_jobs)), random_state=42)
legit_sample = legit_jobs.sample(n=min(SAMPLES_PER_CLASS, len(legit_jobs)), random_state=42)

# Combine and shuffle
df = pd.concat([fake_sample, legit_sample], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

log(f"Balanced total: {len(df)} (Fake: {len(df[df['fraudulent'] == 1])} | Legit: {len(df[df['fraudulent'] == 0])})", force=True)
if VERBOSE:
    print("Estimated embedding time (CPU) roughly 60-90 min for 8K records")

# ============================================
# 2. TEXT PREPROCESSING
# ============================================
log("\nPreprocessing text ...", force=True)

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits (keep basic punctuation)
    text = re.sub(r'[^a-zA-Z\s.,!?]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Combine all text fields
def combine_features(row):
    """Combine all text features into one"""
    parts = []
    
    for col in ['title', 'description', 'requirements', 'company_profile', 
                'industry', 'location']:
        if col in df.columns and pd.notna(row.get(col)):
            parts.append(str(row[col]))
    
    return " ".join(parts)

df['combined_text'] = df.apply(combine_features, axis=1)
df['cleaned_text'] = df['combined_text'].apply(clean_text)

# Remove empty rows
df = df[df['cleaned_text'].str.len() > 10].reset_index(drop=True)
log(f"Cleaned records retained: {len(df)}", force=True)

# ============================================
# 3. GENERATE BERT EMBEDDINGS
# ============================================
log("\nLoading DistilBERT ...", force=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Device: {device}", force=True)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
model.eval()

log("\nGenerating embeddings ...", force=True)
start_time = time.time()

def _print_progress(current, total, start_time):
    """Render a one-line progress bar with percent and ETA."""
    current = min(current, total)
    pct = (current / total) if total else 0.0
    elapsed = time.time() - start_time
    rate = (current / elapsed) if elapsed > 0 else 0.0
    remaining = (total - current) / rate if rate > 0 else 0.0
    # bar visuals
    bar_len = 30
    filled = int(bar_len * pct)
    bar = '█' * filled + '-' * (bar_len - filled)
    msg = f"\rEmbedding: |{bar}| {pct*100:5.1f}%  {current}/{total}  ETA: {int(remaining//60)}m {int(remaining%60)}s"
    sys.stdout.write(msg)
    sys.stdout.flush()

def get_bert_embedding(text, max_length=512):
    """Generate BERT embedding for text"""
    if len(text) == 0:
        return np.zeros(768)
    
    # Tokenize
    inputs = tokenizer(text, 
                      return_tensors='pt', 
                      max_length=max_length, 
                      truncation=True, 
                      padding='max_length')
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use [CLS] token embedding
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings[0]

# Generate embeddings in batches with progress bar
batch_size = 16  # Smaller batch for CPU
all_embeddings = []
total_batches = (len(df) + batch_size - 1) // batch_size

if VERBOSE:
    print(f"Total records: {len(df)}, batch_size: {batch_size}")
    print(f"Approx minutes (CPU rough): {len(df) * 2 // 60}")

for i in range(0, len(df), batch_size):
    batch_texts = df['cleaned_text'].iloc[i:i+batch_size].tolist()
    batch_embeddings = [get_bert_embedding(text) for text in batch_texts]
    all_embeddings.extend(batch_embeddings)
    progress = min(i + batch_size, len(df))
    _print_progress(progress, len(df), start_time)

# finish the progress line
sys.stdout.write("\n")
sys.stdout.flush()

bert_features = np.array(all_embeddings)
log(f"Embeddings: {bert_features.shape}", force=True)

# ============================================
# 4. CREATE ADDITIONAL FEATURES
# ============================================
log("\nCreating numeric features ...", force=True)

def extract_features(row):
    """Extract custom features from job posting"""
    text = str(row.get('combined_text', '')).lower()
    
    features = {}
    
    # Free email providers
    free_emails = ['gmail', 'yahoo', 'hotmail', 'outlook', 'rediff', 'yopmail']
    features['free_email'] = int(any(email in text for email in free_emails))
    
    # WhatsApp contact
    features['whatsapp'] = int('whatsapp' in text or 'whats app' in text)
    
    # Fee/Payment requests
    fee_keywords = ['fee', 'pay', 'payment', 'registration fee', 'deposit', 
                   'advance', 'upfront', 'money transfer']
    features['fee_request'] = int(any(keyword in text for keyword in fee_keywords))
    
    # Salary presence
    salary_indicators = ['inr', 'usd', 'salary', 'lpa', 'per month', 'per annum', 
                        r'\d+k', r'\$\d+', r'₹\d+']
    features['salary'] = int(any(re.search(indicator, text) for indicator in salary_indicators))
    
    # Text length
    features['text_length'] = len(text)
    
    # Number of words
    features['word_count'] = len(text.split())
    
    # Bullet/line count (structure strength)
    features['bullet_count'] = text.count('•') + text.count('- ') + text.count('* ')
    
    # Has any email / any phone number
    features['has_email'] = int(bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)))
    features['has_phone'] = int(bool(re.search(r"\b(\+?\d[\d\s().-]{7,}\d)\b", text)))
    
    # URL/link count (over-promotional pages often have many)
    features['link_count'] = len(re.findall(r"https?://", text))
    
    # Contact/apply CTA words
    cta_words = ['contact', 'apply', 'click', 'call', 'email', 'submit']
    features['cta_words'] = int(any(w in text for w in cta_words))
    
    # Uppercase ratio (shouting/clickbait)
    letters = re.findall(r"[A-Za-z]", row.get('combined_text', ''))
    uppers = [ch for ch in letters if ch.isupper()]
    features['uppercase_ratio'] = (len(uppers) / len(letters)) if letters else 0.0
    
    # Has company profile
    features['has_company'] = int(pd.notna(row.get('company_profile')) and 
                                 len(str(row.get('company_profile', ''))) > 5)
    
    # Has requirements
    features['has_requirements'] = int(pd.notna(row.get('requirements')) and 
                                      len(str(row.get('requirements', ''))) > 5)
    
    return pd.Series(features)

extra_features_df = df.apply(extract_features, axis=1)
extra_features = extra_features_df.values

log(f"Extra features: {extra_features.shape}", force=True)
if VERBOSE:
    print(f"Feature names: {list(extra_features_df.columns)}")

# ============================================
# 5. COMBINE ALL FEATURES
# ============================================
log("\nCombining features ...", force=True)
X = np.hstack([bert_features, extra_features])
y = df['fraudulent'].values
log(f"Feature matrix: {X.shape}; labels: {y.shape}", force=True)

# ============================================
# 6. TRAIN-TEST SPLIT
# ============================================
log("\nTrain/test split (80/20) ...", force=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

log(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}", force=True)

# ============================================
# 7. FEATURE SCALING
# ============================================
log("\nScaling features ...", force=True)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
if VERBOSE:
    print("Scaling complete")

# ============================================
# 8. TRAIN RANDOM FOREST CLASSIFIER
# ============================================
log("\nTraining RandomForest ...", force=True)
if VERBOSE:
    print("Parameters: n_estimators=600, max_depth=30, class_weight=balanced")

rf_base = RandomForestClassifier(
    n_estimators=600,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_base.fit(X_train_scaled, y_train)
log("RandomForest base model training complete", force=True)

# Probability calibration for better confidence reliability
log("Calibrating probabilities (isotonic, cv=5) ...", force=True)
rf_model = CalibratedClassifierCV(estimator=rf_base, method='isotonic', cv=5)
rf_model.fit(X_train_scaled, y_train)
log("Calibration complete", force=True)

# ============================================
# 9. EVALUATE MODEL
# ============================================
log("\nEvaluating ...", force=True)

# Predictions
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)

# Classification report
if VERBOSE:
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fake']))
else:
    # Minimal metrics: class-wise precision/recall/F1 + accuracy
    report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Fake'], output_dict=True)
    legit = report['Legitimate']
    fake = report['Fake']
    print("Class metrics (precision / recall / f1):")
    print(f"  Legitimate: {legit['precision']:.2f} / {legit['recall']:.2f} / {legit['f1-score']:.2f}")
    print(f"  Fake      : {fake['precision']:.2f} / {fake['recall']:.2f} / {fake['f1-score']:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
if VERBOSE:
    print("\nConfusion matrix:")
    print(cm)

# Accuracy
accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
log(f"Accuracy: {accuracy:.2%}", force=True)

# ============================================
# 10. SAVE MODELS
# ============================================
log("\nSaving artifacts ...", force=True)

# Save calibrated model (primary)
joblib.dump(rf_model, 'models/rf_model_calibrated.joblib')
if VERBOSE:
    print("Saved: models/rf_model_calibrated.joblib")

# Also save base RF for reference
joblib.dump(rf_base, 'models/rf_model.joblib')
if VERBOSE:
    print("Saved: models/rf_model.joblib")

# Save scaler
joblib.dump(scaler, 'models/scaler.joblib')
if VERBOSE:
    print("Saved: models/scaler.joblib")

# Save tokenizer
tokenizer.save_pretrained('models/tokenizer')
if VERBOSE:
    print("Saved: models/tokenizer/")

# Save feature names for reference
feature_info = {
    'bert_dims': 768,
    'extra_features': list(extra_features_df.columns),
    'total_features': X.shape[1]
}
joblib.dump(feature_info, 'models/feature_info.joblib')
if VERBOSE:
    print("Saved: models/feature_info.joblib")

log("\nTraining finished.", force=True)
log("Artifacts stored in ./models", force=True)
if VERBOSE:
    print("Run: python app.py  (to start server)")
log("Done.", force=True)
