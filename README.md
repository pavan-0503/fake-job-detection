# Fake Job Detection API# 🔍 Fake Job Alert Detection System



## 🚀 Live DemoA complete machine learning system that detects whether a job posting is legitimate or fake using **DistilBERT embeddings** combined with a **RandomForest classifier**.

Deployed on Railway: `https://your-app-name.up.railway.app`

## 🌟 Features

## 📊 Features

- Detect fake job postings with 89% accuracy- **Advanced ML Model**: Combines DistilBERT (768-dim embeddings) with custom engineered features

- Analyze job text or URLs directly- **Multi-Source Support**: Accepts raw text, structured data, or job posting URLs

- Web scraping support for major job sites- **Web Scraping**: Automatically scrapes job details from LinkedIn, Naukri, Indeed, TimesJobs, etc.

- Expiration detection- **REST API**: Flask backend with CORS support

- Confidence scores with source reputation boost- **Interactive UI**: Beautiful web interface for instant predictions

- **High Accuracy**: Balanced RandomForest with 300 trees for robust classification

## 🔧 Tech Stack

- **ML**: DistilBERT + RandomForest (600 trees, calibrated)## 📋 Requirements

- **Backend**: Flask + Gunicorn

- **Scraping**: Selenium + BeautifulSoup- Python 3.8+

- **Deployment**: Railway- CUDA-capable GPU (optional, for faster training)

- 4GB+ RAM recommended

## 📡 API Usage

## 🚀 Quick Start

### Health Check

```bash### Option A: Automated Setup (Recommended for Windows)

curl https://your-app-name.up.railway.app/health

``````bash

# Run the setup script (creates venv and installs dependencies)

### Predictsetup.bat          # or setup.ps1 for PowerShell

```bash

curl -X POST https://your-app-name.up.railway.app/predict \# Train the model

  -H "Content-Type: application/json" \run_train.bat      # or run_train.ps1

  -d '{"input": "your job description or URL"}'

```# Start the web application

run_app.bat        # or run_app.ps1

## 🎯 Supported Job Sites```

LinkedIn, Naukri, Indeed, Glassdoor, Monster, Foundit, TimesJobs, Shine, JobHai

### Option B: Manual Setup

## 📄 License

Educational project for fake job detection```bash

# Create virtual environment
python -m venv venv

# Activate virtual environment (if execution policy allows)
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat

# OR use Python directly without activation:
venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your `merged_job_postings.csv` in the root directory. The CSV should have these columns:
- `title` - Job title
- `description` - Job description
- `requirements` - Job requirements
- `company_profile` - Company information
- `industry` - Industry type
- `location` - Job location
- `salary_range` - Salary information (optional)
- `fraudulent` - Label (0 = Legitimate, 1 = Fake)

### 3. Train the Model

**Using helper scripts:**
```bash
run_train.bat      # Windows CMD
# or
run_train.ps1      # PowerShell
```

**Or manually:**
```bash
venv\Scripts\python.exe train_model.py
```

This will:
- Load and preprocess the dataset
- Generate DistilBERT embeddings for all job postings
- Extract custom features (free_email, whatsapp, fee_request, etc.)
- Train a RandomForest classifier with 300 trees
- Save models to `models/` directory

**Expected output:**
```
✓ Dataset loaded: XXXX records
✓ BERT embeddings generated: (XXXX, 768)
✓ Model training complete!
✓ Overall Accuracy: XX.XX%
```

### 4. Run the Web Application

**Using helper scripts:**
```bash
run_app.bat        # Windows CMD
# or
run_app.ps1        # PowerShell
```

**Or manually:**
```bash
venv\Scripts\python.exe app.py
```

The server will start at: **http://127.0.0.1:5000**

## 🖥️ Usage

### Web Interface

1. Open http://127.0.0.1:5000 in your browser
2. Paste either:
   - **Job description text** (copy-paste from anywhere)
   - **Job posting URL** (LinkedIn, Naukri, Indeed, etc.)
3. Click "Analyze Job Posting"
4. View prediction with confidence score

### API Endpoints

#### POST /predict
Analyze a single job posting (text or URL)

**Request:**
```json
{
  "input": "job description text or URL"
}
```

**Response:**
```json
{
  "prediction": "Fake Job" or "Legit Job",
  "confidence": 0.95,
  "probability_fake": 0.95,
  "probability_legit": 0.05,
  "source": "text" or "scraped",
  "scraped_info": {
    "title": "...",
    "company": "...",
    "location": "..."
  }
}
```

#### POST /predict/batch
Analyze multiple job postings

**Request:**
```json
{
  "inputs": ["text1", "url1", "text2"]
}
```

**Response:**
```json
{
  "results": [
    {"prediction": "Fake Job", "confidence": 0.95, ...},
    ...
  ]
}
```

#### GET /health
Check server health

#### GET /stats
Get model statistics

### Python API

```python
# Make sure to use the venv Python
# venv\Scripts\python.exe

from predictor import FakeJobPredictor

# Initialize predictor
predictor = FakeJobPredictor()

# Predict from text
result = predictor.predict("Urgent hiring! Work from home...")
print(result['prediction'])  # "Fake Job" or "Legit Job"
print(result['confidence'])  # 0.95

# Predict from structured data
job_data = {
    'title': 'Software Engineer',
    'company': 'Tech Corp',
    'description': 'We are hiring...',
    'requirements': 'Bachelor degree...'
}
result = predictor.predict(job_data)
```

## 🧠 Model Architecture

### Input Features (776 total)

1. **BERT Embeddings (768 features)**
   - Generated using DistilBERT-base-uncased
   - [CLS] token representation of combined job text

2. **Custom Features (8 features)**
   - `free_email`: Detects Gmail/Yahoo/Hotmail contacts
   - `whatsapp`: Detects WhatsApp contact requests
   - `fee_request`: Detects payment/registration fee mentions
   - `salary`: Detects salary information presence
   - `text_length`: Total character count
   - `word_count`: Total word count
   - `has_company`: Company profile presence
   - `has_requirements`: Requirements section presence

### Classifier

- **Algorithm**: RandomForestClassifier
- **Trees**: 300
- **Max Depth**: 20
- **Class Weight**: Balanced (handles imbalanced datasets)
- **Scaling**: StandardScaler normalization

### Text Preprocessing

1. Convert to lowercase
2. Remove URLs
3. Remove email addresses
4. Remove special characters
5. Normalize whitespace

## 📁 Project Structure

```
fake job detection/
├── merged_job_postings.csv    # Training dataset
├── train_model.py              # Training script
├── predictor.py                # Prediction module
├── scraper.py                  # Web scraping module
├── app.py                      # Flask web server
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── templates/
│   └── index.html             # Web interface
└── models/                    # Generated after training
    ├── rf_model.joblib        # Trained RandomForest
    ├── scaler.joblib          # Feature scaler
    ├── feature_info.joblib    # Feature metadata
    └── tokenizer/             # DistilBERT tokenizer
```

## 🧪 Testing

### Test Predictor Module
```bash
test_predictor.bat     # Windows CMD
# or
venv\Scripts\python.exe predictor.py
```

### Test Scraper Module
```bash
venv\Scripts\python.exe scraper.py
```

### Example Test Cases

**Fake Job Indicators:**
- Free email addresses (gmail, yahoo, etc.)
- WhatsApp contact numbers
- Registration/payment fee requests
- Unrealistic salary claims
- Poor grammar and spelling
- Lack of company information

**Legitimate Job Indicators:**
- Professional email domain
- Detailed job requirements
- Company profile/website
- Realistic salary range
- Professional language
- Specific location information

## 🎯 Performance Metrics

The model outputs a detailed classification report including:
- **Precision**: How many predicted fakes are actually fake
- **Recall**: How many actual fakes are detected
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: True positives, false positives, etc.

## 🔧 Troubleshooting

### PowerShell execution policy error
If you get "running scripts is disabled" error:
```powershell
# Option 1: Use batch files instead (.bat)
setup.bat
run_train.bat
run_app.bat

# Option 2: Bypass execution policy for one command
powershell -ExecutionPolicy Bypass -File setup.ps1

# Option 3: Use Python directly
venv\Scripts\python.exe train_model.py
venv\Scripts\python.exe app.py
```

### Model not found error
```bash
# Train the model first
venv\Scripts\python.exe train_model.py
```

### Out of memory during training
- Reduce batch size in `train_model.py`
- Use CPU instead of GPU
- Process dataset in smaller chunks

### Scraping fails
- Some websites block scraping
- Try using raw text instead of URL
- Check internet connection

## 📝 Notes

- **GPU Support**: Automatically uses CUDA if available
- **Training Time**: Depends on dataset size (typical: 10-30 minutes)
- **Prediction Time**: ~1-2 seconds per job posting
- **Supported URLs**: LinkedIn, Naukri, Indeed, TimesJobs, Shine

## 🤝 Contributing

Feel free to enhance the system:
- Add more job portals to scraper
- Improve feature engineering
- Fine-tune model hyperparameters
- Add more test cases

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- **DistilBERT**: Hugging Face Transformers
- **Dataset**: Job posting fraud detection dataset
- **Framework**: Flask, scikit-learn, PyTorch

---

**Made with ❤️ for safer job hunting**
