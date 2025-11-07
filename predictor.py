"""
Fake Job Detection - Predictor Module
Loads trained model and makes predictions on new job postings
"""

import numpy as np
import pandas as pd
import re
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import os
import warnings
from datetime import datetime
import dateparser
warnings.filterwarnings('ignore')


class FakeJobPredictor:
    def __init__(self, model_dir='models'):
        """Initialize the predictor with saved models"""
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Decision threshold (probability of fake >= threshold => Fake)
        try:
            self.threshold = float(os.getenv('THRESHOLD', '0.5'))
        except Exception:
            self.threshold = 0.5
        
        
        # Load model (prefer calibrated if available)
        calibrated_path = os.path.join(model_dir, 'rf_model_calibrated.joblib')
        base_path = os.path.join(model_dir, 'rf_model.joblib')
        model_path = calibrated_path if os.path.exists(calibrated_path) else base_path
        self.rf_model = joblib.load(model_path)
        
        # Load scaler
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        
        # Load feature info
        self.feature_info = joblib.load(os.path.join(model_dir, 'feature_info.joblib'))
        
        # Load DistilBERT tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            os.path.join(model_dir, 'tokenizer')
        )
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        self.bert_model.eval()
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s.,!?]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_bert_embedding(self, text, max_length=512):
        """Generate BERT embedding for text"""
        if len(text) == 0:
            return np.zeros(768)
        
        # Tokenize
        inputs = self.tokenizer(text, 
                               return_tensors='pt', 
                               max_length=max_length, 
                               truncation=True, 
                               padding='max_length')
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        # Use [CLS] token embedding
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings[0]
    
    def extract_features(self, text):
        """Extract custom features from job posting text"""
        text_lower = text.lower()

        features = {}

        # Free email providers
        free_emails = ['gmail', 'yahoo', 'hotmail', 'outlook', 'rediff', 'yopmail']
        features['free_email'] = int(any(email in text_lower for email in free_emails))

        # WhatsApp contact
        features['whatsapp'] = int('whatsapp' in text_lower or 'whats app' in text_lower)

        # Fee/Payment requests
        fee_keywords = ['fee', 'pay', 'payment', 'registration fee', 'deposit', 'advance', 'upfront', 'money transfer']
        features['fee_request'] = int(any(keyword in text_lower for keyword in fee_keywords))

        # Salary presence
        salary_indicators = ['inr', 'usd', 'salary', 'lpa', 'per month', 'per annum', r'\d+k', r'\$\d+', r'₹\d+']
        features['salary'] = int(any(re.search(indicator, text_lower) for indicator in salary_indicators))

        # Text length and word count
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())

        # Has company / requirements (simple heuristics)
        company_keywords = ['company', 'corporation', 'ltd', 'inc', 'pvt', 'limited']
        req_keywords = ['experience', 'qualification', 'skills', 'requirement', 'education']
        features['has_company'] = int(any(keyword in text_lower for keyword in company_keywords))
        features['has_requirements'] = int(any(keyword in text_lower for keyword in req_keywords))

        # Bullet/line count and contact cues
        features['bullet_count'] = text_lower.count('•') + text_lower.count('- ') + text_lower.count('* ')
        features['has_email'] = int(bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text_lower)))
        features['has_phone'] = int(bool(re.search(r"\b(\+?\d[\d\s().-]{7,}\d)\b", text_lower)))
        features['link_count'] = len(re.findall(r"https?://", text_lower))
        cta_words = ['contact', 'apply', 'click', 'call', 'email', 'submit']
        features['cta_words'] = int(any(w in text_lower for w in cta_words))

        # Uppercase ratio
        letters = re.findall(r"[A-Za-z]", text)
        uppers = [ch for ch in letters if ch.isupper()]
        features['uppercase_ratio'] = (len(uppers) / len(letters)) if letters else 0.0

        # Return as array in saved feature order
        return np.array([features[col] for col in self.feature_info['extra_features']])
    
    def is_job_description(self, text):
        """
        Validate if the text is actually a job description using semantic understanding with BERT
        Returns: (is_valid, reason)
        """
        text_lower = text.lower()
        
        # Basic sanity checks
        if len(text) < 50:
            return False, "Text too short to be a job description (minimum 50 characters required)"
        
        words = text.split()
        word_count = len(words)
        
        if word_count < 20:
            return False, f"Content too brief ({word_count} words). A proper job description should have at least 20 words."
        
        # Check if it's just CSV headers or column names
        csv_indicators = ['title\tdescription', 'job_id', 'fraudulent', 'telecommuting', 
                         'has_company_logo', 'has_questions', 'column', 'header', 'dataset']
        if any(indicator in text_lower for indicator in csv_indicators):
            return False, "This appears to be data headers or metadata, not a job description."
        
        # Check for repetitive/garbage text (same word repeated many times)
        word_freq = {}
        for word in words[:100]:  # Check first 100 words
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # If any word appears more than 30% of the time, it's likely garbage
        max_freq = max(word_freq.values()) if word_freq else 0
        if max_freq > len(words[:100]) * 0.3:
            return False, "Text appears to be repetitive or corrupted. Please provide a clean job description."
        
        # === SEMANTIC UNDERSTANDING WITH BERT ===
        # Instead of keyword matching, use BERT to understand the context
        
        # Generate embedding for the input text
        cleaned_text = self.clean_text(text)
        if len(cleaned_text) < 10:
            return False, "Text doesn't contain meaningful content after cleaning."
        
        # Check for explicit college/university/portal indicators first
        edu_portal_indicators = [
            'vtop', 'student portal', 'university portal', 'college portal',
            'student login', 'faculty login', 'parent login', 'alumni login',
            'digital initiative by the institute', 'academic portal',
            'vit-ap', 'vit on top', 'student management system',
            'transcript', 'semester registration', 'fee payment portal',
            'course registration', 'exam results', 'grade report'
        ]
        edu_count = sum(1 for indicator in edu_portal_indicators if indicator in text_lower)
        if edu_count >= 2:
            return False, "This appears to be a university/college portal or academic system, not a job posting."
        
        input_embedding = self.get_bert_embedding(cleaned_text)
        
        # Define reference job posting examples (these will be compared semantically)
        job_reference_texts = [
            "We are hiring for software engineer position with 2 years experience required skills include python java responsibilities include development",
            "Job opening for marketing manager candidate must have MBA experience in digital marketing apply now with resume",
            "Looking for data analyst position requires SQL Python experience responsibilities include data analysis reporting",
            "Sales executive vacancy excellent communication skills required responsibilities include client management",
            "Immediate opening for content writer work from home flexible hours good writing skills required apply with portfolio"
        ]
        
        # Define non-job reference examples (expanded with educational portals)
        non_job_texts = [
            "Sign in to your account browse jobs search companies create profile save jobs get alerts download app",
            "Breaking news today according to sources the company announced new policy trending now viral",
            "College admission open fee structure tuition fees hostel fees apply now for engineering degree courses university campus",
            "Buy now limited offer click here download app subscribe to channel follow us on social media",
            "University portal student faculty staff parent alumni access academics research services digital platform institute",
            "Student login employee login parent portal transcript semester registration exam results grade report course enrollment"
        ]
        
        # Calculate cosine similarity with job references
        job_similarities = []
        for ref_text in job_reference_texts:
            ref_embedding = self.get_bert_embedding(self.clean_text(ref_text))
            similarity = np.dot(input_embedding, ref_embedding) / (
                np.linalg.norm(input_embedding) * np.linalg.norm(ref_embedding)
            )
            job_similarities.append(similarity)
        
        # Calculate cosine similarity with non-job references
        non_job_similarities = []
        for ref_text in non_job_texts:
            ref_embedding = self.get_bert_embedding(self.clean_text(ref_text))
            similarity = np.dot(input_embedding, ref_embedding) / (
                np.linalg.norm(input_embedding) * np.linalg.norm(ref_embedding)
            )
            non_job_similarities.append(similarity)
        
        # Get max similarities
        max_job_sim = max(job_similarities)
        max_non_job_sim = max(non_job_similarities)
        
        # Decision based on semantic similarity
        # If text is more similar to non-job content, reject it
        # Increased margin from 0.05 to 0.10 for better separation
        if max_non_job_sim > max_job_sim:
            return False, f"This doesn't appear to be a job description. The content is more similar to {self._get_content_type(max_non_job_sim, non_job_texts, input_embedding)} (similarity: {max_non_job_sim:.2f} vs job content: {max_job_sim:.2f})."
        
        # If similarity to job content is too low overall, reject
        # Increased threshold from 0.3 to 0.5 for stricter validation
        if max_job_sim < 0.5:
            return False, f"This doesn't appear to be job-related content. Semantic similarity to job postings is too low ({max_job_sim:.2f}, need at least 0.50)."
        
        # All checks passed - looks like a job description
        return True, None
    
    def _get_content_type(self, similarity, reference_texts, input_embedding):
        """Helper to identify which type of non-job content it matches"""
        # Find which reference text had the highest similarity
        similarities = []
        for ref_text in reference_texts:
            ref_embedding = self.get_bert_embedding(self.clean_text(ref_text))
            sim = np.dot(input_embedding, ref_embedding) / (
                np.linalg.norm(input_embedding) * np.linalg.norm(ref_embedding)
            )
            similarities.append(sim)
        
        max_idx = similarities.index(max(similarities))
        
        # Map to content types
        content_types = [
            "website navigation/UI elements",
            "news articles or blog posts",
            "educational/college information", 
            "promotional/spam content",
            "university/college portals",
            "academic management systems"
        ]
        
        if max_idx < len(content_types):
            return content_types[max_idx]
        return "non-job content"
    
    def is_job_expired(self, text):
        """
        Check if a job posting has expired or is no longer accepting applications
        Returns: (is_expired, reason, deadline_found)
        """
        text_lower = text.lower()
        
        # 1. Check for explicit closure phrases
        closure_phrases = [
            'no longer hiring',
            'position closed',
            'applications closed',
            'position filled',
            'vacancy closed',
            'job closed',
            'hiring closed',
            'this position has been filled',
            'we are not accepting applications',
            'applications are closed',
            'deadline passed',
            'no longer accepting',
            'recruitment closed'
        ]
        
        for phrase in closure_phrases:
            if phrase in text_lower:
                return True, f"Job posting explicitly states: '{phrase}'", None
        
        # 2. Extract and parse deadline dates
        # Common deadline patterns
        deadline_patterns = [
            r'(?:apply by|deadline|last date|close(?:s|d)?\s+on|submit by|applications? close(?:s|d)?\s+on)\s*:?\s*([^.\n]+)',
            r'(?:applications? accepted until|applications? open until)\s*:?\s*([^.\n]+)',
            r'(?:valid till|valid until)\s*:?\s*([^.\n]+)'
        ]
        
        current_date = datetime.now()
        
        for pattern in deadline_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                date_str = match.group(1).strip()
                
                # Try to parse the date using dateparser
                try:
                    parsed_date = dateparser.parse(date_str, settings={'PREFER_DATES_FROM': 'past'})
                    
                    if parsed_date:
                        # Compare with current date
                        if parsed_date < current_date:
                            days_ago = (current_date - parsed_date).days
                            return True, f"Application deadline was {parsed_date.strftime('%B %d, %Y')} ({days_ago} days ago)", parsed_date.strftime('%Y-%m-%d')
                        else:
                            # Future deadline - job is still open
                            days_remaining = (parsed_date - current_date).days
                            return False, f"Job is still open. Deadline: {parsed_date.strftime('%B %d, %Y')} ({days_remaining} days remaining)", parsed_date.strftime('%Y-%m-%d')
                except:
                    # If parsing fails, continue to next match
                    continue
        
        # 3. Check for relative date expressions that might indicate closure
        relative_closed_patterns = [
            r'closed\s+(\d+)\s+(day|week|month|year)s?\s+ago',
            r'expired\s+(\d+)\s+(day|week|month|year)s?\s+ago'
        ]
        
        for pattern in relative_closed_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return True, f"Job posting mentions it closed {match.group(0)}", None
        
        # If no expiration indicators found, assume job is still open
        return False, None, None
    
    def predict(self, job_data, source=None):
        """
        Predict if a job posting is fake or legitimate
        
        Args:
            job_data: Can be either:
                - String: raw job posting text
                - Dict: with keys like 'title', 'description', 'company', etc.
            source: Optional source string (e.g., 'Selenium', 'LinkedIn', 'Naukri')
        
        Returns:
            dict: {
                'prediction': 'Fake Job' or 'Legit Job',
                'confidence': float (0-1),
                'probability_fake': float,
                'probability_legit': float
            }
        """
        # Handle different input types
        if isinstance(job_data, str):
            # Raw text input
            combined_text = job_data
        elif isinstance(job_data, dict):
            # Dictionary input - combine all fields
            parts = []
            for key in ['title', 'description', 'requirements', 'company', 
                       'company_profile', 'industry', 'location', 'text']:
                if key in job_data and job_data[key]:
                    parts.append(str(job_data[key]))
            combined_text = " ".join(parts)
            # Extract source from dict if present
            if 'source' in job_data and not source:
                source = job_data.get('source')
        else:
            raise ValueError("Input must be a string or dictionary")
        
        # Validate if it's a job description
        is_valid, validation_error = self.is_job_description(combined_text)
        if not is_valid:
            return {
                'prediction': 'Not a Job Description',
                'confidence': 0.0,
                'probability_fake': 0.0,
                'probability_legit': 0.0,
                'error': validation_error,
                'is_job': False
            }
        
        # Check if job has expired or is closed
        is_expired, expiration_reason, deadline = self.is_job_expired(combined_text)
        if is_expired:
            return {
                'prediction': 'Job Closed',
                'confidence': 1.0,
                'probability_fake': 0.0,
                'probability_legit': 0.0,
                'reason': expiration_reason,
                'deadline': deadline,
                'is_job': True,
                'is_expired': True
            }
        
        # If we found a future deadline, include it in the result metadata
        deadline_info = {}
        if deadline and not is_expired:
            deadline_info = {
                'deadline': deadline,
                'deadline_note': expiration_reason
            }
        
        # Clean text
        cleaned_text = self.clean_text(combined_text)
        
        if len(cleaned_text) < 10:
            return {
                'prediction': 'Not a Job Description',
                'confidence': 0.0,
                'probability_fake': 0.0,
                'probability_legit': 0.0,
                'error': 'Insufficient job-related content after cleaning',
                'is_job': False
            }
        
        # Generate BERT embedding
        bert_embedding = self.get_bert_embedding(cleaned_text)
        
        # Extract additional features
        extra_features = self.extract_features(combined_text)
        
        # Combine features
        X = np.hstack([bert_embedding, extra_features]).reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict probabilities (calibrated if available)
        probabilities = self.rf_model.predict_proba(X_scaled)[0]
        prob_fake = float(probabilities[1])
        prob_legit = float(probabilities[0])

        # Apply configurable decision threshold
        prediction = 1 if prob_fake >= self.threshold else 0

        # Confidence as max probability (simpler and more intuitive)
        confidence = max(prob_legit, prob_fake)
        
        # Apply source reputation boost for legitimate predictions from verified sources
        if prediction == 0:  # Legit job
            trusted_sources = ['selenium', 'linkedin', 'naukri', 'indeed', 'glassdoor', 
                             'monster', 'foundit', 'timesjobs', 'shine']
            if source and any(trusted in source.lower() for trusted in trusted_sources):
                # Boost confidence by 10-15% for verified sources (cap at 0.99)
                boost = 0.12
                confidence = min(0.99, confidence + boost)
                prob_legit = min(0.99, prob_legit + boost)
                prob_fake = max(0.01, prob_fake - boost)
        
        # Prepare result
        result = {
            'prediction': 'Fake Job' if prediction == 1 else 'Legit Job',
            'confidence': float(confidence),
            'probability_legit': prob_legit,
            'probability_fake': prob_fake,
            'threshold': float(self.threshold),
            'is_job': True
        }
        
        # Add deadline info if available
        if deadline_info:
            result.update(deadline_info)
        
        return result
    
    def predict_batch(self, job_data_list):
        """Predict multiple job postings at once"""
        results = []
        for job_data in job_data_list:
            result = self.predict(job_data)
            results.append(result)
        return results


# For standalone testing
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTING FAKE JOB PREDICTOR")
    print("=" * 60)
    
    # Initialize predictor
    predictor = FakeJobPredictor()
    
    # Test case 1: Suspicious job posting
    print("\n📝 Test 1: Suspicious Job Posting")
    print("-" * 60)
    fake_job = """
    Urgent Hiring! Work from home opportunity. 
    Earn 50000 per month. No experience needed.
    Contact us on WhatsApp: 9876543210
    Registration fee: Rs. 500 only.
    Email: contact@gmail.com
    """
    
    result1 = predictor.predict(fake_job)
    print(f"Input: {fake_job[:100]}...")
    print(f"\n🎯 Prediction: {result1['prediction']}")
    print(f"📊 Confidence: {result1['confidence']:.2%}")
    print(f"   - Probability Legit: {result1['probability_legit']:.2%}")
    print(f"   - Probability Fake: {result1['probability_fake']:.2%}")
    
    # Test case 2: Legitimate job posting
    print("\n" + "=" * 60)
    print("📝 Test 2: Legitimate Job Posting")
    print("-" * 60)
    legit_job = {
        'title': 'Software Engineer',
        'company': 'Tech Corp Private Limited',
        'description': 'We are looking for a skilled software engineer with experience in Python and JavaScript.',
        'requirements': 'Bachelor degree in Computer Science, 2+ years experience in web development',
        'location': 'Bangalore, Karnataka',
        'salary': 'Rs. 8-12 LPA'
    }
    
    result2 = predictor.predict(legit_job)
    print(f"Input: {legit_job}")
    print(f"\n🎯 Prediction: {result2['prediction']}")
    print(f"📊 Confidence: {result2['confidence']:.2%}")
    print(f"   - Probability Legit: {result2['probability_legit']:.2%}")
    print(f"   - Probability Fake: {result2['probability_fake']:.2%}")
    
    print("\n" + "=" * 60)
    print("✅ Testing Complete!")
    print("=" * 60)
