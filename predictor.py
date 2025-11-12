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
    def clean_text(self, text):
        """Clean and preprocess text: lowercase, remove URLs, emails, special chars, normalize whitespace"""
        if text is None:
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s.,!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def get_bert_embedding(self, text, max_length=512):
        """Generate BERT embedding for text using DistilBERT [CLS] token"""
        if len(text) == 0:
            return np.zeros(768)
        inputs = self.tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings[0]
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

        # Precompute and cache reference embeddings for semantic checks to speed up validation
        self.job_reference_texts = [
            "We are hiring for software engineer position with 2 years experience required skills include python java responsibilities include development",
            "Job opening for marketing manager candidate must have MBA experience in digital marketing apply now with resume",
            "Looking for data analyst position requires SQL Python experience responsibilities include data analysis reporting",
            "Sales executive vacancy excellent communication skills required responsibilities include client management",
            "Immediate opening for content writer work from home flexible hours good writing skills required apply with portfolio"
        ]
        self.non_job_texts = [
            "Sign in to your account browse jobs search companies create profile save jobs get alerts download app",
            "Breaking news today according to sources the company announced new policy trending now viral",
            "College admission open fee structure tuition fees hostel fees apply now for engineering degree courses university campus",
            "Buy now limited offer click here download app subscribe to channel follow us on social media",
            "University portal student faculty staff parent alumni access academics research services digital platform institute",
            "Student login employee login parent portal transcript semester registration exam results grade report course enrollment"
        ]
        # Expanded with domain-specific non-job samples (encyclopedic + fee structure) to improve semantic separation
        self.non_job_texts.extend([
            "Formula One is the highest class of international racing for single-seater formula racing cars sanctioned by the FIA with seasons consisting of multiple Grands Prix events across the world",
            "The fee structure for the B.Tech Computer Science program includes tuition hostel accommodation and caution deposit with semester wise breakdown of amounts payable",
            # Educational/AI intro samples
            "After completing this unit, you’ll be able to explain fundamental concepts of artificial intelligence, identify the challenges that make defining artificial intelligence difficult, describe the types of tasks artificial intelligence can perform.",
            "Artificial intelligence (AI) has been a dream of many storytellers and sci-fi fans for years. But most people hadn’t given AI much serious thought because it was always something that might happen far into the future.",
            "Researchers and computer scientists haven’t been waiting for tomorrow to arrive, they’ve been working hard to make the dream of AI into a reality. In fact, as you know, we’re already well into the Age of AI.",
            # Generic non-job text
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
            "Hello! Welcome to our platform. Please sign in to continue.",
            "This is a sample paragraph for testing purposes."
        ])

        # Map each non-job text to a content type (aligned by index)
        initial_types = [
            "website navigation/UI elements",  # 0
            "news articles or blog posts",     # 1
            "educational/college information", # 2
            "promotional/spam content",        # 3
            "university/college portals",      # 4
            "academic management systems"      # 5
        ]
        # ...removed legacy mapping code, now only tuple-based non_job_refs...

        # Compute embeddings once
        # Non-job reference texts and their content types defined together
        self.non_job_refs = [
            ("Sign in to your account browse jobs search companies create profile save jobs get alerts download app", "website navigation/UI elements"),
            ("Breaking news today according to sources the company announced new policy trending now viral", "news articles or blog posts"),
            ("College admission open fee structure tuition fees hostel fees apply now for engineering degree courses university campus", "educational/college information"),
            ("Buy now limited offer click here download app subscribe to channel follow us on social media", "promotional/spam content"),
            ("University portal student faculty staff parent alumni access academics research services digital platform institute", "university/college portals"),
            ("Student login employee login parent portal transcript semester registration exam results grade report course enrollment", "academic management systems"),
            ("Formula One is the highest class of international racing for single-seater formula racing cars sanctioned by the FIA with seasons consisting of multiple Grands Prix events across the world", "news articles or blog posts"),
            ("The fee structure for the B.Tech Computer Science program includes tuition hostel accommodation and caution deposit with semester wise breakdown of amounts payable", "educational/college information"),
            # Educational/AI intro samples
            ("After completing this unit, you’ll be able to explain fundamental concepts of artificial intelligence, identify the challenges that make defining artificial intelligence difficult, describe the types of tasks artificial intelligence can perform.", "education/AI course introduction"),
            ("Artificial intelligence (AI) has been a dream of many storytellers and sci-fi fans for years. But most people hadn’t given AI much serious thought because it was always something that might happen far into the future.", "education/AI course introduction"),
            ("Researchers and computer scientists haven’t been waiting for tomorrow to arrive, they’ve been working hard to make the dream of AI into a reality. In fact, as you know, we’re already well into the Age of AI.", "education/AI course introduction"),
            # Entertainment/biography/news samples
            ("George Clooney's breakthrough came with his role as Dr. Doug Ross in the NBC medical drama ER for which he received two Primetime Emmy Award nominations. He established himself as a film star with roles in From Dusk till Dawn, Out of Sight, Three Kings, and O Brother, Where Art Thou?", "entertainment/biography news"),
            ("When Aishwarya Rai rejected a Hollywood movie with Brad Pitt for this reason. She was offered a role in Troy demanding a 6-9 months commitment which she declined due to commitments in India. Brad Pitt later expressed regret over her not joining the film.", "entertainment/celebrity news"),
            # Generic non-job text
            ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.", "generic non-job content"),
            ("Hello! Welcome to our platform. Please sign in to continue.", "website navigation/UI elements"),
            ("This is a sample paragraph for testing purposes.", "generic non-job content")
        ]
        self.non_job_texts = [t for t, _ in self.non_job_refs]
        self.non_job_content_types = [ct for _, ct in self.non_job_refs]
        
        # Compute embeddings once (after method definitions)
        # Will be initialized at end of __init__
        self.job_ref_embeddings = None
        self.non_job_ref_embeddings = None
        # At end of __init__, after all methods are defined, and still inside __init__:
        self.job_ref_embeddings = [self.get_bert_embedding(self.clean_text(t)) for t in self.job_reference_texts]
        self.non_job_ref_embeddings = [self.get_bert_embedding(self.clean_text(t)) for t in self.non_job_texts]
    
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
        Validate if the text is actually a job description, implementing the
        summary's checks (min length/words, keywords, spam, CSV headers, edu content)
        and enhanced with semantic understanding via BERT.
        Returns: (is_valid, reason)
        """
        text_lower = text.lower()
        
        # Basic sanity checks (per summary)
        if len(text) < 20:
            return False, {"valid": False, "reason": "too_short", "message": "Text too short to be a job description (minimum 20 characters required)", "content_type": None}
        
        words = text.split()
        word_count = len(words)
        
        if word_count < 5:
            return False, {"valid": False, "reason": "too_few_words", "message": f"Content too brief ({word_count} words). Need at least 5 words.", "content_type": None}
        
        # Check if it's just CSV headers or column names (use specific CSV-format indicators)
        csv_indicators = ['title\tdescription', 'job_id\t', 'fraudulent\t', 'telecommuting\t', 
                         'has_company_logo', 'has_questions', '\tcolumn\t', '\theader\t']
        if any(indicator in text_lower for indicator in csv_indicators):
            return False, {"valid": False, "reason": "metadata", "message": "This appears to be data headers or metadata, not a job description.", "content_type": "metadata"}
        
        # Job-related keyword presence (per summary)
        job_keywords = [
            'job', 'position', 'role', 'hiring', 'vacancy', 'career', 'employment',
            'work', 'responsibilities', 'requirements', 'qualifications', 'experience',
            'salary', 'apply', 'candidate', 'skills', 'duties', 'recruit', 'team',
            'opportunity', 'developer', 'engineer', 'manager', 'analyst', 'specialist',
            'coordinator', 'assistant', 'executive', 'intern', 'full time', 'part time',
            'remote', 'office', 'company', 'department', 'benefits', 'compensation',
            'fresher', 'opening', 'join', 'wanted'
        ]
        # More precise keyword detection using word boundaries for single-word tokens to avoid matching substrings like 'engineer' within 'engineering'
        job_keyword_hits = []
        for kw in job_keywords:
            if ' ' in kw:  # multi-word phrase exact substring match
                if kw in text_lower:
                    job_keyword_hits.append(kw)
            else:
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    job_keyword_hits.append(kw)
        has_job_keyword = len(job_keyword_hits) > 0

        # Non-job topic detectors
        ui_indicators = [
            'home', 'header', 'footer', 'menu', 'logo', 'analyze job', 'analyze job posting',
            'examples:', 'supported:', 'powered by', 'confidence level', 'source', 'privacy', 'contact',
            'how it works', 'copyright', 'terms', 'detectify', 'ai-powered job verification'
        ]
        ui_count = sum(1 for ind in ui_indicators if ind in text_lower)

        education_info_indicators = [
            'university', 'college', 'course', 'program', 'b.tech', 'btech', 'engineering',
            'fee', 'fees', 'tuition', 'caution deposit', 'hostel', 'accommodation', 'semester',
            'admission', 'eligibility', 'syllabus', 'credits', 'intake'
        ]
        edu_info_count = sum(1 for ind in education_info_indicators if ind in text_lower)

        news_indicators = [
            'breaking news', 'world championship', 'season', 'races', 'grands prix', 'fédération internationale',
            'fia', 'history', 'since', 'wikipedia', 'motorsport', 'premier forms', 'series of'
        ]
        news_count = sum(1 for ind in news_indicators if ind in text_lower)

        # Entertainment/biography/celebrity news indicators
        entertainment_indicators = [
            'film', 'movie', 'tv series', 'television', 'drama', 'episode', 'season', 'cast', 'role as', 'biography',
            'award', 'awards', 'emmy', 'oscar', 'box office', 'hollywood', 'bollywood', 'actor', 'actress', 'director',
            'trailer', 'release', 'soundtrack', 'starred', 'starring', 'festival'
        ]
        entertainment_count = sum(1 for ind in entertainment_indicators if ind in text_lower)

        # Meta/instructional/system text indicators (not an actual job description)
        meta_instruction_indicators = [
            'this is not a command', 'this a text not command', 'should also check', 'you should also check',
            'tell them', 'we should', 'for comparison', 'our ai', 'confidence means', 'score', 'scored',
            'percent confidence', 'model is reasonably certain', 'missing information', 'less-structured sources',
            'example text', 'sample text', 'demo text', 'instruction', 'instructions', 'guideline', 'prompt',
            'consider the following', 'you need to', 'please check', 'explain', 'means our ai'
        ]
        meta_instruction_count = sum(1 for ind in meta_instruction_indicators if ind in text_lower)

        # Structural employment patterns (signals of actual job structure)
        employment_patterns = [
            'responsibilities include', 'requirements include', 'we are hiring', 'we are looking for',
            'apply with', 'apply now', 'skills required', 'experience required', 'candidate must',
            'job description', 'position involves', 'the role', 'the successful candidate', 'join our team'
        ]
        employment_pattern_hits = sum(1 for ptn in employment_patterns if ptn in text_lower)
        
        # Spam/non-job indicators (per summary) - reject if 3+ strong indicators present
        spam_indicators = [
            'registration fee', 'pay to apply', 'deposit', 'upfront payment', 'money transfer',
            'whatsapp', 'telegram', 'click here', 'limited offer', 'guaranteed income',
            'earn per day', 'no experience needed', 'work from home and earn', 'lottery', 'gift card'
        ]
        spam_hits = sum(1 for s in spam_indicators if s in text_lower)
        if spam_hits >= 3:
            return False, {"valid": False, "reason": "spam_content", "message": "Detected multiple spam indicators (>=3).", "content_type": "spam"}

        # UI/Navigation pages (override even if job keywords appear)
        if ui_count >= 3 and employment_pattern_hits == 0:
            return False, {"valid": False, "reason": "website_navigation", "message": "Website UI / navigation content detected.", "content_type": "website navigation/UI elements"}

        # Education info (not portal) - fees/admissions/course details
        if edu_info_count >= 3 and employment_pattern_hits == 0 and has_job_keyword is False:
            return False, {"valid": False, "reason": "education_info", "message": "Educational fee/course/admission information detected.", "content_type": "education",
                           "debug_counts": {"edu_info_count": edu_info_count, "job_keywords": job_keyword_hits, "news_count": news_count}}

        # News/article content
        if news_count >= 3 and employment_pattern_hits == 0 and has_job_keyword is False:
            return False, {"valid": False, "reason": "news_article", "message": "News / encyclopedic topic detected.", "content_type": "news articles or blog posts",
                           "debug_counts": {"edu_info_count": edu_info_count, "job_keywords": job_keyword_hits, "news_count": news_count}}

        # Entertainment/biography/celebrity content (override even if words like 'role' appear)
        if entertainment_count >= 3 and employment_pattern_hits == 0:
            return False, {"valid": False, "reason": "entertainment_news", "message": "Entertainment/biography news detected.", "content_type": "entertainment/celebrity news",
                           "debug_counts": {"entertainment_count": entertainment_count, "job_keywords": job_keyword_hits, "news_count": news_count}}

        # Meta/instructional/system text (guidance/notes, not a job)
        if meta_instruction_count >= 2 and employment_pattern_hits == 0:
            return False, {"valid": False, "reason": "meta_instruction", "message": "Detected instructional/meta text, not a job description.", "content_type": "meta/instruction",
                           "debug_counts": {"meta_instruction_count": meta_instruction_count, "job_keywords": job_keyword_hits}}

        # Check for repetitive/garbage text (same word repeated many times)
        word_freq = {}
        for word in words[:100]:  # Check first 100 words
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # If any word appears more than 30% of the time, it's likely garbage
        max_freq = max(word_freq.values()) if word_freq else 0
        if max_freq > len(words[:100]) * 0.3:
            return False, {"valid": False, "reason": "repetitive_corrupted", "message": "Text appears to be repetitive or corrupted.", "content_type": "noise"}
        
        # === SEMANTIC UNDERSTANDING WITH BERT ===
        # Instead of keyword matching, use BERT to understand the context
        
        # Generate embedding for the input text
        cleaned_text = self.clean_text(text)
        if len(cleaned_text) < 10:
            return False, "Text doesn't contain meaningful content after cleaning."
        
        # Check for explicit college/university/portal indicators first (per summary)
        edu_portal_indicators = [
            'vtop', 'student portal', 'university portal', 'college portal',
            'student login', 'faculty login', 'parent login', 'alumni login',
            'digital initiative by the institute', 'academic portal',
            'vit-ap', 'vit on top', 'student management system',
            'transcript', 'semester registration', 'fee payment portal',
            'course registration', 'exam results', 'grade report'
        ]
        edu_count = sum(1 for indicator in edu_portal_indicators if indicator in text_lower)
        # Align with summary: reject when there are 3 or more such indicators
        if edu_count >= 3:
            return False, {"valid": False, "reason": "educational_portal", "message": "Educational portal / academic system content detected.", "content_type": "education",
                           "debug_counts": {"edu_portal_count": edu_count, "job_keywords": job_keyword_hits}}
        
        input_embedding = self.get_bert_embedding(cleaned_text)
        
        # Calculate cosine similarity with job references (use cached embeddings)
        job_similarities = []
        for ref_embedding in self.job_ref_embeddings:
            similarity = np.dot(input_embedding, ref_embedding) / (
                np.linalg.norm(input_embedding) * np.linalg.norm(ref_embedding)
            )
            job_similarities.append(similarity)
        
        # Calculate cosine similarity with non-job references (use cached embeddings)
        non_job_similarities = []
        for ref_embedding in self.non_job_ref_embeddings:
            similarity = np.dot(input_embedding, ref_embedding) / (
                np.linalg.norm(input_embedding) * np.linalg.norm(ref_embedding)
            )
            non_job_similarities.append(similarity)
        
        # Get max similarities
        max_job_sim = max(job_similarities)
        max_non_job_sim = max(non_job_similarities)
        
        # Decision based on semantic similarity
        # If text is more similar to non-job content by a significant margin, reject it
        # Allow a small margin (0.05) for close calls since scraped pages often contain navigation
        if max_non_job_sim > max_job_sim + 0.05:
            content_type = self._get_content_type(self.non_job_ref_embeddings, self.non_job_texts, input_embedding)
            return False, {"valid": False, "reason": "semantic_non_job", "message": f"Content semantically closer to {content_type}.", "content_type": content_type,
                           "debug_similarity": {"max_job_sim": max_job_sim, "max_non_job_sim": max_non_job_sim,
                                                 "job_keywords": job_keyword_hits, "edu_info_count": edu_info_count, "news_count": news_count}}
        
        # If similarity to job content is too low overall, reject (allow some leniency for scraped pages)
        # Require either decent job similarity or presence of employment structure
        if max_job_sim < 0.45 and employment_pattern_hits == 0:
            return False, {"valid": False, "reason": "low_job_similarity", "message": f"Job similarity too low ({max_job_sim:.2f} < 0.45) and no employment patterns detected.", "content_type": "non_job",
                           "debug_similarity": {"max_job_sim": max_job_sim, "max_non_job_sim": max_non_job_sim,
                                                 "job_keywords": job_keyword_hits, "edu_info_count": edu_info_count, "news_count": news_count}}

        # Enforce keyword presence unless semantic similarity is clearly strong
        if not has_job_keyword and employment_pattern_hits == 0 and max_job_sim < 0.55:
            return False, {"valid": False, "reason": "missing_keywords", "message": "Missing job-related keywords.", "content_type": "non_job",
                           "debug_similarity": {"max_job_sim": max_job_sim, "max_non_job_sim": max_non_job_sim,
                                                 "job_keywords": job_keyword_hits, "edu_info_count": edu_info_count, "news_count": news_count}}
        
        # All checks passed - looks like a job description
        return True, {"valid": True, "reason": None, "message": None, "content_type": "job",
                      "debug_counts": {"edu_info_count": edu_info_count, "news_count": news_count, "job_keywords": job_keyword_hits,
                                        "employment_pattern_hits": employment_pattern_hits}}
    
    def _get_content_type(self, reference_embeddings, reference_texts, input_embedding):
        """Helper to identify which type of non-job content it matches (uses cached embeddings)"""
        # Find which reference embedding had the highest similarity
        similarities = []
        for ref_embedding in reference_embeddings:
            sim = np.dot(input_embedding, ref_embedding) / (
                np.linalg.norm(input_embedding) * np.linalg.norm(ref_embedding)
            )
            similarities.append(sim)
        
        max_idx = similarities.index(max(similarities))
        # Use expanded mapping list
        if hasattr(self, 'non_job_content_types') and max_idx < len(self.non_job_content_types):
            return self.non_job_content_types[max_idx]
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
        
        # 2. Extract and parse deadline/apply-by dates
        # Patterns that precede dates
        deadline_triggers = [
            'last date to apply', 'last date for application', 'last date', 'apply by', 'apply before',
            'application deadline', 'deadline to apply', 'closing date', 'applications close on',
            'apply until', 'apply till', 'apply before'
        ]

        # Regex to capture various date formats following triggers
        date_regex = r"([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})"

        candidates = []
        for trig in deadline_triggers:
            # find occurrences and slice a window after trigger
            idx = 0
            while True:
                pos = text_lower.find(trig, idx)
                if pos == -1:
                    break
                window = text[pos:pos+80]  # small window after trigger
                m = re.search(date_regex, window, flags=re.IGNORECASE)
                if m:
                    candidates.append(m.group(1))
                idx = pos + 1

        # Also attempt standalone date with words like 'deadline:'
        for label in ['deadline', 'closing date', 'apply before', 'apply by']:
            pattern = label + r"[:\s-]*" + date_regex
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                candidates.append(m.group(1))

        parsed_dates = []
        now = datetime.now()
        for ds in candidates:
            try:
                dt = dateparser.parse(ds, settings={'DATE_ORDER': 'DMY', 'PREFER_DAY_OF_MONTH': 'first'})
                if dt:
                    parsed_dates.append(dt)
            except Exception:
                continue

        if parsed_dates:
            # choose the first or earliest reasonable future/past deadline
            # Prefer the nearest future date; if none, take the latest past date
            future_dates = [d for d in parsed_dates if d.date() >= now.date()]
            if future_dates:
                deadline = min(future_dates)
                return False, f"Application deadline {deadline.date().isoformat()}", deadline.date().isoformat()
            else:
                deadline = max(parsed_dates)
                return True, f"Deadline passed on {deadline.date().isoformat()}", deadline.date().isoformat()

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
        is_valid, meta = self.is_job_description(combined_text)
        if not is_valid:
            # Include debug meta when available for easier troubleshooting
            debug_counts = meta.get('debug_counts') if isinstance(meta, dict) else None
            debug_similarity = meta.get('debug_similarity') if isinstance(meta, dict) else None
            return {
                'prediction': 'Not a Job Description',
                'confidence': 0.0,
                'probability_fake': 0.0,
                'probability_legit': 0.0,
                'error': meta.get('message') if isinstance(meta, dict) else str(meta),
                'reason': meta.get('reason') if isinstance(meta, dict) else None,
                'content_type': meta.get('content_type') if isinstance(meta, dict) else None,
                'debug_counts': debug_counts,
                'debug_similarity': debug_similarity,
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
            'is_job': True,
            'content_type': 'job'
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
