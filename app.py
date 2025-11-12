"""
Fake Job Detection - Flask Web Application
Provides REST API and web interface for fake job detection
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
from flask_cors import CORS
from predictor import FakeJobPredictor
from scraper import JobScraper
import os
import re
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize predictor and scraper
predictor = None
scraper = JobScraper()

def init_predictor():
    """Initialize predictor (lazy loading)"""
    global predictor
    if predictor is None:
        # Allow either calibrated or base model to satisfy availability
        has_calibrated = os.path.exists('models/rf_model_calibrated.joblib')
        has_base = os.path.exists('models/rf_model.joblib')
        if not (has_calibrated or has_base):
            return False
        predictor = FakeJobPredictor()
    return True

def is_url(text):
    """Check if text is a URL"""
    try:
        result = urlparse(text.strip())
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except:
        return False

def clean_job_url(url):
    """
    Clean job URL by removing tracking parameters
    Returns: cleaned URL string
    """
    try:
        parsed = urlparse(url)
        
        # Tracking parameters to remove (common across job sites)
        tracking_params = [
            'refid', 'trackingid', 'trk', 'position', 'pagenumber',
            'ebp', 'alternatechannel', 'originaltrkinfo', 'lipi',
            'refid', 'licu', 'utm_source', 'utm_medium', 'utm_campaign'
        ]
        
        # Parse query parameters
        query_params = parse_qs(parsed.query)
        
        # Remove tracking parameters (case-insensitive)
        cleaned_params = {
            k: v for k, v in query_params.items() 
            if k.lower() not in tracking_params
        }
        
        # Reconstruct URL without tracking parameters
        cleaned_query = urlencode(cleaned_params, doseq=True)
        cleaned_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            cleaned_query,
            parsed.fragment
        ))
        
        return cleaned_url
    except:
        return url

def is_valid_job_url(url):
    """
    Validate if URL is a complete job posting URL using semantic understanding
    Returns: (is_valid, error_message, cleaned_url)
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        query = parsed.query.lower()
        
        # Check domain is present
        if not domain:
            return False, "Invalid URL: Missing domain name", url
        
        # === SEMANTIC URL PATTERN DETECTION ===
        # Check for common non-job URL patterns
        
        # Search/Browse pages (multiple jobs listed)
        search_patterns = [
            'search', 'browse', 'find-jobs', 'job-search', 'jobs?',
            'explore', 'discover', 'recommended', 'similar-jobs',
            'all-jobs', 'view-all', 'job-listings', 'results'
        ]
        
        # Count how many search indicators are present
        search_indicators = sum(1 for pattern in search_patterns if pattern in path or pattern in query)
        
        # If URL looks like a search/browse page (2+ indicators)
        if search_indicators >= 2:
            return False, "Invalid URL: This appears to be a job search/browse page with multiple listings. Please click on a specific job to open its individual page, then copy that URL.", url
        
        # Company career pages (not specific job)
        if any(keyword in path for keyword in ['careers', 'about-us', 'company-profile', 'employers']):
            if not any(keyword in path for keyword in ['job', 'position', 'opening', 'vacancy']):
                return False, "Invalid URL: This appears to be a company career page. Please click on a specific job opening.", url
        
        # Check for incomplete/placeholder URLs (collections page with currentJobId)
        if 'collections' in path and 'currentjobid' in query:
            # LinkedIn collections page - not a direct job URL
            return False, "Invalid LinkedIn URL: This is a job search/collections page. Please click on a specific job and copy that URL instead.", url
        
        # Site-specific validation with semantic checks
        if 'linkedin.com' in domain:
            # Valid LinkedIn job URLs: /jobs/view/JOBID
            if '/jobs/view/' not in path and 'currentjobid' in query:
                return False, "Invalid LinkedIn URL: Please click on a specific job posting to open it, then copy that page's URL.", url
            
            # Verify job ID exists in path (basic format check)
            if '/jobs/view/' in path:
                # Extract job ID from path like /jobs/view/1234567890/
                path_parts = path.split('/jobs/view/')
                if len(path_parts) > 1:
                    job_id_part = path_parts[1].strip('/')
                    # LinkedIn job IDs are typically 10 digits
                    if not job_id_part or not job_id_part.split('/')[0].isdigit():
                        return False, "Invalid LinkedIn URL: Job ID format is incorrect.", url
                else:
                    return False, "Invalid LinkedIn URL: Missing job ID in URL.", url
        
        elif 'naukri.com' in domain:
            # Valid Naukri URLs have job-listings in path
            if 'job-listings' not in path and 'job-detail' not in path:
                return False, "Invalid Naukri URL: This doesn't appear to be a job posting URL. Please open a specific job.", url
        
        elif 'indeed.com' in domain:
            # Valid Indeed URLs have viewjob or /jobs/view in path
            if 'viewjob' not in path and '/jobs/' not in path and '/rc/clk' not in path:
                return False, "Invalid Indeed URL: Please open a specific job posting and copy that URL.", url
        
        # Check URL is not too short (likely incomplete)
        if len(url) < 30:
            return False, "Invalid URL: URL appears incomplete or too short.", url
        
        # Check URL has some path or query parameters
        if (not path or path == '/') and not query:
            return False, "Invalid URL: This appears to be just a homepage. Please navigate to a specific job posting.", url
        
        # Clean the URL (remove tracking parameters)
        cleaned_url = clean_job_url(url)
        
        return True, None, cleaned_url
        
    except Exception as e:
        return False, f"Invalid URL format: {str(e)}", url

@app.route('/')
def home():
    """Serve the hero/landing page"""
    return render_template('home.html')

@app.route('/analyze')
def analyze():
    """Serve the main analysis page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    # Report healthy if either calibrated or base model exists
    model_loaded = (
        os.path.exists('models/rf_model_calibrated.joblib') or 
        os.path.exists('models/rf_model.joblib')
    )
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Unified prediction endpoint
    Accepts either raw text or URL
    
    Request body:
    {
        "input": "job description text or URL"
    }
    
    Response:
    {
        "prediction": "Fake Job" or "Legit Job",
        "confidence": 0.95,
        "probability_fake": 0.95,
        "probability_legit": 0.05,
        "source": "text" or "scraped",
        "error": null
    }
    """
    try:
        # Check if model is loaded
        if not init_predictor():
            return jsonify({
                'error': 'Model not found. Please train the model first by running train_model.py',
                'prediction': None,
                'confidence': 0
            }), 503
        
        # Get input from request
        data = request.get_json()
        
        if not data or 'input' not in data:
            return jsonify({
                'error': 'Missing "input" field in request body',
                'prediction': None,
                'confidence': 0
            }), 400
        
        user_input = data['input'].strip()
        
        if not user_input:
            return jsonify({
                'error': 'Input cannot be empty',
                'prediction': None,
                'confidence': 0
            }), 400
        
        # Determine if input is URL or text
        if is_url(user_input):
            # Validate if it's a proper job URL and get cleaned URL
            is_valid, error_msg, cleaned_url = is_valid_job_url(user_input)
            if not is_valid:
                return jsonify({
                    'error': error_msg,
                    'prediction': None,
                    'confidence': 0,
                    'suggestion': 'Please provide a direct link to a specific job posting, not a search page or homepage.'
                }), 400
            
            # Use cleaned URL for scraping (removes tracking parameters)
            # Console logging (avoid Unicode for Windows cp1252 safety)
            print(f"Scraping URL: {cleaned_url}")
            if cleaned_url != user_input:
                print(f"   (Cleaned from: {user_input})")
            
            scraped_data = scraper.scrape_url(cleaned_url)
            
            if scraped_data.get('error'):
                error_msg = scraped_data['error']
                # Provide helpful message for SSL/certificate errors
                if 'SSL' in error_msg or 'CERTIFICATE' in error_msg or 'certificate' in error_msg.lower():
                    return jsonify({
                        'error': 'Unable to access this URL due to security restrictions. This can happen with some job sites that have strict security policies. Please copy the job description text from the page and paste it directly instead.',
                        'prediction': None,
                        'confidence': 0,
                        'suggestion': 'Copy and paste the job description text directly for analysis.',
                        'technical_details': error_msg
                    }), 400
                else:
                    return jsonify({
                        'error': f"Failed to scrape URL: {error_msg}",
                        'prediction': None,
                        'confidence': 0,
                        'suggestion': 'Copy and paste the job description text directly for analysis.'
                    }), 400
            
            # Use scraped text for prediction
            job_text = scraped_data['full_text']
            source = f"scraped ({scraped_data['source']})"
            source_type = scraped_data.get('source', '')
            
            # Debug: Show scraped text length and preview
            print(f"Scraped text length: {len(job_text)} characters")
            print(f"Preview: {job_text[:200]}...")
            
            # Check if scraped content is too short - likely scraping failed
            if len(job_text.strip()) < 50:
                return jsonify({
                    'error': 'Could not extract job content from this URL. The website may require login or block automated access. Please copy the job description text and paste it directly instead.',
                    'prediction': None,
                    'confidence': 0,
                    'suggestion': 'Copy the job description text from the page and paste it here for analysis.'
                }), 400
            
            # Semantic validation will be done in predictor.predict()
            # which uses BERT embeddings to understand if it's a real job description
            
            # Add scraped details to response
            scraped_info = {
                'title': scraped_data.get('title', ''),
                'company': scraped_data.get('company', ''),
                'location': scraped_data.get('location', ''),
                'source_site': scraped_data.get('source', '')
            }
        else:
            # Use text directly
            job_text = user_input
            source = "text"
            source_type = None
            scraped_info = None
        
        # Make prediction (pass source for reputation boost)
        print("Making prediction...")
        result = predictor.predict(job_text, source=source_type)
        
        # Add source info
        result['source'] = source
        if scraped_info:
            result['scraped_info'] = scraped_info
        
        # Success response
        return jsonify(result), 200
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'prediction': None,
            'confidence': 0
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Request body:
    {
        "inputs": ["text1", "url1", "text2", ...]
    }
    
    Response:
    {
        "results": [
            {"prediction": "Fake Job", "confidence": 0.95, ...},
            ...
        ]
    }
    """
    try:
        if not init_predictor():
            return jsonify({
                'error': 'Model not found. Please train the model first.',
                'results': []
            }), 503
        
        data = request.get_json()
        
        if not data or 'inputs' not in data:
            return jsonify({
                'error': 'Missing "inputs" field in request body',
                'results': []
            }), 400
        
        inputs = data['inputs']
        
        if not isinstance(inputs, list):
            return jsonify({
                'error': '"inputs" must be an array',
                'results': []
            }), 400
        
        results = []
        
        for i, user_input in enumerate(inputs):
            try:
                user_input = user_input.strip()
                
                # Check if URL or text
                if is_url(user_input):
                    scraped_data = scraper.scrape_url(user_input)
                    if scraped_data.get('error'):
                        results.append({
                            'error': scraped_data['error'],
                            'prediction': None,
                            'confidence': 0
                        })
                        continue
                    job_text = scraped_data['full_text']
                else:
                    job_text = user_input
                
                # Predict
                result = predictor.predict(job_text)
                results.append(result)
            
            except Exception as e:
                results.append({
                    'error': str(e),
                    'prediction': None,
                    'confidence': 0
                })
        
        return jsonify({'results': results}), 200
    
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'results': []
        }), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Get model statistics"""
    try:
        if not os.path.exists('models/feature_info.joblib'):
            return jsonify({
                'error': 'Model not found',
                'stats': {}
            }), 404
        
        import joblib
        feature_info = joblib.load('models/feature_info.joblib')
        
        return jsonify({
            'stats': {
                'total_features': feature_info['total_features'],
                'bert_dimensions': feature_info['bert_dims'],
                'extra_features': feature_info['extra_features'],
                'model_type': 'RandomForest + DistilBERT'
            }
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'stats': {}
        }), 500

@app.route('/explain', methods=['POST'])
def explain():
    """Explain validation and (if applicable) classification with similarity metrics.
    Request JSON: {"input": "text or url"}
    Response JSON includes validation meta, semantic similarities, and classification if valid.
    """
    try:
        if not init_predictor():
            return jsonify({'error': 'Model not found. Train first.', 'explanation': None}), 503

        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({'error': 'Missing "input" field', 'explanation': None}), 400

        user_input = data['input'].strip()
        if not user_input:
            return jsonify({'error': 'Input cannot be empty', 'explanation': None}), 400

        is_input_url = is_url(user_input)
        job_text = user_input
        source = 'text'
        source_type = None
        scraped_info = None

        if is_input_url:
            is_valid, error_msg, cleaned_url = is_valid_job_url(user_input)
            if not is_valid:
                return jsonify({'error': error_msg, 'explanation': None, 'input_type': 'url'}), 400
            print(f"Explain: scraping URL {cleaned_url}")
            scraped_data = scraper.scrape_url(cleaned_url)
            if scraped_data.get('error'):
                return jsonify({'error': f"Scrape failed: {scraped_data['error']}", 'explanation': None}), 400
            job_text = scraped_data['full_text']
            source = f"scraped ({scraped_data.get('source','')})"
            source_type = scraped_data.get('source')
            scraped_info = {
                'title': scraped_data.get('title',''),
                'company': scraped_data.get('company',''),
                'location': scraped_data.get('location',''),
                'source_site': scraped_data.get('source','')
            }

        # Run validation
        is_valid, meta = predictor.is_job_description(job_text)

        # Always compute semantic similarities for transparency
        cleaned = predictor.clean_text(job_text)
        embedding = predictor.get_bert_embedding(cleaned)

        job_sims = []
        for ref in predictor.job_ref_embeddings:
            sim = float(np.dot(embedding, ref) / (np.linalg.norm(embedding) * np.linalg.norm(ref))) if np.linalg.norm(embedding) and np.linalg.norm(ref) else 0.0
            job_sims.append(sim)
        non_job_sims = []
        for ref in predictor.non_job_ref_embeddings:
            sim = float(np.dot(embedding, ref) / (np.linalg.norm(embedding) * np.linalg.norm(ref))) if np.linalg.norm(embedding) and np.linalg.norm(ref) else 0.0
            non_job_sims.append(sim)

        max_job = max(job_sims) if job_sims else 0.0
        max_non_job = max(non_job_sims) if non_job_sims else 0.0
        top_job_idx = int(job_sims.index(max_job)) if job_sims else -1
        top_non_job_idx = int(non_job_sims.index(max_non_job)) if non_job_sims else -1

        similarity_block = {
            'max_job_similarity': max_job,
            'max_non_job_similarity': max_non_job,
            'job_similarities': job_sims,
            'non_job_similarities': non_job_sims,
            'top_job_reference': predictor.job_reference_texts[top_job_idx] if top_job_idx >= 0 else None,
            'top_non_job_reference': predictor.non_job_texts[top_non_job_idx] if top_non_job_idx >= 0 else None
        }

        explanation = {
            'input_type': 'url' if is_input_url else 'text',
            'source': source,
            'validation': meta,
            'similarities': similarity_block,
            'scraped_info': scraped_info
        }

        # If not a job description, return only validation & similarities
        if not is_valid:
            return jsonify({'explanation': explanation, 'classification': None}), 200

        # If valid, run classification
        classification = predictor.predict(job_text, source=source_type)
        return jsonify({'explanation': explanation, 'classification': classification}), 200

    except Exception as e:
        print(f"Explain endpoint error: {e}")
        return jsonify({'error': f'Internal server error: {e}', 'explanation': None}), 500


if __name__ == '__main__':
    # Run Flask app (match summary: host 0.0.0.0, port 5000, debug True)
    app.run(host='0.0.0.0', port=5000, debug=True)
