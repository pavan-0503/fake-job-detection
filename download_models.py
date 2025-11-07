"""
Model Downloader for Railway Deployment
Downloads models from Google Drive if not present locally
"""

import os
import requests
import zipfile
from tqdm import tqdm

def download_file_from_google_drive(file_id, destination):
    """Download large file from Google Drive with progress bar"""
    URL = "https://drive.google.com/uc?export=download"
    
    session = requests.Session()
    
    # Initial request
    response = session.get(URL, params={'id': file_id, 'confirm': 1}, stream=True)
    
    # Handle virus scan warning for large files
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': file_id, 'confirm': value}
            response = session.get(URL, params=params, stream=True)
    
    # Save file with progress bar
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1 MB
    
    print(f"Downloading models ({total_size / (1024*1024):.2f} MB)...")
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print("Download complete!")

def extract_models(zip_path, extract_to='.'):
    """Extract models zip file"""
    print("Extracting models...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")
    
    # Clean up zip file
    os.remove(zip_path)
    print("Cleaned up zip file")

def ensure_models_exist(google_drive_file_id=None):
    """
    Ensure models exist locally, download from Google Drive if not
    
    Args:
        google_drive_file_id: The FILE_ID from Google Drive share link
                             Extract from: https://drive.google.com/file/d/FILE_ID/view
    """
    models_dir = 'models'
    required_files = [
        'models/rf_model_calibrated.joblib',
        'models/scaler.joblib',
        'models/feature_info.joblib',
        'models/tokenizer'
    ]
    
    # Check if models already exist
    models_exist = all(os.path.exists(f) for f in required_files)
    
    if models_exist:
        print("✓ Models already exist locally")
        return True
    
    # Models don't exist, need to download
    print("⚠ Models not found locally")
    
    if not google_drive_file_id:
        # Try to get from environment variable
        google_drive_file_id = os.getenv('GOOGLE_DRIVE_MODEL_ID')
    
    if not google_drive_file_id:
        print("❌ ERROR: Google Drive file ID not provided!")
        print("   Set GOOGLE_DRIVE_MODEL_ID environment variable")
        print("   or pass file_id to ensure_models_exist()")
        return False
    
    try:
        # Download models.zip
        zip_path = 'models.zip'
        download_file_from_google_drive(google_drive_file_id, zip_path)
        
        # Extract models
        extract_models(zip_path, extract_to='.')
        
        print("✓ Models downloaded and extracted successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        return False

if __name__ == "__main__":
    # For testing locally
    import sys
    
    if len(sys.argv) > 1:
        file_id = sys.argv[1]
        print(f"Testing download with file ID: {file_id}")
        ensure_models_exist(file_id)
    else:
        print("Usage: python download_models.py <google_drive_file_id>")
        print("\nExtract FILE_ID from Google Drive share link:")
        print("https://drive.google.com/file/d/FILE_ID_HERE/view")
