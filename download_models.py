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
    
    session = requests.Session()
    
    print(f"🔄 Attempting to download from Google Drive (ID: {file_id[:10]}...)")
    
    # For large files, Google Drive requires confirmation
    # Try multiple download methods
    
    # Method 1: Direct download with confirmation
    URL = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    
    try:
        response = session.get(URL, stream=True, timeout=180)
        
        # Check content type BEFORE checking status
        content_type = response.headers.get('content-type', '')
        
        # For large files, Google might return HTML with a download warning
        # Check if we got HTML or actual file
        if 'text/html' in content_type and response.status_code == 200:
            print("   → Handling Google Drive virus scan warning for large file...")
            
            # Try alternative download URL
            URL2 = f"https://drive.google.com/u/0/uc?id={file_id}&export=download&confirm=t"
            response = session.get(URL2, stream=True, timeout=180)
            content_type = response.headers.get('content-type', '')
            
            if 'text/html' in content_type:
                raise Exception("File not publicly accessible. Please set sharing to 'Anyone with the link'")
        
        if response.status_code != 200:
            raise Exception(f"Failed to download. Status code: {response.status_code}")
        
        # Save file with progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MB
        
        if total_size > 0:
            print(f"📦 Downloading models ({total_size / (1024*1024):.2f} MB)...")
        else:
            print(f"📦 Downloading models (size unknown, this may take a while)...")
        
        with open(destination, 'wb') as f:
            downloaded = 0
            with tqdm(total=total_size, unit='B', unit_scale=True, disable=total_size==0) as pbar:
                for chunk in response.iter_content(block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))
    
    print(f"✅ Download complete! ({downloaded / (1024*1024):.2f} MB)")

def extract_models(zip_path, extract_to='.'):
    """Extract models zip file, handling different folder structures"""
    print("Extracting models...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # List all files in the zip
        file_list = zip_ref.namelist()
        print(f"📋 Found {len(file_list)} files in zip")
        
        # Ensure models directory exists
        models_dir = os.path.join(extract_to, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Check the structure of files in the zip
        has_models_prefix = any(f.startswith('models/') for f in file_list)
        has_nested_models = any('models/models/' in f for f in file_list)
        
        if has_nested_models:
            # Case 1: models/models/ structure (zipped models folder itself)
            print("🔄 Detected nested models/models/ structure, fixing...")
            for file in file_list:
                if file.startswith('models/models/'):
                    new_path = file.replace('models/models/', 'models/', 1)
                    target_path = os.path.join(extract_to, new_path)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    if not file.endswith('/'):
                        with zip_ref.open(file) as source, open(target_path, 'wb') as target:
                            target.write(source.read())
        
        elif has_models_prefix:
            # Case 2: models/ structure (correct structure)
            print("✅ Correct models/ structure detected")
            zip_ref.extractall(extract_to)
        
        else:
            # Case 3: Files directly in zip (no models/ prefix)
            # Need to extract into models/ directory
            print("🔄 Files are in root of zip, extracting to models/ directory...")
            for file in file_list:
                if not file.endswith('/') and not file.startswith('__MACOSX'):
                    # Extract each file into the models directory
                    target_path = os.path.join(models_dir, file)
                    
                    # Handle subdirectories (like tokenizer/)
                    if '/' in file:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    with zip_ref.open(file) as source, open(target_path, 'wb') as target:
                        target.write(source.read())
                    print(f"   ✓ Extracted: {file} → models/{file}")
    
    print("✅ Extraction complete!")
    
    # Clean up zip file (check if exists first to avoid race condition with multiple workers)
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("🗑️  Cleaned up zip file")

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
        print("✅ Models already exist locally")
        return True
    
    # Check if another worker is downloading (lock file mechanism)
    lock_file = 'models_download.lock'
    if os.path.exists(lock_file):
        print("⏳ Another worker is downloading models, waiting...")
        # Wait up to 60 seconds for the other worker to finish
        import time
        for i in range(60):
            time.sleep(1)
            if all(os.path.exists(f) for f in required_files):
                print("✅ Models downloaded by another worker!")
                return True
        print("⚠️  Timeout waiting for other worker, attempting download anyway...")
    
    # Create lock file to prevent other workers from downloading simultaneously
    try:
        with open(lock_file, 'w') as f:
            f.write('downloading')
    except:
        pass  # If we can't create lock, continue anyway
    
    # Models don't exist, need to download
    print("⚠️  Models not found locally. Attempting download...")
    
    if not google_drive_file_id:
        # Try to get from environment variable
        google_drive_file_id = os.getenv('GOOGLE_DRIVE_MODEL_ID')
    
    if not google_drive_file_id:
        print("❌ ERROR: Google Drive file ID not provided!")
        print("   📋 Set GOOGLE_DRIVE_MODEL_ID environment variable in Railway")
        print("   📋 Format: Go to Railway → Variables → Add:")
        print("      Variable: GOOGLE_DRIVE_MODEL_ID")
        print("      Value: 16cFNpCAVWqM_qZDuZYbFi_iEmWjkyjin")
        # Clean up lock file
        if os.path.exists(lock_file):
            os.remove(lock_file)
        return False
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Download models.zip
        zip_path = 'models.zip'
        print(f"📥 Downloading from Google Drive (File ID: {google_drive_file_id})...")
        download_file_from_google_drive(google_drive_file_id, zip_path)
        
        # Verify zip file was downloaded
        if not os.path.exists(zip_path):
            raise Exception("Zip file was not created after download")
        
        file_size = os.path.getsize(zip_path)
        if file_size < 1000:  # Less than 1KB suggests error page
            with open(zip_path, 'r') as f:
                content = f.read()[:500]
            raise Exception(f"Downloaded file too small ({file_size} bytes). May be error page: {content}")
        
        print(f"✅ Downloaded {file_size / (1024*1024):.2f} MB")
        
        # Check zip file contents before extraction
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_contents = zip_ref.namelist()
                print(f"📦 Zip contains {len(zip_contents)} files/folders")
                # Show first few entries
                for item in zip_contents[:5]:
                    print(f"   - {item}")
                if len(zip_contents) > 5:
                    print(f"   ... and {len(zip_contents) - 5} more")
        except zipfile.BadZipFile:
            raise Exception("Downloaded file is not a valid ZIP file")
        
        # Extract models
        extract_models(zip_path, extract_to='.')
        
        # Verify models exist after extraction
        models_exist_after = all(os.path.exists(f) for f in required_files)
        if not models_exist_after:
            missing = [f for f in required_files if not os.path.exists(f)]
            # List what was actually created
            print("📂 Checking extracted files:")
            if os.path.exists(models_dir):
                for root, dirs, files in os.walk(models_dir):
                    level = root.replace(models_dir, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files[:10]:  # Limit to first 10 files
                        print(f"{subindent}{file}")
            raise Exception(f"Models extracted but missing required files: {missing}")
        
        print("✅ Models downloaded and extracted successfully!")
        
        # Clean up lock file
        if os.path.exists(lock_file):
            os.remove(lock_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        print(f"   🔍 Check that:")
        print(f"      1. Google Drive file ID is correct: {google_drive_file_id}")
        print(f"      2. File sharing is set to 'Anyone with the link'")
        print(f"      3. Railway has internet access (should be fine)")
        
        # Clean up lock file on error
        if os.path.exists(lock_file):
            os.remove(lock_file)
        
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
