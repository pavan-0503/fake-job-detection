"""
Pre-Deployment Verification Script
Checks if all required files exist before deploying to Railway
"""

import os
import sys

print("=" * 70)
print("🚀 RAILWAY DEPLOYMENT VERIFICATION")
print("=" * 70)

# Required files for deployment
required_files = {
    'Procfile': 'Railway deployment configuration',
    'requirements.txt': 'Python dependencies',
    'runtime.txt': 'Python version specification',
    'README.md': 'Project documentation',
    '.gitignore': 'Git ignore rules',
    'app.py': 'Flask application',
    'predictor.py': 'ML predictor',
    'scraper.py': 'Web scraper',
    'train_model.py': 'Model training script',
}

# Required directories
required_dirs = {
    'models': 'Trained models directory',
    'templates': 'HTML templates',
    'static': 'Static assets',
}

# Required model files
required_models = {
    'models/rf_model_calibrated.joblib': 'Calibrated RandomForest model',
    'models/scaler.joblib': 'Feature scaler',
    'models/feature_info.joblib': 'Feature metadata',
    'models/tokenizer': 'BERT tokenizer directory',
}

# Files that should NOT exist (cleaned up)
unwanted_files = [
    'test_predictor.py',
    'test_selenium.py',
    'test_vtop_validation.py',
    'test_expiration.py',
    'run_app.bat',
    'run_train.bat',
    'setup.bat',
]

all_checks_passed = True

# Check required files
print("\n✅ Checking Required Files:")
for file, description in required_files.items():
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"  ✓ {file:25s} ({size:,} bytes) - {description}")
    else:
        print(f"  ✗ {file:25s} - MISSING! {description}")
        all_checks_passed = False

# Check required directories
print("\n✅ Checking Required Directories:")
for dir_name, description in required_dirs.items():
    if os.path.isdir(dir_name):
        file_count = len([f for f in os.listdir(dir_name)])
        print(f"  ✓ {dir_name:25s} ({file_count} items) - {description}")
    else:
        print(f"  ✗ {dir_name:25s} - MISSING! {description}")
        all_checks_passed = False

# Check model files
print("\n✅ Checking Model Files:")
total_model_size = 0
for model_file, description in required_models.items():
    if os.path.exists(model_file):
        if os.path.isdir(model_file):
            print(f"  ✓ {model_file:40s} - {description}")
        else:
            size = os.path.getsize(model_file)
            total_model_size += size
            size_mb = size / (1024 * 1024)
            print(f"  ✓ {model_file:40s} ({size_mb:.2f} MB) - {description}")
    else:
        print(f"  ✗ {model_file:40s} - MISSING! {description}")
        all_checks_passed = False

print(f"\n  Total model size: {total_model_size / (1024 * 1024):.2f} MB")

# Check for unwanted files (should be deleted)
print("\n✅ Checking Cleanup (files that should be removed):")
found_unwanted = False
for file in unwanted_files:
    if os.path.exists(file):
        print(f"  ⚠ {file:25s} - Should be removed before deployment")
        found_unwanted = True

if not found_unwanted:
    print("  ✓ All test/script files have been cleaned up")

# Check venv is in gitignore
print("\n✅ Checking .gitignore:")
if os.path.exists('.gitignore'):
    with open('.gitignore', 'r') as f:
        gitignore_content = f.read()
        if 'venv/' in gitignore_content:
            print("  ✓ venv/ is in .gitignore")
        else:
            print("  ⚠ venv/ should be in .gitignore")
            all_checks_passed = False
else:
    print("  ✗ .gitignore is missing")
    all_checks_passed = False

# Check Procfile content
print("\n✅ Checking Procfile Configuration:")
if os.path.exists('Procfile'):
    with open('Procfile', 'r') as f:
        procfile_content = f.read().strip()
        if 'gunicorn' in procfile_content and 'app:app' in procfile_content:
            print(f"  ✓ Procfile: {procfile_content}")
        else:
            print(f"  ⚠ Procfile may be incorrect: {procfile_content}")
            all_checks_passed = False

# Check requirements.txt has gunicorn
print("\n✅ Checking requirements.txt:")
if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r') as f:
        requirements = f.read()
        if 'gunicorn' in requirements:
            print("  ✓ gunicorn is in requirements.txt")
        else:
            print("  ✗ gunicorn is MISSING from requirements.txt")
            all_checks_passed = False
        
        # Count dependencies
        dep_count = len([line for line in requirements.split('\n') if line.strip() and not line.startswith('#')])
        print(f"  ✓ Total dependencies: {dep_count}")

# Final summary
print("\n" + "=" * 70)
if all_checks_passed:
    print("✅ ALL CHECKS PASSED! Ready for deployment to Railway")
    print("\n📋 Next Steps:")
    print("  1. Initialize git: git init")
    print("  2. Add files: git add .")
    print("  3. Commit: git commit -m 'Initial commit'")
    print("  4. Create GitHub repo and push")
    print("  5. Deploy on Railway from GitHub")
    print("\n📖 Read RAILWAY_DEPLOYMENT_GUIDE.md for detailed steps")
else:
    print("❌ SOME CHECKS FAILED! Fix the issues above before deploying")
    sys.exit(1)

print("=" * 70)
