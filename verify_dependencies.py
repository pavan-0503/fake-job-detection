#!/usr/bin/env python3
"""
Comprehensive Dependency Verification Script
Checks all Python imports against requirements.txt
"""

import re
import sys
from pathlib import Path

# Define which files are used in production (deployed to Railway)
PRODUCTION_FILES = [
    'app.py',
    'predictor.py',
    'scraper.py',
    'download_models.py'
]

# Requirements from requirements.txt (normalized names)
REQUIREMENTS = {
    'torch', 'transformers', 'sklearn', 'scikit-learn', 'pandas', 'numpy',
    'flask', 'flask_cors', 'gunicorn', 'beautifulsoup4', 'bs4',
    'joblib', 'requests', 'tqdm'
}

# Python standard library modules (don't need to be in requirements.txt)
STDLIB_MODULES = {
    're', 'os', 'sys', 'time', 'warnings', 'argparse', 'datetime',
    'urllib', 'zipfile', 'pathlib', 'json', 'collections', 'typing',
    'functools', 'itertools', 'math', 'random', 'string', 'io',
    'tempfile', 'shutil', 'subprocess', 'threading', 'multiprocessing',
    'importlib', 'inspect', 'ast', 'copy', 'pickle', 'gzip', 'bz2',
    'csv', 'configparser', 'logging', 'traceback', 'contextlib',
    'weakref', 'gc', 'signal', 'socket', 'select', 'queue', 'heapq',
    'bisect', 'array', 'struct', 'codecs', 'unicodedata', 'base64',
    'hashlib', 'hmac', 'secrets', 'uuid', 'enum', 'decimal', 'fractions',
    'statistics', 'difflib', 'pprint', 'textwrap', 'reprlib', 'dataclasses'
}

def normalize_module_name(module):
    """Normalize module names for comparison"""
    # Handle 'from X import Y' - we only care about X
    module = module.split('.')[0]
    
    # Special mappings
    mappings = {
        'sklearn': 'scikit-learn',
        'bs4': 'beautifulsoup4',
        'flask_cors': 'flask-cors'
    }
    
    return mappings.get(module, module)

def extract_imports(file_path):
    """Extract all imports from a Python file"""
    imports = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find 'import X' statements
    for match in re.finditer(r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*)', content, re.MULTILINE):
        imports.add(match.group(1))
    
    # Find 'from X import Y' statements
    for match in re.finditer(r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*)', content, re.MULTILINE):
        imports.add(match.group(1))
    
    return imports

def main():
    print("=" * 70)
    print("🔍 COMPREHENSIVE DEPENDENCY VERIFICATION")
    print("=" * 70)
    
    issues_found = []
    all_production_imports = set()
    
    # Check each production file
    for filename in PRODUCTION_FILES:
        file_path = Path(filename)
        if not file_path.exists():
            print(f"⚠️  Warning: {filename} not found (skipping)")
            continue
        
        print(f"\n📄 Checking {filename}...")
        imports = extract_imports(file_path)
        
        # Separate into stdlib and external
        external_imports = set()
        for module in imports:
            if module not in STDLIB_MODULES:
                external_imports.add(module)
                all_production_imports.add(module)
        
        print(f"   Found {len(imports)} imports ({len(external_imports)} external)")
        
        # Check if all external imports are in requirements
        for module in external_imports:
            normalized = normalize_module_name(module)
            if normalized not in REQUIREMENTS:
                issue = f"❌ {filename}: imports '{module}' but it's NOT in requirements.txt"
                issues_found.append(issue)
                print(f"   {issue}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 70)
    
    print(f"\n✅ Production files checked: {len(PRODUCTION_FILES)}")
    print(f"✅ Total external imports found: {len(all_production_imports)}")
    print(f"   {', '.join(sorted(all_production_imports))}")
    
    if issues_found:
        print(f"\n❌ ISSUES FOUND: {len(issues_found)}")
        for issue in issues_found:
            print(f"   {issue}")
        print("\n⚠️  DEPLOYMENT WILL FAIL - Fix these issues before deploying!")
        return 1
    else:
        print("\n✅ ALL CHECKS PASSED!")
        print("✅ All imports are either:")
        print("   - In requirements.txt")
        print("   - Part of Python standard library")
        print("\n🚀 Ready for deployment!")
        return 0

if __name__ == '__main__':
    sys.exit(main())
