#!/usr/bin/env python
"""
Script to prepare files for Kaggle upload
Creates zip files for:
1. Data (noisy variants)
2. Code (src, config, scripts)
"""

import zipfile
from pathlib import Path
import shutil


def create_data_zip():
    """Create zip file for noisy data variants"""
    print("\n" + "="*70)
    print("CREATING DATA ZIP")
    print("="*70)
    
    data_dir = Path('data/noisy')
    if not data_dir.exists():
        print("❌ Error: data/noisy/ not found!")
        print("   Run preprocessing first: python scripts/run_preprocessing.py")
        return False
    
    output_zip = Path('mvtec_noisy_data.zip')
    
    print(f"Compressing {data_dir}...")
    print("This may take 5-10 minutes...")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in data_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(data_dir.parent)
                zipf.write(file_path, arcname)
                if len(list(zipf.namelist())) % 100 == 0:
                    print(f"  Compressed {len(list(zipf.namelist()))} files...")
    
    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"\n✅ Created: {output_zip}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   Files: {len(list(zipf.namelist()))}")
    
    return True


def create_code_zip():
    """Create zip file for code"""
    print("\n" + "="*70)
    print("CREATING CODE ZIP")
    print("="*70)
    
    output_zip = Path('project_code.zip')
    
    # Files/folders to include
    include = [
        'src/',
        'config/',
        'scripts/',
        'requirements.txt',
        'README.md'
    ]
    
    # Files/folders to exclude
    exclude_patterns = [
        '__pycache__',
        '*.pyc',
        '.git',
        '.vscode',
        'data/',
        'results/',
        'checkpoints/',
        'logs/',
        '*.zip'
    ]
    
    print("Compressing code files...")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in include:
            item_path = Path(item)
            
            if not item_path.exists():
                print(f"⚠️  Skipping {item} (not found)")
                continue
            
            if item_path.is_file():
                zipf.write(item_path, item_path)
                print(f"  ✓ {item}")
            else:
                for file_path in item_path.rglob('*'):
                    if file_path.is_file():
                        # Check if should exclude
                        should_exclude = False
                        for pattern in exclude_patterns:
                            if pattern in str(file_path):
                                should_exclude = True
                                break
                        
                        if not should_exclude:
                            zipf.write(file_path, file_path)
    
    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"\n✅ Created: {output_zip}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   Files: {len(list(zipf.namelist()))}")
    
    return True


def print_instructions():
    """Print upload instructions"""
    print("\n" + "="*70)
    print("UPLOAD INSTRUCTIONS")
    print("="*70)
    
    print("\n📦 Step 1: Upload Data")
    print("   1. Go to https://www.kaggle.com/datasets")
    print("   2. Click 'New Dataset'")
    print("   3. Upload: mvtec_noisy_data.zip")
    print("   4. Title: 'MVTec AD Noisy Variants'")
    print("   5. Click 'Create'")
    
    print("\n📦 Step 2: Upload Code")
    print("   Option A: Upload as Dataset")
    print("   1. Go to https://www.kaggle.com/datasets")
    print("   2. Click 'New Dataset'")
    print("   3. Upload: project_code.zip")
    print("   4. Title: 'Anomaly Detection Code'")
    print("   5. Click 'Create'")
    
    print("\n   Option B: Use GitHub (Recommended)")
    print("   1. Push code to GitHub")
    print("   2. In Kaggle Notebook:")
    print("      !git clone https://github.com/your-username/your-repo.git")
    
    print("\n📦 Step 3: Create Kaggle Notebook")
    print("   1. Go to https://www.kaggle.com/code")
    print("   2. Click 'New Notebook'")
    print("   3. Enable GPU (Settings → Accelerator → GPU P100/T4)")
    print("   4. Add Data Sources:")
    print("      - mvtec-ad-noisy-variants")
    print("      - anomaly-detection-code (if using Option A)")
    print("   5. Copy code from: notebooks/kaggle_training.ipynb")
    
    print("\n✅ Ready to upload!")
    print("\nNext: Follow WEEK4-5_KAGGLE_GUIDE.md for detailed steps")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("KAGGLE UPLOAD PREPARATION")
    print("="*70)
    
    # Create data zip
    data_success = create_data_zip()
    
    # Create code zip
    code_success = create_code_zip()
    
    # Print instructions
    if data_success and code_success:
        print_instructions()
    else:
        print("\n❌ Some files failed to create")
        print("   Check errors above and try again")


if __name__ == '__main__':
    main()
