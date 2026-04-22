#!/usr/bin/env python
"""
Script chạy toàn bộ preprocessing pipeline (Tuần 2-3)

Thứ tự:
1. EDA (đã có trong notebooks/01_eda.ipynb)
2. Noise Injection (tạo 4 variants)
3. Quality Report (verify noise injection)
4. Test DataLoader

Usage:
    python scripts/run_preprocessing.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.noise_injector import NoiseInjector
from src.data.quality_report import DataQualityReporter
from src.data.dataset_loader import test_dataloader


def main():
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE - TUẦN 2-3")
    print("="*70)
    
    # Step 1: EDA (manual - run notebook)
    print("\nStep 1: EDA")
    print("   Please run: notebooks/01_eda.ipynb")
    print("   This will generate dataset statistics and visualizations")
    
    # Step 2: Noise Injection
    print("\n" + "="*70)
    print("Step 2: NOISE INJECTION")
    print("="*70)
    
    # Check if noisy variants already exist
    noisy_root = Path('data/noisy')
    variants_exist = []
    for variant in ['clean', 'noisy-5', 'noisy-10', 'noisy-20']:
        variant_path = noisy_root / variant
        if variant_path.exists():
            variants_exist.append(variant)
    
    overwrite = False
    if variants_exist:
        print(f"\n⚠️  Found existing variants: {', '.join(variants_exist)}")
        print("\nOptions:")
        print("  1. Skip noise injection (use existing data)")
        print("  2. Recreate all variants (delete and regenerate)")
        print("  3. Exit and manually delete data/noisy/ folder first")
        
        choice = input("\nYour choice (1, 2, or 3): ").strip()
        
        if choice == '3':
            print("\n📝 To manually delete:")
            print("   1. Close this terminal")
            print("   2. Close VSCode/Jupyter if open")
            print("   3. Open PowerShell as Administrator:")
            print("      Remove-Item -Recurse -Force data\\noisy\\")
            print("   4. Run this script again")
            return
        elif choice == '2':
            overwrite = True
            print("\n🗑️  Will try to recreate all variants...")
            print("⚠️  If you see permission errors, please:")
            print("   - Close VSCode/Jupyter")
            print("   - Close File Explorer")
            print("   - Run PowerShell as Administrator")
            input("\nPress Enter to continue or Ctrl+C to cancel...")
        else:
            print("\n✅ Using existing variants. Skipping to quality report...")
    
    if not variants_exist or overwrite:
        injector = NoiseInjector(
            dataset_root='dataset',
            output_root='data/noisy',
            seed=42
        )
        
        # Create all 4 variants
        injector.create_all_variants(overwrite=overwrite)
    
    # Step 3: Quality Report
    print("\n" + "="*70)
    print("Step 3: DATA QUALITY REPORT")
    print("="*70)
    
    reporter = DataQualityReporter(
        noisy_root='data/noisy',
        output_dir='data/quality_reports'
    )
    
    # Generate full report for 2 representative categories
    reporter.generate_full_report(categories=['bottle', 'carpet'])
    
    # Step 4: Test DataLoader
    print("\n" + "="*70)
    print("🧪 Step 4: TEST DATALOADER")
    print("="*70)
    
    test_dataloader()
    
    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE COMPLETED!")
    print("="*70)
    print("\nOutput locations:")
    print("   - Noisy variants: data/noisy/")
    print("   - Quality reports: data/quality_reports/")
    print("   - Manifests: data/noisy/*/manifest_*.json")
    print("\nNext steps:")
    print("   1. Review quality reports in data/quality_reports/")
    print("   2. Check manifests to verify noise injection")
    print("   3. Proceed to Tuần 4-5: Core Experiments")
    print("="*70)


if __name__ == '__main__':
    main()
