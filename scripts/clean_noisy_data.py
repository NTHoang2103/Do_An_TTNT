#!/usr/bin/env python
"""
Script để xóa noisy variants và bắt đầu lại từ đầu

Usage:
    python scripts/clean_noisy_data.py
    python scripts/clean_noisy_data.py --variant noisy-10  # Xóa 1 variant cụ thể
"""

import shutil
import argparse
import os
import stat
import time
from pathlib import Path


def safe_rmtree(path):
    """
    Safely remove directory tree on Windows
    
    Windows-specific: Files can be locked by antivirus, explorer, etc.
    Retry with delay if permission denied.
    """
    def handle_remove_readonly(func, path, exc):
        """Error handler for Windows readonly files"""
        if not os.access(path, os.W_OK):
            # Change file to be writable
            os.chmod(path, stat.S_IWUSR)
            func(path)
        else:
            raise
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path, onerror=handle_remove_readonly)
            return True
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"   ⚠️  Permission denied, retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(2)
            else:
                print(f"\n❌ ERROR: Cannot delete {path}")
                print(f"   Reason: {e}")
                print(f"\n💡 Solutions:")
                print(f"   1. Close any programs that might be using these files")
                print(f"   2. Close File Explorer if viewing this folder")
                print(f"   3. Run this script as Administrator")
                print(f"   4. Manually delete the folder")
                return False


def clean_all_variants():
    """Xóa tất cả noisy variants"""
    noisy_root = Path('data/noisy')
    
    if not noisy_root.exists():
        print("✅ No noisy data found. Nothing to clean.")
        return
    
    variants = ['clean', 'noisy-5', 'noisy-10', 'noisy-20']
    deleted = []
    failed = []
    
    for variant in variants:
        variant_path = noisy_root / variant
        if variant_path.exists():
            print(f"🗑️  Deleting {variant}...")
            if safe_rmtree(variant_path):
                deleted.append(variant)
            else:
                failed.append(variant)
    
    if deleted:
        print(f"\n✅ Deleted {len(deleted)} variants: {', '.join(deleted)}")
    if failed:
        print(f"\n❌ Failed to delete {len(failed)} variants: {', '.join(failed)}")
    if not deleted and not failed:
        print("✅ No variants found to delete.")


def clean_variant(variant_name):
    """Xóa 1 variant cụ thể"""
    variant_path = Path('data/noisy') / variant_name
    
    if not variant_path.exists():
        print(f"✅ Variant '{variant_name}' not found. Nothing to clean.")
        return
    
    print(f"🗑️  Deleting {variant_name}...")
    if safe_rmtree(variant_path):
        print(f"✅ Deleted {variant_name}")
    else:
        print(f"❌ Failed to delete {variant_name}")


def clean_quality_reports():
    """Xóa quality reports"""
    reports_dir = Path('data/quality_reports')
    
    if not reports_dir.exists():
        print("✅ No quality reports found.")
        return
    
    # Delete all files except .gitkeep
    deleted_count = 0
    for file in reports_dir.iterdir():
        if file.name != '.gitkeep':
            file.unlink()
            deleted_count += 1
    
    print(f"✅ Deleted {deleted_count} quality report files")


def main():
    parser = argparse.ArgumentParser(description='Clean noisy data variants')
    parser.add_argument('--variant', type=str, help='Specific variant to delete (clean, noisy-5, noisy-10, noisy-20)')
    parser.add_argument('--reports', action='store_true', help='Also delete quality reports')
    parser.add_argument('--all', action='store_true', help='Delete everything (variants + reports)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CLEAN NOISY DATA")
    print("="*70)
    
    if args.all:
        print("\n⚠️  This will delete ALL noisy variants and quality reports!")
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        
        if confirm == 'yes':
            clean_all_variants()
            clean_quality_reports()
        else:
            print("❌ Cancelled.")
    
    elif args.variant:
        print(f"\n⚠️  This will delete variant: {args.variant}")
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        
        if confirm == 'yes':
            clean_variant(args.variant)
        else:
            print("❌ Cancelled.")
    
    elif args.reports:
        print("\n⚠️  This will delete all quality reports!")
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        
        if confirm == 'yes':
            clean_quality_reports()
        else:
            print("❌ Cancelled.")
    
    else:
        # Default: clean all variants
        print("\n⚠️  This will delete ALL noisy variants!")
        print("Quality reports will be kept.")
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        
        if confirm == 'yes':
            clean_all_variants()
        else:
            print("❌ Cancelled.")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == '__main__':
    main()
