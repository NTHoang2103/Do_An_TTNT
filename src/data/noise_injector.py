# 3 loại noise: Label/Instance/Feature
# Fix seed=42, manifest JSON, reproducible

import numpy as np
import json
import shutil
import os
from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm
import cv2


class NoiseInjector:
    """
    Inject 3 loại noise vào MVTec AD training set:
    
    1. Label Noise: Inject ảnh defect rõ ràng vào train set với nhãn 'normal'
    2. Instance Noise: Inject ảnh defect NHỎ (<5% diện tích) - QC bỏ qua
    3. Feature Noise: Thêm Gaussian noise vào ảnh sau preprocessing
    
    Simulate tình huống thực tế nhà máy:
    - Label Noise: Công nhân đặt nhầm sản phẩm lỗi vào dây chuyền tốt
    - Instance Noise: Defect nhỏ, ánh sáng kém → QC nhầm là sản phẩm tốt
    - Feature Noise: Camera kém, sensor noise, nhiễu điện từ
    """
    
    def __init__(self, dataset_root='dataset', output_root='data/noisy', seed=42):
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.seed = seed
        
        # Fix seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Categories
        self.categories = [
            'bottle', 'cable', 'capsule', 'carpet', 'grid',
            'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
            'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
        ]
    
    def calculate_defect_area_ratio(self, mask_path):
        """
        Tính tỷ lệ diện tích defect trên ảnh từ ground truth mask
        
        Returns:
            ratio: float (0-1)
        """
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return 0.0
        
        total_pixels = mask.shape[0] * mask.shape[1]
        defect_pixels = np.sum(mask > 0)
        ratio = defect_pixels / total_pixels
        
        return ratio
    
    def get_small_defect_images(self, category, threshold=0.05):
        """
        Lấy danh sách ảnh có defect nhỏ (<5% diện tích) cho Instance Noise
        
        Args:
            category: MVTec category name
            threshold: Defect area threshold (default 5%)
        
        Returns:
            small_defect_images: List of (image_path, mask_path, ratio)
        """
        test_path = self.dataset_root / category / 'test'
        gt_path = self.dataset_root / category / 'ground_truth'
        
        small_defect_images = []
        
        # Iterate through all defect types
        for defect_type in test_path.iterdir():
            if not defect_type.is_dir() or defect_type.name == 'good':
                continue
            
            # Check corresponding ground truth
            gt_defect_path = gt_path / defect_type.name
            if not gt_defect_path.exists():
                continue
            
            # Check each image
            for img_file in defect_type.glob('*.png'):
                mask_file = gt_defect_path / img_file.name.replace('.png', '_mask.png')
                
                if mask_file.exists():
                    ratio = self.calculate_defect_area_ratio(mask_file)
                    
                    if ratio < threshold and ratio > 0:  # Small but not zero
                        small_defect_images.append({
                            'image_path': str(img_file),
                            'mask_path': str(mask_file),
                            'defect_type': defect_type.name,
                            'defect_ratio': ratio
                        })
        
        return small_defect_images
    
    def inject_label_noise(self, category, noise_ratio, output_dir):
        """
        LABEL NOISE: Inject ảnh defect rõ ràng vào train set
        
        Simulate: Công nhân đặt nhầm sản phẩm lỗi rõ vào dây chuyền tốt
        """
        train_good_path = self.dataset_root / category / 'train' / 'good'
        test_path = self.dataset_root / category / 'test'
        
        # Count normal images
        normal_images = list(train_good_path.glob('*.png'))
        num_normal = len(normal_images)
        
        # Calculate number of noisy images to inject
        num_noise = int(num_normal * noise_ratio)
        
        # Collect all defect images
        defect_images = []
        for defect_type in test_path.iterdir():
            if defect_type.is_dir() and defect_type.name != 'good':
                defect_images.extend(list(defect_type.glob('*.png')))
        
        # Random sample defect images
        if len(defect_images) < num_noise:
            print(f"WARNING: Not enough defect images for {category}. Using all {len(defect_images)} images.")
            sampled_defects = defect_images
        else:
            sampled_defects = random.sample(defect_images, num_noise)
        
        # Copy to output
        output_good_dir = output_dir / category / 'train' / 'good'
        output_good_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy normal images
        for img in normal_images:
            shutil.copy(img, output_good_dir / img.name)
        
        # Copy (inject) defect images with new names
        injected_files = []
        for idx, defect_img in enumerate(sampled_defects):
            new_name = f'injected_label_{idx:04d}.png'
            shutil.copy(defect_img, output_good_dir / new_name)
            injected_files.append({
                'type': 'label_noise',
                'source': str(defect_img),
                'target': str(output_good_dir / new_name)
            })
        
        return injected_files
    
    def inject_instance_noise(self, category, noise_ratio, output_dir):
        """
        INSTANCE NOISE: Inject ảnh defect NHỎ (<5% diện tích)
        
        Simulate: Defect nhỏ, ánh sáng kém → QC bỏ qua, nhầm là sản phẩm tốt
        """
        train_good_path = self.dataset_root / category / 'train' / 'good'
        
        # Get small defect images
        small_defects = self.get_small_defect_images(category, threshold=0.05)
        
        if len(small_defects) == 0:
            print(f"WARNING: No small defect images found for {category}. Skipping instance noise.")
            return []
        
        # Count normal images
        normal_images = list(train_good_path.glob('*.png'))
        num_normal = len(normal_images)
        
        # Calculate number of instance noise to inject
        num_instance_noise = int(num_normal * noise_ratio * 0.3)  # 30% of total noise budget
        
        # Sample small defects
        if len(small_defects) < num_instance_noise:
            sampled_small_defects = small_defects
        else:
            sampled_small_defects = random.sample(small_defects, num_instance_noise)
        
        # Copy to output
        output_good_dir = output_dir / category / 'train' / 'good'
        output_good_dir.mkdir(parents=True, exist_ok=True)
        
        # Inject small defect images
        injected_files = []
        for idx, defect_info in enumerate(sampled_small_defects):
            new_name = f'injected_instance_{idx:04d}.png'
            shutil.copy(defect_info['image_path'], output_good_dir / new_name)
            injected_files.append({
                'type': 'instance_noise',
                'source': defect_info['image_path'],
                'target': str(output_good_dir / new_name),
                'defect_ratio': defect_info['defect_ratio']
            })
        
        return injected_files
    
    def apply_feature_noise(self, image, sigma=0.02):
        """
        FEATURE NOISE: Thêm Gaussian noise vào ảnh
        
        Simulate: Camera kém, sensor noise, nhiễu điện từ trong nhà máy
        
        Args:
            image: numpy array (H, W, C), normalized to [0, 1]
            sigma: Noise standard deviation
        
        Returns:
            noisy_image: numpy array with added noise
        """
        noise = np.random.normal(0, sigma, image.shape)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 1)
        return noisy_image
    
    def create_noisy_variant(self, noise_ratio, variant_name, overwrite=False):
        """
        Tạo 1 noisy variant với tỷ lệ noise cho trước
        
        Args:
            noise_ratio: float (0.05, 0.10, 0.20)
            variant_name: str ('noisy-5', 'noisy-10', 'noisy-20')
            overwrite: bool, if True will delete existing variant
        """
        output_dir = self.output_root / variant_name
        
        # Check if variant already exists
        if output_dir.exists() and not overwrite:
            manifest_path = output_dir / f'manifest_{variant_name}.json'
            if manifest_path.exists():
                print(f"\n⚠️  {variant_name} already exists at: {output_dir}")
                print(f"   Manifest found: {manifest_path}")
                print(f"   Skipping creation. Use overwrite=True to recreate.")
                return None
        
        # Delete existing if overwrite=True
        if output_dir.exists() and overwrite:
            print(f"\n🗑️  Deleting existing {variant_name}...")
            self._safe_rmtree(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            'variant_name': variant_name,
            'noise_ratio_target': noise_ratio,
            'seed': self.seed,
            'categories': {}
        }
        
        print(f"\n{'='*60}")
        print(f"Creating {variant_name} (target noise ratio: {noise_ratio*100:.0f}%)")
        print(f"{'='*60}")
        
        for category in tqdm(self.categories, desc="Processing categories"):
            # Inject Label Noise (70% of noise budget)
            label_noise_files = self.inject_label_noise(
                category, 
                noise_ratio * 0.7,  # 70% label noise
                output_dir
            )
            
            # Inject Instance Noise (30% of noise budget)
            instance_noise_files = self.inject_instance_noise(
                category,
                noise_ratio,  # Will use 30% internally
                output_dir
            )
            
            # Calculate actual noise ratio
            train_good_path = output_dir / category / 'train' / 'good'
            total_images = len(list(train_good_path.glob('*.png')))
            num_injected = len(label_noise_files) + len(instance_noise_files)
            actual_ratio = num_injected / total_images if total_images > 0 else 0
            
            manifest['categories'][category] = {
                'total_images': total_images,
                'injected_images': num_injected,
                'actual_noise_ratio': actual_ratio,
                'label_noise': label_noise_files,
                'instance_noise': instance_noise_files
            }
            
            print(f"  {category}: {num_injected}/{total_images} = {actual_ratio*100:.1f}%")
        
        # Save manifest
        manifest_path = output_dir / f'manifest_{variant_name}.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n{variant_name} created successfully!")
        print(f"Manifest saved to: {manifest_path}")
        
        return manifest
    
    def create_all_variants(self, overwrite=False):
        """
        Tạo tất cả 4 variants: clean, noisy-5, noisy-10, noisy-20
        
        Args:
            overwrite: bool, if True will delete and recreate existing variants
        """
        # Clean variant (0% noise) - just copy original
        clean_dir = self.output_root / 'clean'
        
        if clean_dir.exists() and not overwrite:
            print("\n⚠️  Clean variant already exists. Skipping.")
        else:
            if clean_dir.exists() and overwrite:
                print(f"\n🗑️  Deleting existing clean variant...")
                self._safe_rmtree(clean_dir)
            
            print("\n" + "="*60)
            print("Creating clean variant (0% noise)")
            print("="*60)
            
            for category in tqdm(self.categories, desc="Copying clean data"):
                src_train = self.dataset_root / category / 'train'
                dst_train = clean_dir / category / 'train'
                if src_train.exists():
                    shutil.copytree(src_train, dst_train, dirs_exist_ok=True)
            
            print("Clean variant created!")
        
        # Noisy variants
        self.create_noisy_variant(0.05, 'noisy-5', overwrite=overwrite)
        self.create_noisy_variant(0.10, 'noisy-10', overwrite=overwrite)
        self.create_noisy_variant(0.20, 'noisy-20', overwrite=overwrite)
        
        print("\n" + "="*60)
        print("All variants created successfully!")
        print("="*60)
    
    def _safe_rmtree(self, path):
        """
        Safely remove directory tree on Windows
        
        Windows-specific: Files can be locked by antivirus, explorer, etc.
        Retry with delay if permission denied.
        """
        import time
        import stat
        
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
                return
            except PermissionError as e:
                if attempt < max_retries - 1:
                    print(f"   ⚠️  Permission denied, retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(2)
                else:
                    print(f"\n❌ ERROR: Cannot delete {path}")
                    print(f"   Reason: {e}")
                    print(f"\n💡 Solutions:")
                    print(f"   1. Close any programs that might be using these files")
                    print(f"   2. Run this script as Administrator")
                    print(f"   3. Manually delete the folder: {path}")
                    print(f"   4. Use: python scripts/clean_noisy_data.py --all")
                    raise


# ========== MAIN ==========
if __name__ == '__main__':
    injector = NoiseInjector(
        dataset_root='dataset',
        output_root='data/noisy',
        seed=42
    )
    
    # Create all 4 variants
    injector.create_all_variants()
