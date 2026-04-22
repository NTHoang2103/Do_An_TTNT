# 7-bước preprocessing:
# quality_check → CLAHE → alignment → augmentation → resize → normalize → verify

import cv2
import numpy as np
import yaml
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')


class MVTecPreprocessor:
    """
    Preprocessing pipeline đặc thù cho Anomaly Detection trên MVTec AD
    
    7 bước:
    1. Image Quality Check
    2. Illumination Normalization (CLAHE)
    3. Alignment & Crop
    4. Augmentation (cẩn thận - không tạo fake anomaly)
    5. Resize
    6. Normalize (ImageNet stats)
    7. Verify
    """
    
    def __init__(self, config_path='config/preprocess_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.quality_cfg = self.config['quality_check']
        self.illum_cfg = self.config['illumination']
        self.align_cfg = self.config['alignment']
        self.aug_cfg = self.config['augmentation']
        self.norm_cfg = self.config['normalization']
        
        # Category groups
        self.texture_cats = self.config['category_groups']['texture']
        self.object_cats = self.config['category_groups']['object']
    
    # ========== BƯỚC 1: IMAGE QUALITY CHECK ==========
    def check_image_quality(self, image_path):
        """
        Kiểm tra chất lượng ảnh trước khi preprocessing
        
        Returns:
            (is_valid, reason): (bool, str)
        """
        try:
            # Try load
            img = cv2.imread(str(image_path))
            if img is None:
                return False, "Cannot load image"
            
            # Check channels
            if len(img.shape) != 3 or img.shape[2] != 3:
                return False, f"Wrong channels: {img.shape}"
            
            # Check size (basic check)
            h, w = img.shape[:2]
            if h < 100 or w < 100:
                return False, f"Image too small: {h}x{w}"
            
            # Check overexposed
            mean_pixel = img.mean()
            if mean_pixel > self.quality_cfg['max_mean_pixel']:
                return False, f"Overexposed: mean={mean_pixel:.1f}"
            
            # Check underexposed
            if mean_pixel < self.quality_cfg['min_mean_pixel']:
                return False, f"Underexposed: mean={mean_pixel:.1f}"
            
            # Check blank/uniform
            std_pixel = img.std()
            if std_pixel < self.quality_cfg['min_std_pixel']:
                return False, f"Blank/uniform: std={std_pixel:.1f}"
            
            return True, "OK"
        
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    # ========== BƯỚC 2: ILLUMINATION NORMALIZATION ==========
    def apply_clahe(self, image, category):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Chỉ áp dụng cho texture categories
        """
        if category not in self.illum_cfg['clahe']['apply_to']:
            return image
        
        if not self.illum_cfg['clahe']['enabled']:
            return image
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.illum_cfg['clahe']['clip_limit'],
            tileGridSize=tuple(self.illum_cfg['clahe']['tile_grid_size'])
        )
        l_clahe = clahe.apply(l)
        
        # Merge back
        lab_clahe = cv2.merge([l_clahe, a, b])
        image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return image_clahe
    
    def apply_global_hist_eq(self, image, category):
        """
        Apply global histogram equalization
        Chỉ áp dụng cho một số object categories
        """
        if category not in self.illum_cfg['global_hist_eq']['apply_to']:
            return image
        
        if not self.illum_cfg['global_hist_eq']['enabled']:
            return image
        
        # Convert to YUV
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        image_eq = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        return image_eq
    
    # ========== BƯỚC 3: ALIGNMENT & CROP ==========
    def align_and_crop(self, image, category):
        """
        Aspect-ratio preserving resize + center crop
        """
        h, w = image.shape[:2]
        target_size = self.align_cfg['target_size']
        
        # Aspect-ratio preserving resize
        if h > w:
            new_h = target_size
            new_w = int(w * target_size / h)
        else:
            new_w = target_size
            new_h = int(h * target_size / w)
        
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Padding to square
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        
        # Mirror padding for texture, zero padding for objects
        if category in self.texture_cats:
            border_mode = cv2.BORDER_REFLECT
        else:
            border_mode = cv2.BORDER_CONSTANT
        
        image_padded = cv2.copyMakeBorder(
            image_resized,
            pad_h, target_size - new_h - pad_h,
            pad_w, target_size - new_w - pad_w,
            border_mode
        )
        
        return image_padded
    
    # ========== BƯỚC 4: AUGMENTATION ==========
    def get_augmentation_pipeline(self, category, is_train=True):
        """
        Tạo augmentation pipeline đặc thù cho anomaly detection
        
        CẢNH BÁO: Augmentation sai = tạo fake normal → model học nhầm
        """
        if not is_train:
            return A.Compose([])
        
        transforms = []
        
        # Horizontal flip
        if self.aug_cfg['horizontal_flip']['enabled']:
            if category in self.aug_cfg['horizontal_flip']['apply_to']:
                transforms.append(
                    A.HorizontalFlip(p=self.aug_cfg['horizontal_flip']['probability'])
                )
        
        # Vertical flip (chỉ texture)
        if self.aug_cfg['vertical_flip']['enabled']:
            if category in self.aug_cfg['vertical_flip']['apply_to']:
                transforms.append(
                    A.VerticalFlip(p=self.aug_cfg['vertical_flip']['probability'])
                )
        
        # Small rotation
        if self.aug_cfg['rotation']['enabled']:
            transforms.append(
                A.Rotate(
                    limit=self.aug_cfg['rotation']['degrees'],
                    p=self.aug_cfg['rotation']['probability'],
                    border_mode=cv2.BORDER_REFLECT
                )
            )
        
        # Gaussian blur (nhẹ)
        if self.aug_cfg['gaussian_blur']['enabled']:
            transforms.append(
                A.GaussianBlur(
                    blur_limit=tuple(self.aug_cfg['gaussian_blur']['kernel_size']),
                    sigma_limit=tuple(self.aug_cfg['gaussian_blur']['sigma']),
                    p=self.aug_cfg['gaussian_blur']['probability']
                )
            )
        
        # Brightness jitter (rất nhẹ)
        if self.aug_cfg['brightness_jitter']['enabled']:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=self.aug_cfg['brightness_jitter']['brightness'],
                    contrast_limit=0,  # No contrast change
                    p=self.aug_cfg['brightness_jitter']['probability']
                )
            )
        
        return A.Compose(transforms)
    
    # ========== BƯỚC 5 & 6: RESIZE & NORMALIZE ==========
    def get_normalization_pipeline(self):
        """
        Resize về 224x224 + normalize với ImageNet stats
        """
        return A.Compose([
            A.CenterCrop(
                height=self.norm_cfg['crop_size'],
                width=self.norm_cfg['crop_size']
            ),
            A.Normalize(
                mean=self.norm_cfg['mean'],
                std=self.norm_cfg['std']
            ),
            ToTensorV2()
        ])
    
    # ========== PIPELINE TỔNG HỢP ==========
    def preprocess(self, image_path, category, is_train=True, apply_augmentation=True):
        """
        Pipeline đầy đủ 7 bước
        
        Args:
            image_path: Path to image
            category: MVTec category name
            is_train: Training mode or not
            apply_augmentation: Apply augmentation or not
        
        Returns:
            tensor: Preprocessed image tensor (C, H, W)
            metadata: Dict with preprocessing info
        """
        metadata = {'steps': []}
        
        # Bước 1: Quality check
        is_valid, reason = self.check_image_quality(image_path)
        metadata['steps'].append(f"Quality check: {reason}")
        if not is_valid:
            raise ValueError(f"Image quality check failed: {reason}")
        
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
        metadata['original_shape'] = image.shape
        
        # Bước 2: Illumination normalization
        image = self.apply_clahe(image, category)
        metadata['steps'].append("CLAHE applied" if category in self.texture_cats else "CLAHE skipped")
        
        image = self.apply_global_hist_eq(image, category)
        metadata['steps'].append("Hist EQ applied" if category in ['bottle', 'toothbrush'] else "Hist EQ skipped")
        
        # Bước 3: Alignment & crop
        image = self.align_and_crop(image, category)
        metadata['steps'].append(f"Aligned to {self.align_cfg['target_size']}x{self.align_cfg['target_size']}")
        
        # Bước 4: Augmentation (if training)
        if is_train and apply_augmentation:
            aug_pipeline = self.get_augmentation_pipeline(category, is_train=True)
            augmented = aug_pipeline(image=image)
            image = augmented['image']
            metadata['steps'].append("Augmentation applied")
        else:
            metadata['steps'].append("Augmentation skipped")
        
        # Bước 5 & 6: Resize + Normalize
        norm_pipeline = self.get_normalization_pipeline()
        normalized = norm_pipeline(image=image)
        tensor = normalized['image']
        metadata['steps'].append(f"Normalized to {tensor.shape}")
        
        # Bước 7: Verify
        assert tensor.shape[0] == 3, f"Wrong channels: {tensor.shape}"
        assert tensor.shape[1] == self.norm_cfg['crop_size'], f"Wrong height: {tensor.shape}"
        assert tensor.shape[2] == self.norm_cfg['crop_size'], f"Wrong width: {tensor.shape}"
        metadata['steps'].append("Verification passed")
        
        return tensor, metadata


# ========== HELPER FUNCTIONS ==========
def test_preprocessor():
    """Test preprocessing pipeline"""
    preprocessor = MVTecPreprocessor()
    
    # Test on a sample image
    test_image = Path('dataset/bottle/train/good/000.png')
    if test_image.exists():
        tensor, metadata = preprocessor.preprocess(test_image, 'bottle', is_train=True)
        print("Preprocessing test passed!")
        print(f"Output shape: {tensor.shape}")
        print(f"Steps: {metadata['steps']}")
    else:
        print("WARNING: Test image not found")


if __name__ == '__main__':
    test_preprocessor()
