# DataLoader chuẩn Anomalib cho MVTec + 4 noisy variants + external

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Optional, List, Tuple
import yaml


class MVTecDataset(Dataset):
    """
    Dataset cho MVTec AD với support cho noisy variants
    
    Compatible với Anomalib framework
    """
    
    def __init__(
        self,
        root: str,
        category: str,
        split: str = 'train',
        variant: str = 'clean',
        transform=None,
        mask_transform=None
    ):
        """
        Args:
            root: Root path to dataset (e.g., 'data/noisy')
            category: MVTec category name
            split: 'train' or 'test'
            variant: 'clean', 'noisy-5', 'noisy-10', 'noisy-20'
            transform: Image transform
            mask_transform: Mask transform (for test set)
        """
        self.root = Path(root)
        self.category = category
        self.split = split
        self.variant = variant
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Paths
        self.data_path = self.root / variant / category / split
        
        # Load samples
        self.samples = self._load_samples()
        
        # Add has_normal attribute required by Anomalib
        self.has_normal = any(s['label'] == 0 for s in self.samples)
        
        print(f"Loaded {len(self.samples)} samples from {variant}/{category}/{split}")
    
    def _load_samples(self) -> List[dict]:
        """Load all samples (images + masks if test)"""
        samples = []
        
        if self.split == 'train':
            # Train: only 'good' folder
            good_path = self.data_path / 'good'
            if good_path.exists():
                for img_path in sorted(good_path.glob('*.png')):
                    samples.append({
                        'image_path': str(img_path),
                        'mask_path': None,
                        'label': 0,  # Normal
                        'defect_type': 'good'
                    })
        
        else:  # test
            # Test: 'good' + defect folders
            for defect_folder in sorted(self.data_path.iterdir()):
                if not defect_folder.is_dir():
                    continue
                
                defect_type = defect_folder.name
                is_defect = (defect_type != 'good')
                
                for img_path in sorted(defect_folder.glob('*.png')):
                    # Find corresponding mask
                    mask_path = None
                    if is_defect:
                        # Mask in original dataset (not in noisy variants)
                        original_gt_path = Path('dataset') / self.category / 'ground_truth' / defect_type
                        mask_file = original_gt_path / img_path.name.replace('.png', '_mask.png')
                        if mask_file.exists():
                            mask_path = str(mask_file)
                    
                    samples.append({
                        'image_path': str(img_path),
                        'mask_path': mask_path,
                        'label': 1 if is_defect else 0,
                        'defect_type': defect_type
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys: 'image', 'mask', 'label', 'image_path', 'mask_path'
        """
        sample = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(sample['image_path']).convert('RGB')
            
            # Load mask (if exists)
            if sample['mask_path'] is not None and Path(sample['mask_path']).exists():
                mask = Image.open(sample['mask_path']).convert('L')
            else:
                # Create empty mask for normal images
                mask = Image.fromarray(np.zeros(image.size[::-1], dtype=np.uint8))
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
                # Ensure transform didn't return None
                if image is None:
                    raise ValueError(f"Transform returned None for image: {sample['image_path']}")
            else:
                # Convert to tensor if no transform
                from torchvision import transforms
                image = transforms.ToTensor()(image)
            
            if self.mask_transform:
                mask = self.mask_transform(mask)
                if mask is None:
                    raise ValueError(f"Mask transform returned None")
            else:
                # Default: convert to tensor
                mask = torch.from_numpy(np.array(mask)).float() / 255.0
            
            # Final validation
            if image is None:
                raise ValueError(f"Image is None after transform")
            if mask is None:
                raise ValueError(f"Mask is None after transform")
            
            return {
                'image': image,
                'mask': mask,
                'label': sample['label'],
                'image_path': sample['image_path'],
                'mask_path': sample['mask_path'],
                'defect_type': sample['defect_type']
            }
        
        except Exception as e:
            print(f"\n❌ Error loading sample {idx}:")
            print(f"   Path: {sample['image_path']}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            raise e


class ExternalDataset(Dataset):
    """
    Dataset cho external products (sản phẩm ngoài MVTec)
    """
    
    def __init__(
        self,
        root: str = 'data/external',
        split: str = 'normal',
        transform=None
    ):
        """
        Args:
            root: Root path to external data
            split: 'normal' or 'defect'
            transform: Image transform
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
        # Load samples
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} external samples from {split}")
    
    def _load_samples(self) -> List[dict]:
        """Load external samples"""
        samples = []
        
        split_path = self.root / self.split
        if split_path.exists():
            for img_path in sorted(split_path.glob('*.png')) + sorted(split_path.glob('*.jpg')):
                samples.append({
                    'image_path': str(img_path),
                    'label': 0 if self.split == 'normal' else 1
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': sample['label'],
            'image_path': sample['image_path']
        }


def get_mvtec_dataloaders(
    category: str,
    variant: str = 'clean',
    batch_size: int = 32,
    num_workers: int = 4,
    transform=None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and test dataloaders for MVTec AD
    
    Args:
        category: MVTec category name
        variant: 'clean', 'noisy-5', 'noisy-10', 'noisy-20'
        batch_size: Batch size
        num_workers: Number of workers
        transform: Transform to apply
    
    Returns:
        train_loader, test_loader
    """
    # Default transform if not provided
    if transform is None:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Custom collate function to handle string fields
    def custom_collate(batch):
        """Custom collate to handle mixed types (tensors + strings)"""
        from torch.utils.data._utils.collate import default_collate
        
        # Separate tensor fields and non-tensor fields
        tensor_batch = {}
        list_batch = {}
        
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                # Stack tensors
                tensor_batch[key] = default_collate([d[key] for d in batch])
            else:
                # Keep as list for strings/None
                list_batch[key] = [d[key] for d in batch]
        
        # Merge
        return {**tensor_batch, **list_batch}
    
    # Train dataset (from noisy variants)
    train_dataset = MVTecDataset(
        root='data/noisy',
        category=category,
        split='train',
        variant=variant,
        transform=transform
    )
    
    # Test dataset (always use original dataset, not noisy)
    # Create a simple wrapper to load from original dataset
    test_dataset = MVTecDataset(
        root='dataset',
        category=category,
        split='test',
        variant='',  # Empty variant for original dataset
        transform=transform
    )
    
    # Override the data_path to point to original dataset
    test_dataset.data_path = Path('dataset') / category / 'test'
    test_dataset.samples = test_dataset._load_samples()
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    return train_loader, test_loader


def test_dataloader():
    """Test dataloader"""
    print("Testing MVTec DataLoader...")
    
    # Test clean variant
    train_loader, test_loader = get_mvtec_dataloaders(
        category='bottle',
        variant='clean',
        batch_size=4,
        num_workers=0
    )
    
    # Test train loader
    print("\nTrain loader:")
    batch = next(iter(train_loader))
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Label: {batch['label']}")
    print(f"  Defect types: {batch['defect_type']}")
    
    # Test test loader
    print("\nTest loader:")
    batch = next(iter(test_loader))
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Mask shape: {batch['mask'].shape}")
    print(f"  Label: {batch['label']}")
    print(f"  Defect types: {batch['defect_type']}")
    
    print("\nDataLoader test passed!")


if __name__ == '__main__':
    test_dataloader()
