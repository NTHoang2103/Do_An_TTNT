import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Tuple


class MVTecDataset(Dataset):
    """
    Dataset cho MVTec AD + noisy variants
    Compatible với Anomalib
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str = 'train',
        variant: str = None,   # 🔥 FIX: dùng None thay vì ''
        transform=None,
        mask_transform=None
    ):
        self.root = Path(root)
        self.category = category
        self.split = split
        self.variant = variant
        self.transform = transform
        self.mask_transform = mask_transform

        # 🔥 FIX PATH CLEAN
        if self.variant:
            self.data_path = self.root / variant / category / split
        else:
            self.data_path = self.root / category / split

        # Load samples
        self.samples = self._load_samples()

        # Required by Anomalib
        self.has_normal = any(s['label'] == 0 for s in self.samples)
        self.label_index = {'Normal': 0, 'Anomalous': 1}

        print(f"Loaded {len(self.samples)} samples from {self.data_path}")

    def _load_samples(self) -> List[dict]:
        samples = []

        if self.split == 'train':
            good_path = self.data_path / 'good'
            if good_path.exists():
                for img_path in sorted(good_path.glob('*.png')):
                    samples.append({
                        'image_path': str(img_path),
                        'mask_path': None,
                        'label': 0,
                        'defect_type': 'good'
                    })

        else:  # test
            for defect_folder in sorted(self.data_path.iterdir()):
                if not defect_folder.is_dir():
                    continue

                defect_type = defect_folder.name
                is_defect = defect_type != 'good'

                for img_path in sorted(defect_folder.glob('*.png')):
                    mask_path = None

                    if is_defect:
                        gt_path = Path('dataset') / self.category / 'ground_truth' / defect_type
                        mask_file = gt_path / img_path.name.replace('.png', '_mask.png')
                        if mask_file.exists():
                            mask_path = str(mask_file)

                    samples.append({
                        'image_path': str(img_path),
                        'mask_path': mask_path,
                        'label': 1 if is_defect else 0,
                        'defect_type': defect_type
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            image = Image.open(sample['image_path']).convert('RGB')

            # Mask
            if sample['mask_path'] and Path(sample['mask_path']).exists():
                mask = Image.open(sample['mask_path']).convert('L')
            else:
                mask = Image.fromarray(np.zeros(image.size[::-1], dtype=np.uint8))

            # Transform
            if self.transform:
                image = self.transform(image)
            else:
                from torchvision import transforms
                image = transforms.ToTensor()(image)

            if self.mask_transform:
                mask = self.mask_transform(mask)
            else:
                mask = torch.from_numpy(np.array(mask)).float() / 255.0

            # 🔥 QUAN TRỌNG NHẤT (FIX LỖI CỦA BẠN)
            return {
                'image': image,
                'mask': mask,
                'label': sample['label'],
                'label_index': sample['label'],   # ✅ FIX
                'image_path': sample['image_path'],
                'mask_path': sample['mask_path'],
                'defect_type': sample['defect_type']
            }

        except Exception as e:
            print(f"\n❌ Error loading sample {idx}")
            print(f"Path: {sample['image_path']}")
            print(f"Error: {e}")
            raise e


# =========================
# DataLoader factory
# =========================

def get_mvtec_dataloaders(
    category: str,
    variant: str = 'clean',
    batch_size: int = 32,
    num_workers: int = 4,
    transform=None
) -> Tuple[DataLoader, DataLoader]:

    if transform is None:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # Custom collate
    def custom_collate(batch):
        from torch.utils.data._utils.collate import default_collate

        tensor_batch = {}
        list_batch = {}

        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                tensor_batch[key] = default_collate([d[key] for d in batch])
            else:
                list_batch[key] = [d[key] for d in batch]

        return {**tensor_batch, **list_batch}

    # Train (noisy)
    train_dataset = MVTecDataset(
        root='data/noisy',
        category=category,
        split='train',
        variant=variant,
        transform=transform
    )

    # Test (original dataset)
    test_dataset = MVTecDataset(
        root='dataset',
        category=category,
        split='test',
        variant=None,   # 🔥 FIX
        transform=transform
    )

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


# =========================
# TEST
# =========================

if __name__ == '__main__':
    train_loader, test_loader = get_mvtec_dataloaders(
        category='bottle',
        variant='clean',
        batch_size=4,
        num_workers=0
    )

    print("\nTrain batch:")
    batch = next(iter(train_loader))
    print(batch['image'].shape, batch['label'])

    print("\nTest batch:")
    batch = next(iter(test_loader))
    print(batch['image'].shape, batch['mask'].shape)
