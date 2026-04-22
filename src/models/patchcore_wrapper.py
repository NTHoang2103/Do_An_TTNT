import torch
from pathlib import Path
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import AnomalibDataModule
import yaml


class CustomDataModule(AnomalibDataModule):
    """Clean + compatible với Anomalib"""

    def __init__(self, train_loader, test_loader, train_batch_size=32, eval_batch_size=32, num_workers=4):
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers
        )

        self._train_loader = train_loader
        self._test_loader = test_loader

        # 🔥 BẮT BUỘC (fix lỗi test_data)
        self.test_data = test_loader.dataset

    def _setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self._train_loader

    def test_dataloader(self):
        return self._test_loader

    def val_dataloader(self):
        return self._test_loader


class PatchCoreWrapper:

    def __init__(self, config_path='config/model_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.patchcore_config = self.config['patchcore']
        self.device = self.config['hardware']['device']

    def create_model(self):
        return Patchcore(
            backbone=self.patchcore_config['backbone'],
            layers=self.patchcore_config['layers'],
            coreset_sampling_ratio=self.patchcore_config['coreset_sampling_ratio'],
            num_neighbors=self.patchcore_config['num_neighbors']
        )

    def train(self, category, variant='clean', output_dir='results/patchcore'):

        print(f"\n{'='*60}")
        print(f"Training PatchCore: {category} / {variant}")
        print(f"{'='*60}")

        output_path = Path(output_dir) / variant / category
        output_path.mkdir(parents=True, exist_ok=True)

        model = self.create_model()

        from src.data.dataset_loader import get_mvtec_dataloaders

        train_loader, test_loader = get_mvtec_dataloaders(
            category=category,
            variant=variant,
            batch_size=self.patchcore_config['training']['batch_size'],
            num_workers=4  # 🔥 fix warning
        )

        datamodule = CustomDataModule(
            train_loader,
            test_loader,
            train_batch_size=self.patchcore_config['training']['batch_size'],
            eval_batch_size=self.patchcore_config['training']['batch_size'],
            num_workers=4
        )

        engine = Engine(default_root_dir=str(output_path))

        engine.fit(model=model, datamodule=datamodule)

        results = engine.test(model=model, datamodule=datamodule)

        metrics = {
            'category': category,
            'variant': variant,
            'image_AUROC': results[0].get('image_AUROC', 0.0),
            'pixel_AUROC': results[0].get('pixel_AUROC', 0.0),
            'image_F1': results[0].get('image_F1Score', 0.0),
            'pixel_PRO': results[0].get('pixel_PRO', 0.0)
        }

        print(f"\n✅ Results:")
        print(f"   Image AUROC: {metrics['image_AUROC']:.4f}")
        print(f"   Pixel AUROC: {metrics['pixel_AUROC']:.4f}")
        print(f"   Image F1: {metrics['image_F1']:.4f}")
        print(f"   Pixel PRO: {metrics['pixel_PRO']:.4f}")

        return metrics
