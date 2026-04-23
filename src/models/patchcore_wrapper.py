"""
PatchCore Wrapper for Anomalib (Compatible with anomalib 2.x)

Exp A1-A4: Train PatchCore on 4 noise variants
"""

import torch
from pathlib import Path
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import AnomalibDataModule
import yaml


class CustomDataModule(AnomalibDataModule):
    """
    Minimal AnomalibDataModule wrapper around PyTorch DataLoaders
    """

    def __init__(self, train_loader, test_loader,
                 train_batch_size=32, eval_batch_size=32, num_workers=8):
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
        )
        self._train_loader = train_loader
        self._test_loader = test_loader

        # IMPORTANT: expose dataset cho anomalib
        self.train_data = train_loader.dataset
        self.test_data = test_loader.dataset

    def setup(self, stage=None):
        """Required by Lightning but not used"""
        pass

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._test_loader  # dùng test làm validation

    def test_dataloader(self):
        return self._test_loader


class PatchCoreWrapper:
    def __init__(self, config_path='config/model_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.patchcore_config = self.config['patchcore']
        self.device = self.config['hardware']['device']

    def create_model(self):
        """Create PatchCore model (anomalib 2.x)"""
        model = Patchcore(
            backbone=self.patchcore_config['backbone'],
            layers=self.patchcore_config['layers'],
            coreset_sampling_ratio=self.patchcore_config['coreset_sampling_ratio'],
            num_neighbors=self.patchcore_config['num_neighbors'],
        )
        return model

    def train(self, category, variant='clean', output_dir='results/patchcore'):
        print(f"\n{'='*60}")
        print(f"Training PatchCore: {category} / {variant}")
        print(f"{'='*60}")

        output_path = Path(output_dir) / variant / category
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Model
        model = self.create_model()

        # 2. Load custom dataloader
        from src.data.dataset_loader import get_mvtec_dataloaders

        train_loader, test_loader = get_mvtec_dataloaders(
            category=category,
            variant=variant,
            batch_size=self.patchcore_config['training']['batch_size'],
            num_workers=self.patchcore_config['training']['num_workers'],
        )

        # 3. Wrap thành datamodule
        datamodule = CustomDataModule(
            train_loader=train_loader,
            test_loader=test_loader,
            train_batch_size=self.patchcore_config['training']['batch_size'],
            eval_batch_size=self.patchcore_config['training']['batch_size'],
            num_workers=self.patchcore_config['training']['num_workers'],
        )

        # 4. Engine (anomalib 2.x)
        engine = Engine(default_root_dir=str(output_path))

        # 5. Train
        engine.fit(model=model, datamodule=datamodule)

        # 6. Test
        results = engine.test(model=model, datamodule=datamodule)

        # 7. Extract metrics
        result = results[0] if isinstance(results, list) else results

        metrics = {
            'category': category,
            'variant': variant,
            'image_AUROC': result.get('image_AUROC', 0.0),
            'pixel_AUROC': result.get('pixel_AUROC', 0.0),
            'image_F1': result.get('image_F1Score', 0.0),
            'pixel_PRO': result.get('pixel_PRO', 0.0),
        }

        print(f"\n✅ Results:")
        print(f"   Image AUROC: {metrics['image_AUROC']:.4f}")
        print(f"   Pixel AUROC: {metrics['pixel_AUROC']:.4f}")
        print(f"   Image F1:    {metrics['image_F1']:.4f}")
        print(f"   Pixel PRO:   {metrics['pixel_PRO']:.4f}")

        return metrics

    def train_all_variants(self, category, output_dir='results/patchcore'):
        variants = self.config['experiments']['noise_variants']
        all_results = []

        for variant in variants:
            results = self.train(category, variant, output_dir)
            all_results.append(results)

        return all_results


# ===== TEST =====
if __name__ == '__main__':
    wrapper = PatchCoreWrapper()
    results = wrapper.train('bottle', 'clean')
    print("\nTest done:", results)
