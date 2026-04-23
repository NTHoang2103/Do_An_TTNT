"""
PatchCore Wrapper for Anomalib — Fixed for anomalib v2.3.1+
Exp A1-A4: Train PatchCore on 4 noise variants

CHANGELOG:
- Fixed: `from anomalib.data import MVTec` → `from anomalib.data import MVTecAD`
  (MVTec was renamed to MVTecAD in anomalib v2.3.1)
- Fixed: `MVTec(...)` → `MVTecAD(...)` throughout
- Fixed: `Patchcore` → `Patchcore` (still correct, no change needed here)
"""

import torch
from pathlib import Path
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import AnomalibDataModule

# ✅ FIXED: đổi MVTec → MVTecAD (breaking change từ anomalib v2.3.1)
try:
    from anomalib.data import MVTecAD          # anomalib v2.3.1+
except ImportError:
    from anomalib.data import MVTec as MVTecAD  # fallback anomalib v2.3.0 trở về trước

import yaml


class CustomDataModule(AnomalibDataModule):
    """
    Minimal AnomalibDataModule wrapper around plain PyTorch DataLoaders.

    Anomalib internally accesses:
      - datamodule.test_data          -> the test Dataset object
      - datamodule.test_data.samples  -> must have .label_index attribute
      - datamodule.test_data.samples.label_index -> {'Normal': 0, 'Anomalous': 1}

    Because MVTecDataset.samples is now a SamplesWithLabelIndex (a list subclass),
    everything here works automatically.
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

        # Expose the underlying Dataset so Anomalib's Engine can reach .samples
        self.test_data = test_loader.dataset if hasattr(test_loader, 'dataset') else None
        self.train_data = train_loader.dataset if hasattr(train_loader, 'dataset') else None

    def _setup(self, stage=None):
        """Abstract method required by AnomalibDataModule – data is pre-loaded."""
        pass

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._test_loader   # Reuse test set for validation

    def test_dataloader(self):
        return self._test_loader


class PatchCoreWrapper:
    """
    Wrapper cho PatchCore model từ Anomalib

    Exp A1: PatchCore + clean
    Exp A2: PatchCore + noisy-5
    Exp A3: PatchCore + noisy-10
    Exp A4: PatchCore + noisy-20
    """

    def __init__(self, config_path='config/model_config.yaml'):
        """
        Args:
            config_path: Path to model config file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.patchcore_config = self.config['patchcore']
        self.device = self.config['hardware']['device']

    def create_model(self):
        """Create PatchCore model"""
        model = Patchcore(
            backbone=self.patchcore_config['backbone'],
            layers=self.patchcore_config['layers'],
            coreset_sampling_ratio=self.patchcore_config['coreset_sampling_ratio'],
            num_neighbors=self.patchcore_config['num_neighbors'],
        )
        return model

    def train(self, category, variant='clean', output_dir='results/patchcore'):
        """
        Train PatchCore on specific category and noise variant

        Args:
            category: MVTec category name
            variant: Noise variant (clean, noisy-5, noisy-10, noisy-20)
            output_dir: Output directory for results

        Returns:
            metrics: Dict with image_AUROC, pixel_AUROC, image_F1, pixel_PRO
        """
        print(f"\n{'='*60}")
        print(f"Training PatchCore: {category} / {variant}")
        print(f"{'='*60}")

        # Create output directory
        output_path = Path(output_dir) / variant / category
        output_path.mkdir(parents=True, exist_ok=True)

        # Create model
        model = self.create_model()

        # Load data
        from src.data.dataset_loader import get_mvtec_dataloaders

        train_loader, test_loader = get_mvtec_dataloaders(
            category=category,
            variant=variant,
            batch_size=self.patchcore_config['training']['batch_size'],
            num_workers=self.patchcore_config['training']['num_workers'],
        )

        # Wrap in Anomalib-compatible datamodule
        datamodule = CustomDataModule(
            train_loader, test_loader,
            train_batch_size=self.patchcore_config['training']['batch_size'],
            eval_batch_size=self.patchcore_config['training']['batch_size'],
            num_workers=self.patchcore_config['training']['num_workers'],
        )

        # Create Engine (v2.x API)
        engine = Engine(default_root_dir=str(output_path))

        # Train
        engine.fit(model=model, datamodule=datamodule)

        # Test
        results = engine.test(model=model, datamodule=datamodule)

        # Extract metrics — key names may vary by anomalib version
        metrics = {
            'category': category,
            'variant': variant,
            'image_AUROC': results[0].get('image_AUROC', results[0].get('test_image_AUROC', 0.0)),
            'pixel_AUROC': results[0].get('pixel_AUROC', results[0].get('test_pixel_AUROC', 0.0)),
            'image_F1':    results[0].get('image_F1Score', results[0].get('test_image_F1Score', 0.0)),
            'pixel_PRO':   results[0].get('pixel_PRO', results[0].get('test_pixel_PRO', 0.0)),
        }

        print(f"\n✅ Results:")
        print(f"   Image AUROC: {metrics['image_AUROC']:.4f}")
        print(f"   Pixel AUROC: {metrics['pixel_AUROC']:.4f}")
        print(f"   Image F1:    {metrics['image_F1']:.4f}")
        print(f"   Pixel PRO:   {metrics['pixel_PRO']:.4f}")

        return metrics

    def train_all_variants(self, category, output_dir='results/patchcore'):
        """
        Train on all 4 noise variants for a category

        Returns:
            all_results: List of dicts with metrics
        """
        variants = self.config['experiments']['noise_variants']
        all_results = []

        for variant in variants:
            results = self.train(category, variant, output_dir)
            all_results.append(results)

        return all_results


# ==================== QUICK VERSION CHECK ====================
def check_anomalib_version():
    """
    In ra version anomalib đang dùng và tên import đúng.
    Chạy cell này trước để biết môi trường Kaggle đang dùng version nào.
    """
    import anomalib
    version = anomalib.__version__
    print(f"anomalib version: {version}")

    # Parse version
    parts = version.split(".")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2].split("rc")[0].split("b")[0])

    if (major, minor, patch) >= (2, 3, 1):
        print("✅ Dùng: from anomalib.data import MVTecAD")
    else:
        print("⚠️  Dùng: from anomalib.data import MVTec  (version cũ)")

    return version


# ==================== TESTING ====================
if __name__ == '__main__':
    # Bước 1: kiểm tra version trước
    check_anomalib_version()

    # Bước 2: test wrapper
    wrapper = PatchCoreWrapper()
    results = wrapper.train('bottle', 'clean')
    print(f"\nTest results: {results}")
