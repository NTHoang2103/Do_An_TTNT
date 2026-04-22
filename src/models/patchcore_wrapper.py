"""
PatchCore Wrapper for Anomalib
Exp A1-A4: Train PatchCore on 4 noise variants
"""

import torch
from pathlib import Path
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import AnomalibDataModule
import yaml


class CustomDataModule(AnomalibDataModule):
    """Custom DataModule wrapper for PyTorch DataLoaders"""
    
    def __init__(self, train_loader, test_loader):
        super().__init__()
        self._train_loader = train_loader
        self._test_loader = test_loader
    
    def _setup(self, stage=None):
        """Setup method required by AnomalibDataModule (abstract method)"""
        # Data is already loaded, nothing to setup
        pass
    
    def train_dataloader(self):
        """Return training dataloader"""
        return self._train_loader
    
    def test_dataloader(self):
        """Return test dataloader"""
        return self._test_loader
    
    def val_dataloader(self):
        """Return validation dataloader (use test for validation)"""
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
            num_neighbors=self.patchcore_config['num_neighbors']
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
            results: Dict with metrics (AUROC, F1, PRO)
        """
        print(f"\n{'='*60}")
        print(f"Training PatchCore: {category} / {variant}")
        print(f"{'='*60}")
        
        # Create output directory
        output_path = Path(output_dir) / variant / category
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create model
        model = self.create_model()
        
        # Use custom dataloader for all variants
        from src.data.dataset_loader import get_mvtec_dataloaders
        
        train_loader, test_loader = get_mvtec_dataloaders(
            category=category,
            variant=variant,
            batch_size=self.patchcore_config['training']['batch_size'],
            num_workers=self.patchcore_config['training']['num_workers']
        )
        
        # Wrap in Anomalib-compatible datamodule
        datamodule = CustomDataModule(train_loader, test_loader)
        
        # Create Engine (v1.0+ API)
        engine = Engine(
            default_root_dir=str(output_path)
        )
        
        # Train
        engine.fit(model=model, datamodule=datamodule)
        
        # Test
        results = engine.test(model=model, datamodule=datamodule)
        
        # Extract metrics
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


# ========== TESTING ==========
if __name__ == '__main__':
    # Test PatchCore wrapper
    wrapper = PatchCoreWrapper()
    
    # Train on bottle + clean
    results = wrapper.train('bottle', 'clean')
    print(f"\nTest results: {results}")
