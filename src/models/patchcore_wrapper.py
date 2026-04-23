"""
PatchCore Wrapper for Anomalib
Exp A1-A4: Train PatchCore on 4 noise variants
"""

import torch
from pathlib import Path
from anomalib.models import Patchcore
from anomalib.data import MVTecAD  # Changed from MVTec in v2.3+
from anomalib.engine import Engine
import yaml


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
        
        # Create datamodule (custom path for noisy variants)
        if variant == 'clean':
            # Use original MVTec dataset
            datamodule = MVTecAD(
                root=f'dataset/{category}',
                category=category,
                train_batch_size=self.patchcore_config['training']['batch_size'],
                eval_batch_size=self.patchcore_config['training']['batch_size'],
                num_workers=self.patchcore_config['training']['num_workers']
            )
        else:
            # Use noisy variant - point to noisy data folder
            datamodule = MVTecAD(
                root=f'data/noisy/{variant}/{category}',
                category=category,
                train_batch_size=self.patchcore_config['training']['batch_size'],
                eval_batch_size=self.patchcore_config['training']['batch_size'],
                num_workers=self.patchcore_config['training']['num_workers']
            )
            
            print(f"Loaded {len(datamodule.train_data)} samples from {variant}/{category}/train")
            print(f"Loaded {len(datamodule.test_data)} samples from /{category}/test")
        
        # Create trainer
        engine = Engine(
            default_root_dir=str(output_path),
            accelerator='gpu' if self.device == 'cuda' else 'cpu',
            devices=1,
            max_epochs=self.patchcore_config['training']['max_epochs'],
            logger=False
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
    
    def _create_custom_datamodule(self, train_loader, test_loader):
        """
        Create Anomalib-compatible datamodule from custom loaders
        
        This is a placeholder - needs proper implementation
        """
        # TODO: Implement proper datamodule wrapper
        # For now, return None and handle in train()
        return None
    
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
