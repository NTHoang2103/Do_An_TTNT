"""
EfficientAD Wrapper for Anomalib
Exp A9-A12: Train EfficientAD on 4 noise variants
"""

import torch
from pathlib import Path
from anomalib.models import EfficientAd
from anomalib.engine import Engine
import yaml


class EfficientADWrapper:
    """
    Wrapper cho EfficientAD model từ Anomalib
    
    Exp A9: EfficientAD + clean
    Exp A10: EfficientAD + noisy-5
    Exp A11: EfficientAD + noisy-10
    Exp A12: EfficientAD + noisy-20
    """
    
    def __init__(self, config_path='config/model_config.yaml'):
        """
        Args:
            config_path: Path to model config file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.efficientad_config = self.config['efficientad']
        self.device = self.config['hardware']['device']
    
    def create_model(self):
        """Create EfficientAD model"""
        model = EfficientAd(
            teacher_out_channels=384,
            model_size='small',  # or 'medium'
            lr=self.efficientad_config['training']['learning_rate'],
            weight_decay=self.efficientad_config['training']['weight_decay']
        )
        return model
    
    def train(self, category, variant='clean', output_dir='results/efficientad'):
        """
        Train EfficientAD on specific category and noise variant
        
        Args:
            category: MVTec category name
            variant: Noise variant
            output_dir: Output directory
        
        Returns:
            results: Dict with metrics
        """
        print(f"\n{'='*60}")
        print(f"Training EfficientAD: {category} / {variant}")
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
            batch_size=self.efficientad_config['training']['batch_size'],
            num_workers=self.efficientad_config['training']['num_workers']
        )
        
        # Wrap in Anomalib-compatible datamodule
        datamodule = self._create_custom_datamodule(train_loader, test_loader)
        
        # Create trainer
        engine = Engine(
            task='segmentation',
            model=model,
            datamodule=datamodule,
            max_epochs=self.efficientad_config['training']['max_epochs'],
            default_root_dir=str(output_path)
        )
        
        # Train
        engine.fit()
        
        # Test
        results = engine.test()
        
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
        
        # Save results
        import json
        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _create_custom_datamodule(self, train_loader, test_loader):
        """
        Create Anomalib-compatible datamodule
        
        Placeholder - needs proper implementation
        """
        # TODO: Implement proper datamodule wrapper
        return None
    
    def train_all_variants(self, category, output_dir='results/efficientad'):
        """Train on all 4 noise variants"""
        variants = self.config['experiments']['noise_variants']
        all_results = []
        
        for variant in variants:
            results = self.train(category, variant, output_dir)
            all_results.append(results)
        
        return all_results


# ========== TESTING ==========
if __name__ == '__main__':
    # Test EfficientAD wrapper
    wrapper = EfficientADWrapper()
    
    # Train on bottle + clean
    results = wrapper.train('bottle', 'clean')
    print(f"\nTest results: {results}")
