"""
SoftPatch Wrapper
Exp A5-A8: Train SoftPatch on 4 noise variants
Exp B1-B9: Ablation study with different discriminators
"""

import torch
import torch.nn as nn
from pathlib import Path
import yaml
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from torchvision import models
from tqdm import tqdm


class SoftPatchWrapper:
    """
    Wrapper cho SoftPatch model
    
    Exp A5-A8: SoftPatch + 4 variants (LOF discriminator)
    Exp B1-B9: Ablation study (LOF/Gaussian/KNN on 3 noise levels)
    """
    
    def __init__(self, config_path='config/model_config.yaml', discriminator='LOF'):
        """
        Args:
            config_path: Path to model config file
            discriminator: 'LOF', 'Gaussian', or 'KNN'
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.softpatch_config = self.config['softpatch']
        self.device = self.config['hardware']['device']
        self.discriminator_type = discriminator
        
        # Feature extractor
        self.feature_extractor = None
        self.discriminator = None
    
    def create_feature_extractor(self):
        """Create feature extractor (WideResNet50)"""
        backbone = models.wide_resnet50_2(pretrained=True)
        
        # Extract features from layer2 and layer3
        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3
        )
        
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        return self.feature_extractor
    
    def extract_features(self, dataloader):
        """
        Extract features from dataloader
        
        Returns:
            features: numpy array of shape (N, D)
        """
        features_list = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                images = batch['image'].to(self.device)
                
                # Extract features
                feats = self.feature_extractor(images)
                
                # Adaptive average pooling
                feats = nn.functional.adaptive_avg_pool2d(feats, (1, 1))
                feats = feats.view(feats.size(0), -1)
                
                features_list.append(feats.cpu().numpy())
        
        features = np.concatenate(features_list, axis=0)
        return features
    
    def create_discriminator(self, features):
        """
        Create discriminator based on type
        
        Args:
            features: Training features (N, D)
        
        Returns:
            discriminator: Fitted discriminator
        """
        if self.discriminator_type == 'LOF':
            # Local Outlier Factor
            lof_config = self.softpatch_config['lof']
            discriminator = LocalOutlierFactor(
                n_neighbors=lof_config['n_neighbors'],
                contamination=lof_config['contamination'],
                novelty=True  # For prediction on new data
            )
            discriminator.fit(features)
        
        elif self.discriminator_type == 'Gaussian':
            # Gaussian Mixture Model
            gaussian_config = self.softpatch_config['gaussian']
            discriminator = GaussianMixture(
                n_components=1,
                covariance_type=gaussian_config['covariance_type']
            )
            discriminator.fit(features)
        
        elif self.discriminator_type == 'KNN':
            # K-Nearest Neighbors
            knn_config = self.softpatch_config['knn']
            discriminator = NearestNeighbors(
                n_neighbors=knn_config['n_neighbors'],
                algorithm='auto'
            )
            discriminator.fit(features)
        
        else:
            raise ValueError(f"Unknown discriminator: {self.discriminator_type}")
        
        return discriminator
    
    def predict(self, features):
        """
        Predict anomaly scores
        
        Args:
            features: Test features (N, D)
        
        Returns:
            scores: Anomaly scores (N,)
        """
        if self.discriminator_type == 'LOF':
            # LOF: negative outlier factor (higher = more anomalous)
            scores = -self.discriminator.score_samples(features)
        
        elif self.discriminator_type == 'Gaussian':
            # Gaussian: negative log likelihood
            scores = -self.discriminator.score_samples(features)
        
        elif self.discriminator_type == 'KNN':
            # KNN: average distance to k neighbors
            distances, _ = self.discriminator.kneighbors(features)
            scores = distances.mean(axis=1)
        
        return scores
    
    def train(self, category, variant='clean', output_dir='results/softpatch'):
        """
        Train SoftPatch on specific category and noise variant
        
        Args:
            category: MVTec category name
            variant: Noise variant
            output_dir: Output directory
        
        Returns:
            results: Dict with metrics
        """
        print(f"\n{'='*60}")
        print(f"Training SoftPatch ({self.discriminator_type}): {category} / {variant}")
        print(f"{'='*60}")
        
        # Create output directory
        output_path = Path(output_dir) / self.discriminator_type / variant / category
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        from src.data.dataset_loader import get_mvtec_dataloaders
        
        train_loader, test_loader = get_mvtec_dataloaders(
            category=category,
            variant=variant,
            batch_size=self.softpatch_config['training']['batch_size'],
            num_workers=self.softpatch_config['training']['num_workers']
        )
        
        # Create feature extractor
        if self.feature_extractor is None:
            self.create_feature_extractor()
        
        # Extract training features
        print("Extracting training features...")
        train_features = self.extract_features(train_loader)
        
        # Fit discriminator
        print(f"Fitting {self.discriminator_type} discriminator...")
        self.discriminator = self.create_discriminator(train_features)
        
        # Extract test features and predict
        print("Extracting test features...")
        test_features = self.extract_features(test_loader)
        
        print("Predicting anomaly scores...")
        scores = self.predict(test_features)
        
        # Get ground truth labels
        test_labels = []
        for batch in test_loader:
            test_labels.extend(batch['label'])
        test_labels = np.array(test_labels)
        
        # Calculate metrics
        from sklearn.metrics import roc_auc_score, f1_score
        
        # AUROC
        auroc = roc_auc_score(test_labels, scores)
        
        # F1 (need to threshold scores)
        threshold = np.percentile(scores, 95)  # Top 5% as anomalies
        predictions = (scores > threshold).astype(int)
        f1 = f1_score(test_labels, predictions)
        
        metrics = {
            'category': category,
            'variant': variant,
            'discriminator': self.discriminator_type,
            'image_AUROC': auroc,
            'image_F1': f1,
            'pixel_AUROC': 0.0,  # SoftPatch doesn't do pixel-level
            'pixel_PRO': 0.0
        }
        
        print(f"\n✅ Results:")
        print(f"   Image AUROC: {metrics['image_AUROC']:.4f}")
        print(f"   Image F1: {metrics['image_F1']:.4f}")
        
        # Save results
        import json
        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def train_all_variants(self, category, output_dir='results/softpatch'):
        """Train on all 4 noise variants"""
        variants = self.config['experiments']['noise_variants']
        all_results = []
        
        for variant in variants:
            results = self.train(category, variant, output_dir)
            all_results.append(results)
        
        return all_results


# ========== TESTING ==========
if __name__ == '__main__':
    # Test SoftPatch wrapper
    wrapper = SoftPatchWrapper(discriminator='LOF')
    
    # Train on bottle + clean
    results = wrapper.train('bottle', 'clean')
    print(f"\nTest results: {results}")
