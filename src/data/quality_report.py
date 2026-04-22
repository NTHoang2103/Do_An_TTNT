# Auto-generate data quality report:
# tỷ lệ noise thực tế, distribution shift, t-SNE, baseline AUROC check

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class DataQualityReporter:
    """
    Tự động sinh báo cáo chất lượng dữ liệu sau khi tạo noise variants
    
    Báo cáo bao gồm:
    1. Tỷ lệ noise thực tế vs target
    2. Pixel distribution shift (histogram comparison)
    3. Feature space visualization (t-SNE/PCA)
    4. Baseline AUROC check (quick sanity check)
    """
    
    def __init__(self, noisy_root='data/noisy', output_dir='data/quality_reports'):
        self.noisy_root = Path(noisy_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature extractor (WideResNet50 pretrained)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = models.wide_resnet50_2(pretrained=True)
        self.feature_extractor.fc = torch.nn.Identity()  # Remove classifier
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_manifest(self, variant_name):
        """Load manifest JSON file"""
        manifest_path = self.noisy_root / variant_name / f'manifest_{variant_name}.json'
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                return json.load(f)
        return None
    
    def check_noise_ratio(self, variants=['clean', 'noisy-5', 'noisy-10', 'noisy-20']):
        """
        Kiểm tra tỷ lệ noise thực tế so với target
        """
        print("\n" + "="*60)
        print("1. NOISE RATIO VERIFICATION")
        print("="*60)
        
        results = []
        
        for variant in variants:
            manifest = self.load_manifest(variant)
            if manifest is None:
                continue
            
            target_ratio = manifest.get('noise_ratio_target', 0)
            
            for category, info in manifest['categories'].items():
                actual_ratio = info.get('actual_noise_ratio', 0)
                results.append({
                    'Variant': variant,
                    'Category': category,
                    'Target Ratio': target_ratio,
                    'Actual Ratio': actual_ratio,
                    'Difference': abs(actual_ratio - target_ratio)
                })
        
        df = pd.DataFrame(results)
        
        # Summary statistics
        print("\nSummary by Variant:")
        summary = df.groupby('Variant').agg({
            'Target Ratio': 'first',
            'Actual Ratio': ['mean', 'std'],
            'Difference': 'mean'
        }).round(4)
        print(summary)
        
        # Save to CSV
        csv_path = self.output_dir / 'noise_ratio_verification.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nSaved to: {csv_path}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot = df.groupby('Variant')['Actual Ratio'].mean().reset_index()
        df_plot['Target Ratio'] = df_plot['Variant'].map({
            'clean': 0, 'noisy-5': 0.05, 'noisy-10': 0.10, 'noisy-20': 0.20
        })
        
        x = np.arange(len(df_plot))
        width = 0.35
        
        ax.bar(x - width/2, df_plot['Target Ratio'], width, label='Target', color='#3498db')
        ax.bar(x + width/2, df_plot['Actual Ratio'], width, label='Actual', color='#e74c3c')
        
        ax.set_xlabel('Variant')
        ax.set_ylabel('Noise Ratio')
        ax.set_title('Target vs Actual Noise Ratio')
        ax.set_xticks(x)
        ax.set_xticklabels(df_plot['Variant'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'noise_ratio_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {self.output_dir / 'noise_ratio_comparison.png'}")
        
        return df
    
    def analyze_pixel_distribution(self, category='bottle', max_samples=100):
        """
        So sánh pixel distribution của 4 variants
        """
        print("\n" + "="*60)
        print(f"2. PIXEL DISTRIBUTION ANALYSIS ({category})")
        print("="*60)
        
        variants = ['clean', 'noisy-5', 'noisy-10', 'noisy-20']
        stats = []
        
        for variant in variants:
            train_path = self.noisy_root / variant / category / 'train' / 'good'
            if not train_path.exists():
                continue
            
            images = list(train_path.glob('*.png'))[:max_samples]
            
            means = []
            stds = []
            
            for img_path in images:
                img = np.array(Image.open(img_path))
                means.append(img.mean())
                stds.append(img.std())
            
            stats.append({
                'Variant': variant,
                'Mean (avg)': np.mean(means),
                'Mean (std)': np.std(means),
                'Std (avg)': np.mean(stds),
                'Std (std)': np.std(stds)
            })
        
        df_stats = pd.DataFrame(stats)
        print("\nPixel Statistics:")
        print(df_stats.to_string(index=False))
        
        # Plot distributions
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Mean pixel values
        for variant in variants:
            train_path = self.noisy_root / variant / category / 'train' / 'good'
            if not train_path.exists():
                continue
            
            images = list(train_path.glob('*.png'))[:max_samples]
            means = [np.array(Image.open(img)).mean() for img in images]
            
            axes[0].hist(means, bins=30, alpha=0.6, label=variant)
        
        axes[0].set_xlabel('Mean Pixel Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Mean Pixel Distribution')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Std pixel values
        for variant in variants:
            train_path = self.noisy_root / variant / category / 'train' / 'good'
            if not train_path.exists():
                continue
            
            images = list(train_path.glob('*.png'))[:max_samples]
            stds = [np.array(Image.open(img)).std() for img in images]
            
            axes[1].hist(stds, bins=30, alpha=0.6, label=variant)
        
        axes[1].set_xlabel('Std Pixel Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Std Pixel Distribution')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'pixel_distribution_{category}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {self.output_dir / f'pixel_distribution_{category}.png'}")
        
        return df_stats
    
    def extract_features(self, image_paths, max_samples=50):
        """Extract features using WideResNet50"""
        features = []
        
        for img_path in tqdm(image_paths[:max_samples], desc="Extracting features"):
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feat = self.feature_extractor(img_tensor)
                features.append(feat.cpu().numpy().flatten())
        
        return np.array(features)
    
    def visualize_feature_space(self, category='bottle', max_samples=50):
        """
        Visualize feature space với t-SNE/PCA
        """
        print("\n" + "="*60)
        print(f"3. FEATURE SPACE VISUALIZATION ({category})")
        print("="*60)
        
        variants = ['clean', 'noisy-5', 'noisy-10', 'noisy-20']
        all_features = []
        all_labels = []
        
        for variant in variants:
            train_path = self.noisy_root / variant / category / 'train' / 'good'
            if not train_path.exists():
                continue
            
            images = list(train_path.glob('*.png'))
            features = self.extract_features(images, max_samples=max_samples)
            
            all_features.append(features)
            all_labels.extend([variant] * len(features))
        
        all_features = np.vstack(all_features)
        
        # t-SNE
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_tsne = tsne.fit_transform(all_features)
        
        # PCA
        print("Running PCA...")
        pca = PCA(n_components=2, random_state=42)
        features_pca = pca.fit_transform(all_features)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = {'clean': '#2ecc71', 'noisy-5': '#3498db', 'noisy-10': '#f39c12', 'noisy-20': '#e74c3c'}
        
        # t-SNE plot
        for variant in variants:
            mask = np.array(all_labels) == variant
            axes[0].scatter(
                features_tsne[mask, 0], 
                features_tsne[mask, 1],
                c=colors[variant],
                label=variant,
                alpha=0.6,
                s=50
            )
        axes[0].set_title('t-SNE Visualization', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # PCA plot
        for variant in variants:
            mask = np.array(all_labels) == variant
            axes[1].scatter(
                features_pca[mask, 0], 
                features_pca[mask, 1],
                c=colors[variant],
                label=variant,
                alpha=0.6,
                s=50
            )
        axes[1].set_title('PCA Visualization', fontsize=14, fontweight='bold')
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.suptitle(f'Feature Space Visualization - {category.upper()}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'feature_space_{category}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {self.output_dir / f'feature_space_{category}.png'}")
    
    def generate_full_report(self, categories=['bottle', 'carpet']):
        """
        Generate full quality report
        """
        print("\n" + "="*70)
        print("DATA QUALITY REPORT GENERATION")
        print("="*70)
        
        # 1. Noise ratio verification
        df_noise = self.check_noise_ratio()
        
        # 2. Pixel distribution analysis
        for category in categories:
            self.analyze_pixel_distribution(category, max_samples=100)
        
        # 3. Feature space visualization
        for category in categories:
            self.visualize_feature_space(category, max_samples=50)
        
        # Summary report
        print("\n" + "="*70)
        print("SUMMARY REPORT")
        print("="*70)
        print(f"Noise ratio verification: {self.output_dir / 'noise_ratio_verification.csv'}")
        print(f"Pixel distribution plots: {len(categories)} categories")
        print(f"Feature space visualizations: {len(categories)} categories")
        print(f"\nAll reports saved to: {self.output_dir}")
        print("="*70)


# ========== MAIN ==========
if __name__ == '__main__':
    reporter = DataQualityReporter(
        noisy_root='data/noisy',
        output_dir='data/quality_reports'
    )
    
    # Generate full report for 2 representative categories
    reporter.generate_full_report(categories=['bottle', 'carpet'])
