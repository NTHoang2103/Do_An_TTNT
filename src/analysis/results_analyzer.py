"""
Results Analysis for Week 4-5
Analyze and visualize experiment results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


class ResultsAnalyzer:
    """
    Analyze experiment results and generate insights
    
    Tasks:
    1. Compare models across noise levels
    2. Find critical threshold (inflection point)
    3. Per-category error analysis
    4. Ablation study analysis
    """
    
    def __init__(self, results_path):
        """
        Args:
            results_path: Path to results CSV or JSON
        """
        self.results_path = Path(results_path)
        
        # Load results
        if self.results_path.suffix == '.csv':
            self.df = pd.read_csv(self.results_path)
        elif self.results_path.suffix == '.json':
            with open(self.results_path, 'r') as f:
                data = json.load(f)
            self.df = pd.DataFrame(data)
        
        # Output directory
        self.output_dir = Path('results/analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compare_models(self):
        """
        Compare models across noise levels
        
        RQ1: Which model is most robust to noisy data?
        """
        print("\n" + "="*60)
        print("ANALYSIS 1: Model Comparison")
        print("="*60)
        
        # Group by model and variant
        summary = self.df.groupby(['model', 'variant'])['image_AUROC'].agg(['mean', 'std'])
        print("\nImage AUROC by Model and Variant:")
        print(summary)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = self.df['model'].unique()
        variants = ['clean', 'noisy-5', 'noisy-10', 'noisy-20']
        
        for model in models:
            model_data = self.df[self.df['model'] == model]
            means = []
            stds = []
            
            for variant in variants:
                variant_data = model_data[model_data['variant'] == variant]
                if len(variant_data) > 0:
                    means.append(variant_data['image_AUROC'].mean())
                    stds.append(variant_data['image_AUROC'].std())
                else:
                    means.append(0)
                    stds.append(0)
            
            ax.errorbar(variants, means, yerr=stds, marker='o', label=model, capsize=5)
        
        ax.set_xlabel('Noise Variant')
        ax.set_ylabel('Image AUROC')
        ax.set_title('Model Performance vs Noise Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300)
        print(f"✅ Saved: {self.output_dir / 'model_comparison.png'}")
        
        return summary
    
    def find_critical_threshold(self, model='PatchCore'):
        """
        Find critical threshold (inflection point)
        
        RQ2: At what noise level does performance degrade significantly?
        """
        print("\n" + "="*60)
        print(f"ANALYSIS 2: Critical Threshold for {model}")
        print("="*60)
        
        # Filter by model
        model_data = self.df[self.df['model'] == model]
        
        # Calculate mean AUROC per variant
        variants = ['clean', 'noisy-5', 'noisy-10', 'noisy-20']
        noise_levels = [0, 5, 10, 20]
        
        means = []
        for variant in variants:
            variant_data = model_data[model_data['variant'] == variant]
            means.append(variant_data['image_AUROC'].mean())
        
        # Find inflection point (where degradation > 10%)
        baseline = means[0]  # clean performance
        degradations = [(baseline - m) / baseline * 100 for m in means]
        
        critical_idx = None
        for i, deg in enumerate(degradations):
            if deg > 10:  # >10% degradation
                critical_idx = i
                break
        
        if critical_idx:
            print(f"\n✅ Critical threshold found at: {noise_levels[critical_idx]}% noise")
            print(f"   Degradation: {degradations[critical_idx]:.2f}%")
        else:
            print("\n✅ No critical threshold found (degradation < 10%)")
        
        # Plot AUROC curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # AUROC vs Noise Level
        ax1.plot(noise_levels, means, marker='o', linewidth=2)
        ax1.axhline(y=baseline * 0.9, color='r', linestyle='--', label='10% degradation')
        if critical_idx:
            ax1.axvline(x=noise_levels[critical_idx], color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Noise Level (%)')
        ax1.set_ylabel('Image AUROC')
        ax1.set_title(f'{model}: AUROC vs Noise Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Degradation vs Noise Level
        ax2.plot(noise_levels, degradations, marker='s', linewidth=2, color='orange')
        ax2.axhline(y=10, color='r', linestyle='--', label='10% threshold')
        if critical_idx:
            ax2.axvline(x=noise_levels[critical_idx], color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Noise Level (%)')
        ax2.set_ylabel('Performance Degradation (%)')
        ax2.set_title(f'{model}: Degradation vs Noise Level')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'critical_threshold_{model}.png', dpi=300)
        print(f"✅ Saved: {self.output_dir / f'critical_threshold_{model}.png'}")
        
        return critical_idx, degradations
    
    def per_category_analysis(self):
        """
        Per-category error analysis
        
        Which categories are most affected by noise?
        """
        print("\n" + "="*60)
        print("ANALYSIS 3: Per-Category Error Analysis")
        print("="*60)
        
        # Calculate degradation per category
        categories = self.df['category'].unique()
        
        degradations = []
        for category in categories:
            cat_data = self.df[self.df['category'] == category]
            
            clean_auroc = cat_data[cat_data['variant'] == 'clean']['image_AUROC'].mean()
            noisy20_auroc = cat_data[cat_data['variant'] == 'noisy-20']['image_AUROC'].mean()
            
            if clean_auroc > 0:
                deg = (clean_auroc - noisy20_auroc) / clean_auroc * 100
            else:
                deg = 0
            
            degradations.append({
                'category': category,
                'clean_AUROC': clean_auroc,
                'noisy20_AUROC': noisy20_auroc,
                'degradation_%': deg
            })
        
        deg_df = pd.DataFrame(degradations)
        deg_df = deg_df.sort_values('degradation_%', ascending=False)
        
        print("\nTop 5 Most Affected Categories:")
        print(deg_df.head())
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.barh(deg_df['category'], deg_df['degradation_%'])
        ax.set_xlabel('Performance Degradation (%)')
        ax.set_ylabel('Category')
        ax.set_title('Per-Category Degradation (Clean → Noisy-20)')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_category_degradation.png', dpi=300)
        print(f"✅ Saved: {self.output_dir / 'per_category_degradation.png'}")
        
        return deg_df
    
    def ablation_analysis(self):
        """
        Ablation study analysis
        
        Which discriminator is best for SoftPatch?
        """
        print("\n" + "="*60)
        print("ANALYSIS 4: Ablation Study (Discriminators)")
        print("="*60)
        
        # Filter ablation experiments
        ablation_data = self.df[self.df['experiment'].str.startswith('B')]
        
        if len(ablation_data) == 0:
            print("⚠️  No ablation data found")
            return None
        
        # Group by discriminator and variant
        summary = ablation_data.groupby(['discriminator', 'variant'])['image_AUROC'].agg(['mean', 'std'])
        print("\nAblation Results:")
        print(summary)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        discriminators = ablation_data['discriminator'].unique()
        variants = ['noisy-5', 'noisy-10', 'noisy-20']
        
        for disc in discriminators:
            disc_data = ablation_data[ablation_data['discriminator'] == disc]
            means = []
            
            for variant in variants:
                variant_data = disc_data[disc_data['variant'] == variant]
                if len(variant_data) > 0:
                    means.append(variant_data['image_AUROC'].mean())
                else:
                    means.append(0)
            
            ax.plot(variants, means, marker='o', label=disc, linewidth=2)
        
        ax.set_xlabel('Noise Variant')
        ax.set_ylabel('Image AUROC')
        ax.set_title('Ablation Study: Discriminator Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ablation_discriminators.png', dpi=300)
        print(f"✅ Saved: {self.output_dir / 'ablation_discriminators.png'}")
        
        return summary
    
    def generate_full_report(self):
        """Generate full analysis report"""
        print("\n" + "="*70)
        print("GENERATING FULL ANALYSIS REPORT")
        print("="*70)
        
        # Run all analyses
        model_comparison = self.compare_models()
        critical_threshold = self.find_critical_threshold('PatchCore')
        category_analysis = self.per_category_analysis()
        ablation_results = self.ablation_analysis()
        
        # Save summary
        with open(self.output_dir / 'analysis_summary.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("WEEK 4-5 ANALYSIS SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write("1. MODEL COMPARISON\n")
            f.write(str(model_comparison) + "\n\n")
            
            f.write("2. CRITICAL THRESHOLD\n")
            f.write(f"Critical threshold: {critical_threshold}\n\n")
            
            f.write("3. PER-CATEGORY ANALYSIS\n")
            f.write(str(category_analysis) + "\n\n")
            
            f.write("4. ABLATION STUDY\n")
            f.write(str(ablation_results) + "\n")
        
        print(f"\n✅ Full report saved to: {self.output_dir / 'analysis_summary.txt'}")


# ========== MAIN ==========
if __name__ == '__main__':
    # Test analyzer
    analyzer = ResultsAnalyzer('results/all_results_20260422_120000.csv')
    analyzer.generate_full_report()
