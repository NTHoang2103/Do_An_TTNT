"""
Experiment Runner for Week 4-5
Runs all 21 experiments automatically
"""

import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm

from src.models.patchcore_wrapper import PatchCoreWrapper
from src.models.softpatch_wrapper import SoftPatchWrapper
from src.models.efficientad_wrapper import EfficientADWrapper


class ExperimentRunner:
    """
    Run all 21 experiments for Week 4-5
    
    Group A: Noisy Data Robustness (12 experiments)
    - A1-A4: PatchCore (4 variants)
    - A5-A8: SoftPatch (4 variants)
    - A9-A12: EfficientAD (4 variants)
    
    Group B: Ablation Study (9 experiments)
    - B1-B9: SoftPatch with LOF/Gaussian/KNN on 3 noise levels
    """
    
    def __init__(self, config_path='config/model_config.yaml'):
        """
        Args:
            config_path: Path to model config
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path(self.config['logging']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.all_results = []
    
    def run_group_a_patchcore(self, categories=None):
        """
        Run Exp A1-A4: PatchCore on 4 noise variants
        
        Args:
            categories: List of categories to test (default: all 15)
        
        Returns:
            results: List of dicts with metrics
        """
        print("\n" + "="*70)
        print("GROUP A: PATCHCORE (Exp A1-A4)")
        print("="*70)
        
        if categories is None:
            categories = self.config['experiments']['categories']
        
        variants = self.config['experiments']['noise_variants']
        
        wrapper = PatchCoreWrapper()
        results = []
        
        for category in tqdm(categories, desc="Categories"):
            for variant in variants:
                exp_name = f"A{variants.index(variant)+1}"
                print(f"\n[{exp_name}] PatchCore: {category} / {variant}")
                
                try:
                    metrics = wrapper.train(category, variant)
                    metrics['experiment'] = exp_name
                    metrics['model'] = 'PatchCore'
                    results.append(metrics)
                    self.all_results.append(metrics)
                except Exception as e:
                    print(f"❌ Error: {e}")
                    results.append({
                        'experiment': exp_name,
                        'model': 'PatchCore',
                        'category': category,
                        'variant': variant,
                        'error': str(e)
                    })
        
        return results
    
    def run_group_a_softpatch(self, categories=None):
        """
        Run Exp A5-A8: SoftPatch on 4 noise variants
        
        Returns:
            results: List of dicts with metrics
        """
        print("\n" + "="*70)
        print("GROUP A: SOFTPATCH (Exp A5-A8)")
        print("="*70)
        
        if categories is None:
            categories = self.config['experiments']['categories']
        
        variants = self.config['experiments']['noise_variants']
        
        wrapper = SoftPatchWrapper(discriminator='LOF')
        results = []
        
        for category in tqdm(categories, desc="Categories"):
            for variant in variants:
                exp_name = f"A{variants.index(variant)+5}"
                print(f"\n[{exp_name}] SoftPatch: {category} / {variant}")
                
                try:
                    metrics = wrapper.train(category, variant)
                    metrics['experiment'] = exp_name
                    metrics['model'] = 'SoftPatch'
                    results.append(metrics)
                    self.all_results.append(metrics)
                except Exception as e:
                    print(f"❌ Error: {e}")
                    results.append({
                        'experiment': exp_name,
                        'model': 'SoftPatch',
                        'category': category,
                        'variant': variant,
                        'error': str(e)
                    })
        
        return results
    
    def run_group_a_efficientad(self, categories=None):
        """
        Run Exp A9-A12: EfficientAD on 4 noise variants
        
        Returns:
            results: List of dicts with metrics
        """
        print("\n" + "="*70)
        print("GROUP A: EFFICIENTAD (Exp A9-A12)")
        print("="*70)
        
        if categories is None:
            categories = self.config['experiments']['categories']
        
        variants = self.config['experiments']['noise_variants']
        
        wrapper = EfficientADWrapper()
        results = []
        
        for category in tqdm(categories, desc="Categories"):
            for variant in variants:
                exp_name = f"A{variants.index(variant)+9}"
                print(f"\n[{exp_name}] EfficientAD: {category} / {variant}")
                
                try:
                    metrics = wrapper.train(category, variant)
                    metrics['experiment'] = exp_name
                    metrics['model'] = 'EfficientAD'
                    results.append(metrics)
                    self.all_results.append(metrics)
                except Exception as e:
                    print(f"❌ Error: {e}")
                    results.append({
                        'experiment': exp_name,
                        'model': 'EfficientAD',
                        'category': category,
                        'variant': variant,
                        'error': str(e)
                    })
        
        return results
    
    def run_group_b_ablation(self, categories=None):
        """
        Run Exp B1-B9: Ablation study
        SoftPatch with LOF/Gaussian/KNN on 3 noise levels
        
        Returns:
            results: List of dicts with metrics
        """
        print("\n" + "="*70)
        print("GROUP B: ABLATION STUDY (Exp B1-B9)")
        print("="*70)
        
        if categories is None:
            categories = self.config['experiments']['categories']
        
        discriminators = self.config['ablation']['discriminators']
        noise_levels = self.config['ablation']['noise_levels']
        
        results = []
        exp_counter = 1
        
        for discriminator in discriminators:
            wrapper = SoftPatchWrapper(discriminator=discriminator)
            
            for noise_level in noise_levels:
                exp_name = f"B{exp_counter}"
                
                for category in tqdm(categories, desc=f"[{exp_name}] {discriminator}/{noise_level}"):
                    print(f"\n[{exp_name}] SoftPatch-{discriminator}: {category} / {noise_level}")
                    
                    try:
                        metrics = wrapper.train(category, noise_level)
                        metrics['experiment'] = exp_name
                        metrics['model'] = f'SoftPatch-{discriminator}'
                        results.append(metrics)
                        self.all_results.append(metrics)
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        results.append({
                            'experiment': exp_name,
                            'model': f'SoftPatch-{discriminator}',
                            'category': category,
                            'variant': noise_level,
                            'discriminator': discriminator,
                            'error': str(e)
                        })
                
                exp_counter += 1
        
        return results
    
    def run_all(self, categories=None, skip_groups=None):
        """
        Run all 21 experiments
        
        Args:
            categories: List of categories (default: all 15)
            skip_groups: List of groups to skip (e.g., ['patchcore', 'ablation'])
        
        Returns:
            all_results: List of all results
        """
        if skip_groups is None:
            skip_groups = []
        
        print("\n" + "="*70)
        print("RUNNING ALL EXPERIMENTS (Week 4-5)")
        print("="*70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Group A: PatchCore
        if 'patchcore' not in skip_groups:
            self.run_group_a_patchcore(categories)
        
        # Group A: SoftPatch
        if 'softpatch' not in skip_groups:
            self.run_group_a_softpatch(categories)
        
        # Group A: EfficientAD
        if 'efficientad' not in skip_groups:
            self.run_group_a_efficientad(categories)
        
        # Group B: Ablation
        if 'ablation' not in skip_groups:
            self.run_group_b_ablation(categories)
        
        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETED!")
        print("="*70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total experiments: {len(self.all_results)}")
        
        # Save results
        self.save_results()
        
        return self.all_results
    
    def save_results(self):
        """Save all results to CSV and JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as CSV
        df = pd.DataFrame(self.all_results)
        csv_path = self.results_dir / f'all_results_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Results saved to: {csv_path}")
        
        # Save as JSON
        json_path = self.results_dir / f'all_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        print(f"✅ Results saved to: {json_path}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary of results"""
        df = pd.DataFrame(self.all_results)
        
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        
        # Group by model and variant
        if 'image_AUROC' in df.columns:
            summary = df.groupby(['model', 'variant'])['image_AUROC'].agg(['mean', 'std', 'count'])
            print("\nImage AUROC by Model and Variant:")
            print(summary)
        
        # Best performing model per variant
        if 'image_AUROC' in df.columns:
            print("\nBest Model per Variant:")
            best = df.loc[df.groupby('variant')['image_AUROC'].idxmax()]
            print(best[['variant', 'model', 'image_AUROC']])


# ========== MAIN ==========
if __name__ == '__main__':
    # Run all experiments
    runner = ExperimentRunner()
    
    # Test on 1 category first
    results = runner.run_all(categories=['bottle'])
    
    print(f"\nTotal results: {len(results)}")
