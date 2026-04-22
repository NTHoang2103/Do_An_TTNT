#!/usr/bin/env python
"""
Main script to run all Week 4-5 experiments

Usage:
    # Run all experiments on all categories
    python scripts/run_experiments.py
    
    # Run on specific categories
    python scripts/run_experiments.py --categories bottle carpet
    
    # Skip certain groups
    python scripts/run_experiments.py --skip patchcore ablation
    
    # Run only one group
    python scripts/run_experiments.py --only softpatch
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.runner import ExperimentRunner
from src.analysis.results_analyzer import ResultsAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Run Week 4-5 experiments')
    parser.add_argument('--categories', nargs='+', help='Categories to test (default: all 15)')
    parser.add_argument('--skip', nargs='+', help='Groups to skip (patchcore, softpatch, efficientad, ablation)')
    parser.add_argument('--only', type=str, help='Run only this group')
    parser.add_argument('--analyze', action='store_true', help='Run analysis after experiments')
    parser.add_argument('--results', type=str, help='Path to results file for analysis only')
    
    args = parser.parse_args()
    
    # Analysis only mode
    if args.results:
        print("\n" + "="*70)
        print("ANALYSIS MODE")
        print("="*70)
        
        analyzer = ResultsAnalyzer(args.results)
        analyzer.generate_full_report()
        return
    
    # Run experiments
    runner = ExperimentRunner()
    
    # Determine which groups to run
    skip_groups = args.skip if args.skip else []
    
    if args.only:
        # Run only specified group
        skip_groups = ['patchcore', 'softpatch', 'efficientad', 'ablation']
        skip_groups.remove(args.only)
    
    # Run
    results = runner.run_all(
        categories=args.categories,
        skip_groups=skip_groups
    )
    
    print(f"\n✅ Completed {len(results)} experiments")
    
    # Run analysis if requested
    if args.analyze:
        print("\n" + "="*70)
        print("RUNNING ANALYSIS")
        print("="*70)
        
        # Find latest results file
        results_dir = Path('results')
        results_files = list(results_dir.glob('all_results_*.csv'))
        
        if results_files:
            latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
            analyzer = ResultsAnalyzer(latest_results)
            analyzer.generate_full_report()
        else:
            print("⚠️  No results file found for analysis")


if __name__ == '__main__':
    main()
