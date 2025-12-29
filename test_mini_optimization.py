"""
Mini Optimization Test

Runs a small-scale NSGA-III optimization to verify the full pipeline works.
Much faster than full optimization (completes in ~30-60 minutes vs hours).

Usage:
    python test_mini_optimization.py --data_root /path/to/kaggle_vindr_data

Author: David
"""

import argparse
import os
from pathlib import Path
from dataset import prepare_dataset
from training import full_evaluation
from optimization import run_optimization
from config import ExperimentConfig

def main():
    parser = argparse.ArgumentParser(description='Mini optimization test')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to Kaggle VinDr dataset')
    parser.add_argument('--output_dir', type=str, default='./test_output',
                        help='Output directory (default: ./test_output)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--pop_size', type=int, default=6,
                        help='Population size (default: 6, small for testing)')
    parser.add_argument('--generations', type=int, default=3,
                        help='Number of generations (default: 3)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Epochs per training (default: 2)')
    args = parser.parse_args()

    print("="*70)
    print("MINI NSGA-III OPTIMIZATION TEST")
    print("="*70)
    print("\nThis is a quick test with minimal settings:")
    print(f"  Population size: {args.pop_size}")
    print(f"  Generations: {args.generations}")
    print(f"  Epochs per training: {args.epochs}")
    print(f"\nEstimated time: ~{args.pop_size * args.generations * 2} minutes")
    print("(Full optimization would use: pop_size=50, generations=30, epochs=20+)")

    # Load dataset
    print("\n[1/3] Loading dataset...")
    train_paths, train_labels, val_paths, val_labels = prepare_dataset(
        dataset_name="kaggle_vindr_png",
        data_root=args.data_root,
        val_split=0.2,
        seed=42
    )

    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_paths)} samples")
    print(f"  Val: {len(val_paths)} samples")

    # Create evaluation function
    print("\n[2/3] Setting up optimization...")

    def make_eval_fn(tp, tl, vp, vl):
        def eval_fn(hp_config):
            # Override epochs for faster testing
            hp_config = hp_config.copy()
            hp_config['epochs'] = args.epochs
            return full_evaluation(
                hp_config, tp, tl, vp, vl,
                device=args.device,
                verbose=False  # Reduce output clutter
            )
        return eval_fn

    # Configure optimization (minimal settings)
    config = ExperimentConfig()
    config.nsga3.pop_size = args.pop_size
    config.nsga3.n_generations = args.generations
    config.nsga3.surrogate_ratio = 0.5  # Use less surrogate for small test
    config.nsga3.n_objectives = 4
    config.nsga3.checkpoint_frequency = 1  # Save every generation

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run optimization
    print(f"\n[3/3] Running NSGA-III optimization...")
    print(f"Output directory: {args.output_dir}")
    print("\nProgress will be shown below:\n")

    try:
        results = run_optimization(
            hp_space=config.hyperparameter_space,
            nsga_config=config.nsga3,
            eval_function=make_eval_fn(train_paths, train_labels, val_paths, val_labels),
            output_dir=args.output_dir,
            seed=42,
            verbose=True
        )

        # Display results
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE!")
        print("="*70)

        print(f"\nStatistics:")
        print(f"  Pareto solutions found: {len(results['pareto_configs'])}")
        print(f"  True evaluations: {results['n_true_evals']}")
        print(f"  Surrogate evaluations: {results['n_surrogate_evals']}")
        print(f"  Total evaluations: {results['n_true_evals'] + results['n_surrogate_evals']}")

        print(f"\nTop 3 solutions (by AUC):")

        # Sort by AUC (index 2)
        sorted_indices = sorted(
            range(len(results['pareto_F'])),
            key=lambda i: results['pareto_F'][i][2],
            reverse=True
        )

        for rank, idx in enumerate(sorted_indices[:3], 1):
            cfg = results['pareto_configs'][idx]
            obj = results['pareto_F'][idx]

            print(f"\n  Solution {rank}:")
            print(f"    Backbone: {cfg['backbone']}")
            print(f"    Optimizer: {cfg['optimizer']}, LR: {cfg['learning_rate']:.2e}")
            print(f"    Batch size: {cfg['batch_size']}")
            print(f"    Metrics:")
            print(f"      Sensitivity: {obj[0]:.4f}")
            print(f"      Specificity: {obj[1]:.4f}")
            print(f"      AUC: {obj[2]:.4f}")
            print(f"      Model Size: {obj[3]:.2f}M")

        print(f"\nResults saved to: {args.output_dir}/final_results.pkl")
        print(f"Checkpoints saved to: {args.output_dir}/checkpoints/")

        # Try to create simple visualization
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            F = results['pareto_F']

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Pareto Front (Mini Test)', fontsize=14)

            # Sensitivity vs Specificity
            axes[0].scatter(F[:, 0], F[:, 1], alpha=0.7, s=100)
            axes[0].set_xlabel('Sensitivity', fontsize=12)
            axes[0].set_ylabel('Specificity', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title('Diagnostic Performance')

            # AUC vs Model Size
            axes[1].scatter(F[:, 2], F[:, 3], alpha=0.7, s=100, c=F[:, 0], cmap='viridis')
            axes[1].set_xlabel('AUC', fontsize=12)
            axes[1].set_ylabel('Model Size (M parameters)', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_title('Accuracy vs Efficiency')
            cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
            cbar.set_label('Sensitivity', fontsize=10)

            plt.tight_layout()
            plot_path = f"{args.output_dir}/pareto_front_mini.png"
            plt.savefig(plot_path, dpi=150)
            print(f"\nVisualization saved to: {plot_path}")

        except ImportError:
            print("\nMatplotlib not available, skipping visualization")
        except Exception as e:
            print(f"\nCould not create visualization: {e}")

        print("\n" + "="*70)
        print("[OK] MINI OPTIMIZATION TEST PASSED!")
        print("="*70)

        print("\nNext steps:")
        print("  1. Review the results and Pareto front")
        print("  2. Increase settings for production runs:")
        print("     - pop_size: 50 (currently {})".format(args.pop_size))
        print("     - generations: 30 (currently {})".format(args.generations))
        print("     - epochs: 20-30 (currently {})".format(args.epochs))
        print("  3. Edit example_kaggle_dataset.py for full optimization")

    except Exception as e:
        print("\n" + "="*70)
        print("[X] ERROR DURING OPTIMIZATION")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  - Check that single config test works first")
        print("  - Reduce --pop_size and --generations for faster debugging")
        print("  - Check logs in output directory")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
