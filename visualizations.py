"""
Visualization Module for NSGA-III Mammography HPO Research

Generates publication-quality figures for thesis/paper:
1. Pareto Front Visualizations (2D, 3D, parallel coordinates)
2. Convergence Plots
3. Hyperparameter Analysis
4. Objective Trade-off Analysis
5. Transfer Learning Comparison
6. Surrogate Model Performance

Author: David
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pickle
import os

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Objective names for labeling
OBJECTIVE_NAMES = ['Sensitivity', 'Specificity', 'AUC', 'Model Size (M)', 'Inference Time (ms)']
OBJECTIVE_SHORT = ['Sens', 'Spec', 'AUC', 'Size', 'Time']


def load_results(results_path: str) -> Dict:
    """Load optimization results from pickle file."""
    with open(results_path, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# 1. PARETO FRONT VISUALIZATIONS
# =============================================================================

def plot_pareto_2d(
    pareto_F: np.ndarray,
    obj_x: int = 2,  # AUC
    obj_y: int = 0,  # Sensitivity
    highlight_idx: List[int] = None,
    title: str = "Pareto Front",
    save_path: str = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    2D Pareto front scatter plot.
    
    Args:
        pareto_F: Objective values array (n_solutions, n_objectives)
        obj_x: Index of x-axis objective
        obj_y: Index of y-axis objective
        highlight_idx: Indices of solutions to highlight
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all points
    scatter = ax.scatter(
        pareto_F[:, obj_x], 
        pareto_F[:, obj_y],
        c=pareto_F[:, 3],  # Color by model size
        cmap='viridis_r',
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Model Size (M)', fontsize=12)
    
    # Highlight specific solutions
    if highlight_idx:
        ax.scatter(
            pareto_F[highlight_idx, obj_x],
            pareto_F[highlight_idx, obj_y],
            s=200,
            facecolors='none',
            edgecolors='red',
            linewidths=2,
            label='Selected Solutions'
        )
        # Add labels
        for idx in highlight_idx:
            ax.annotate(
                f'S{idx}',
                (pareto_F[idx, obj_x], pareto_F[idx, obj_y]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold'
            )
    
    ax.set_xlabel(OBJECTIVE_NAMES[obj_x])
    ax.set_ylabel(OBJECTIVE_NAMES[obj_y])
    ax.set_title(title)
    
    if highlight_idx:
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_pareto_3d(
    pareto_F: np.ndarray,
    obj_x: int = 0,  # Sensitivity
    obj_y: int = 1,  # Specificity
    obj_z: int = 2,  # AUC
    highlight_idx: List[int] = None,
    title: str = "3D Pareto Front",
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """3D Pareto front visualization."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all points
    scatter = ax.scatter(
        pareto_F[:, obj_x],
        pareto_F[:, obj_y],
        pareto_F[:, obj_z],
        c=pareto_F[:, 3],  # Color by model size
        cmap='viridis_r',
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Highlight specific solutions
    if highlight_idx:
        ax.scatter(
            pareto_F[highlight_idx, obj_x],
            pareto_F[highlight_idx, obj_y],
            pareto_F[highlight_idx, obj_z],
            s=200,
            facecolors='none',
            edgecolors='red',
            linewidths=2
        )
    
    ax.set_xlabel(OBJECTIVE_NAMES[obj_x])
    ax.set_ylabel(OBJECTIVE_NAMES[obj_y])
    ax.set_zlabel(OBJECTIVE_NAMES[obj_z])
    ax.set_title(title)
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Model Size (M)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_parallel_coordinates(
    pareto_F: np.ndarray,
    pareto_configs: List[Dict] = None,
    highlight_idx: List[int] = None,
    title: str = "Pareto Solutions - Parallel Coordinates",
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Parallel coordinates plot showing all objectives.
    Each line represents one Pareto-optimal solution.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_solutions, n_objectives = pareto_F.shape
    
    # Normalize objectives to [0, 1] for visualization
    F_norm = np.zeros_like(pareto_F)
    for i in range(n_objectives):
        min_val, max_val = pareto_F[:, i].min(), pareto_F[:, i].max()
        if max_val > min_val:
            F_norm[:, i] = (pareto_F[:, i] - min_val) / (max_val - min_val)
        else:
            F_norm[:, i] = 0.5
    
    # For minimization objectives (size, time), invert so "up" is always better
    F_norm[:, 3] = 1 - F_norm[:, 3]  # Model size
    F_norm[:, 4] = 1 - F_norm[:, 4]  # Inference time
    
    x = np.arange(n_objectives)
    
    # Color by AUC
    colors = plt.cm.RdYlGn(F_norm[:, 2])
    
    # Plot each solution
    for i in range(n_solutions):
        linewidth = 3 if highlight_idx and i in highlight_idx else 1
        alpha = 1.0 if highlight_idx and i in highlight_idx else 0.5
        ax.plot(x, F_norm[i], c=colors[i], linewidth=linewidth, alpha=alpha)
    
    # Highlight selected solutions with markers
    if highlight_idx:
        for idx in highlight_idx:
            ax.plot(x, F_norm[idx], 'o-', markersize=8, linewidth=2, 
                   label=f'S{idx} (AUC={pareto_F[idx, 2]:.3f})')
    
    # Set axis labels
    ax.set_xticks(x)
    labels = [f'{OBJECTIVE_SHORT[i]}\n({pareto_F[:, i].min():.2f}-{pareto_F[:, i].max():.2f})' 
              for i in range(n_objectives)]
    ax.set_xticklabels(labels)
    ax.set_ylabel('Normalized Value (higher = better)')
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    
    if highlight_idx:
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add note about normalization
    ax.text(0.02, 0.02, 'Note: Size and Time inverted so higher = better', 
            transform=ax.transAxes, fontsize=9, style='italic')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_pairwise_objectives(
    pareto_F: np.ndarray,
    highlight_idx: List[int] = None,
    title: str = "Pairwise Objective Trade-offs",
    save_path: str = None,
    figsize: Tuple[int, int] = (14, 12)
):
    """Pairwise scatter plot matrix of all objectives."""
    n_obj = pareto_F.shape[1]
    
    fig, axes = plt.subplots(n_obj, n_obj, figsize=figsize)
    
    for i in range(n_obj):
        for j in range(n_obj):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                ax.hist(pareto_F[:, i], bins=10, color='steelblue', alpha=0.7, edgecolor='black')
                ax.set_ylabel('Count' if j == 0 else '')
            else:
                # Off-diagonal: scatter
                ax.scatter(pareto_F[:, j], pareto_F[:, i], 
                          c='steelblue', alpha=0.6, s=50, edgecolors='black', linewidths=0.3)
                
                if highlight_idx:
                    ax.scatter(pareto_F[highlight_idx, j], pareto_F[highlight_idx, i],
                              c='red', s=100, marker='*', edgecolors='black', linewidths=0.5)
            
            # Labels
            if i == n_obj - 1:
                ax.set_xlabel(OBJECTIVE_SHORT[j])
            if j == 0:
                ax.set_ylabel(OBJECTIVE_SHORT[i])
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# 2. CONVERGENCE PLOTS
# =============================================================================

def plot_convergence(
    history: Dict,
    title: str = "Optimization Convergence",
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 4)
):
    """
    Plot convergence metrics over generations.
    
    Args:
        history: Dict with 'generation', 'best_auc', 'pareto_size'
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    generations = history['generation']
    
    # Best AUC over generations
    ax1 = axes[0]
    ax1.plot(generations, history['best_auc'], 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best AUC')
    ax1.set_title('Best AUC Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Pareto front size
    ax2 = axes[1]
    ax2.plot(generations, history['pareto_size'], 'g-s', linewidth=2, markersize=6)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Pareto Front Size')
    ax2.set_title('Pareto Front Evolution')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_evaluation_efficiency(
    n_true_evals: int,
    n_surrogate_evals: int,
    n_generations: int,
    title: str = "Surrogate-Assisted Evaluation Efficiency",
    save_path: str = None,
    figsize: Tuple[int, int] = (8, 5)
):
    """Pie/bar chart showing true vs surrogate evaluations."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Pie chart
    ax1 = axes[0]
    sizes = [n_true_evals, n_surrogate_evals]
    labels = [f'True Evaluations\n({n_true_evals})', f'Surrogate Predictions\n({n_surrogate_evals})']
    colors = ['#2ecc71', '#3498db']
    explode = (0.05, 0)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title('Evaluation Distribution')
    
    # Bar chart - evaluations per generation
    ax2 = axes[1]
    total = n_true_evals + n_surrogate_evals
    pop_size = total // n_generations
    
    gens = np.arange(1, n_generations + 1)
    # Estimate: Gen 1 all true, then ~30% true
    true_per_gen = [pop_size] + [int(pop_size * 0.3)] * (n_generations - 1)
    surr_per_gen = [0] + [pop_size - int(pop_size * 0.3)] * (n_generations - 1)
    
    ax2.bar(gens, true_per_gen, label='True', color='#2ecc71', alpha=0.8)
    ax2.bar(gens, surr_per_gen, bottom=true_per_gen, label='Surrogate', color='#3498db', alpha=0.8)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Evaluations')
    ax2.set_title('Evaluations per Generation')
    ax2.legend()
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# 3. HYPERPARAMETER ANALYSIS
# =============================================================================

def plot_hyperparameter_distribution(
    pareto_configs: List[Dict],
    pareto_F: np.ndarray,
    params_to_plot: List[str] = None,
    title: str = "Hyperparameter Distribution in Pareto Solutions",
    save_path: str = None,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Visualize distribution of hyperparameters in Pareto-optimal solutions.
    """
    if params_to_plot is None:
        params_to_plot = ['backbone', 'optimizer', 'loss_function', 'learning_rate', 
                         'dropout_rate', 'unfreeze_strategy', 'batch_size', 'focal_gamma']
    
    # Filter to params that exist
    params_to_plot = [p for p in params_to_plot if p in pareto_configs[0]]
    
    n_params = len(params_to_plot)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_params > 1 else [axes]
    
    for idx, param in enumerate(params_to_plot):
        ax = axes[idx]
        values = [cfg[param] for cfg in pareto_configs]
        aucs = pareto_F[:, 2]
        
        if isinstance(values[0], (int, float)) and not isinstance(values[0], bool):
            # Continuous: scatter with AUC color
            scatter = ax.scatter(range(len(values)), values, c=aucs, cmap='RdYlGn', 
                               s=80, edgecolors='black', linewidths=0.5)
            ax.set_ylabel(param)
            plt.colorbar(scatter, ax=ax, label='AUC')
        else:
            # Categorical: bar chart
            unique_vals = list(set(values))
            counts = [values.count(v) for v in unique_vals]
            colors = plt.cm.Set2(np.linspace(0, 1, len(unique_vals)))
            bars = ax.bar(range(len(unique_vals)), counts, color=colors, edgecolor='black')
            ax.set_xticks(range(len(unique_vals)))
            ax.set_xticklabels([str(v)[:10] for v in unique_vals], rotation=45, ha='right')
            ax.set_ylabel('Count')
        
        ax.set_title(param)
    
    # Hide empty subplots
    for idx in range(len(params_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_backbone_comparison(
    pareto_configs: List[Dict],
    pareto_F: np.ndarray,
    title: str = "Backbone Architecture Comparison",
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 5)
):
    """Compare performance across backbone architectures."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Group by backbone
    backbones = list(set(cfg['backbone'] for cfg in pareto_configs))
    backbone_data = {b: [] for b in backbones}
    
    for cfg, obj in zip(pareto_configs, pareto_F):
        backbone_data[cfg['backbone']].append(obj)
    
    metrics = ['AUC', 'Sensitivity', 'Specificity']
    metric_idx = [2, 0, 1]
    
    for ax, metric, m_idx in zip(axes, metrics, metric_idx):
        data = []
        labels = []
        for b in backbones:
            if backbone_data[b]:
                values = [obj[m_idx] for obj in backbone_data[b]]
                data.append(values)
                labels.append(b.replace('_', '\n'))
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Backbone')
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# 4. TRADE-OFF ANALYSIS
# =============================================================================

def plot_sensitivity_specificity_tradeoff(
    pareto_F: np.ndarray,
    highlight_idx: List[int] = None,
    title: str = "Sensitivity-Specificity Trade-off",
    save_path: str = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Sensitivity vs Specificity plot with medical context.
    Shows the clinical trade-off between catching cancers vs false alarms.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sens = pareto_F[:, 0]
    spec = pareto_F[:, 1]
    auc = pareto_F[:, 2]
    
    # Scatter plot colored by AUC
    scatter = ax.scatter(spec, sens, c=auc, cmap='RdYlGn', s=100, 
                        edgecolors='black', linewidths=0.5, alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('AUC')
    
    # Highlight selected solutions
    if highlight_idx:
        ax.scatter(spec[highlight_idx], sens[highlight_idx], s=200,
                  facecolors='none', edgecolors='red', linewidths=2)
        for idx in highlight_idx:
            ax.annotate(f'S{idx}', (spec[idx], sens[idx]), 
                       xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    # Add reference regions
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Min acceptable sensitivity')
    ax.axvline(x=0.7, color='blue', linestyle='--', alpha=0.5, label='Min acceptable specificity')
    
    # Shade ideal region
    ax.fill_between([0.7, 1.0], 0.8, 1.0, alpha=0.1, color='green', label='Ideal region')
    
    ax.set_xlabel('Specificity (True Negative Rate)')
    ax.set_ylabel('Sensitivity (True Positive Rate)')
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower left')
    
    # Add medical context annotation
    ax.text(0.95, 0.05, 'High Sensitivity:\nFewer missed cancers\nMore false alarms', 
            transform=ax.transAxes, ha='right', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_efficiency_vs_performance(
    pareto_F: np.ndarray,
    highlight_idx: List[int] = None,
    title: str = "Model Efficiency vs Performance",
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 5)
):
    """AUC vs Model Size and Inference Time."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    auc = pareto_F[:, 2]
    size = pareto_F[:, 3]
    time = pareto_F[:, 4]
    
    # AUC vs Size
    ax1 = axes[0]
    scatter1 = ax1.scatter(size, auc, c=pareto_F[:, 0], cmap='coolwarm', 
                          s=100, edgecolors='black', linewidths=0.5)
    if highlight_idx:
        ax1.scatter(size[highlight_idx], auc[highlight_idx], s=200,
                   facecolors='none', edgecolors='red', linewidths=2)
    ax1.set_xlabel('Model Size (M parameters)')
    ax1.set_ylabel('AUC')
    ax1.set_title('AUC vs Model Size')
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Sensitivity')
    
    # AUC vs Time
    ax2 = axes[1]
    scatter2 = ax2.scatter(time, auc, c=pareto_F[:, 1], cmap='coolwarm',
                          s=100, edgecolors='black', linewidths=0.5)
    if highlight_idx:
        ax2.scatter(time[highlight_idx], auc[highlight_idx], s=200,
                   facecolors='none', edgecolors='red', linewidths=2)
    ax2.set_xlabel('Inference Time (ms)')
    ax2.set_ylabel('AUC')
    ax2.set_title('AUC vs Inference Time')
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Specificity')
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# 5. TRANSFER LEARNING COMPARISON
# =============================================================================

def plot_transfer_learning_comparison(
    vindr_results: np.ndarray,
    inbreast_results: Dict[int, Dict],
    solution_labels: List[str] = None,
    title: str = "Transfer Learning: VinDr → INbreast",
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Compare performance on source (VinDr) vs target (INbreast) dataset.
    
    Args:
        vindr_results: Pareto front from VinDr optimization
        inbreast_results: Dict mapping solution idx to INbreast metrics
        solution_labels: Optional labels for solutions
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    indices = list(inbreast_results.keys())
    if solution_labels is None:
        solution_labels = [f'S{i}' for i in indices]
    
    metrics = ['Sensitivity', 'Specificity', 'AUC']
    metric_idx = [0, 1, 2]
    
    x = np.arange(len(indices))
    width = 0.35
    
    for ax, metric, m_idx in zip(axes, metrics, metric_idx):
        vindr_vals = [vindr_results[i, m_idx] for i in indices]
        inbreast_vals = [inbreast_results[i][metric.lower()] for i in indices]
        
        bars1 = ax.bar(x - width/2, vindr_vals, width, label='VinDr', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, inbreast_vals, width, label='INbreast', color='coral', alpha=0.8)
        
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(solution_labels, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Add value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{bar.get_height():.2f}', ha='center', fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{bar.get_height():.2f}', ha='center', fontsize=9)
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_generalization_scatter(
    vindr_auc: List[float],
    inbreast_auc: List[float],
    labels: List[str] = None,
    title: str = "Hyperparameter Generalization",
    save_path: str = None,
    figsize: Tuple[int, int] = (7, 6)
):
    """Scatter plot comparing AUC on both datasets."""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(vindr_auc, inbreast_auc, s=100, c='steelblue', edgecolors='black', linewidths=0.5)
    
    # Add diagonal line (perfect transfer)
    lims = [min(min(vindr_auc), min(inbreast_auc)) - 0.05,
            max(max(vindr_auc), max(inbreast_auc)) + 0.05]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Transfer')
    
    # Add labels
    if labels:
        for i, label in enumerate(labels):
            ax.annotate(label, (vindr_auc[i], inbreast_auc[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('VinDr AUC')
    ax.set_ylabel('INbreast AUC')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Add correlation
    corr = np.corrcoef(vindr_auc, inbreast_auc)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# 6. SUMMARY TABLE
# =============================================================================

def create_pareto_summary_table(
    pareto_F: np.ndarray,
    pareto_configs: List[Dict],
    save_path: str = None
) -> pd.DataFrame:
    """Create a formatted summary table of Pareto solutions."""
    
    rows = []
    for i, (obj, cfg) in enumerate(zip(pareto_F, pareto_configs)):
        rows.append({
            'Solution': f'S{i}',
            'Backbone': cfg['backbone'],
            'LR': f"{cfg['learning_rate']:.1e}",
            'Optimizer': cfg['optimizer'],
            'Loss': cfg['loss_function'],
            'Sensitivity': f"{obj[0]:.4f}",
            'Specificity': f"{obj[1]:.4f}",
            'AUC': f"{obj[2]:.4f}",
            'Size (M)': f"{obj[3]:.2f}",
            'Time (ms)': f"{obj[4]:.1f}"
        })
    
    df = pd.DataFrame(rows)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")
    
    return df


# =============================================================================
# 7. GENERATE ALL FIGURES
# =============================================================================

def generate_all_thesis_figures(
    results_path: str,
    output_dir: str,
    highlight_solutions: List[int] = None
):
    """
    Generate all thesis figures from optimization results.
    
    Args:
        results_path: Path to final_results.pkl
        output_dir: Directory to save figures
        highlight_solutions: Indices of solutions to highlight (e.g., [5, 8, 13])
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results = load_results(results_path)
    pareto_F = results['pareto_F']
    pareto_configs = results['pareto_configs']
    history = results['history']
    
    if highlight_solutions is None:
        # Auto-select: best AUC, best sensitivity, best balanced
        best_auc_idx = np.argmax(pareto_F[:, 2])
        best_sens_idx = np.argmax(pareto_F[:, 0])
        # Best balanced: highest sum of sens + spec with AUC > median
        median_auc = np.median(pareto_F[:, 2])
        balanced_scores = pareto_F[:, 0] + pareto_F[:, 1]
        balanced_scores[pareto_F[:, 2] < median_auc] = -1
        best_balanced_idx = np.argmax(balanced_scores)
        highlight_solutions = list(set([best_auc_idx, best_sens_idx, best_balanced_idx]))
    
    print(f"Generating figures with highlighted solutions: {highlight_solutions}")
    print(f"Output directory: {output_dir}\n")
    
    # 1. Pareto Front 2D
    plot_pareto_2d(pareto_F, obj_x=2, obj_y=0, highlight_idx=highlight_solutions,
                   title="Pareto Front: AUC vs Sensitivity",
                   save_path=os.path.join(output_dir, "fig_pareto_2d_auc_sens.png"))
    
    plot_pareto_2d(pareto_F, obj_x=0, obj_y=1, highlight_idx=highlight_solutions,
                   title="Pareto Front: Sensitivity vs Specificity",
                   save_path=os.path.join(output_dir, "fig_pareto_2d_sens_spec.png"))
    
    # 2. Pareto Front 3D
    plot_pareto_3d(pareto_F, highlight_idx=highlight_solutions,
                   title="3D Pareto Front: Sensitivity, Specificity, AUC",
                   save_path=os.path.join(output_dir, "fig_pareto_3d.png"))
    
    # 3. Parallel Coordinates
    plot_parallel_coordinates(pareto_F, pareto_configs, highlight_idx=highlight_solutions,
                             save_path=os.path.join(output_dir, "fig_parallel_coordinates.png"))
    
    # 4. Pairwise Objectives
    plot_pairwise_objectives(pareto_F, highlight_idx=highlight_solutions,
                            save_path=os.path.join(output_dir, "fig_pairwise_objectives.png"))
    
    # 5. Convergence
    plot_convergence(history, save_path=os.path.join(output_dir, "fig_convergence.png"))
    
    # 6. Evaluation Efficiency
    plot_evaluation_efficiency(
        results['n_true_evals'], results['n_surrogate_evals'],
        len(history['generation']),
        save_path=os.path.join(output_dir, "fig_evaluation_efficiency.png")
    )
    
    # 7. Hyperparameter Distribution
    plot_hyperparameter_distribution(pareto_configs, pareto_F,
                                    save_path=os.path.join(output_dir, "fig_hyperparameter_dist.png"))
    
    # 8. Backbone Comparison
    plot_backbone_comparison(pareto_configs, pareto_F,
                            save_path=os.path.join(output_dir, "fig_backbone_comparison.png"))
    
    # 9. Sensitivity-Specificity Trade-off
    plot_sensitivity_specificity_tradeoff(pareto_F, highlight_idx=highlight_solutions,
                                         save_path=os.path.join(output_dir, "fig_sens_spec_tradeoff.png"))
    
    # 10. Efficiency vs Performance
    plot_efficiency_vs_performance(pareto_F, highlight_idx=highlight_solutions,
                                  save_path=os.path.join(output_dir, "fig_efficiency_performance.png"))
    
    # 11. Summary Table
    df = create_pareto_summary_table(pareto_F, pareto_configs,
                                    save_path=os.path.join(output_dir, "table_pareto_summary.csv"))
    print("\nPareto Summary Table:")
    print(df.to_string(index=False))
    
    print(f"\n{'='*60}")
    print(f"All figures saved to: {output_dir}")
    print(f"{'='*60}")
    
    return results


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Visualization module for NSGA-III Mammography HPO")
    print("\nUsage in Colab:")
    print("""
    from visualizations import generate_all_thesis_figures
    
    # Generate all figures
    results = generate_all_thesis_figures(
        results_path="/content/drive/MyDrive/nsga3_outputs/final_results.pkl",
        output_dir="/content/drive/MyDrive/nsga3_outputs/figures",
        highlight_solutions=[5, 8, 13]  # Your best solutions
    )
    """)
