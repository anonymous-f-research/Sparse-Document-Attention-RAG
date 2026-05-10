"""
Visualization Utilities for Training and Evaluation

This module provides plotting functions for:
- Training data distribution
- Training metrics (loss, learning rate, perplexity)
- Evaluation comparisons
- Attention mask patterns
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_style("whitegrid")
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not found. Plots will use default matplotlib styling.")

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def get_color_palette(name: str, n: int):
    """Get color palette, using seaborn if available, otherwise matplotlib."""
    if HAS_SEABORN:
        return sns.color_palette(name, n)
    else:
        # Fallback to matplotlib default colors
        cmap = plt.cm.get_cmap('tab10' if n <= 10 else 'tab20')
        return [cmap(i / n) for i in range(n)]


def _resolve_x_values(values: List[float], x_values: Optional[List[float]]) -> List[float]:
    """Resolve x-axis values, falling back to a 1-based sequence."""
    if x_values is not None and len(x_values) == len(values):
        return x_values
    return list(range(1, len(values) + 1))


def _add_top_step_axis(ax, epoch_values: List[float], step_values: Optional[List[float]]):
    """Add a top axis showing global steps aligned to epoch ticks."""
    if not epoch_values or step_values is None:
        return
    if len(epoch_values) != len(step_values):
        return

    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xticks(epoch_values)
    top_ax.set_xticklabels([str(int(step)) for step in step_values], fontsize=9)
    top_ax.set_xlabel("Global Step", fontsize=11, fontweight='bold')


def _add_top_epoch_axis(ax, epoch_step_values: Optional[List[float]], epoch_values: Optional[List[float]]):
    """Add epoch markers and a top axis aligned to step-based plots."""
    if epoch_step_values is None or epoch_values is None:
        return
    if len(epoch_step_values) == 0 or len(epoch_values) == 0:
        return
    if len(epoch_step_values) != len(epoch_values):
        return

    for step in epoch_step_values:
        ax.axvline(step, color='gray', linestyle='--', linewidth=0.8, alpha=0.25)

    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xticks(epoch_step_values)
    top_ax.set_xticklabels([str(int(epoch)) for epoch in epoch_values], fontsize=9)
    top_ax.set_xlabel("Epoch", fontsize=11, fontweight='bold')


def plot_label_distribution(
    label_counts: Dict[str, int],
    title: str = "Label Distribution",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot bar chart of label distribution.

    Args:
        label_counts: Dict mapping labels to counts
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    colors = get_color_palette("husl", len(labels))

    bars = ax.bar(labels, counts, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig


def plot_training_metrics(
    metrics_history: Dict[str, List[float]],
    save_dir: Optional[str] = None,
) -> List[Figure]:
    """
    Plot training metrics over time.

    Args:
        metrics_history: Dict with keys like 'train_loss', 'val_loss', 'learning_rate'
        save_dir: Optional directory to save figures

    Returns:
        List of matplotlib figures
    """
    figures = []

    train_loss = metrics_history.get("train_loss", [])
    val_loss = metrics_history.get("val_loss", [])
    train_perplexity = metrics_history.get("train_perplexity", [])
    val_perplexity = metrics_history.get("val_perplexity", [])

    epoch_numbers = metrics_history.get("epoch_numbers")
    if not epoch_numbers:
        epoch_count = max(len(train_loss), len(train_perplexity))
        epoch_numbers = list(range(1, epoch_count + 1))
    epoch_end_steps = metrics_history.get("epoch_end_steps")

    val_epochs = metrics_history.get("val_epochs")
    val_step_indices = metrics_history.get("val_step_indices")

    # Plot 1: Training Loss by epoch (separate from validation)
    if train_loss:
        x_epochs = _resolve_x_values(train_loss, epoch_numbers)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_epochs, train_loss, linewidth=2, marker='o', markersize=4, alpha=0.85, color='tab:blue')
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title('Training Loss (Epoch)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        _add_top_step_axis(ax, x_epochs, _resolve_x_values(train_loss, epoch_end_steps) if epoch_end_steps else None)

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, 'train_loss_epochs.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved train loss plot to {save_path}")
            legacy_path = os.path.join(save_dir, 'loss_curve.png')
            fig.savefig(legacy_path, dpi=300, bbox_inches='tight')
            print(f"Saved train loss plot to {legacy_path}")

        figures.append(fig)

    # Plot 2: Validation Loss by epoch (separate from training)
    if val_loss:
        x_val_epochs = _resolve_x_values(val_loss, val_epochs)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_val_epochs, val_loss, linewidth=2, marker='s', markersize=4, alpha=0.85, color='tab:orange')
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title('Validation Loss (Epoch)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        _add_top_step_axis(ax, x_val_epochs, _resolve_x_values(val_loss, val_step_indices) if val_step_indices else None)

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, 'val_loss_epochs.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved validation loss plot to {save_path}")

        figures.append(fig)

    # Plot 3: Training Loss by global step (every step)
    train_loss_steps = metrics_history.get("train_loss_steps", [])
    train_loss_step_indices = metrics_history.get("train_loss_step_indices")
    if train_loss_steps:
        x_steps = _resolve_x_values(train_loss_steps, train_loss_step_indices)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_steps, train_loss_steps, linewidth=1.8, alpha=0.8, color='tab:blue')
        ax.set_xlabel('Global Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title('Training Loss (Every Step)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        _add_top_epoch_axis(ax, epoch_end_steps, epoch_numbers)

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, 'train_loss_steps.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved step-level train loss plot to {save_path}")

        figures.append(fig)

    # Plot 4: Learning Rate (step-level)
    learning_rate = metrics_history.get("learning_rate", [])
    learning_rate_steps = metrics_history.get("learning_rate_steps")
    if learning_rate:
        x_lr = _resolve_x_values(learning_rate, learning_rate_steps)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_lr, learning_rate, linewidth=2, color='green', alpha=0.8)
        ax.set_xlabel('Global Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax.set_title('Learning Rate Schedule (Every Step)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        if all(lr > 0 for lr in learning_rate):
            ax.set_yscale('log')
        else:
            ax.set_yscale('linear')
            ax.text(
                0.01,
                0.98,
                "Linear scale used (learning rate includes zero)",
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=9,
                color='dimgray',
            )
        _add_top_epoch_axis(ax, epoch_end_steps, epoch_numbers)

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, 'learning_rate.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved learning rate plot to {save_path}")

        figures.append(fig)

    # Plot 5: Gradient Norm (step-level)
    grad_norm = metrics_history.get("grad_norm", [])
    grad_norm_steps = metrics_history.get("grad_norm_steps")
    if grad_norm:
        x_grad = _resolve_x_values(grad_norm, grad_norm_steps)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_grad, grad_norm, linewidth=2, color='purple', alpha=0.8)
        ax.set_xlabel('Global Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gradient Norm', fontsize=12, fontweight='bold')
        ax.set_title('Gradient Norm (Every Step)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        _add_top_epoch_axis(ax, epoch_end_steps, epoch_numbers)

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, 'gradient_norm.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved gradient norm plot to {save_path}")

        figures.append(fig)

    # Plot 6: Training Perplexity by epoch (separate from validation)
    if train_perplexity:
        x_epochs = _resolve_x_values(train_perplexity, epoch_numbers)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_epochs, train_perplexity, linewidth=2, marker='o', markersize=4, alpha=0.85, color='tab:green')
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
        ax.set_title('Training Perplexity (Epoch)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        _add_top_step_axis(ax, x_epochs, _resolve_x_values(train_perplexity, epoch_end_steps) if epoch_end_steps else None)

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, 'train_perplexity_epochs.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved train perplexity plot to {save_path}")
            legacy_path = os.path.join(save_dir, 'perplexity.png')
            fig.savefig(legacy_path, dpi=300, bbox_inches='tight')
            print(f"Saved train perplexity plot to {legacy_path}")

        figures.append(fig)

    # Plot 7: Validation Perplexity by epoch (separate from training)
    if val_perplexity:
        x_val_epochs = _resolve_x_values(val_perplexity, val_epochs)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_val_epochs, val_perplexity, linewidth=2, marker='s', markersize=4, alpha=0.85, color='tab:red')
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
        ax.set_title('Validation Perplexity (Epoch)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        _add_top_step_axis(ax, x_val_epochs, _resolve_x_values(val_perplexity, val_step_indices) if val_step_indices else None)

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, 'val_perplexity_epochs.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved validation perplexity plot to {save_path}")

        figures.append(fig)

    return figures


def plot_accuracy_comparison(
    comparison_data: Dict[str, float],
    mode: str = "nli",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot bar chart comparing accuracy across different configurations.

    Args:
        comparison_data: Dict mapping config names to accuracy values
        mode: 'nli' or 'qa'
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = list(comparison_data.keys())
    accuracies = list(comparison_data.values())
    colors = get_color_palette("Set2", len(configs))

    bars = ax.bar(configs, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    metric_name = "Exact Match" if mode == "qa" else "Accuracy"
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} Comparison Across Configurations', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=15, ha='right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved accuracy comparison to {save_path}")

    return fig


def plot_perplexity_comparison(
    perplexity_data: Dict[str, float],
    title: str = "WikiText Perplexity Comparison Across Configurations",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot bar chart comparing perplexity across configurations.

    Args:
        perplexity_data: Dict mapping config names to perplexity values
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = list(perplexity_data.keys())
    perplexities = [float(perplexity_data[c]) for c in configs]
    colors = get_color_palette("Set2", len(configs))

    bars = ax.bar(configs, perplexities, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

    # Add exact values on bars
    for bar, value in zip(bars, perplexities):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.6f}",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
        )

    ax.set_ylabel("Perplexity", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    if perplexities:
        ax.set_ylim(0, max(perplexities) * 1.18)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved perplexity comparison to {save_path}")

    return fig


def plot_confusion_matrix(
    confusion_matrix: Dict[str, Dict[str, int]],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot confusion matrix heatmap.

    Args:
        confusion_matrix: Nested dict with true_label -> pred_label -> count
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Convert to numpy array
    labels = sorted(confusion_matrix.keys())
    matrix = np.zeros((len(labels), len(labels)))

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            matrix[i, j] = confusion_matrix[true_label].get(pred_label, 0)

    # Normalize by row (true labels)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(matrix, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot heatmap
    im = ax.imshow(normalized_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = int(matrix[i, j])
            percentage = normalized_matrix[i, j]
            text_color = "white" if percentage > 0.5 else "black"
            ax.text(j, i, f'{count}\n({percentage:.2f})',
                   ha="center", va="center", color=text_color, fontsize=10)

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Frequency', fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    return fig


def plot_f1_scores_comparison(
    f1_scores: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot grouped bar chart comparing F1 scores across configurations.

    Args:
        f1_scores: Dict mapping config name -> label -> f1_score
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    configs = list(f1_scores.keys())
    labels = list(f1_scores[configs[0]].keys())

    x = np.arange(len(labels))
    width = 0.8 / len(configs)
    colors = get_color_palette("Set1", len(configs))

    for i, config in enumerate(configs):
        scores = [f1_scores[config][label] for label in labels]
        offset = (i - len(configs) / 2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=config,
                     color=colors[i], alpha=0.8, edgecolor='black')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:  # Only show if bar is visible
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('F1 Scores Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved F1 scores comparison to {save_path}")

    return fig


def plot_attention_mask_heatmap(
    mask: np.ndarray,
    tokens: Optional[List[str]] = None,
    title: str = "Attention Mask",
    max_tokens: int = 60,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot attention mask as a heatmap.

    Args:
        mask: Attention mask array (seq_len, seq_len)
        tokens: Optional list of token strings
        title: Plot title
        max_tokens: Maximum tokens to display
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Truncate if needed
    n = min(mask.shape[0], max_tokens)
    mask_truncated = mask[:n, :n]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    im = ax.imshow(mask_truncated, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    if tokens:
        tokens_truncated = [t[:20] for t in tokens[:n]]  # Truncate long tokens
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(tokens_truncated, fontsize=6)
        ax.set_yticklabels(tokens_truncated, fontsize=6)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right")

    ax.set_xlabel('Key Position', fontsize=11, fontweight='bold')
    ax.set_ylabel('Query Position', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention (1=allowed, 0=blocked)', fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention mask heatmap to {save_path}")

    return fig


def plot_data_statistics(
    stats: Dict,
    save_dir: Optional[str] = None,
) -> List[Figure]:
    """
    Plot data statistics from the statistics dictionary.

    Args:
        stats: Statistics dictionary
        save_dir: Optional directory to save figures

    Returns:
        List of matplotlib figures
    """
    figures = []
    mode = stats.get("mode", "nli")

    # Plot 1: Label distribution (for NLI)
    if mode == "nli" and "train_label_distribution" in stats:
        fig = plot_label_distribution(
            stats["train_label_distribution"],
            title="Training Label Distribution",
            save_path=os.path.join(save_dir, "train_label_dist.png") if save_dir else None,
        )
        figures.append(fig)

        if "val_label_distribution" in stats:
            fig = plot_label_distribution(
                stats["val_label_distribution"],
                title="Validation Label Distribution",
                save_path=os.path.join(save_dir, "val_label_dist.png") if save_dir else None,
            )
            figures.append(fig)

    # Plot 2: Length distributions
    if "train_prompt_length" in stats or "train_answer_length" in stats:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if "train_prompt_length" in stats:
            train_lengths = stats["train_prompt_length"]
            axes[0].bar(['Mean', 'Min', 'Max'],
                       [train_lengths['mean'], train_lengths['min'], train_lengths['max']],
                       color=['skyblue', 'lightgreen', 'salmon'], alpha=0.8, edgecolor='black')
            axes[0].set_ylabel('Prompt Length (chars)', fontweight='bold')
            axes[0].set_title('Training Prompt Length Statistics', fontweight='bold')
            axes[0].grid(axis='y', alpha=0.3)

        if "val_prompt_length" in stats:
            val_lengths = stats["val_prompt_length"]
            axes[1].bar(['Mean', 'Min', 'Max'],
                       [val_lengths['mean'], val_lengths['min'], val_lengths['max']],
                       color=['skyblue', 'lightgreen', 'salmon'], alpha=0.8, edgecolor='black')
            axes[1].set_ylabel('Prompt Length (chars)', fontweight='bold')
            axes[1].set_title('Validation Prompt Length Statistics', fontweight='bold')
            axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, 'length_statistics.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved length statistics to {save_path}")

        figures.append(fig)

    return figures


def create_training_summary_plot(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create a comprehensive 2x2 summary plot of training metrics.

    Args:
        metrics_history: Dict with training metrics
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Summary', fontsize=16, fontweight='bold')

    train_loss = metrics_history.get("train_loss", [])
    val_loss = metrics_history.get("val_loss", [])
    train_perplexity = metrics_history.get("train_perplexity", [])
    val_perplexity = metrics_history.get("val_perplexity", [])
    epoch_numbers = metrics_history.get("epoch_numbers")
    if not epoch_numbers:
        epoch_count = max(len(train_loss), len(train_perplexity))
        epoch_numbers = list(range(1, epoch_count + 1))
    epoch_end_steps = metrics_history.get("epoch_end_steps")
    val_epochs = metrics_history.get("val_epochs")

    # Plot 1: Loss
    if train_loss:
        x_train_epochs = _resolve_x_values(train_loss, epoch_numbers)
        axes[0, 0].plot(x_train_epochs, train_loss, label='Train', linewidth=2, marker='o', markersize=3)
    if val_loss:
        x_val_epochs = _resolve_x_values(val_loss, val_epochs)
        axes[0, 0].plot(x_val_epochs, val_loss, label='Validation', linewidth=2, marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch', fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontweight='bold')
    axes[0, 0].set_title('Loss Over Time', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    if train_loss:
        _add_top_step_axis(
            axes[0, 0],
            _resolve_x_values(train_loss, epoch_numbers),
            _resolve_x_values(train_loss, epoch_end_steps) if epoch_end_steps else None,
        )

    # Plot 2: Learning Rate
    learning_rate = metrics_history.get("learning_rate", [])
    learning_rate_steps = metrics_history.get("learning_rate_steps")
    if learning_rate:
        x_lr = _resolve_x_values(learning_rate, learning_rate_steps)
        axes[0, 1].plot(x_lr, learning_rate, linewidth=2, color='green')
        axes[0, 1].set_xlabel('Step', fontweight='bold')
        axes[0, 1].set_ylabel('Learning Rate', fontweight='bold')
        axes[0, 1].set_title('Learning Rate Schedule', fontweight='bold')
        if all(lr > 0 for lr in learning_rate):
            axes[0, 1].set_yscale('log')
        else:
            axes[0, 1].set_yscale('linear')
        axes[0, 1].grid(alpha=0.3)
        _add_top_epoch_axis(axes[0, 1], epoch_end_steps, epoch_numbers)

    # Plot 3: Perplexity
    if train_perplexity:
        x_train_epochs = _resolve_x_values(train_perplexity, epoch_numbers)
        axes[1, 0].plot(x_train_epochs, train_perplexity, label='Train', linewidth=2, marker='o', markersize=3)
    if val_perplexity:
        x_val_epochs = _resolve_x_values(val_perplexity, val_epochs)
        axes[1, 0].plot(x_val_epochs, val_perplexity, label='Validation', linewidth=2, marker='s', markersize=3)
    axes[1, 0].set_xlabel('Epoch', fontweight='bold')
    axes[1, 0].set_ylabel('Perplexity', fontweight='bold')
    axes[1, 0].set_title('Perplexity Over Time', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    if train_perplexity:
        _add_top_step_axis(
            axes[1, 0],
            _resolve_x_values(train_perplexity, epoch_numbers),
            _resolve_x_values(train_perplexity, epoch_end_steps) if epoch_end_steps else None,
        )

    # Plot 4: Gradient Norm (if available)
    grad_norm = metrics_history.get("grad_norm", [])
    grad_norm_steps = metrics_history.get("grad_norm_steps")
    if grad_norm:
        x_grad = _resolve_x_values(grad_norm, grad_norm_steps)
        axes[1, 1].plot(x_grad, grad_norm, linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Step', fontweight='bold')
        axes[1, 1].set_ylabel('Gradient Norm', fontweight='bold')
        axes[1, 1].set_title('Gradient Norm Over Time', fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        _add_top_epoch_axis(axes[1, 1], epoch_end_steps, epoch_numbers)
    else:
        # If no gradient norm, show training samples processed
        if train_loss:
            steps = range(1, len(train_loss) + 1)
            axes[1, 1].plot(steps, linewidth=2, color='orange')
            axes[1, 1].set_xlabel('Step', fontweight='bold')
            axes[1, 1].set_ylabel('Epoch', fontweight='bold')
            axes[1, 1].set_title('Training Progress', fontweight='bold')
            axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training summary to {save_path}")

    return fig
