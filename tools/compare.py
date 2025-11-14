import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_data(labels_file="audio_labels.json", calibration_file="calibration_results.json"):
    """Load both manual labels and automated metrics."""
    with open(labels_file, 'r') as f:
        labels = json.load(f)

    with open(calibration_file, 'r') as f:
        calibration = json.load(f)

    # Create a lookup dict for calibration results by file path
    calibration_lookup = {result['file']: result for result in calibration}

    return labels, calibration_lookup


def analyze_by_label(labels, calibration_lookup):
    """
    Group metrics by manual label (clean/dirty) and calculate statistics.
    """
    metrics_by_label = {
        'clean': defaultdict(list),
        'dirty': defaultdict(list)
    }

    matched_count = 0
    unmatched_files = []

    # Iterate through manual labels
    for file_path, label_data in labels.items():
        manual_label = label_data['label']

        # Find corresponding calibration result
        if file_path in calibration_lookup:
            matched_count += 1
            metrics = calibration_lookup[file_path]

            # Group all metrics by label
            for metric_name, value in metrics.items():
                if metric_name != 'file':  # Skip the filename field
                    metrics_by_label[manual_label][metric_name].append(value)
        else:
            unmatched_files.append(file_path)

    print(f"\n{'=' * 70}")
    print(f"DATA MATCHING")
    print(f"{'=' * 70}")
    print(f"Manual labels: {len(labels)}")
    print(f"Matched with calibration: {matched_count}")
    print(f"Unmatched: {len(unmatched_files)}")
    if unmatched_files:
        print(f"\nFirst few unmatched files:")
        for f in unmatched_files[:5]:
            print(f"  {Path(f).name}")

    return metrics_by_label


def calculate_statistics(metrics_by_label):
    """
    Calculate mean, median, std for each metric by label.
    """
    stats = {}

    for label in ['clean', 'dirty']:
        stats[label] = {}
        for metric_name, values in metrics_by_label[label].items():
            if values:  # Only if we have data
                stats[label][metric_name] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }

    return stats


def print_comparison_report(stats):
    """
    Print a detailed comparison report.
    """
    print(f"\n{'=' * 70}")
    print(f"METRIC COMPARISON: CLEAN vs DIRTY")
    print(f"{'=' * 70}")

    # Get all metric names
    if not stats.get('clean') or not stats.get('dirty'):
        print("Not enough data for comparison!")
        return

    metric_names = list(stats['clean'].keys())

    for metric_name in metric_names:
        clean_stats = stats['clean'].get(metric_name, {})
        dirty_stats = stats['dirty'].get(metric_name, {})

        if not clean_stats or not dirty_stats:
            continue

        print(f"\n{metric_name.upper().replace('_', ' ')}:")
        print(f"  {'':20} {'CLEAN':>15} {'DIRTY':>15} {'Difference':>15}")
        print(f"  {'-' * 65}")

        clean_mean = clean_stats['mean']
        dirty_mean = dirty_stats['mean']
        diff = dirty_mean - clean_mean
        diff_pct = (diff / clean_mean * 100) if clean_mean != 0 else 0

        print(f"  {'Mean':20} {clean_mean:>15.6f} {dirty_mean:>15.6f} {diff:>+15.6f}")
        print(f"  {'Median':20} {clean_stats['median']:>15.6f} {dirty_stats['median']:>15.6f}")
        print(f"  {'Std Dev':20} {clean_stats['std']:>15.6f} {dirty_stats['std']:>15.6f}")
        print(f"  {'Range':20} [{clean_stats['min']:.4f}, {clean_stats['max']:.4f}]")
        print(f"  {'':20} [{dirty_stats['min']:.4f}, {dirty_stats['max']:.4f}]")
        print(f"  {'Sample Size':20} {clean_stats['count']:>15} {dirty_stats['count']:>15}")

        if abs(diff_pct) > 0:
            print(f"  → Dirty is {abs(diff_pct):.1f}% {'higher' if diff > 0 else 'lower'} than clean")


def suggest_thresholds(stats):
    """
    Suggest optimal thresholds based on the separation between clean and dirty.
    """
    print(f"\n{'=' * 70}")
    print(f"SUGGESTED THRESHOLDS")
    print(f"{'=' * 70}")
    print("(Values that best separate clean from dirty)\n")

    if not stats.get('clean') or not stats.get('dirty'):
        print("Not enough data for threshold suggestions!")
        return {}

    thresholds = {}

    for metric_name in stats['clean'].keys():
        clean_stats = stats['clean'].get(metric_name, {})
        dirty_stats = stats['dirty'].get(metric_name, {})

        if not clean_stats or not dirty_stats:
            continue

        clean_mean = clean_stats['mean']
        dirty_mean = dirty_stats['mean']

        # Threshold halfway between the means
        threshold = (clean_mean + dirty_mean) / 2

        # Calculate separation (how well this metric distinguishes)
        clean_std = clean_stats['std']
        dirty_std = dirty_stats['std']
        pooled_std = np.sqrt((clean_std ** 2 + dirty_std ** 2) / 2)
        separation = abs(dirty_mean - clean_mean) / (pooled_std + 1e-10)

        thresholds[metric_name] = threshold

        # Higher is bad for most metrics
        if dirty_mean > clean_mean:
            print(f"{metric_name:30} > {threshold:.6f}  (separation: {separation:.2f})")
        else:
            print(f"{metric_name:30} < {threshold:.6f}  (separation: {separation:.2f})")

    print("\nSeparation score interpretation:")
    print("  > 2.0 = Excellent discriminator")
    print("  1.0-2.0 = Good discriminator")
    print("  0.5-1.0 = Moderate discriminator")
    print("  < 0.5 = Poor discriminator")

    return thresholds


def create_comparison_plots(metrics_by_label, output_file="metric_comparison.png"):
    """
    Create side-by-side box plots comparing clean vs dirty for each metric.
    """
    metric_names = list(metrics_by_label['clean'].keys())
    n_metrics = len(metric_names)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Clean vs Dirty: Metric Distributions', fontsize=16)

    axes = axes.flatten()

    for i, metric_name in enumerate(metric_names):
        if i >= len(axes):
            break

        ax = axes[i]

        clean_values = metrics_by_label['clean'][metric_name]
        dirty_values = metrics_by_label['dirty'][metric_name]

        # Box plot
        bp = ax.boxplot([clean_values, dirty_values],
                        labels=['Clean', 'Dirty'],
                        patch_artist=True)

        # Color the boxes
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')

        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)

        # Add mean markers
        ax.plot([1], [np.mean(clean_values)], 'D', color='darkgreen',
                markersize=8, label='Mean')
        ax.plot([2], [np.mean(dirty_values)], 'D', color='darkred', markersize=8)

    # Remove empty subplots
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_file}")
    plt.close()


def generate_classifier_code(thresholds):
    """
    Generate ready-to-use classifier code based on the thresholds.
    """
    print(f"\n{'=' * 70}")
    print(f"GENERATED CLASSIFIER CODE")
    print(f"{'=' * 70}\n")

    print("def classify_audio(metrics):")
    print('    """')
    print('    Classify audio as clean or dirty based on calibrated thresholds.')
    print('    Generated from manual labeling comparison.')
    print('    """')
    print("    dirty_indicators = 0")
    print()

    for metric_name, threshold in thresholds.items():
        print(f"    if metrics['{metric_name}'] > {threshold:.6f}:")
        print(f"        dirty_indicators += 1")
        print()

    print("    # Consider dirty if 2 or more indicators triggered")
    print("    return 'dirty' if dirty_indicators >= 2 else 'clean'")
    print()
    print("\n# Usage:")
    print("# from noise_floor import analyze_noise_floor")
    print("# metrics = analyze_noise_floor('song.mp3')")
    print("# result = classify_audio(metrics)")


def main(labels_file="audio_labels.json", calibration_file="calibration_results.json"):
    """
    Main comparison function.
    """
    # Load data
    labels, calibration_lookup = load_data(labels_file, calibration_file)

    # Group metrics by label
    metrics_by_label = analyze_by_label(labels, calibration_lookup)

    # Calculate statistics
    stats = calculate_statistics(metrics_by_label)

    # Print comparison report
    print_comparison_report(stats)

    # Suggest thresholds
    thresholds = suggest_thresholds(stats)

    # Create visualizations
    create_comparison_plots(metrics_by_label)

    # Generate classifier code
    generate_classifier_code(thresholds)

    return stats, thresholds


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python compare_results.py <audio_labels.json> <calibration_results.json>")
        sys.exit(1)

    labels_file = sys.argv[1]
    calibration_file = sys.argv[2]

    main(labels_file, calibration_file)