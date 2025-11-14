import librosa
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy import stats


def enhanced_audio_analysis(file_path, chunk_size_sec=1.0, percentile=10):
    """
    Comprehensive audio quality analysis with expanded metrics.
    """
    try:
        y, sr = librosa.load(file_path, sr=None, duration=300)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    chunk_size_samples = int(chunk_size_sec * sr)

    if len(y) < chunk_size_samples:
        print(f"Skipping {file_path}: too short")
        return None

    # === ORIGINAL METRICS ===

    # 1. Noise floor
    rms_values = librosa.feature.rms(y=y, frame_length=chunk_size_samples,
                                     hop_length=chunk_size_samples)[0]

    if len(rms_values) == 0:
        return None

    quiet_threshold = np.percentile(rms_values, percentile)
    quiet_chunks = rms_values[rms_values <= quiet_threshold]
    noise_floor_absolute = np.mean(quiet_chunks)

    # 2. Normalized noise floor
    overall_rms = np.sqrt(np.mean(y ** 2))
    noise_floor_ratio = noise_floor_absolute / (overall_rms + 1e-10)

    # 3. Clipping detection
    clipping_rate = np.sum(np.abs(y) > 0.99) / len(y)

    # 4. Spectral flatness
    spec_flat = np.mean(librosa.feature.spectral_flatness(y=y))

    # 5. Dynamic range
    dynamic_range = np.max(rms_values) / (np.min(rms_values) + 1e-10)

    # 6. High frequency noise
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    high_freq_mask = freqs > 8000
    high_freq_energy = np.mean(stft[high_freq_mask, :])
    total_energy = np.mean(stft)
    hf_ratio = high_freq_energy / (total_energy + 1e-10)

    # === NEW METRICS ===

    # 7. Crest Factor (peak-to-RMS ratio)
    # Lower values suggest over-compression
    peak = np.max(np.abs(y))
    crest_factor = peak / (overall_rms + 1e-10)

    # 8. Zero Crossing Rate variance
    # High variance can indicate distortion
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_variance = np.var(zcr)
    zcr_mean = np.mean(zcr)

    # 9. Spectral Contrast
    # Measures difference between peaks and valleys in spectrum
    # Lower values might indicate muddy/unclear audio
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(contrast)
    spectral_contrast_std = np.std(contrast)

    # 10. Spectral Rolloff variance
    # Measures consistency of frequency content
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_rolloff_mean = np.mean(rolloff)
    spectral_rolloff_std = np.std(rolloff)

    # 11. Spectral Centroid variance
    # Indicates brightness consistency
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_centroid_mean = np.mean(centroid)
    spectral_centroid_std = np.std(centroid)

    # 12. RMS Energy variance
    # High variance might indicate inconsistent levels
    rms_std = np.std(rms_values)
    rms_coefficient_of_variation = rms_std / (np.mean(rms_values) + 1e-10)

    # 13. Spectral Bandwidth
    # Wider bandwidth can indicate richer/cleaner sound
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_bandwidth_mean = np.mean(bandwidth)
    spectral_bandwidth_std = np.std(bandwidth)

    # 14. Low frequency energy
    # Check for rumble or poor low-end
    low_freq_mask = freqs < 100
    low_freq_energy = np.mean(stft[low_freq_mask, :])
    lf_ratio = low_freq_energy / (total_energy + 1e-10)

    # 15. Mid frequency clarity
    mid_freq_mask = (freqs > 300) & (freqs < 3000)
    mid_freq_energy = np.mean(stft[mid_freq_mask, :])
    mf_ratio = mid_freq_energy / (total_energy + 1e-10)

    # 16. Temporal irregularity
    # Sudden changes might indicate artifacts
    rms_diff = np.abs(np.diff(rms_values))
    temporal_irregularity = np.mean(rms_diff)

    # 17. Silent frames ratio
    # Too many silent frames might indicate dropout issues
    silent_threshold = overall_rms * 0.01
    silent_frames = np.sum(rms_values < silent_threshold) / len(rms_values)

    # 18. Peak consistency
    # Check if peaks are consistent or erratic
    peak_frames = rms_values > (np.mean(rms_values) + np.std(rms_values))
    peak_ratio = np.sum(peak_frames) / len(rms_values)

    return {
        'file': str(file_path),
        # Original metrics
        'noise_floor_absolute': float(noise_floor_absolute),
        'noise_floor_ratio': float(noise_floor_ratio),
        'clipping_rate': float(clipping_rate),
        'spectral_flatness': float(spec_flat),
        'dynamic_range': float(dynamic_range),
        'high_freq_ratio': float(hf_ratio),
        # New metrics
        'crest_factor': float(crest_factor),
        'zcr_mean': float(zcr_mean),
        'zcr_variance': float(zcr_variance),
        'spectral_contrast_mean': float(spectral_contrast_mean),
        'spectral_contrast_std': float(spectral_contrast_std),
        'spectral_rolloff_mean': float(spectral_rolloff_mean),
        'spectral_rolloff_std': float(spectral_rolloff_std),
        'spectral_centroid_mean': float(spectral_centroid_mean),
        'spectral_centroid_std': float(spectral_centroid_std),
        'rms_std': float(rms_std),
        'rms_coefficient_of_variation': float(rms_coefficient_of_variation),
        'spectral_bandwidth_mean': float(spectral_bandwidth_mean),
        'spectral_bandwidth_std': float(spectral_bandwidth_std),
        'low_freq_ratio': float(lf_ratio),
        'mid_freq_ratio': float(mf_ratio),
        'temporal_irregularity': float(temporal_irregularity),
        'silent_frames_ratio': float(silent_frames),
        'peak_ratio': float(peak_ratio)
    }


def analyze_labeled_corpus(labels_file="audio_labels.json", output_file="enhanced_analysis.json"):
    """
    Analyze all labeled files with enhanced metrics.
    """
    with open(labels_file, 'r') as f:
        labels = json.load(f)

    print(f"\n{'=' * 70}")
    print(f"ENHANCED AUDIO ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Analyzing {len(labels)} labeled files...\n")

    results = []
    for i, (file_path, label_data) in enumerate(labels.items(), 1):
        print(f"[{i}/{len(labels)}] {Path(file_path).name}...", end=" ")

        metrics = enhanced_audio_analysis(file_path)

        if metrics:
            metrics['label'] = label_data['label']
            results.append(metrics)
            print("✓")
        else:
            print("✗")

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Enhanced analysis saved to {output_file}")

    return results


def compare_clean_vs_dirty(results):
    """
    Statistical comparison of all metrics between clean and dirty.
    """
    # Group by label
    clean_metrics = defaultdict(list)
    dirty_metrics = defaultdict(list)

    for result in results:
        label = result['label']
        for key, value in result.items():
            if key not in ['file', 'label']:
                if label == 'clean':
                    clean_metrics[key].append(value)
                else:
                    dirty_metrics[key].append(value)

    # Calculate statistics and discriminative power
    comparisons = []

    print(f"\n{'=' * 70}")
    print(f"STATISTICAL COMPARISON: CLEAN vs DIRTY")
    print(f"{'=' * 70}\n")

    for metric_name in clean_metrics.keys():
        clean_vals = np.array(clean_metrics[metric_name])
        dirty_vals = np.array(dirty_metrics[metric_name])

        # Basic statistics
        clean_mean = np.mean(clean_vals)
        dirty_mean = np.mean(dirty_vals)
        clean_std = np.std(clean_vals)
        dirty_std = np.std(dirty_vals)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((clean_std ** 2 + dirty_std ** 2) / 2)
        cohens_d = abs(dirty_mean - clean_mean) / (pooled_std + 1e-10)

        # T-test
        t_stat, p_value = stats.ttest_ind(clean_vals, dirty_vals)

        # Percent difference
        pct_diff = ((dirty_mean - clean_mean) / (abs(clean_mean) + 1e-10)) * 100

        comparisons.append({
            'metric': metric_name,
            'clean_mean': clean_mean,
            'dirty_mean': dirty_mean,
            'clean_std': clean_std,
            'dirty_std': dirty_std,
            'cohens_d': cohens_d,
            'p_value': p_value,
            'pct_diff': pct_diff,
            'direction': 'higher' if dirty_mean > clean_mean else 'lower'
        })

    # Sort by effect size (discriminative power)
    comparisons.sort(key=lambda x: x['cohens_d'], reverse=True)

    # Print top discriminators
    print("TOP DISCRIMINATING METRICS (sorted by effect size):")
    print(f"{'=' * 70}\n")
    print(f"{'Metric':<30} {'Effect Size':>12} {'P-value':>10} {'Direction':>15}")
    print(f"{'-' * 70}")

    for comp in comparisons[:15]:  # Top 15
        significance = "***" if comp['p_value'] < 0.001 else "**" if comp['p_value'] < 0.01 else "*" if comp[
                                                                                                            'p_value'] < 0.05 else ""
        print(
            f"{comp['metric']:<30} {comp['cohens_d']:>12.3f} {comp['p_value']:>10.4f}{significance:>3} {'Dirty ' + comp['direction']:>15}")

    print("\nEffect Size Interpretation:")
    print("  > 0.8 = Large effect (excellent discriminator)")
    print("  0.5-0.8 = Medium effect (good discriminator)")
    print("  0.2-0.5 = Small effect (weak discriminator)")
    print("  < 0.2 = Negligible effect (poor discriminator)")

    print("\nP-value: * p<0.05, ** p<0.01, *** p<0.001")

    # Detailed comparison of top metrics
    print(f"\n{'=' * 70}")
    print("DETAILED STATISTICS FOR TOP 5 METRICS")
    print(f"{'=' * 70}\n")

    for comp in comparisons[:5]:
        print(f"{comp['metric'].upper().replace('_', ' ')}:")
        print(f"  Clean:  mean={comp['clean_mean']:.6f}, std={comp['clean_std']:.6f}")
        print(f"  Dirty:  mean={comp['dirty_mean']:.6f}, std={comp['dirty_std']:.6f}")
        print(f"  Difference: {comp['pct_diff']:+.1f}% (dirty is {comp['direction']})")
        print(f"  Effect size: {comp['cohens_d']:.3f}")
        print(f"  Statistical significance: p={comp['p_value']:.4f}")
        print()

    return comparisons


def visualize_top_discriminators(results, comparisons, top_n=8, output_file="top_discriminators.png"):
    """
    Visualize the top discriminating metrics.
    """
    # Get top metrics
    top_metrics = [comp['metric'] for comp in comparisons[:top_n]]

    # Organize data
    clean_data = defaultdict(list)
    dirty_data = defaultdict(list)

    for result in results:
        for metric in top_metrics:
            if metric in result:
                if result['label'] == 'clean':
                    clean_data[metric].append(result[metric])
                else:
                    dirty_data[metric].append(result[metric])

    # Create plots
    n_rows = (top_n + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4 * n_rows))
    fig.suptitle('Top Discriminating Metrics: Clean vs Dirty', fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for i, metric in enumerate(top_metrics):
        ax = axes[i]

        clean_vals = clean_data[metric]
        dirty_vals = dirty_data[metric]

        # Box plot
        bp = ax.boxplot([clean_vals, dirty_vals],
                        labels=['Clean', 'Dirty'],
                        patch_artist=True,
                        showmeans=True)

        # Color boxes
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')

        # Add effect size to title
        effect_size = comparisons[i]['cohens_d']
        p_val = comparisons[i]['p_value']
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        ax.set_title(f"{metric.replace('_', ' ').title()}\n(Effect: {effect_size:.2f}{sig})",
                     fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add mean lines
        ax.plot([1], [np.mean(clean_vals)], 'D', color='darkgreen',
                markersize=8, label='Mean', zorder=3)
        ax.plot([2], [np.mean(dirty_vals)], 'D', color='darkred',
                markersize=8, zorder=3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_file}")
    plt.close()


def suggest_new_classifier(comparisons, threshold_percentile=50):
    """
    Suggest a new classifier based on the best discriminating metrics.
    """
    print(f"\n{'=' * 70}")
    print("SUGGESTED NEW CLASSIFIER")
    print(f"{'=' * 70}\n")

    # Select metrics with large effect sizes (> 0.5) and significant p-values
    good_metrics = [c for c in comparisons if c['cohens_d'] > 0.5 and c['p_value'] < 0.05]

    if not good_metrics:
        print("⚠ No metrics with strong discriminative power found!")
        good_metrics = comparisons[:5]  # Use top 5 anyway

    print(f"Using {len(good_metrics)} metrics with strong discriminative power:\n")

    print("def classify_audio_v2(metrics):")
    print('    """')
    print('    Enhanced classifier based on discriminative analysis.')
    print('    """')
    print("    score = 0")
    print()

    for i, comp in enumerate(good_metrics[:8], 1):  # Use top 8
        metric = comp['metric']

        # Calculate threshold (midpoint between means)
        threshold = (comp['clean_mean'] + comp['dirty_mean']) / 2

        # Determine weight based on effect size
        if comp['cohens_d'] > 1.0:
            weight = 3  # Strong discriminator
        elif comp['cohens_d'] > 0.7:
            weight = 2  # Good discriminator
        else:
            weight = 1  # Moderate discriminator

        operator = '>' if comp['direction'] == 'higher' else '<'

        print(f"    # {metric} (effect size: {comp['cohens_d']:.2f})")
        print(f"    if metrics['{metric}'] {operator} {threshold:.6f}:")
        print(f"        score += {weight}  # Weight: {weight}")
        print()

    # Suggest threshold
    max_score = sum(3 if c['cohens_d'] > 1.0 else 2 if c['cohens_d'] > 0.7 else 1
                    for c in good_metrics[:8])
    suggested_threshold = max_score // 2

    print(f"    # Classify as dirty if score exceeds threshold")
    print(f"    # Max possible score: {max_score}")
    print(f"    return 'dirty' if score >= {suggested_threshold} else 'clean'")
    print()


def main(labels_file="audio_labels.json"):
    """
    Complete enhanced analysis workflow.
    """
    # Step 1: Analyze all labeled files
    results = analyze_labeled_corpus(labels_file, "enhanced_analysis.json")

    if len(results) < 10:
        print("\n⚠ Not enough valid results for analysis!")
        return

    # Step 2: Statistical comparison
    comparisons = compare_clean_vs_dirty(results)

    # Step 3: Visualize top discriminators
    visualize_top_discriminators(results, comparisons)

    # Step 4: Suggest new classifier
    suggest_new_classifier(comparisons)

    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
    print("\nNext steps:")
    print("1. Review the top discriminating metrics above")
    print("2. Check the visualization: top_discriminators.png")
    print("3. Copy the suggested classifier code")
    print("4. Test it with: python classifier.py --batch audio_labels.json")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python enhanced_analysis.py <audio_labels.json>")
        sys.exit(1)

    labels_file = sys.argv[1]
    main(labels_file)