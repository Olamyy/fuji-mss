import json
import sys
from pathlib import Path

from tools.enhanced_analysis import enhanced_audio_analysis


def classify_audio_v2(metrics):
    """
    Enhanced classifier based on discriminative analysis.
    """
    score = 0

    # high_freq_ratio (effect size: 0.82)
    if metrics['high_freq_ratio'] < 0.104012:
        score += 2  # Weight: 2

    # spectral_rolloff_mean (effect size: 0.76)
    if metrics['spectral_rolloff_mean'] < 4591.042845:
        score += 2  # Weight: 2

    # spectral_contrast_std (effect size: 0.73)
    if metrics['spectral_contrast_std'] < 14.140853:
        score += 2  # Weight: 2

    # spectral_centroid_mean (effect size: 0.71)
    if metrics['spectral_centroid_mean'] < 2274.122939:
        score += 2  # Weight: 2

    # spectral_centroid_std (effect size: 0.70)
    if metrics['spectral_centroid_std'] < 853.532278:
        score += 2  # Weight: 2

    # spectral_bandwidth_mean (effect size: 0.70)
    if metrics['spectral_bandwidth_mean'] < 2651.900872:
        score += 2  # Weight: 2

    # mid_freq_ratio (effect size: 0.68)
    if metrics['mid_freq_ratio'] > 4.562974:
        score += 1  # Weight: 1

    # Classify as dirty if score exceeds threshold
    # Max possible score: 13
    return 'dirty' if score >= 6 else 'clean'


def load_manual_labels(labels_file="audio_labels.json"):
    """Load manual labels from JSON file."""
    if not Path(labels_file).exists():
        return None

    with open(labels_file, 'r') as f:
        labels = json.load(f)

    return labels


def find_label_for_file(audio_file, labels):
    """
    Find the manual label for a given audio file.
    Handles different path formats and normalizes them.
    """
    audio_path = str(Path(audio_file).resolve())

    # Try exact match first
    if audio_path in labels:
        return labels[audio_path]['label']

    # Try matching by filename only
    audio_filename = Path(audio_file).name
    for path, data in labels.items():
        if Path(path).name == audio_filename:
            return data['label']

    return None


def print_metrics_detail(metrics, predicted, actual=None):
    """Print detailed metric information."""
    print(f"\n{'=' * 70}")
    print("METRIC DETAILS")
    print(f"{'=' * 70}")

    thresholds = {
        'noise_floor_absolute': 0.075134,
        'noise_floor_ratio': 0.412846,
        'clipping_rate': 0.000912,
        'spectral_flatness': 0.002445,
        'dynamic_range': 229722487.677820,
        'high_freq_ratio': 0.104012
    }

    print(f"{'Metric':<25} {'Value':>15} {'Threshold':>15} {'Status':>12}")
    print(f"{'-' * 70}")

    for metric_name, threshold in thresholds.items():
        value = metrics[metric_name]

        # Determine if this metric indicates dirty
        if metric_name == 'dynamic_range':
            is_dirty_indicator = value > threshold
        else:
            is_dirty_indicator = value > threshold

        status = "🚩 DIRTY" if is_dirty_indicator else "✓ Clean"

        print(f"{metric_name:<25} {value:>15.6f} {threshold:>15.6f} {status:>12}")

    print(f"\n{'Predicted:':<15} {predicted.upper()}")
    if actual:
        match = "✓" if predicted == actual else "✗"
        print(f"{'Actual:':<15} {actual.upper()} {match}")


def validate_single_file(audio_file, labels_file="audio_labels.json", verbose=False):
    """
    Classify a single audio file and compare with manual label if available.
    """
    from tools.noise_floor import analyze_noise_floor

    print(f"\n{'=' * 70}")
    print(f"Processing: {Path(audio_file).name}")
    print(f"{'=' * 70}")

    # Analyze audio
    metrics = enhanced_audio_analysis(audio_file)

    if metrics is None:
        print("✗ Could not analyze audio file.")
        return None

    # Classify
    predicted = classify_audio_v2(metrics)

    # Load manual labels
    labels = load_manual_labels(labels_file)

    if labels is None:
        print(f"\n⚠ No manual labels file found at: {labels_file}")
        print(f"Predicted classification: {predicted.upper()}")
        if verbose:
            print_metrics_detail(metrics, predicted)
        return {'predicted': predicted, 'actual': None, 'match': None}

    # Find actual label
    actual = find_label_for_file(audio_file, labels)

    if actual is None:
        print(f"\n⚠ No manual label found for this file")
        print(f"Predicted classification: {predicted.upper()}")
        if verbose:
            print_metrics_detail(metrics, predicted)
        return {'predicted': predicted, 'actual': None, 'match': None}

    # # Compare
    # match = predicted == actual
    #
    # print(f"\nPredicted: {predicted.upper()}")
    # print(f"Actual:    {actual.upper()}")
    # print(f"Result:    {'✓ MATCH' if match else '✗ MISMATCH'}")
    # print(f"Dirty indicators triggered: {dirty_count}/6")
    #
    # if verbose or not match:
    #     print_metrics_detail(metrics, predicted, actual)
    #
    # return {
    #     'predicted': predicted,
    #     'actual': actual,
    #     'match': match,
    #     'dirty_count': dirty_count,
    #     'metrics': metrics
    # }


def validate_batch(labels_file="audio_labels.json", show_mismatches_only=False):
    """
    Validate classifier against all manually labeled files.
    """

    labels = load_manual_labels(labels_file)

    if labels is None:
        print(f"✗ Could not load labels from {labels_file}")
        return

    print(f"\n{'=' * 70}")
    print(f"BATCH VALIDATION")
    print(f"{'=' * 70}")
    print(f"Total labeled files: {len(labels)}")
    print(f"Starting validation...\n")

    results = {
        'correct': [],
        'incorrect': [],
        'failed': []
    }

    for i, (file_path, label_data) in enumerate(labels.items(), 1):
        actual = label_data['label']
        filename = Path(file_path).name

        print(f"[{i}/{len(labels)}] {filename}...", end=" ")

        # Analyze
        metrics = enhanced_audio_analysis(file_path)

        if metrics is None:
            print("✗ FAILED")
            results['failed'].append({
                'file': file_path,
                'actual': actual
            })
            continue

        # Classify
        predicted = classify_audio_v2(metrics)
        match = predicted == actual

        result = {
            'file': file_path,
            'filename': filename,
            'predicted': predicted,
            'actual': actual,
            # 'dirty_count': dirty_count,
            'metrics': metrics
        }

        if match:
            print("✓ MATCH")
            results['correct'].append(result)
        else:
            print(f"✗ MISMATCH (predicted: {predicted}, actual: {actual})")
            results['incorrect'].append(result)

    # Print summary
    total = len(labels)
    correct = len(results['correct'])
    incorrect = len(results['incorrect'])
    failed = len(results['failed'])
    accuracy = (correct / (correct + incorrect) * 100) if (correct + incorrect) > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total files:      {total}")
    print(f"Correct:          {correct} ({correct / total * 100:.1f}%)")
    print(f"Incorrect:        {incorrect} ({incorrect / total * 100:.1f}%)")
    print(f"Failed to analyze: {failed} ({failed / total * 100:.1f}%)")
    print(f"\nAccuracy:         {accuracy:.1f}%")

    # Show confusion matrix
    if correct + incorrect > 0:
        true_positives = sum(1 for r in results['correct'] if r['actual'] == 'dirty')
        true_negatives = sum(1 for r in results['correct'] if r['actual'] == 'clean')
        false_positives = sum(1 for r in results['incorrect'] if r['predicted'] == 'dirty')
        false_negatives = sum(1 for r in results['incorrect'] if r['predicted'] == 'clean')

        print(f"\n{'=' * 70}")
        print(f"CONFUSION MATRIX")
        print(f"{'=' * 70}")
        print(f"                  Predicted Clean    Predicted Dirty")
        print(f"Actual Clean      {true_negatives:>15}    {false_positives:>15}")
        print(f"Actual Dirty      {false_negatives:>15}    {true_positives:>15}")

    # Show mismatches in detail
    if results['incorrect']:
        print(f"\n{'=' * 70}")
        print(f"MISMATCHES ({len(results['incorrect'])} files)")
        print(f"{'=' * 70}")

        for result in results['incorrect']:
            print(f"\n{result['filename']}")
            print(f"  Predicted: {result['predicted'].upper()}")
            print(f"  Actual:    {result['actual'].upper()}")
            print(f"  Dirty indicators: {result['dirty_count']}/6")
            print(f"  Noise floor: {result['metrics']['noise_floor_absolute']:.6f}")
            print(f"  Spectral flatness: {result['metrics']['spectral_flatness']:.6f}")

    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file:  python classifier.py <audio_file> [labels.json]")
        print("  Batch mode:   python classifier.py --batch [labels.json]")
        print("  Verbose:      python classifier.py <audio_file> [labels.json] --verbose")
        sys.exit(1)

    if sys.argv[1] == "--batch":
        # Batch validation mode
        labels_file = sys.argv[2] if len(sys.argv) > 2 else "audio_labels.json"
        validate_batch(labels_file)
    else:
        # Single file mode
        audio_file = sys.argv[1]
        labels_file = sys.argv[2] if len(sys.argv) > 2 else "audio_labels.json"
        verbose = "--verbose" in sys.argv or "-v" in sys.argv

        validate_single_file(audio_file, labels_file, verbose)