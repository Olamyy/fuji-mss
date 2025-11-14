import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def analyze_noise_floor(file_path, chunk_size_sec=1.0, percentile=10):
    """
    Analyzes the noise floor of an audio file.
    Returns a dict with multiple quality metrics.
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

    # 1. Noise floor (your original metric)
    rms_values = librosa.feature.rms(y=y, frame_length=chunk_size_samples,
                                     hop_length=chunk_size_samples)[0]

    if len(rms_values) == 0:
        return None

    quiet_threshold = np.percentile(rms_values, percentile)
    quiet_chunks = rms_values[rms_values <= quiet_threshold]
    noise_floor_absolute = np.mean(quiet_chunks)

    # 2. Normalized noise floor (ratio to overall level)
    overall_rms = np.sqrt(np.mean(y ** 2))
    noise_floor_ratio = noise_floor_absolute / (overall_rms + 1e-10)

    # 3. Clipping detection
    clipping_rate = np.sum(np.abs(y) > 0.99) / len(y)

    # 4. Spectral flatness (higher = more noise-like)
    spec_flat = np.mean(librosa.feature.spectral_flatness(y=y))

    # 5. Dynamic range
    dynamic_range = np.max(rms_values) / (np.min(rms_values) + 1e-10)

    # 6. High frequency noise (energy above 8kHz)
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    high_freq_mask = freqs > 8000
    high_freq_energy = np.mean(stft[high_freq_mask, :])
    total_energy = np.mean(stft)
    hf_ratio = high_freq_energy / (total_energy + 1e-10)

    return {
        'file': str(file_path),
        'noise_floor_absolute': float(noise_floor_absolute),
        'noise_floor_ratio': float(noise_floor_ratio),
        'clipping_rate': float(clipping_rate),
        'spectral_flatness': float(spec_flat),
        'dynamic_range': float(dynamic_range),
        'high_freq_ratio': float(hf_ratio)
    }


def calibrate_threshold(audio_dir, output_json="calibration_results.json"):
    """
    Analyze all audio files in a directory and generate calibration data.

    Args:
        audio_dir: Path to directory containing audio files
        output_json: Path to save results
    """
    audio_dir = Path(audio_dir)

    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
    audio_files = [f for f in audio_dir.rglob('*') if f.suffix.lower() in audio_extensions]

    print(f"Found {len(audio_files)} audio files")
    print("Analyzing... (this may take a while)\n")

    results = []
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        metrics = analyze_noise_floor(audio_file)
        if metrics:
            results.append(metrics)

    if not results:
        print("\nNo valid audio files processed!")
        return

    # Save raw results
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_json}")

    # Generate statistics and visualizations
    generate_report(results)

    return results


def generate_report(results):
    """
    Generate statistical report and visualizations.
    """
    metrics = ['noise_floor_absolute', 'noise_floor_ratio', 'clipping_rate',
               'spectral_flatness', 'dynamic_range', 'high_freq_ratio']

    print("\n" + "=" * 70)
    print("CALIBRATION REPORT")
    print("=" * 70)

    for metric in metrics:
        values = [r[metric] for r in results]

        print(f"\n{metric.upper().replace('_', ' ')}:")
        print(f"  Min:    {np.min(values):.6f}")
        print(f"  10th:   {np.percentile(values, 10):.6f}")
        print(f"  25th:   {np.percentile(values, 25):.6f}")
        print(f"  Median: {np.median(values):.6f}")
        print(f"  75th:   {np.percentile(values, 75):.6f}")
        print(f"  90th:   {np.percentile(values, 90):.6f}")
        print(f"  Max:    {np.max(values):.6f}")

    print("\n" + "=" * 70)
    print("RECOMMENDED THRESHOLDS (adjust based on your manual inspection):")
    print("=" * 70)

    print(f"\nnoise_floor_absolute > {np.percentile([r['noise_floor_absolute'] for r in results], 75):.6f}")
    print(f"noise_floor_ratio > {np.percentile([r['noise_floor_ratio'] for r in results], 75):.6f}")
    print(f"clipping_rate > {np.percentile([r['clipping_rate'] for r in results], 90):.6f}")
    print(f"spectral_flatness > {np.percentile([r['spectral_flatness'] for r in results], 75):.6f}")
    print(f"dynamic_range < {np.percentile([r['dynamic_range'] for r in results], 25):.6f}")
    print(f"high_freq_ratio > {np.percentile([r['high_freq_ratio'] for r in results], 75):.6f}")

    create_visualizations(results, metrics)


def create_visualizations(results, metrics):
    """
    Create histograms for each metric.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Audio Quality Metrics Distribution', fontsize=16)

    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]

        ax = axes[i]
        ax.hist(values, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.median(values), color='red', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(values):.4f}')
        ax.axvline(np.percentile(values, 75), color='orange', linestyle='--',
                   linewidth=2, label=f'75th: {np.percentile(values, 75):.4f}')
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quality_metrics_distribution.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to quality_metrics_distribution.png")
    plt.close()


def manual_inspection_helper(results_json="calibration_results.json", n_samples=10):
    """
    Help manually inspect edge cases to set better thresholds.
    """
    with open(results_json, 'r') as f:
        results = json.load(f)

    # Sort by noise floor
    results.sort(key=lambda x: x['noise_floor_absolute'])

    print("\n" + "=" * 70)
    print("MANUAL INSPECTION HELPER")
    print("=" * 70)

    print(f"\n{n_samples} CLEANEST files (lowest noise floor):")
    print("-" * 70)
    for r in results[:n_samples]:
        print(f"{Path(r['file']).name:<50} | Noise: {r['noise_floor_absolute']:.6f}")

    print(f"\n{n_samples} DIRTIEST files (highest noise floor):")
    print("-" * 70)
    for r in results[-n_samples:]:
        print(f"{Path(r['file']).name:<50} | Noise: {r['noise_floor_absolute']:.6f}")

    print("\n" + "=" * 70)
    print("INSTRUCTIONS:")
    print("=" * 70)
    print("1. Listen to a few songs from each category")
    print("2. Find the noise floor value that best separates clean from dirty")
    print("3. Update your threshold accordingly")
    print("\nTIP: The threshold should be between the 'cleanest dirty' and 'dirtiest clean' songs")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python calibrate.py <audio_directory>")
        print("  python calibrate.py inspect <calibration_results.json>")
        sys.exit(1)

    if sys.argv[1] == "inspect":
        json_file = sys.argv[2] if len(sys.argv) > 2 else "calibration_results.json"
        manual_inspection_helper(json_file)
    else:
        audio_dir = sys.argv[1]
        results = calibrate_threshold(audio_dir)

        if results:
            print("\n" + "=" * 70)
            print("NEXT STEPS:")
            print("=" * 70)
            print("1. Review the visualizations and statistics above")
            print("2. Run: python calibrate.py inspect calibration_results.json")
            print("3. Listen to the cleanest and dirtiest files")
            print("4. Choose your threshold based on where clean/dirty separate")
            print("=" * 70)