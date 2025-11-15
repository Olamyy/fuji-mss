import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import json
import argparse


class FujiAudioProfiler:
    """Profile audio quality and characteristics of Fuji music dataset"""

    def __init__(self, sr: int = 22050):
        self.sr = sr

    def analyze_audio_quality(self, audio_path: str) -> Dict:
        """
        Analyze audio quality metrics
        Returns dict with quality indicators
        """
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)

        metrics = {}

        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        noise_floor = np.percentile(rms, 10)
        signal_level = np.percentile(rms, 90)
        snr_estimate = 20 * np.log10(signal_level / (noise_floor + 1e-10))
        metrics['snr_estimate_db'] = float(snr_estimate)

        clipping_threshold = 0.99
        clipped_samples = np.sum(np.abs(y) > clipping_threshold)
        metrics['clipping_ratio'] = float(clipped_samples / len(y))

        metrics['dynamic_range_db'] = float(20 * np.log10(np.max(np.abs(y)) / (np.mean(np.abs(y)) + 1e-10)))

        stft = librosa.stft(y)
        magnitude = np.abs(stft)

        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        metrics['spectral_centroid_mean'] = float(np.mean(spectral_centroid))

        # High frequency content (above 8kHz) - lower in low-quality recordings
        nyquist = sr / 2
        freq_bins = librosa.fft_frequencies(sr=sr)
        hf_mask = freq_bins > 8000
        hf_energy = np.mean(magnitude[hf_mask, :])
        total_energy = np.mean(magnitude)
        metrics['high_freq_ratio'] = float(hf_energy / (total_energy + 1e-10))

        # 5. Compression artifact detection (using spectral flatness)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        metrics['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
        metrics['spectral_flatness_std'] = float(np.std(spectral_flatness))

        # 6. Duration
        metrics['duration_seconds'] = float(len(y) / sr)

        return metrics

    def analyze_rhythm(self, audio_path: str) -> Dict:
        """
        Analyze rhythmic characteristics
        """
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)

        rhythm_metrics = {}

        # 1. Tempo estimation
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        rhythm_metrics['tempo_bpm'] = float(tempo)
        rhythm_metrics['num_beats'] = len(beats)

        # 2. Beat strength/consistency
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        rhythm_metrics['onset_strength_mean'] = float(np.mean(onset_env))
        rhythm_metrics['onset_strength_std'] = float(np.std(onset_env))

        # 3. Rhythmic regularity (using tempogram)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        rhythm_metrics['tempogram_std'] = float(np.std(tempogram))

        # 4. Percussive vs harmonic content
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = np.sum(y_harmonic ** 2)
        percussive_energy = np.sum(y_percussive ** 2)
        total_energy = harmonic_energy + percussive_energy
        rhythm_metrics['percussive_ratio'] = float(percussive_energy / (total_energy + 1e-10))

        return rhythm_metrics

    def detect_live_recording(self, audio_path: str) -> Dict:
        """
        Heuristics to detect if recording is live vs studio
        """
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)

        live_indicators = {}

        # 1. Background noise level (higher in live)
        rms = librosa.feature.rms(y=y)[0]
        noise_consistency = np.std(rms[rms < np.percentile(rms, 20)])
        live_indicators['noise_consistency'] = float(noise_consistency)

        # 2. Spectral spread (wider in live due to crowd, reverb)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        live_indicators['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))

        # 3. Reverb estimation (more in live venues)
        # Simple reverb estimate using autocorrelation
        autocorr = np.correlate(y, y, mode='full')[len(y):]
        autocorr = autocorr[:int(sr * 0.5)]  # First 0.5 seconds
        reverb_estimate = np.max(autocorr[int(sr * 0.05):]) / (autocorr[0] + 1e-10)
        live_indicators['reverb_estimate'] = float(reverb_estimate)

        # 4. Crowd noise detection (energy in 200-500 Hz range during "quiet" moments)
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        freq_bins = librosa.fft_frequencies(sr=sr)
        crowd_freq_mask = (freq_bins >= 200) & (freq_bins <= 500)

        # Look at quiet moments
        quiet_frames = rms < np.percentile(rms, 30)
        if np.sum(quiet_frames) > 0:
            crowd_energy = np.mean(magnitude[crowd_freq_mask, :][:, quiet_frames])
            live_indicators['crowd_noise_estimate'] = float(crowd_energy)
        else:
            live_indicators['crowd_noise_estimate'] = 0.0

        return live_indicators

    def profile_dataset(self, audio_dir: str, sample_size: int = 100,
                        output_path: str = 'fuji_dataset_profile.csv') -> pd.DataFrame:
        """
        Profile a sample of the dataset
        """
        audio_files = list(Path(audio_dir).glob('**/*.mp3')) + \
                      list(Path(audio_dir).glob('**/*.wav')) + \
                      list(Path(audio_dir).glob('**/*.m4a'))

        # Sample if dataset is large
        if len(audio_files) > sample_size:
            audio_files = np.random.choice(audio_files, sample_size, replace=False)

        results = []

        for i, audio_path in enumerate(audio_files):
            print(f"Processing {i + 1}/{len(audio_files)}: {audio_path.name}")

            try:
                # Combine all metrics
                profile = {'filename': audio_path.name}
                profile.update(self.analyze_audio_quality(str(audio_path)))
                profile.update(self.analyze_rhythm(str(audio_path)))
                profile.update(self.detect_live_recording(str(audio_path)))

                results.append(profile)

            except Exception as e:
                print(f"Error processing {audio_path.name}: {e}")
                continue

        # Convert to DataFrame
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)

        return df

    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics and quality categories
        """
        report = {
            'dataset_summary': {
                'total_files': len(df),
                'avg_duration_minutes': df['duration_seconds'].mean() / 60,
                'total_duration_hours': df['duration_seconds'].sum() / 3600,
            },
            'quality_metrics': {
                'avg_snr_db': df['snr_estimate_db'].mean(),
                'high_quality_ratio': (df['snr_estimate_db'] > 20).mean(),
                'medium_quality_ratio': ((df['snr_estimate_db'] >= 10) & (df['snr_estimate_db'] <= 20)).mean(),
                'low_quality_ratio': (df['snr_estimate_db'] < 10).mean(),
                'clipping_issues': (df['clipping_ratio'] > 0.001).mean(),
            },
            'rhythm_characteristics': {
                'avg_tempo_bpm': df['tempo_bpm'].mean(),
                'tempo_std': df['tempo_bpm'].std(),
                'avg_percussive_ratio': df['percussive_ratio'].mean(),
            },
            'live_vs_studio_estimate': {
                'likely_live_ratio': (df['reverb_estimate'] > 0.3).mean(),
                'avg_reverb': df['reverb_estimate'].mean(),
            }
        }

        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile audio dataset and generate a quality report.")
    parser.add_argument('--input', type=str, default="/content/drive/MyDrive/fuji-mss/songs", help='Path to the directory containing audio files.')
    parser.add_argument('--output', type=str, default='/content/drive/MyDrive/fuji-mss/fuji_dataset_profile.csv', help='Output CSV file for the profile data.')
    parser.add_argument('--sample_size', type=int, default=100, help='Number of audio files to sample from the dataset.')

    args = parser.parse_args()

    profiler = FujiAudioProfiler(sr=22050)

    print("Starting dataset profiling...")
    df = profiler.profile_dataset(args.audio_directory, sample_size=args.sample_size, output_path=args.output)

    report = profiler.generate_quality_report(df)
    print(json.dumps(report, indent=2))

    print(f"\nDataset profile saved to {args.output}")