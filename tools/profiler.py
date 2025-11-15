import os

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import json
import argparse
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple


class FujiAudioProfiler:
    """Profile audio quality and characteristics of Fuji music dataset"""

    def __init__(self, sr: int = 22050):
        self.sr = sr

    @staticmethod
    def _safe_div(n: float, d: float, default: float = 0.0) -> float:
        return n / d if d not in (0.0, None) else default

    @staticmethod
    def _safe_log10(x: float, default: float = -120.0) -> float:
        return np.log10(x) if x > 0 else default

    def load_audio(self, audio_path: str, max_duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Load audio once; optionally trim to max_duration seconds."""
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        if max_duration:
            max_samples = int(max_duration * sr)
            if len(y) > max_samples:
                y = y[:max_samples]
        return y, sr

    def analyze_audio_quality(self, y: np.ndarray, sr: int) -> Dict:
        """Compute core audio quality metrics."""
        metrics = {}
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        noise_floor = np.percentile(rms, 10)
        signal_level = np.percentile(rms, 90)
        snr_estimate = 20 * self._safe_log10(self._safe_div(signal_level, noise_floor + 1e-10, 1.0))
        metrics['snr_estimate_db'] = float(snr_estimate)

        clipping_threshold = 0.99
        clipped_samples = np.sum(np.abs(y) > clipping_threshold)
        metrics['clipping_ratio'] = float(self._safe_div(clipped_samples, len(y)))

        metrics['dynamic_range_db'] = float(
            20 * self._safe_log10(self._safe_div(np.max(np.abs(y)), (np.mean(np.abs(y)) + 1e-10), 1.0))
        )

        # Reuse single STFT
        stft = librosa.stft(y)
        magnitude = np.abs(stft)

        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        metrics['spectral_centroid_mean'] = float(np.mean(spectral_centroid))

        nyquist = sr / 2
        freq_bins = librosa.fft_frequencies(sr=sr)
        hf_mask = freq_bins > min(8000, nyquist * 0.9)
        hf_energy = np.mean(magnitude[hf_mask, :]) if np.any(hf_mask) else 0.0
        total_energy = np.mean(magnitude)
        metrics['high_freq_ratio'] = float(self._safe_div(hf_energy, total_energy + 1e-10))

        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        metrics['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
        metrics['spectral_flatness_std'] = float(np.std(spectral_flatness))

        metrics['duration_seconds'] = float(len(y) / sr)
        return metrics

    def analyze_rhythm(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze rhythmic characteristics."""
        rhythm_metrics = {}
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        except Exception:
            tempo, beats = 0.0, []

        rhythm_metrics['tempo_bpm'] = float(tempo)
        rhythm_metrics['num_beats'] = int(len(beats))

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        rhythm_metrics['onset_strength_mean'] = float(np.mean(onset_env))
        rhythm_metrics['onset_strength_std'] = float(np.std(onset_env))

        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        rhythm_metrics['tempogram_std'] = float(np.std(tempogram))

        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = float(np.sum(y_harmonic ** 2))
        percussive_energy = float(np.sum(y_percussive ** 2))
        total_energy = harmonic_energy + percussive_energy
        rhythm_metrics['percussive_ratio'] = float(self._safe_div(percussive_energy, total_energy + 1e-10))
        return rhythm_metrics

    def detect_live_recording(self, y: np.ndarray, sr: int) -> Dict:
        """Heuristics to detect live vs studio characteristics."""
        live_indicators = {}
        rms = librosa.feature.rms(y=y)[0]
        low_rms_mask = rms < np.percentile(rms, 20)
        if np.any(low_rms_mask):
            noise_consistency = float(np.std(rms[low_rms_mask]))
        else:
            noise_consistency = 0.0
        live_indicators['noise_consistency'] = noise_consistency

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        live_indicators['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))

        autocorr = np.correlate(y, y, mode='full')[len(y):]
        autocorr = autocorr[:int(sr * 0.5)]
        if len(autocorr) > int(sr * 0.05):
            reverb_estimate = float(self._safe_div(
                np.max(autocorr[int(sr * 0.05):]),
                (autocorr[0] + 1e-10)
            ))
        else:
            reverb_estimate = 0.0
        live_indicators['reverb_estimate'] = reverb_estimate

        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        freq_bins = librosa.fft_frequencies(sr=sr)
        crowd_freq_mask = (freq_bins >= 200) & (freq_bins <= 500)
        quiet_frames = rms < np.percentile(rms, 30)
        if np.sum(quiet_frames) > 0 and np.any(crowd_freq_mask):
            crowd_energy = float(np.mean(magnitude[crowd_freq_mask, :][:, quiet_frames]))
        else:
            crowd_energy = 0.0
        live_indicators['crowd_noise_estimate'] = crowd_energy
        return live_indicators

    def profile_file(self, audio_path: Path, max_duration: Optional[float] = None) -> Dict:
        """Profile a single file end-to-end."""
        y, sr = self.load_audio(str(audio_path), max_duration=max_duration)
        profile = {'filename': audio_path.name}
        profile.update(self.analyze_audio_quality(y, sr))
        profile.update(self.analyze_rhythm(y, sr))
        profile.update(self.detect_live_recording(y, sr))
        return profile

    def profile_dataset(self,
                        audio_dir: str,
                        sample_size: int = 100,
                        output_path: str = '/content/drive/MyDrive/fuji-mss/fuji_dataset_profile.csv',
                        save_interval: int = 10,
                        extensions: Optional[List[str]] = None,
                        seed: Optional[int] = None,
                        max_duration: Optional[float] = None,
                        jobs: int = 1) -> pd.DataFrame:
        """
        Profile a sample of the dataset with checkpointing.
        Added: extensions, seed, max_duration, jobs (threads).
        """
        if extensions is None:
            extensions = ['mp3', 'wav', 'm4a']
        if seed is not None:
            random.seed(seed)

        all_audio_files: List[Path] = []
        for ext in extensions:
            all_audio_files.extend(Path(audio_dir).glob(f'**/*.{ext}'))

        results: List[Dict] = []
        processed_filenames: set = set()

        if os.path.exists(output_path):
            try:
                existing_df = pd.read_csv(output_path)
                results = existing_df.to_dict('records')
                processed_filenames = set(existing_df['filename'])
                logging.info(f"Resuming from checkpoint: {len(processed_filenames)} files.")
            except Exception as e:
                logging.warning(f"Failed to read checkpoint: {e}. Starting fresh.")

        files_to_process = [f for f in all_audio_files if f.name not in processed_filenames]

        if sample_size and len(files_to_process) > sample_size:
            files_to_process = random.sample(files_to_process, sample_size)

        logging.info(f"Total files found: {len(all_audio_files)} | New to process: {len(files_to_process)}")

        def _process(path: Path) -> Optional[Dict]:
            try:
                return self.profile_file(path, max_duration=max_duration)
            except Exception as e:
                logging.error(f"Error processing {path.name}: {e}")
                return None

        if jobs > 1:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                futures = {ex.submit(_process, p): p for p in files_to_process}
                for i, fut in enumerate(as_completed(futures), 1):
                    res = fut.result()
                    if res:
                        results.append(res)
                    if i % save_interval == 0 or i == len(files_to_process):
                        pd.DataFrame(results).to_csv(output_path, index=False)
                        logging.info(f"Checkpoint (threaded) saved at {i}/{len(files_to_process)}")
        else:
            for i, audio_path in enumerate(files_to_process, 1):
                logging.info(f"Processing {i}/{len(files_to_process)}: {audio_path.name}")
                res = _process(audio_path)
                if res:
                    results.append(res)
                if i % save_interval == 0 or i == len(files_to_process):
                    pd.DataFrame(results).to_csv(output_path, index=False)
                    logging.info(f"Checkpoint saved at {i}/{len(files_to_process)}")

        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logging.info("Profiling complete. Final dataset saved.")
        return df

    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics and quality categories.
        Handles empty DataFrame gracefully.
        """
        if df.empty:
            return {
                'dataset_summary': {'total_files': 0, 'avg_duration_minutes': 0.0, 'total_duration_hours': 0.0},
                'quality_metrics': {},
                'rhythm_characteristics': {},
                'live_vs_studio_estimate': {}
            }
        report = {
            'dataset_summary': {
                'total_files': len(df),
                'avg_duration_minutes': float(df['duration_seconds'].mean() / 60.0),
                'total_duration_hours': float(df['duration_seconds'].sum() / 3600.0),
            },
            'quality_metrics': {
                'avg_snr_db': float(df['snr_estimate_db'].mean()),
                'high_quality_ratio': float((df['snr_estimate_db'] > 20).mean()),
                'medium_quality_ratio': float(((df['snr_estimate_db'] >= 10) & (df['snr_estimate_db'] <= 20)).mean()),
                'low_quality_ratio': float((df['snr_estimate_db'] < 10).mean()),
                'clipping_issues_ratio': float((df['clipping_ratio'] > 0.001).mean()),
            },
            'rhythm_characteristics': {
                'avg_tempo_bpm': float(df['tempo_bpm'].mean()),
                'tempo_std': float(df['tempo_bpm'].std()),
                'avg_percussive_ratio': float(df['percussive_ratio'].mean()),
            },
            'live_vs_studio_estimate': {
                'likely_live_ratio': float((df['reverb_estimate'] > 0.3).mean()),
                'avg_reverb': float(df['reverb_estimate'].mean()),
            }
        }
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile audio dataset and generate a quality report.")
    parser.add_argument('--input', type=str, default="/content/drive/MyDrive/fuji-mss/songs", help='Path to the directory containing audio files.')
    parser.add_argument('--output', type=str, default='/content/drive/MyDrive/fuji-mss/fuji_dataset_profile.csv', help='Output CSV file for the profile data.')
    parser.add_argument('--sample', type=int, default=200, help='Number of audio files to sample from the dataset.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for sampling.')
    parser.add_argument('--max-duration', type=float, default=None, help='Trim audio to this many seconds if set.')
    parser.add_argument('--extensions', type=str, default='mp3,wav,m4a', help='Comma separated list of extensions.')
    parser.add_argument('--jobs', type=int, default=1, help='Number of parallel threads.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging.')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='[%(levelname)s] %(message)s'
    )

    profiler = FujiAudioProfiler(sr=22050)
    logging.info("Starting dataset profiling...")

    df = profiler.profile_dataset(
        args.input,
        sample_size=args.sample,
        output_path=args.output,
        extensions=[e.strip() for e in args.extensions.split(',') if e.strip()],
        seed=args.seed,
        max_duration=args.max_duration,
        jobs=args.jobs
    )

    report = profiler.generate_quality_report(df)
    print(json.dumps(report, indent=2))
    logging.info(f"Dataset profile saved to {args.output}")
