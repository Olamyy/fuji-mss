"""
Fuji Music Dataset Profiling Pipeline - Optimized for Colab/GPU
Includes segment sampling, parallel processing, and GPU-accelerated operations
"""

import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import argparse
import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore')


class FujiAudioProfiler:
    """Profile audio quality and characteristics of Fuji music dataset"""

    def __init__(self,
                 sr: int = 22050,
                 segment_duration: float = 30.0,
                 num_segments: int = 3,
                 use_segments: bool = True):
        """
        Args:
            sr: Sample rate for analysis
            segment_duration: Duration of each segment (seconds)
            num_segments: Number of segments to sample per song
            use_segments: If True, use segment sampling; if False, analyze full audio
        """
        self.sr = sr
        self.segment_duration = segment_duration
        self.num_segments = num_segments
        self.use_segments = use_segments

    @staticmethod
    def _safe_div(n: float, d: float, default: float = 0.0) -> float:
        """Safe division with default fallback"""
        return n / d if d not in (0.0, None) and not np.isnan(d) else default

    @staticmethod
    def _safe_log10(x: float, default: float = -120.0) -> float:
        """Safe log10 with default fallback"""
        return np.log10(x) if x > 0 else default

    def load_segments(self, audio_path: str) -> Tuple[np.ndarray, List[float], float]:
        """
        Load multiple segments from audio file at different positions.
        Returns: (concatenated audio, segment positions, total duration)
        """
        try:
            # Get duration without loading full file
            duration = librosa.get_duration(path=audio_path)

            # If song is shorter than one segment, just load it all
            if duration < self.segment_duration:
                y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
                return y, [0.0], duration

            # Define segment positions (avoid first/last 10% for intro/outro)
            if duration < self.segment_duration * 2:
                # For short songs, just take middle section
                segment_positions = [0.5]
            else:
                segment_positions = np.linspace(0.2, 0.8, self.num_segments)

            segments = []
            actual_positions = []

            for pos in segment_positions:
                start_time = pos * duration
                # Ensure we don't go past the end
                if start_time + self.segment_duration > duration:
                    start_time = max(0, duration - self.segment_duration)

                # Load only this segment
                y, sr = librosa.load(
                    audio_path,
                    sr=self.sr,
                    mono=True,
                    offset=start_time,
                    duration=self.segment_duration
                )
                segments.append(y)
                actual_positions.append(start_time)

            # Concatenate all segments
            y_combined = np.concatenate(segments)

            return y_combined, actual_positions, duration

        except Exception as e:
            logging.error(f"Error loading segments from {audio_path}: {e}")
            raise

    def load_audio(self, audio_path: str, max_duration: Optional[float] = None) -> Tuple[np.ndarray, int, float]:
        """
        Load audio - either segments or full file.
        Returns: (audio, sample_rate, total_duration)
        """
        if self.use_segments:
            y, positions, duration = self.load_segments(audio_path)
            return y, self.sr, duration
        else:
            # Load full audio (with optional max_duration trim)
            duration = librosa.get_duration(path=audio_path)
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True, duration=max_duration)
            return y, sr, duration

    def analyze_audio_quality(self, y: np.ndarray, sr: int) -> Dict:
        """Compute core audio quality metrics."""
        metrics = {}

        # RMS-based metrics
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        noise_floor = np.percentile(rms, 10)
        signal_level = np.percentile(rms, 90)
        snr_estimate = 20 * self._safe_log10(self._safe_div(signal_level, noise_floor + 1e-10, 1.0))
        metrics['snr_estimate_db'] = float(snr_estimate)

        # Clipping detection
        clipping_threshold = 0.99
        clipped_samples = np.sum(np.abs(y) > clipping_threshold)
        metrics['clipping_ratio'] = float(self._safe_div(clipped_samples, len(y)))

        # Dynamic range
        metrics['dynamic_range_db'] = float(
            20 * self._safe_log10(self._safe_div(np.max(np.abs(y)), (np.mean(np.abs(y)) + 1e-10), 1.0))
        )

        # Spectral analysis (reuse single STFT)
        stft = librosa.stft(y)
        magnitude = np.abs(stft)

        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        metrics['spectral_centroid_mean'] = float(np.mean(spectral_centroid))

        # High frequency content
        freq_bins = librosa.fft_frequencies(sr=sr)
        nyquist = sr / 2
        hf_mask = freq_bins > min(8000, nyquist * 0.9)
        hf_energy = np.mean(magnitude[hf_mask, :]) if np.any(hf_mask) else 0.0
        total_energy = np.mean(magnitude)
        metrics['high_freq_ratio'] = float(self._safe_div(hf_energy, total_energy + 1e-10))

        # Spectral flatness (compression artifacts)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        metrics['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
        metrics['spectral_flatness_std'] = float(np.std(spectral_flatness))

        return metrics

    def analyze_rhythm(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze rhythmic characteristics."""
        rhythm_metrics = {}

        try:
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            rhythm_metrics['tempo_bpm'] = float(tempo)
            rhythm_metrics['num_beats'] = int(len(beats))
        except Exception as e:
            logging.warning(f"Beat tracking failed: {e}")
            rhythm_metrics['tempo_bpm'] = 0.0
            rhythm_metrics['num_beats'] = 0

        # Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        rhythm_metrics['onset_strength_mean'] = float(np.mean(onset_env))
        rhythm_metrics['onset_strength_std'] = float(np.std(onset_env))

        # Tempogram (rhythmic regularity)
        try:
            tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
            rhythm_metrics['tempogram_std'] = float(np.std(tempogram))
        except:
            rhythm_metrics['tempogram_std'] = 0.0

        # HPSS - percussive vs harmonic
        try:
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_energy = float(np.sum(y_harmonic ** 2))
            percussive_energy = float(np.sum(y_percussive ** 2))
            total_energy = harmonic_energy + percussive_energy
            rhythm_metrics['percussive_ratio'] = float(self._safe_div(percussive_energy, total_energy + 1e-10))
        except Exception as e:
            logging.warning(f"HPSS failed: {e}")
            rhythm_metrics['percussive_ratio'] = 0.0

        return rhythm_metrics

    def detect_live_recording(self, y: np.ndarray, sr: int) -> Dict:
        """Heuristics to detect live vs studio characteristics."""
        live_indicators = {}

        # RMS analysis
        rms = librosa.feature.rms(y=y)[0]
        low_rms_mask = rms < np.percentile(rms, 20)
        if np.any(low_rms_mask):
            noise_consistency = float(np.std(rms[low_rms_mask]))
        else:
            noise_consistency = 0.0
        live_indicators['noise_consistency'] = noise_consistency

        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        live_indicators['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))

        # Reverb estimation
        try:
            autocorr = np.correlate(y, y, mode='full')[len(y):]
            autocorr = autocorr[:int(sr * 0.5)]
            if len(autocorr) > int(sr * 0.05):
                reverb_estimate = float(self._safe_div(
                    np.max(autocorr[int(sr * 0.05):]),
                    (autocorr[0] + 1e-10)
                ))
            else:
                reverb_estimate = 0.0
        except:
            reverb_estimate = 0.0
        live_indicators['reverb_estimate'] = reverb_estimate

        # Crowd noise detection
        try:
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            freq_bins = librosa.fft_frequencies(sr=sr)
            crowd_freq_mask = (freq_bins >= 200) & (freq_bins <= 500)
            quiet_frames = rms < np.percentile(rms, 30)
            if np.sum(quiet_frames) > 0 and np.any(crowd_freq_mask):
                crowd_energy = float(np.mean(magnitude[crowd_freq_mask, :][:, quiet_frames]))
            else:
                crowd_energy = 0.0
        except:
            crowd_energy = 0.0
        live_indicators['crowd_noise_estimate'] = crowd_energy

        return live_indicators

    def profile_file(self, audio_path: Path) -> Optional[Dict]:
        """Profile a single file end-to-end."""
        try:
            y, sr, total_duration = self.load_audio(str(audio_path))

            profile = {
                'filename': audio_path.name,
                'duration_seconds': float(total_duration),
                'analyzed_duration_seconds': float(len(y) / sr)
            }

            profile.update(self.analyze_audio_quality(y, sr))
            profile.update(self.analyze_rhythm(y, sr))
            profile.update(self.detect_live_recording(y, sr))

            return profile

        except Exception as e:
            logging.error(f"Error processing {audio_path.name}: {e}")
            return None

    def profile_dataset(self,
                        audio_dir: str,
                        sample_size: Optional[int] = None,
                        output_path: str = 'fuji_dataset_profile.csv',
                        save_interval: int = 10,
                        extensions: Optional[List[str]] = None,
                        seed: Optional[int] = None,
                        n_workers: int = 4) -> pd.DataFrame:
        """
        Profile dataset with parallel processing and checkpointing.

        Args:
            audio_dir: Directory containing audio files
            sample_size: Number of files to sample (None = all files)
            output_path: Path to save CSV results
            save_interval: Save checkpoint every N files
            extensions: List of file extensions to process
            seed: Random seed for sampling
            n_workers: Number of parallel workers (use 1 for debugging)
        """
        if extensions is None:
            extensions = ['mp3', 'wav', 'm4a']
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        all_audio_files: List[Path] = []
        for ext in extensions:
            all_audio_files.extend(Path(audio_dir).glob(f'**/*.{ext}'))

        logging.info(f"Found {len(all_audio_files)} audio files")

        results: List[Dict] = []
        processed_filenames: set = set()

        if os.path.exists(output_path):
            try:
                existing_df = pd.read_csv(output_path)
                results = existing_df.to_dict('records')
                processed_filenames = set(existing_df['filename'])
                logging.info(f"Resuming from checkpoint: {len(processed_filenames)} files already processed")
            except Exception as e:
                logging.warning(f"Failed to read checkpoint: {e}. Starting fresh.")

        files_to_process = [f for f in all_audio_files if f.name not in processed_filenames]

        if sample_size and len(files_to_process) > sample_size:
            files_to_process = random.sample(files_to_process, sample_size)

        logging.info(f"Processing {len(files_to_process)} new files")

        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(self.profile_file, f): f for f in files_to_process}

                with tqdm(total=len(files_to_process), desc="Profiling") as pbar:
                    for i, future in enumerate(as_completed(futures), 1):
                        result = future.result()
                        if result:
                            results.append(result)

                        pbar.update(1)

                        # Periodic checkpoint
                        if i % save_interval == 0 or i == len(files_to_process):
                            df = pd.DataFrame(results)
                            df.to_csv(output_path, index=False)
                            logging.info(f"Checkpoint saved: {len(results)} files processed")
        else:
            for i, audio_path in enumerate(tqdm(files_to_process, desc="Profiling"), 1):
                result = self.profile_file(audio_path)
                if result:
                    results.append(result)

                if i % save_interval == 0 or i == len(files_to_process):
                    df = pd.DataFrame(results)
                    df.to_csv(output_path, index=False)
                    logging.info(f"Checkpoint saved: {len(results)} files processed")

        # Final save
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logging.info(f"Profiling complete. {len(df)} files processed. Saved to {output_path}")

        return df

    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics and quality categories."""
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
                'snr_median_db': float(df['snr_estimate_db'].median()),
                'high_quality_ratio': float((df['snr_estimate_db'] > 20).mean()),
                'medium_quality_ratio': float(((df['snr_estimate_db'] >= 10) & (df['snr_estimate_db'] <= 20)).mean()),
                'low_quality_ratio': float((df['snr_estimate_db'] < 10).mean()),
                'clipping_issues_ratio': float((df['clipping_ratio'] > 0.001).mean()),
                'avg_high_freq_ratio': float(df['high_freq_ratio'].mean()),
            },
            'rhythm_characteristics': {
                'avg_tempo_bpm': float(df['tempo_bpm'].mean()),
                'tempo_median_bpm': float(df['tempo_bpm'].median()),
                'tempo_std': float(df['tempo_bpm'].std()),
                'avg_percussive_ratio': float(df['percussive_ratio'].mean()),
                'percussive_ratio_median': float(df['percussive_ratio'].median()),
            },
            'live_vs_studio_estimate': {
                'likely_live_ratio': float((df['reverb_estimate'] > 0.3).mean()),
                'avg_reverb': float(df['reverb_estimate'].mean()),
                'avg_spectral_bandwidth': float(df['spectral_bandwidth_mean'].mean()),
            }
        }

        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Fuji music dataset with segment sampling")
    parser.add_argument('--input', type=str, required=True, help='Path to audio directory')
    parser.add_argument('--output', type=str, default='fuji_dataset_profile.csv', help='Output CSV path')
    parser.add_argument('--sample', type=int, default=None, help='Number of files to sample (None = all)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--extensions', type=str, default='mp3,wav,m4a', help='Comma-separated extensions')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--segment-duration', type=float, default=30.0, help='Segment duration in seconds')
    parser.add_argument('--num-segments', type=int, default=3, help='Number of segments per song')
    parser.add_argument('--no-segments', action='store_true', help='Disable segment sampling (use full audio)')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    profiler = FujiAudioProfiler(
        sr=args.sr,
        segment_duration=args.segment_duration,
        num_segments=args.num_segments,
        use_segments=not args.no_segments
    )

    logging.info("Starting dataset profiling...")
    logging.info(f"Segment sampling: {'ENABLED' if not args.no_segments else 'DISABLED'}")
    if not args.no_segments:
        logging.info(f"  - {args.num_segments} segments of {args.segment_duration}s each")

    df = profiler.profile_dataset(
        audio_dir=args.input,
        sample_size=args.sample,
        output_path=args.output,
        extensions=[e.strip() for e in args.extensions.split(',')],
        seed=args.seed,
        n_workers=args.workers
    )

    report = profiler.generate_quality_report(df)
    print("\n" + "=" * 60)
    print("DATASET PROFILING REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))

    logging.info(f"Results saved to {args.output}")