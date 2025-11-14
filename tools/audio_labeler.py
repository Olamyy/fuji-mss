import librosa
import sounddevice as sd
import numpy as np
from pathlib import Path
import json
import random
from datetime import datetime


class AudioLabeler:
    def __init__(self, audio_dir, labels_file="audio_labels.json", snippet_duration=10):
        """
        Interactive tool for labeling audio quality.

        Args:
            audio_dir: Directory containing audio files
            labels_file: JSON file to save labels
            snippet_duration: Duration of audio snippets to play (seconds)
        """
        self.audio_dir = Path(audio_dir)
        self.labels_file = labels_file
        self.snippet_duration = snippet_duration

        # Find all audio files
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
        self.audio_files = [f for f in self.audio_dir.rglob('*')
                            if f.suffix.lower() in audio_extensions]

        # Load existing labels if they exist
        self.labels = self.load_labels()

        # Separate labeled and unlabeled files
        self.update_file_lists()

        print(f"\n{'=' * 70}")
        print(f"Audio Quality Labeling Tool")
        print(f"{'=' * 70}")
        print(f"Total files: {len(self.audio_files)}")
        print(f"Already labeled: {len(self.labeled_files)}")
        print(f"Remaining: {len(self.unlabeled_files)}")
        print(f"{'=' * 70}\n")

    def load_labels(self):
        """Load existing labels from JSON file."""
        if Path(self.labels_file).exists():
            with open(self.labels_file, 'r') as f:
                return json.load(f)
        return {}

    def save_labels(self):
        """Save labels to JSON file."""
        with open(self.labels_file, 'w') as f:
            json.dump(self.labels, f, indent=2)
        print(f"✓ Labels saved to {self.labels_file}")

    def update_file_lists(self):
        """Update lists of labeled and unlabeled files."""
        labeled_paths = set(self.labels.keys())
        self.labeled_files = [f for f in self.audio_files if str(f) in labeled_paths]
        self.unlabeled_files = [f for f in self.audio_files if str(f) not in labeled_paths]

    def play_snippet(self, file_path, start_time=None):
        """
        Play a random snippet from the audio file.

        Args:
            file_path: Path to audio file
            start_time: If provided, start from this time (seconds). Otherwise random.
        """
        try:
            # Load full file metadata to get duration
            duration = librosa.get_duration(path=file_path)

            # Choose random start time if not provided
            if start_time is None:
                # Avoid the very start and end
                max_start = max(0, duration - self.snippet_duration - 5)
                start_time = random.uniform(5, max_start) if max_start > 5 else 0

            # Load snippet
            y, sr = librosa.load(file_path, sr=None, offset=start_time,
                                 duration=self.snippet_duration)

            # Normalize to prevent clipping
            y = y / (np.max(np.abs(y)) + 1e-10) * 0.8

            print(f"\n▶ Playing snippet from {start_time:.1f}s to {start_time + self.snippet_duration:.1f}s")
            print(f"  (Total duration: {duration:.1f}s)")

            # Play audio
            sd.play(y, sr)
            sd.wait()

            return True

        except Exception as e:
            print(f"✗ Error playing {file_path}: {e}")
            return False

    def label_file(self, file_path):
        """
        Interactive labeling session for a single file.
        """
        print(f"\n{'=' * 70}")
        print(f"File: {file_path.name}")
        print(f"Path: {file_path}")
        print(f"{'=' * 70}")

        while True:
            # Play a random snippet
            if not self.play_snippet(file_path):
                return 'skip'

            # Get user input
            print("\nOptions:")
            print("  [c] Clean - Good quality recording")
            print("  [d] Dirty - Noisy/bad quality recording")
            print("  [p] Play another snippet from this song")
            print("  [s] Skip this file for now")
            print("  [q] Quit and save")
            print("  [u] Undo last label")

            choice = input("\nYour choice: ").lower().strip()

            if choice == 'c':
                return 'clean'
            elif choice == 'd':
                return 'dirty'
            elif choice == 'p':
                continue  # Play another snippet
            elif choice == 's':
                return 'skip'
            elif choice == 'q':
                return 'quit'
            elif choice == 'u':
                return 'undo'
            else:
                print("Invalid choice. Please try again.")

    def run(self):
        """
        Main labeling loop.
        """
        if not self.unlabeled_files:
            print("\n✓ All files have been labeled!")
            self.show_stats()
            return

        # Shuffle for variety
        random.shuffle(self.unlabeled_files)

        last_labeled = []  # Track last few labels for undo

        try:
            for i, file_path in enumerate(self.unlabeled_files, 1):
                print(f"\n\nProgress: {i}/{len(self.unlabeled_files)} unlabeled files")

                label = self.label_file(file_path)

                if label == 'quit':
                    print("\nQuitting...")
                    break
                elif label == 'skip':
                    print("Skipped.")
                    continue
                elif label == 'undo':
                    if last_labeled:
                        undo_path, undo_label = last_labeled.pop()
                        del self.labels[str(undo_path)]
                        self.unlabeled_files.append(undo_path)
                        print(f"✓ Undone: {undo_path.name} (was: {undo_label})")
                        self.save_labels()
                    else:
                        print("Nothing to undo.")
                    continue
                elif label in ['clean', 'dirty']:
                    # Save label
                    self.labels[str(file_path)] = {
                        'label': label,
                        'timestamp': datetime.now().isoformat(),
                        'filename': file_path.name
                    }
                    last_labeled.append((file_path, label))
                    print(f"✓ Labeled as: {label.upper()}")

                    # Auto-save every 5 labels
                    if len(last_labeled) % 5 == 0:
                        self.save_labels()
                        print(f"\n--- Auto-saved ({len(self.labels)} total labels) ---")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")

        finally:
            # Save on exit
            self.save_labels()
            self.show_stats()

    def show_stats(self):
        """Display labeling statistics."""
        if not self.labels:
            return

        clean_count = sum(1 for v in self.labels.values() if v['label'] == 'clean')
        dirty_count = sum(1 for v in self.labels.values() if v['label'] == 'dirty')
        total = len(self.labels)

        print(f"\n{'=' * 70}")
        print(f"LABELING STATISTICS")
        print(f"{'=' * 70}")
        print(f"Total labeled: {total}")
        print(f"Clean: {clean_count} ({clean_count / total * 100:.1f}%)")
        print(f"Dirty: {dirty_count} ({dirty_count / total * 100:.1f}%)")
        print(f"Remaining: {len(self.audio_files) - total}")
        print(f"{'=' * 70}\n")

    def resume(self):
        """Resume labeling from where you left off."""
        self.update_file_lists()
        self.run()


def export_for_training(labels_file="audio_labels.json",
                        output_clean="clean_files.txt",
                        output_dirty="dirty_files.txt"):
    """
    Export labeled files into separate lists for training.
    """
    with open(labels_file, 'r') as f:
        labels = json.load(f)

    clean_files = [path for path, data in labels.items() if data['label'] == 'clean']
    dirty_files = [path for path, data in labels.items() if data['label'] == 'dirty']

    with open(output_clean, 'w') as f:
        f.write('\n'.join(clean_files))

    with open(output_dirty, 'w') as f:
        f.write('\n'.join(dirty_files))

    print(f"✓ Exported {len(clean_files)} clean files to {output_clean}")
    print(f"✓ Exported {len(dirty_files)} dirty files to {output_dirty}")


def compare_heuristic_labels():
    with open('audio_labels.json', 'r') as f:
        manual_labels = json.load(f)

    # Load calibration results
    with open('calibration_results.json', 'r') as f:
        auto_results = json.load(f)

    # Compare
    for result in auto_results:
        file_path = result['file']
        if file_path in manual_labels:
            manual = manual_labels[file_path]['label']
            auto = 'dirty' if result['noise_floor_absolute'] > YOUR_THRESHOLD else 'clean'
            if manual != auto:
                print(f"Mismatch: {file_path}")
                print(f"  Manual: {manual}, Auto: {auto}")
                print(f"  Noise floor: {result['noise_floor_absolute']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python label_audio.py <audio_directory>")
        print("  python label_audio.py export <labels_file>")
        sys.exit(1)

    if sys.argv[1] == "export":
        # Export mode
        labels_file = sys.argv[2] if len(sys.argv) > 2 else "audio_labels.json"
        export_for_training(labels_file)
    else:
        # Labeling mode
        audio_dir = sys.argv[1]
        labeler = AudioLabeler(audio_dir)
        labeler.run()