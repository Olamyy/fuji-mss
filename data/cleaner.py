import re
from pathlib import Path
from pydub import AudioSegment


DATA_DIR = Path("/Users/olamilekan/Desktop/research/fujidb_songs/songs")
EXTENSION = "mp3"
GLOB_PATH = f"**/*.{EXTENSION}"


class DatasetCleaner:
    def __init__(self, dry_run: bool = False):
        self.files = Path(DATA_DIR)
        self.dry_run = dry_run

    @staticmethod
    def get_60_second_ranges(song_length: int):
        ranges = []
        num_full_ranges = int(song_length // 60)
        for i in range(num_full_ranges):
            start = i * 60
            end = (i + 1) * 60
            ranges.append((start, end))

        if song_length % 60 != 0:
            start = num_full_ranges * 60
            end = song_length
            ranges.append((start, end))

        return ranges

    def split_audio(self):
        for file in self.files.glob(GLOB_PATH):
            sound = AudioSegment.from_file(file, format=EXTENSION)
            song_length_in_seconds = sound.duration_seconds
            song_ranges = self.get_60_second_ranges(song_length_in_seconds)
            for song_range in song_ranges:
                start, end = song_range
                split_sound = sound[start * 1000:end * 1000]
                split_sound.export(f"{file.stem}_{start}_{end}.wav",
                                   format="wav",
                                   parameters=["-ar", "16000",  "-ac", "1", "-c:a"])

    def clean(self):
        self.clean_filenames()

    def clean_filenames(self):
        for file in self.files.glob(GLOB_PATH):
            existing_name = file.stem
            new_name = self.clean_name(existing_name)
            if existing_name != new_name:
                new_file_path = file.with_name(f"{new_name}.{EXTENSION}")
                if not self.dry_run:
                    file.rename(new_file_path)
                    print(f"Renamed {file} to {new_file_path}")
                else:
                    print(f"Would rename {file} to {new_file_path}")

    @staticmethod
    def clean_name(existing_name: str):
        new_name = existing_name.rsplit('.', 1)[0]
        new_name = new_name.lower()
        new_name = new_name.replace(' ', '_').replace('-', '_')
        return re.sub(r'[^a-z0-9_]', '', new_name)
