import os


DATA_PATH = os.environ["DATA_PATH"]

CONFIG = {
    "data_path": DATA_PATH,
    "urls_path": f"{DATA_PATH}/urls",
    "songs_path": f"{DATA_PATH}/songs",
    "youtube_dl" : {
        "downloader": {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],

            'ignoreerrors': True,
            'writemetadata': True,
            'writeinfojson': True,
            'noplaylist': True,
            'quiet': True,
        }
    }
}