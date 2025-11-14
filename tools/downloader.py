import yt_dlp
import os


def download_songs_from_file(urls: list[str], config: dict[str, str]) -> None:
    for url in urls:
        try:
            with yt_dlp.YoutubeDL(config) as ydl:
                ydl.download([url])
        except Exception as e:
            print(f"Failed to download {url}: {e}")


if __name__ == "__main__":
    from tools.config import CONFIG
    URLS_PATH = CONFIG["urls_path"]
    SONGS_DIR = CONFIG["songs_path"]
    _config = CONFIG["youtube_dl"]["downloader"]
    _config['outtmpl'] = os.path.join(SONGS_DIR, '%(id)s.%(ext)s')
    _config["sleeprequests"] = 30
    for filename in os.listdir(URLS_PATH):
        if filename.endswith(".txt"):
            filepath = os.path.join(URLS_PATH, filename)
            with open(filepath, "r") as f:
                _urls = [line.strip() for line in f.readlines()]
                print(f"Downloading {len(_urls)} songs from {filepath}...")
                download_songs_from_file(
                    urls=_urls,
                    config=_config
                )
                print(f"Finished downloading songs from {filepath}.")

    print("Finished downloading songs. Check the songs directory.")

