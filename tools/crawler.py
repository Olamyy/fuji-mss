import os
import yt_dlp

from tools.config import CONFIG


def search_songs(query: str, sleep: float = 2.0) -> set[str]:
    ydl_search_opts = {
        'ignoreerrors': True,
        'quiet': True,
        'sleeprequests': sleep,
    }
    print(f"Searching for: '{search_string}'")
    video_urls = set()
    try:
        with yt_dlp.YoutubeDL(ydl_search_opts) as ydl:
            result = ydl.extract_info(search_string, download=False)
            if 'entries' in result:
                for entry in result['entries']:
                    if entry and 'webpage_url' in entry:
                        video_urls.add(entry['webpage_url'])
    except Exception as e:
        print(f"Error during search for '{query}': {e}")
    # time.sleep(sleep / 2) # Not sure if needed since yt_dlp has sleeprequests
    return video_urls

def write_urls_per_query(query: str, results: set[str]) -> None:
    formatted_filename = f"{query.replace(' ', '_')}.txt"
    with open(formatted_filename, "w") as f:
        for url in results:
            f.write(f"{url}\n")
    print(f"Saved {len(results)} URLs to {formatted_filename}")

if __name__ == "__main__":
    search_queries = [
        "Sikiru Ayinde Barrister",
        "Kollington Ayinla",
        "K1 De Ultimate",
        "Wasiu Ayinde",
        "Pasuma",
        "Ayinla Omowura",
        "Sunny T Adesokan"
    ]
    _limit = 20
    _sleep = 20
    data_path = CONFIG["data_path"]
    urls_path = CONFIG["urls_path"]
    os.makedirs(urls_path, exist_ok=True)
    for _query in search_queries:
        search_string = f"ytsearch{_limit}:{_query}"
        _results = search_songs(_query, sleep=_sleep)
        print(f"Found {len(_results)} songs for '{_query}'")
        write_urls_per_query(
            query=_query,
            results=_results
        )
