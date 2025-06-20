import os
import requests
import tempfile
import threading
import time
import traceback
import functools
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from typing import List, Tuple, Optional
import yt_dlp

FETCH_URL = "https://www.youtube.com"


class RESOLUTIONS:
    SD      = 720,  480  # Standard
    HD_720  = 1280, 720  # 720P
    HD_1080 = 1920, 1080 # 1080p
    HD_1440 = 2560, 1440 # 1440p / QHD
    HD_2160 = 3840, 2160 # 2160p / 4K
    HD_4320 = 7680, 4320 # 4320p / 8K


def cookies_to_netscape(cookies, max_age=48 * 3600) -> str:
    """
    Converts Selenium Chrome driver cookies to Netscape format
    Returns the cookies as a string
    """
    expiry = int(time.time()) + max_age
    lines = ["# Netscape HTTP Cookie File"]
    for c in cookies:
        domain = c["domain"]
        include_subdomains = "TRUE" if domain.startswith(".") else "FALSE"
        path = c["path"]
        secure = "TRUE" if c.get("secure", False) else "FALSE"
        expiry = c.get("expiry", expiry)
        name = c["name"]
        value = c["value"]
        lines.append(f"{domain}\t{include_subdomains}\t{path}\t{secure}\t{expiry}\t{name}\t{value}")
    return "\n".join(lines)


def fetch_cookies(fetch_url:str = "https://youtube.com") -> os.PathLike:
    """
    Gets cookies from YouTube using Selenium
    Converts them to Netscape format for YT-DLP
    Saves them to a tempfile
    Returns the tempfile path
    """
    start = time.time()
    options = [
        "--headless=new",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--window-size=64,64",
        "--disable-extensions",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-default-apps",
        "--disable-translate",
        "--disable-features=TranslateUI",
    ]
    chrome_options = ChromeOptions()
    for o in options:
        chrome_options.add_argument(o)
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(fetch_url)
    driver.implicitly_wait(5)
    cookies = driver.get_cookies()
    driver.quit()
    cookies = cookies_to_netscape(cookies)
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.cookies') as f:
        f.write(cookies)
        cookie_file = f.name
    print(f"Cookie fetch took {time.time() - start} seconds")
    return cookie_file


def download_video(
    video_url:str,
    video_format:dict,
    output_path:os.PathLike,
    cookie_file:os.PathLike | None = None
) -> os.PathLike:
    """Download video from YouTube"""
    print(f"Downloading video {video_url} to {output_path}")
    
    if cookie_file is None:
        print(f"Getting YouTube cookies from {FETCH_URL}")
        cookie_file = fetch_cookies()

    # TODO: Generate output temppath if output_path is not specified
    format_id = video_format['format_id']
    actual_width = video_format.get('width')
    actual_height = video_format.get('height')
    print(f"Selected video format: {format_id}")

    ydl_opts = {
        "cookiefile": cookie_file,
        "format": format_id,
        "outtmpl": str(output_path),
        "quiet": False,
        "no_warnings": True,
        "fixup": "never",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    return output_path


def fetch_video_metadata(
    video_url:str,
    cookie_file:str | None = None,
    **kw
) -> dict:
    """Fetch video metadata with YT-DLP, cookies optional"""
    conf = kw.copy()
    conf.update({"quiet": True})
    if cookie_file:
        conf.update({"cookiefile": cookie_file})
    with yt_dlp.YoutubeDL(conf) as ydl:
        info = ydl.extract_info(video_url, download=False)
        return info.get('formats', [])


def get_matching_format(formats:List[dict], resolution:Tuple[int, int], extension="mp4") -> list[dict]:
    # Get video matches
    expected_width, expected_height = resolution
    matching_formats = [
        f for f in formats
        if f.get('vcodec') != 'none' and f.get('acodec') == 'none'
        and f.get('width') == expected_width and f.get('height') == expected_height
    ]
    if not len(matching_formats):
        raise ValueError(f"No suitable formats found for resolution {resolution}")
    # Select highest bitrate version
    vid_format = max(matching_formats, key=lambda f: f.get('tbr') or 0)
    return vid_format


class YouTubeHandler:
    """
    A class to handle YouTube fetch operations with retry.
    Fetches and supplies cookies to YT-DLP.
    """
    def __init__(self, fetch_url="https://www.youtube.com", max_retries=4):
        self.fetch_url = fetch_url
        self.max_retries = max_retries
        self.refreshing_cookies = True
        self.refresh_cookies()

    def refresh_cookies(self) -> None:
        """
        Refresh YouTube cookies with Selenium.
        Takes 5-10 seconds.
        """
        self.cookie_file = fetch_cookies()
        self.refreshing_cookies = False
    
    def vid_to_url(self, vid:str) -> str:
        return f"{self.fetch_url}/watch?v={vid}"

    def fetch_video_metadata(self, vid:str, **kw) -> dict:
        return fetch_video_metadata(self.vid_to_url(vid), cookie_file=self.cookie_file, **kw)

    def _download_video(self, vid:str, *args, **kw) -> os.PathLike:
        while self.refreshing_cookies:
            time.sleep(0.1) # wait for cookies to stop refreshing
        attempt = 0
        while attempt < self.max_retries:
            try:
                if attempt:
                    print(f"Starting download attempt {attempt}")
                result = download_video(self.vid_to_url(vid), *args, cookie_file=self.cookie_file, **kw)
                return result
            except Exception as e:
                print(traceback.print_exc(e))
                attempt += 1
                if attempt == self.max_retries:
                    print(f"Failed to download video {vid} after {attempt} tries - {e}")
                    raise e
                else:
                    print(f"Failed to download video {vid} - {e}")

    def download_video(self, vid:str, resolution:Tuple[int, int], **kw) -> os.PathLike:
        metadata = self.fetch_video_metadata(vid)
        video_format = get_matching_format(metadata, resolution)
        return self._download_video(vid, video_format, **kw)
    
    def download_video_by_format(self, vid:str, video_format:str, **kw) -> os.PathLike:
        return self._download_video(vid, {"format_id":video_format}, **kw)

    def search_videos_by_resolution(
        self,
        search_term: str,
        resolution: Tuple[int, int],
        max_results: int = 10
    ) -> dict[str, dict]:
        """
        Searches YouTube for videos matching the given term and resolution.
        This function is fairly slow as it has to fetch the metadata for each 
        video in the search sequentially.
        Returns a dictionary of {video_id: format_dict} for matches.
        """
        search_query = f"ytsearch{max_results}:{search_term}"
        ydl_opts = {
            "quiet": False,
            "cookiefile": self.cookie_file,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_result = ydl.extract_info(search_query, download=False)
            if "entries" not in search_result:
                return {}

            matching_videos = {}
            for entry in search_result["entries"]:
                video_id = entry.get("id")
                formats = entry.get("formats", [])
                try:
                    fmt = get_matching_format(formats, resolution)
                    matching_videos[video_id] = fmt
                except ValueError:
                    continue

            return matching_videos
    
    def search_videos_raw(
        self,
        search_term: str,
        max_results: int = 10
    ) -> list[dict]:
        """
        Searches YouTube for videos matching the given term
        This function is fairly slow as it has to fetch the metadata for each 
        video in the search sequentially.
        Returns a list of dicts for matches.
        """
        search_query = f"ytsearch{max_results}:{search_term}"
        ydl_opts = {
            "quiet": False,
            "cookiefile": self.cookie_file,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_result = ydl.extract_info(search_query, download=False)
            if "entries" not in search_result:
                return {}
            
            return search_result["entries"]

if __name__ == "__main__":
    resolution = RESOLUTIONS.HD_2160
    start_time = time.time()
    downloader = YouTubeHandler()
    videos = downloader.search_videos_by_resolution("Nature 4K 1 Minute", resolution, max_results=4)
    elapsed_time = time.time() - start_time
    print(f"Search took {elapsed_time:.2f} seconds")
    # print("\n Found IDs:")
    # for k, v in videos.items():
    #     print(f"  {k}, ...")
    import json
    print(json.dumps(videos, indent=4))


    # get first video in list
    vid = [k for k,v in videos.items()][0]
    downloader.download_video(vid, resolution, output_path="test.mp4")
