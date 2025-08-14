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
        "--window-size=1920,1080",  # Larger window size
        "--disable-extensions",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-default-apps",
        "--disable-translate",
        "--disable-features=TranslateUI",
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",  # Better user agent
        "--disable-blink-features=AutomationControlled",  # Hide automation
        "--disable-web-security",
    ]
    chrome_options = ChromeOptions()
    for o in options:
        chrome_options.add_argument(o)
    
    # Add experimental options to avoid detection
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        
        # Set navigator.webdriver to undefined to avoid detection
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        driver.get(fetch_url)
        driver.implicitly_wait(10)  # Longer wait
        
        # Wait a bit for page to fully load
        time.sleep(3)
        
        cookies = driver.get_cookies()
        driver.quit()
        
        if not cookies:
            print("Warning: No cookies retrieved from browser session")
            return None
            
        cookies_netscape = cookies_to_netscape(cookies)
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.cookies') as f:
            f.write(cookies_netscape)
            cookie_file = f.name
        
        print(f"Cookie fetch took {time.time() - start} seconds")
        return cookie_file
        
    except Exception as e:
        print(f"Error fetching cookies with Selenium: {e}")
        return None


def try_browser_cookies() -> Optional[str]:
    """
    Try to extract cookies from browser using yt-dlp's built-in functionality.
    This is often more reliable than Selenium.
    """
    import tempfile
    
    try:
        # Try to extract cookies from Chrome first
        cookie_file = tempfile.NamedTemporaryFile(delete=False, suffix='.cookies').name
        
        # Use yt-dlp to extract cookies from browser
        import subprocess
        result = subprocess.run([
            'yt-dlp', 
            '--cookies-from-browser', 'chrome',
            '--print-to-file', 'cookies', cookie_file,
            '--no-download',
            'https://www.youtube.com/watch?v=dQw4w9WgXcQ'  # Test video
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(cookie_file) and os.path.getsize(cookie_file) > 0:
            print("Successfully extracted cookies from Chrome browser")
            return cookie_file
        else:
            os.unlink(cookie_file)
            
    except Exception as e:
        print(f"Failed to extract cookies from Chrome: {e}")
    
    try:
        # Try Firefox as fallback
        cookie_file = tempfile.NamedTemporaryFile(delete=False, suffix='.cookies').name
        
        result = subprocess.run([
            'yt-dlp', 
            '--cookies-from-browser', 'firefox',
            '--print-to-file', 'cookies', cookie_file,
            '--no-download',
            'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(cookie_file) and os.path.getsize(cookie_file) > 0:
            print("Successfully extracted cookies from Firefox browser")
            return cookie_file
        else:
            os.unlink(cookie_file)
            
    except Exception as e:
        print(f"Failed to extract cookies from Firefox: {e}")
    
    return None


def try_manual_cookies() -> Optional[str]:
    """
    Try to use manually provided cookie file.
    """
    # Check for manual cookie file from environment
    manual_cookie_file = os.getenv("YOUTUBE_COOKIES_FILE")
    if manual_cookie_file and os.path.exists(manual_cookie_file):
        print(f"✅ Using manual cookie file: {manual_cookie_file}")
        return manual_cookie_file
    
    # Check for common cookie file locations
    cookie_locations = [
        "youtube_cookies.txt",
        "cookies.txt",
        os.path.expanduser("~/youtube_cookies.txt"),
        os.path.expanduser("~/cookies.txt")
    ]
    
    for cookie_file in cookie_locations:
        if os.path.exists(cookie_file):
            print(f"✅ Found manual cookie file: {cookie_file}")
            return cookie_file
    
    return None


def download_video(
    video_url:str,
    video_format:dict,
    output_path:os.PathLike,
    cookie_file:os.PathLike | None = None
) -> os.PathLike:
    """Download video from YouTube"""
    print(f"Downloading video {video_url} to {output_path}")

    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            print(f"Cleaned up existing file before download: {output_path}")
        except OSError as e:
            print(f"Warning: Could not clean up existing file {output_path}: {e}")

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

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print(f"Warning: Download produced empty or invalid file: {output_path}")
            if os.path.exists(output_path):
                os.remove(output_path)
            raise Exception("Download failed or file is empty")
            
        return output_path
        
    except Exception as e:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"Cleaned up failed download file: {output_path}")
            except OSError as cleanup_error:
                print(f"Warning: Could not clean up failed download {output_path}: {cleanup_error}")
        raise e  # Re-raise the original exception


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
        self.cookie_file = None
        self.youtube_accessible = True  # Track if YouTube is accessible
        self.refresh_cookies()

    def refresh_cookies(self) -> None:
        """
        Refresh YouTube cookies with multiple fallback methods.
        Takes 5-10 seconds.
        """
        self.refreshing_cookies = True
        
        # Method 1: Try manual cookies first (most reliable)
        print("Trying manual cookie files...")
        self.cookie_file = try_manual_cookies()
        
        if self.cookie_file:
            print("✅ Successfully loaded manual cookies")
            self.refreshing_cookies = False
            self.youtube_accessible = True
            return
        
        # Method 2: Try browser cookie extraction
        print("Trying to extract cookies from browser...")
        self.cookie_file = try_browser_cookies()
        
        if self.cookie_file:
            print("✅ Successfully got cookies from browser")
            self.refreshing_cookies = False
            self.youtube_accessible = True
            return
        
        # Method 3: Try Selenium approach (fallback)
        print("Browser cookie extraction failed, trying Selenium...")
        self.cookie_file = fetch_cookies()
        
        if self.cookie_file:
            print("✅ Successfully got cookies from Selenium")
            self.refreshing_cookies = False
            self.youtube_accessible = True
            return
        
        # Method 4: Try without cookies (last resort)
        print("⚠️ All cookie methods failed, will try without cookies")
        self.cookie_file = None
        self.youtube_accessible = False  # Mark as potentially inaccessible
        self.refreshing_cookies = False

    def _test_youtube_access(self) -> bool:
        """
        Test if YouTube is accessible with current cookies.
        """
        try:
            # Test with a simple metadata fetch
            test_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": True,
            }
            
            if self.cookie_file:
                test_opts["cookiefile"] = self.cookie_file
            
            with yt_dlp.YoutubeDL(test_opts) as ydl:
                # Test with a known video
                info = ydl.extract_info("https://www.youtube.com/watch?v=dQw4w9WgXcQ", download=False)
                return info is not None
                
        except Exception as e:
            error_str = str(e)
            if "Sign in to confirm you're not a bot" in error_str:
                return False
            # Other errors might be temporary
            return True

    def vid_to_url(self, vid:str) -> str:
        return f"{self.fetch_url}/watch?v={vid}"

    def fetch_video_metadata(self, vid:str, **kw) -> dict:
        return fetch_video_metadata(self.vid_to_url(vid), cookie_file=self.cookie_file, **kw)

    def _download_video(self, vid:str, *args, **kw) -> os.PathLike:
        while self.refreshing_cookies:
            time.sleep(0.1) # wait for cookies to stop refreshing
        
        attempt = 0
        output_path = None
        while attempt < self.max_retries:
            try:
                if attempt:
                    print(f"Starting download attempt {attempt}")
                
                if 'output_path' in kw:
                    output_path = kw['output_path']
                    if os.path.exists(output_path):
                        try:
                            os.remove(output_path)
                            print(f"Cleaned up existing file before retry: {output_path}")
                        except OSError as e:
                            print(f"Warning: Could not clean up existing file {output_path}: {e}")
                
                result = download_video(self.vid_to_url(vid), *args, cookie_file=self.cookie_file, **kw)
                return result
                
            except Exception as e:
                error_str = str(e)
                print(f"Download attempt {attempt + 1} failed: {error_str}")
                
                if output_path and os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                        print(f"Cleaned up failed download attempt: {output_path}")
                    except OSError as cleanup_error:
                        print(f"Warning: Could not clean up failed attempt {output_path}: {cleanup_error}")
                
                # If it's a bot detection error, try refreshing cookies
                if "Sign in to confirm you're not a bot" in error_str or "blocked" in error_str.lower():
                    print("🤖 Bot detection triggered, refreshing cookies...")
                    self.refresh_cookies()
                
                attempt += 1
                if attempt == self.max_retries:
                    print(f"Failed to download video {vid} after {attempt} tries - {e}")
                    raise e
                else:
                    # Wait longer between retries
                    wait_time = min(30 * attempt, 120)  # Progressive backoff, max 2 minutes
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

    def download_video(self, vid:str, resolution:Tuple[int, int], **kw) -> os.PathLike:
        try:
            metadata = self.fetch_video_metadata(vid)
            video_format = get_matching_format(metadata, resolution)
            return self._download_video(vid, video_format, **kw)
        except Exception as e:
            print(f"Error getting video metadata for {vid}: {e}")
            raise

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
        # Quick accessibility check
        if not self.youtube_accessible:
            print("⚠️ YouTube marked as inaccessible, skipping search")
            return {}
            
        search_query = f"ytsearch{max_results}:{search_term}"
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "ignoreerrors": True,  # Don't stop on individual video errors
        }
        
        if self.cookie_file:
            ydl_opts["cookiefile"] = self.cookie_file

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_result = ydl.extract_info(search_query, download=False)
                if "entries" not in search_result:
                    return {}

                matching_videos = {}
                for entry in search_result["entries"]:
                    if entry is None:  # Skip failed extractions
                        continue
                        
                    video_id = entry.get("id")
                    if not video_id:
                        continue
                        
                    formats = entry.get("formats", [])
                    try:
                        fmt = get_matching_format(formats, resolution)
                        matching_videos[video_id] = fmt
                    except ValueError:
                        continue

                return matching_videos
                
        except Exception as e:
            error_str = str(e)
            print(f"Search failed for '{search_term}': {e}")
            
            # Check for bot detection
            if "Sign in to confirm you're not a bot" in error_str:
                print("🤖 Bot detection encountered, marking YouTube as inaccessible")
                self.youtube_accessible = False
                
            return {}

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
        # Quick accessibility check
        if not self.youtube_accessible:
            print("⚠️ YouTube marked as inaccessible, skipping search")
            return []
            
        search_query = f"ytsearch{max_results}:{search_term}"
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "ignoreerrors": True,  # Don't stop on individual video errors
        }
        
        if self.cookie_file:
            ydl_opts["cookiefile"] = self.cookie_file

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_result = ydl.extract_info(search_query, download=False)
                if "entries" not in search_result:
                    return []

                # Filter out None entries (failed extractions)
                valid_entries = [entry for entry in search_result["entries"] if entry is not None]
                print(f"Successfully found {len(valid_entries)} videos for search: '{search_term}'")
                return valid_entries
                
        except Exception as e:
            error_str = str(e)
            print(f"Search failed for '{search_term}': {e}")
            
            # Check for bot detection
            if "Sign in to confirm you're not a bot" in error_str:
                print("🤖 Bot detection encountered, marking YouTube as inaccessible")
                self.youtube_accessible = False
                
            return []

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