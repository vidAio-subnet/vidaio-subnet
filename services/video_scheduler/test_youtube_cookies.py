#!/usr/bin/env python3
"""
YouTube Cookie Test Script

This script helps diagnose and fix YouTube cookie issues for the video scheduler.

Usage:
    python test_youtube_cookies.py

This will test various methods of accessing YouTube and help identify the best approach.
"""

import sys
import os
import subprocess
import tempfile
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_browser_cookies():
    """Test extracting cookies from browser."""
    print("🔍 Testing browser cookie extraction...")
    
    browsers = ['chrome', 'firefox', 'edge', 'safari']
    
    for browser in browsers:
        print(f"   Testing {browser}...")
        try:
            result = subprocess.run([
                'yt-dlp', 
                '--cookies-from-browser', browser,
                '--list-formats',
                'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"   ✅ {browser}: SUCCESS")
                return browser
            else:
                print(f"   ❌ {browser}: Failed - {result.stderr.strip()}")
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ {browser}: Timeout")
        except FileNotFoundError:
            print(f"   ❓ {browser}: Not found")
        except Exception as e:
            print(f"   ❌ {browser}: Error - {e}")
    
    return None

def test_youtube_handler():
    """Test the YouTube handler with different methods."""
    print("\n🧪 Testing YouTube handler...")
    
    try:
        from youtube_requests import YouTubeHandler
        
        handler = YouTubeHandler()
        print("   ✅ YouTubeHandler created successfully")
        
        # Test search
        print("   🔍 Testing video search...")
        results = handler.search_videos_raw("test", max_results=2)
        
        if results:
            print(f"   ✅ Search successful: Found {len(results)} videos")
            return True
        else:
            print("   ❌ Search failed: No results returned")
            return False
            
    except Exception as e:
        print(f"   ❌ YouTubeHandler test failed: {e}")
        return False

def test_yt_dlp_basic():
    """Test basic yt-dlp functionality."""
    print("\n🔧 Testing basic yt-dlp...")
    
    try:
        result = subprocess.run([
            'yt-dlp',
            '--list-formats',
            'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ Basic yt-dlp works")
            return True
        else:
            print(f"   ❌ Basic yt-dlp failed: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        print(f"   ❌ yt-dlp test error: {e}")
        return False

def main():
    """Run all tests and provide recommendations."""
    print("🎬 YouTube Cookie Diagnostic Tool")
    print("=" * 50)
    
    # Test 1: Basic yt-dlp
    basic_works = test_yt_dlp_basic()
    
    # Test 2: Browser cookies  
    working_browser = test_browser_cookies()
    
    # Test 3: YouTube handler
    handler_works = test_youtube_handler()
    
    # Provide recommendations
    print("\n📋 RECOMMENDATIONS:")
    print("=" * 50)
    
    if not basic_works:
        print("❌ CRITICAL: yt-dlp is not working properly")
        print("   Solutions:")
        print("   1. Install/update yt-dlp: pip install -U yt-dlp")
        print("   2. Check your internet connection")
        return
    
    if working_browser:
        print(f"✅ GOOD: Browser cookies work with {working_browser}")
        print(f"   Your YouTube worker should work with {working_browser} browser cookies")
        print(f"   Make sure you're logged into YouTube in {working_browser}")
    else:
        print("⚠️  WARNING: No browser cookies are working")
        print("   Solutions:")
        print("   1. Log into YouTube in Chrome or Firefox")
        print("   2. Make sure browser profile is accessible")
        print("   3. Try running browser as the same user")
    
    if not handler_works:
        print("⚠️  WARNING: YouTube handler is having issues")
        print("   This might be temporary YouTube rate limiting")
        print("   Try again in a few minutes")
    
    # Environment setup
    print("\n🔧 SETUP RECOMMENDATIONS:")
    print("=" * 50)
    
    if working_browser:
        print("Add this to your .env file:")
        print("ENABLE_YOUTUBE_COLOR_TRANSFORM=false")
        print("YOUTUBE_QUEUE_THRESHOLD=10")
        print("YOUTUBE_QUEUE_MAX_SIZE=50")
    else:
        print("Consider using Pexels-only mode until YouTube access is fixed:")
        print("# Comment out YouTube worker startup")
        print("# Focus on Pexels content with color transformation")
    
    print("\n🚀 Next steps:")
    if working_browser and handler_works:
        print("   ✅ Everything looks good! You can start the YouTube worker")
        print("   Run: python start_youtube_worker.py")
    elif working_browser:
        print("   ⚠️  YouTube access is limited but might work")
        print("   Try: python start_youtube_worker.py")
        print("   Monitor logs for issues")
    else:
        print("   ❌ YouTube access is blocked")
        print("   Stick with Pexels-only mode for now")
        print("   Try again later or use a VPN")

if __name__ == "__main__":
    main() 