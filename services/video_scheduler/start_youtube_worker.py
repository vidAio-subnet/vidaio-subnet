#!/usr/bin/env python3
"""
YouTube Worker Startup Script

This script starts the YouTube video processing worker as a separate process.
The YouTube worker runs independently from the main video scheduler to avoid
blocking the main processing pipeline with long YouTube download times.

Usage:
    python start_youtube_worker.py

Environment Variables:
    YOUTUBE_QUEUE_THRESHOLD=20        # Minimum videos to keep in YouTube queue  
    YOUTUBE_QUEUE_MAX_SIZE=100        # Maximum videos in YouTube queue
    ENABLE_YOUTUBE_COLOR_TRANSFORM=false  # Enable color space transformation for YouTube videos
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from youtube_worker import main

if __name__ == "__main__":
    print("🎬 Starting YouTube Video Worker...")
    print("📋 Configuration:")
    print(f"   YouTube Queue Threshold: {os.getenv('YOUTUBE_QUEUE_THRESHOLD', '20')}")
    print(f"   YouTube Queue Max Size: {os.getenv('YOUTUBE_QUEUE_MAX_SIZE', '100')}")
    print(f"   Color Transformation: {os.getenv('ENABLE_YOUTUBE_COLOR_TRANSFORM', 'false')}")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 YouTube Worker stopped by user")
    except Exception as e:
        print(f"❌ Error starting YouTube Worker: {e}")
        sys.exit(1) 