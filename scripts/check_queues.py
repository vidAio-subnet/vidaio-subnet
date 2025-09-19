#!/usr/bin/env python3
"""
Quick script to check video scheduler queue sizes
"""

import sys
import os

# Add the services directory to Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'video_scheduler'))

from services.video_scheduler.redis_utils import (
    get_redis_connection,
    get_5s_queue_size,
    get_10s_queue_size,
    get_organic_upscaling_queue_size,
    get_organic_compression_queue_size,
    get_pexels_queue_size,
    get_youtube_queue_size,
    is_scheduler_ready
)

def main():
    """Check and display all queue sizes"""
    try:
        redis_conn = get_redis_connection()
        
        print("=" * 50)
        print("📊 VIDEO SCHEDULER QUEUE STATUS")
        print("=" * 50)
        
        # Scheduler readiness status
        ready = is_scheduler_ready(redis_conn)
        status_icon = "🟢" if ready else "🔴"
        print(f"{status_icon} Scheduler Ready: {'YES' if ready else 'NO'}")
        print()
        
        # Synthetic queues
        print("🎬 SYNTHETIC QUEUES:")
        print(f"   5s chunks:  {get_5s_queue_size(redis_conn):>4}")
        print(f"   10s chunks: {get_10s_queue_size(redis_conn):>4}")
        print(f"   Compressed chunks: {get_organic_compression_queue_size(redis_conn):>4}")
        print(f"   Upscaling chunks: {get_organic_upscaling_queue_size(redis_conn):>4}")
        print()
        
        # Source queues
        print("🎥 SOURCE QUEUES:")
        print(f"   Pexels IDs:  {get_pexels_queue_size(redis_conn):>4}")
        print(f"   YouTube IDs: {get_youtube_queue_size(redis_conn):>4}")
        print()
        
        
        # Total counts
        total_synthetic = get_5s_queue_size(redis_conn) + get_10s_queue_size(redis_conn) + get_organic_compression_queue_size(redis_conn) + get_organic_upscaling_queue_size(redis_conn)
        total_source = get_pexels_queue_size(redis_conn) + get_youtube_queue_size(redis_conn)
        
        print("📈 TOTALS:")
        print(f"   Total Synthetic: {total_synthetic:>4}")
        print(f"   Total Source:    {total_source:>4}")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error checking queues: {str(e)}")
        print("💡 Make sure Redis is running and accessible")
        sys.exit(1)

if __name__ == "__main__":
    main() 