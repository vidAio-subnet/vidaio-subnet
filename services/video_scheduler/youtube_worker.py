import asyncio
import os
import time
import random
from typing import List, Dict, Optional
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path

from redis_utils import (
    get_redis_connection,
    push_5s_chunks,
    push_10s_chunks,
    push_20s_chunks,
    push_youtube_video_ids,
    get_youtube_queue_size,
    pop_youtube_video_id,
    get_5s_queue_size,
    get_10s_queue_size,
    get_20s_queue_size,
)
from video_utils import download_trim_downscale_youtube_video, apply_video_transformations
from youtube_scraper import populate_database, get_1080p_videos, get_2160p_videos, get_4320p_videos
from youtube_requests import YouTubeHandler, RESOLUTIONS
from vidaio_subnet_core import CONFIG
from vidaio_subnet_core.utilities.storage_client import storage_client

load_dotenv()

class YouTubeWorker:
    """
    YouTube video processing worker that runs as a separate process.
    
    This worker:
    1. Maintains a queue of YouTube video IDs
    2. Downloads and processes YouTube videos into chunks
    3. Applies optional color space transformation
    4. Uploads processed chunks to storage
    5. Feeds chunks into the main synthetic video queues
    """
    
    def __init__(self):
        self.redis_conn = get_redis_connection()
        self.youtube_handler = YouTubeHandler()
        
        # Configuration
        self.youtube_queue_threshold = int(os.getenv("YOUTUBE_QUEUE_THRESHOLD", "20"))
        self.youtube_queue_max_size = int(os.getenv("YOUTUBE_QUEUE_MAX_SIZE", "100"))
        self.enable_color_transform = os.getenv("ENABLE_YOUTUBE_COLOR_TRANSFORM", "false").lower() == "true"
        self.enable_youtube_worker = os.getenv("ENABLE_YOUTUBE_WORKER", "true").lower() == "true"
        
        # Track YouTube accessibility
        self.youtube_accessible = True
        self.failed_attempts = 0
        self.max_failed_attempts = 3
        
        # Search terms for YouTube video discovery
        self.search_terms = [
            "4K nature documentary",
            "wildlife 4K footage",
            "4K city timelapse",
            "4K landscape scenery",
            "4K ocean waves",
            "4K mountain vista",
            "4K forest ambience",
            "4K sunset sunrise",
            "4K aerial drone footage",
            "4K space astronomy",
            "4K architecture buildings",
            "4K technology future",
            "4K abstract art",
            "4K slow motion",
            "4K travel destinations"
        ]
        
        logger.info("🎬 YouTube Worker initialized")
        logger.info(f"📊 YouTube queue threshold: {self.youtube_queue_threshold}")
        logger.info(f"📈 YouTube queue max size: {self.youtube_queue_max_size}")
        logger.info(f"🎨 Color transformation: {'ENABLED' if self.enable_color_transform else 'DISABLED'}")
        logger.info(f"🔧 YouTube worker: {'ENABLED' if self.enable_youtube_worker else 'DISABLED'}")
        
        # Check if YouTube is accessible on startup
        if self.enable_youtube_worker:
            self._check_youtube_accessibility()

    def _check_youtube_accessibility(self):
        """Check if YouTube is accessible and log the result."""
        try:
            # Test with a simple search
            test_results = self.youtube_handler.search_videos_raw("test", max_results=1)
            
            if test_results and len(test_results) > 0:
                self.youtube_accessible = True
                self.failed_attempts = 0
                logger.info("✅ YouTube accessibility test passed")
            else:
                self.youtube_accessible = False
                self.failed_attempts += 1
                logger.warning("⚠️ YouTube accessibility test failed - no results returned")
                
        except Exception as e:
            self.youtube_accessible = False
            self.failed_attempts += 1
            logger.error(f"❌ YouTube accessibility test failed: {str(e)}")
            
        if not self.youtube_accessible:
            logger.warning("🚨 YouTube appears to be inaccessible!")
            logger.warning("💡 Consider:")
            logger.warning("   1. Setting ENABLE_YOUTUBE_WORKER=false")
            logger.warning("   2. Using manual cookies (see fix_youtube_access.md)")
            logger.warning("   3. Focusing on Pexels content with color transforms")

    async def populate_youtube_queue(self):
        """Populate the YouTube video queue with video IDs from database."""
        
        # Check if YouTube worker is disabled
        if not self.enable_youtube_worker:
            logger.debug("YouTube worker disabled, skipping queue population")
            return
            
        # Check if we've had too many failures
        if self.failed_attempts >= self.max_failed_attempts:
            logger.warning(f"⚠️ YouTube marked as inaccessible after {self.failed_attempts} failures")
            logger.warning("💡 To retry, restart the worker or set ENABLE_YOUTUBE_WORKER=false")
            return
            
        try:
            current_size = get_youtube_queue_size(self.redis_conn)
            if current_size >= self.youtube_queue_threshold:
                logger.debug(f"YouTube queue has sufficient videos ({current_size}), skipping population")
                return
            
            needed = self.youtube_queue_max_size - current_size
            logger.info(f"Populating YouTube queue with {needed} videos...")
            
            # Import here to avoid circular imports
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from youtube_scraper import Video, Base
            
            # Create temporary database session
            engine = create_engine("sqlite:///youtube_videos.db")
            Base.metadata.create_all(bind=engine)
            SessionLocal = sessionmaker(bind=engine)
            session = SessionLocal()
            
            # Try to populate database if it's empty
            existing_videos = session.query(Video).count()
            if existing_videos < 10:
                logger.info("Database seems empty, searching for new videos...")
                
                # Try multiple search terms to increase success rate
                search_attempts = 0
                max_search_attempts = 3  # Reduced from 5
                videos_found = 0
                
                while videos_found < 5 and search_attempts < max_search_attempts:
                    search_term = random.choice(self.search_terms)
                    logger.info(f"Searching YouTube for: '{search_term}' (attempt {search_attempts + 1})")
                    
                    try:
                        added_videos = populate_database(session, search_term, count=10)
                        videos_found += len(added_videos)
                        logger.info(f"✅ Found {len(added_videos)} videos for '{search_term}'")
                        
                        if len(added_videos) > 0:
                            self.youtube_accessible = True
                            self.failed_attempts = 0
                            break  # Success, stop trying
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Search failed for '{search_term}': {str(e)}")
                        self.failed_attempts += 1
                    
                    search_attempts += 1
                    
                    # Wait between attempts to avoid rate limiting
                    if search_attempts < max_search_attempts:
                        logger.info("Waiting 30s before next search attempt...")
                        await asyncio.sleep(30)
                
                if videos_found == 0:
                    self.youtube_accessible = False
                    self.failed_attempts += 1
                    logger.error("❌ Could not find any YouTube videos after multiple attempts")
                    logger.error("🚨 YOUTUBE ACCESS BLOCKED")
                    logger.error("💡 Solutions:")
                    logger.error("   1. Set ENABLE_YOUTUBE_WORKER=false in your .env")
                    logger.error("   2. Use manual cookies (see fix_youtube_access.md)")
                    logger.error("   3. Check: python test_youtube_cookies.py")
                    logger.error("   4. Focus on Pexels content with ENABLE_COLOR_TRANSFORM=true")
                    return
            
            # Get videos by resolution/task type
            task_types = ["HD24K", "SD24K", "4K28K"]
            video_entries = []
            
            for task_type in task_types:
                if task_type == "HD24K":
                    videos = get_2160p_videos(session)[:needed//3]
                elif task_type == "SD24K":
                    videos = get_2160p_videos(session)[:needed//3]
                elif task_type == "4K28K":
                    videos = get_4320p_videos(session)[:needed//3]
                
                for video in videos:
                    video_entries.append({
                        "vid": video.id,
                        "task_type": task_type,
                        "source": "youtube"
                    })
            
            if video_entries:
                push_youtube_video_ids(self.redis_conn, video_entries)
                logger.info(f"✅ Added {len(video_entries)} YouTube videos to queue")
                self.youtube_accessible = True
                self.failed_attempts = 0
            else:
                logger.warning("⚠️ No YouTube videos found to add to queue")
                self.failed_attempts += 1
                
            session.close()
            
        except Exception as e:
            self.failed_attempts += 1
            logger.error(f"❌ Error populating YouTube queue: {str(e)}")
            logger.error("💡 This might be due to YouTube bot detection. Consider:")
            logger.error("   1. Set ENABLE_YOUTUBE_WORKER=false")
            logger.error("   2. Use manual cookies (see fix_youtube_access.md)")
            logger.error("   3. Run: python test_youtube_cookies.py")

    async def process_youtube_video(self, video_data: Dict[str, str], chunk_duration: int) -> List[Dict[str, str]]:
        """
        Process a single YouTube video into chunks.
        
        Args:
            video_data: Dictionary containing video info
            chunk_duration: Duration of chunks to create
            
        Returns:
            List of processed chunk data
        """
        video_id = video_data["vid"]
        task_type = video_data["task_type"]
        
        logger.info(f"🎬 Processing YouTube video: {video_id} (task: {task_type})")
        
        try:
            # Download and chunk the video
            challenge_local_paths, video_ids = download_trim_downscale_youtube_video(
                clip_duration=chunk_duration,
                youtube_video_id=video_id,
                task_type=task_type,
                youtube_handler=self.youtube_handler
            )
            
            if challenge_local_paths is None:
                logger.error(f"❌ Failed to download YouTube video: {video_id}")
                return []
            
            # Number of transformations to apply per chunk (default: 3)
            transformations_per_chunk = int(os.getenv("TRANSFORMATIONS_PER_CHUNK", "3"))
            
            # Apply color space transformation if enabled
            transformed_paths = []
            if self.enable_color_transform:
                logger.info(f"🎨 Applying {transformations_per_chunk} color transformations to {len(challenge_local_paths)} YouTube chunks...")
                
                for i, challenge_local_path in enumerate(challenge_local_paths):
                    # Get the corresponding trimmed file path for scoring reference
                    challenge_local_path_obj = Path(challenge_local_path)
                    if challenge_local_path_obj.name.endswith('_downscale.mp4'):
                        # Convert downscale path to trim path for scoring reference
                        trim_path = str(challenge_local_path_obj.parent / challenge_local_path_obj.name.replace('_downscale.mp4', '_trim.mp4'))
                    else:
                        # Fallback: assume it's already a trim path
                        trim_path = str(challenge_local_path_obj.parent / f"{challenge_local_path_obj.stem}_trim.mp4")
                    
                    # Apply multiple transformations to the same source chunk (downscaled version)
                    for transform_idx in range(transformations_per_chunk):
                        try:
                            # Create a unique output path for each transformation
                            transformed_path = str(challenge_local_path_obj.parent / f"{challenge_local_path_obj.stem}_ytransform_{transform_idx}{challenge_local_path_obj.suffix}")
                            
                            # Apply RANDOM transformation to the downscaled file for better variation
                            # No need to specify transformation_index - it defaults to None which selects random
                            transformed_path = apply_video_transformations(
                                challenge_local_path, 
                                transformed_path, 
                                preserve_original=True
                            )
                            
                            transformed_paths.append(transformed_path)
                            logger.debug(f"✅ Applied YouTube random transformation ({transform_idx + 1}/{transformations_per_chunk}) to chunk {i+1}")
                            
                        except Exception as e:
                            logger.error(f"❌ Failed to apply YouTube transformation {transform_idx + 1} to chunk {challenge_local_path}: {str(e)}")
                            # If transformation fails, use the original path
                            transformed_paths.append(challenge_local_path)
                    
                    # Clean up the downscaled file after all transformations since it's no longer needed
                    if os.path.exists(challenge_local_path):
                        os.unlink(challenge_local_path)
                        logger.debug(f"🧹 Cleaned up downscaled file after transformations: {challenge_local_path}")
                    
                    # Preserve the original trimmed file with the original video ID for scoring reference
                    # All transformations of the same source video will reference this same file
                    if challenge_local_path_obj.name.endswith('_downscale.mp4'):
                        original_trim_path = str(challenge_local_path_obj.parent / challenge_local_path_obj.name.replace('_downscale.mp4', '_trim.mp4'))
                    else:
                        original_trim_path = str(challenge_local_path_obj.parent / f"{challenge_local_path_obj.stem}_trim.mp4")
                    
                    if os.path.exists(original_trim_path):
                        logger.debug(f"📁 Preserving original YouTube trimmed reference file for scoring: {original_trim_path}")
                    else:
                        logger.warning(f"⚠️ Original YouTube trimmed reference file not found: {original_trim_path}")
            else:
                # No transformation - use original paths directly
                transformed_paths = challenge_local_paths
                
                # Still preserve trimmed files for scoring when no transformation is applied
                for challenge_local_path in challenge_local_paths:
                    challenge_local_path_obj = Path(challenge_local_path)
                    if challenge_local_path_obj.name.endswith('_downscale.mp4'):
                        original_trim_path = str(challenge_local_path_obj.parent / challenge_local_path_obj.name.replace('_downscale.mp4', '_trim.mp4'))
                        if os.path.exists(original_trim_path):
                            logger.debug(f"📁 Preserving trimmed reference file for scoring: {original_trim_path}")
                        else:
                            logger.warning(f"⚠️ Trimmed reference file not found: {original_trim_path}")
            
            # Upload chunks and create sharing links
            uploaded_video_chunks = []
            for i, chunk_path in enumerate(transformed_paths):
                try:
                    # Generate unique video ID for transformed chunks
                    base_video_id = video_ids[i // transformations_per_chunk if self.enable_color_transform else i]
                    if self.enable_color_transform:
                        transform_idx = i % transformations_per_chunk
                        video_chunk_id = f"{base_video_id}_yt_{transform_idx}"
                    else:
                        video_chunk_id = base_video_id
                    
                    object_name = f"youtube_{video_chunk_id}.mp4"
                    
                    # Upload to storage
                    await storage_client.upload_file(object_name, chunk_path)
                    sharing_link = await storage_client.get_presigned_url(object_name)
                    
                    # Clean up local file
                    os.unlink(chunk_path)
                    
                    if sharing_link:
                        uploaded_video_chunks.append({
                            "video_id": str(video_chunk_id),
                            "uploaded_object_name": object_name,
                            "sharing_link": sharing_link,
                            "task_type": task_type,
                            "source": "youtube"
                        })
                        logger.debug(f"✅ Uploaded YouTube chunk: {object_name}")
                    else:
                        logger.error(f"❌ Failed to get sharing link for YouTube chunk: {object_name}")
                        
                except Exception as e:
                    logger.error(f"❌ Error uploading YouTube chunk {chunk_path}: {str(e)}")
                    # Clean up local file even on error
                    try:
                        os.unlink(chunk_path)
                    except:
                        pass
            
            logger.info(f"✅ Successfully processed {len(uploaded_video_chunks)} chunks from YouTube video {video_id}")
            return uploaded_video_chunks
            
        except Exception as e:
            logger.error(f"❌ Error processing YouTube video {video_id}: {str(e)}")
            return []

    async def contribute_to_synthetic_queues(self):
        """
        Contribute YouTube video chunks to the synthetic video queues.
        This helps diversify the video pool beyond just Pexels content.
        """
        
        # Skip if YouTube worker is disabled or inaccessible
        if not self.enable_youtube_worker:
            logger.debug("YouTube worker disabled, skipping contribution")
            return
            
        if self.failed_attempts >= self.max_failed_attempts:
            logger.debug("YouTube marked as inaccessible, skipping contribution")
            return
            
        try:
            # Check which queues need YouTube content
            queue_sizes = {
                5: get_5s_queue_size(self.redis_conn),
                10: get_10s_queue_size(self.redis_conn),
                20: get_20s_queue_size(self.redis_conn)
            }
            
            # Define contribution thresholds (contribute when queue is getting low)
            contribution_threshold = 15
            contribution_amount = 5  # Number of chunks to contribute per duration
            
            for duration, queue_size in queue_sizes.items():
                if queue_size < contribution_threshold:
                    logger.info(f"🎬 Contributing {contribution_amount} YouTube chunks to {duration}s queue (current size: {queue_size})")
                    
                    youtube_chunks = []
                    processed_count = 0
                    
                    while processed_count < contribution_amount:
                        video_data = pop_youtube_video_id(self.redis_conn)
                        if video_data is None:
                            logger.warning(f"⚠️ YouTube video queue is empty, replenishing...")
                            await self.populate_youtube_queue()
                            
                            # If still no videos after replenishment, give up
                            if not self.youtube_accessible:
                                logger.warning("⚠️ YouTube not accessible, stopping contribution")
                                break
                            
                            break
                        
                        chunks = await self.process_youtube_video(video_data, duration)
                        if chunks:
                            youtube_chunks.extend(chunks[:contribution_amount - processed_count])
                            processed_count += len(chunks)
                            if processed_count >= contribution_amount:
                                break
                    
                    if youtube_chunks:
                        # Push to appropriate queue
                        if duration == 5:
                            push_5s_chunks(self.redis_conn, youtube_chunks)
                        elif duration == 10:
                            push_10s_chunks(self.redis_conn, youtube_chunks)
                        elif duration == 20:
                            push_20s_chunks(self.redis_conn, youtube_chunks)
                        
                        logger.info(f"✅ Successfully contributed {len(youtube_chunks)} YouTube chunks to {duration}s queue")
                    else:
                        logger.warning(f"⚠️ No YouTube chunks processed for {duration}s queue")
                        
        except Exception as e:
            logger.error(f"❌ Error contributing to synthetic queues: {str(e)}")
            self.failed_attempts += 1

    async def run(self):
        """Main worker loop."""
        logger.info("🚀 Starting YouTube Worker...")
        
        # Check if YouTube worker is disabled
        if not self.enable_youtube_worker:
            logger.info("🔧 YouTube worker is DISABLED")
            logger.info("💡 To enable, set ENABLE_YOUTUBE_WORKER=true in .env")
            logger.info("📋 For now, the system will use Pexels content only")
            
            # Keep the worker running but inactive
            while True:
                logger.info("😴 YouTube worker sleeping (disabled)")
                await asyncio.sleep(300)  # Sleep for 5 minutes
                continue
        
        while True:
            try:
                cycle_start = time.time()
                
                # Check if we should stop trying
                if self.failed_attempts >= self.max_failed_attempts:
                    logger.error("🚨 YouTube worker has failed too many times")
                    logger.error("💡 Consider setting ENABLE_YOUTUBE_WORKER=false")
                    logger.error("📋 System will continue with Pexels content only")
                    await asyncio.sleep(300)  # Sleep for 5 minutes
                    continue
                
                # Populate YouTube video queue if needed
                await self.populate_youtube_queue()
                
                # Contribute to synthetic queues
                await self.contribute_to_synthetic_queues()
                
                # Log status
                youtube_queue_size = get_youtube_queue_size(self.redis_conn)
                logger.info(f"📊 YouTube queue size: {youtube_queue_size}")
                logger.info(f"🎯 YouTube accessibility: {'✅ GOOD' if self.youtube_accessible else '❌ BLOCKED'}")
                logger.info(f"🔢 Failed attempts: {self.failed_attempts}/{self.max_failed_attempts}")
                
                cycle_time = time.time() - cycle_start
                logger.info(f"⏱️ YouTube worker cycle completed in {cycle_time:.2f}s")
                
                # Wait before next cycle
                await asyncio.sleep(30)  # 30-second cycles
                
            except Exception as e:
                logger.error(f"❌ Error in YouTube worker main loop: {str(e)}")
                self.failed_attempts += 1
                await asyncio.sleep(60)  # Wait longer on error

async def main():
    """Entry point for the YouTube worker."""
    worker = YouTubeWorker()
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main()) 
