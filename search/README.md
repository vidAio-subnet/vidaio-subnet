# Utility
## 1. search/init_db_from_src_files.py
    Construct MongoDB from config['VIDEO_DIR'] with original video files

## 2. search/utility/generate_test_videos.py
    Generate test videos (trim, downscaled) from original files
    Those test videos will be stored in config['TEST_VIDEO_DIR']

## 3. search/utility/update_hashes_in_db.py
    Update hashes field in MongoDB

# Test
## 1. search/modules/hashengine.py
    Hash search test

## 2. search/modules/video_search_engine.py
    Video search test

## 3. search/test/upscaler.py
    Upscaler offline test

## 4. search/test/validator.py
    Validator based online test