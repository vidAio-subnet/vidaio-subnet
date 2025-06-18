from pymongo import MongoClient
from search.modules.search_config import search_config
import asyncio
import os
import json
from moviepy.editor import VideoFileClip
import cv2
from datetime import datetime

async def main():
    client = MongoClient(search_config['MONGO_URI'])
    db_name = search_config['DB_NAME']
    collection_name = search_config['COLLECTION_NAME']
    video_dir = search_config['VIDEO_DIR']

    try:
        if collection_name not in client[db_name].list_collection_names():
            client[db_name].create_collection(collection_name)

        collection = client[db_name][collection_name]
        collection.delete_many({})
        print(f"Successfully deleted all documents from {collection_name}")

        total_files = len(os.listdir(video_dir))
        for idx, filename in enumerate(os.listdir(video_dir), 1):
            print(f"Processing file {idx}/{total_files}: {filename}")
            file_path = os.path.join(video_dir, filename)
            
            if os.path.isfile(file_path):
                cap = cv2.VideoCapture(file_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                video_doc = {
                    "filename": filename,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "frame_count": frame_count,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
                
                collection.insert_one(video_doc)
                print(f"✅ Inserted {filename} into database")
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(main())
