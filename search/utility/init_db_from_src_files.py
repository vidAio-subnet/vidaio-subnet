from pymongo import MongoClient
from search.modules.config import config
import asyncio
import os
import json
from moviepy.editor import VideoFileClip
import cv2

async def main():
    # Initialize MongoDB client
    client = MongoClient(config['mongo_uri'])
    db_name = config['db_name']
    collection_name = config['collection']
    video_dir = config['video_dir']

    try:
        # Check if collection exists, if not create it
        if collection_name not in client[db_name].list_collection_names():
            client[db_name].create_collection(collection_name)

        # Delete all documents in the collection
        collection = client[db_name][collection_name]
        collection.delete_many({})
        print(f"Successfully deleted all documents from {collection_name}")

        # Iterate through all files in video_dir
        for filename in os.listdir(video_dir):
            file_path = os.path.join(video_dir, filename)
            
            # Check if it's a file (not a directory)
            if os.path.isfile(file_path):
                # Create document for each video file
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
                    "frame_count": frame_count
                }
                
                # Insert document into collection
                collection.insert_one(video_doc)
                print(f"Added {filename} to database")
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(main())
