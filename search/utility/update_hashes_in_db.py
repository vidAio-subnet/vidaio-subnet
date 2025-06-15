from pymongo import MongoClient
from search.modules.search_config import search_config
import os
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from search.modules.hash_engine import video_to_phashes

db_name = search_config['DB_NAME']
collection_name = search_config['COLLECTION_NAME']
video_dir = search_config['VIDEO_DIR']

def process_single_doc(doc):
    filename = doc.get('filename')
    if filename:
        file_path = os.path.join(video_dir, filename)
        if os.path.exists(file_path):
            raw_hashes, *_ = video_to_phashes(file_path, 16)
            return doc["_id"], raw_hashes
    return None, None

def process_docs(docs):
    docs_list = list(docs)
    total_docs = len(docs_list)

    max_workers = 2

    print(f"Using {max_workers} processes")

    hash_dict = {}

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_doc, doc): doc for doc in docs_list}

            with tqdm(total=total_docs, desc="Processing videos") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            doc_id, hashes = result
                            hash_dict[doc_id] = hashes
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted! Cancelling futures...")
        executor.shutdown(cancel_futures=True)

    return hash_dict

def main():
    client = MongoClient(search_config['MONGO_URI'])

    try:
        if collection_name not in client[db_name].list_collection_names():
            client[db_name].create_collection(collection_name)

        collection = client[db_name][collection_name]
        docs = list(collection.find())
        hash_dict = process_docs(docs)

        for doc in tqdm(docs, desc="Updating hashes in DB"):
            hashes = hash_dict.get(doc["_id"], None)    
            if hashes:
                hashes = [str(h) for h in hashes]
                update = {"hashes": hashes}
            else:
                update = {"hashes": None}
            collection.update_one({"_id": doc["_id"]}, {"$set": update})

    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    main()