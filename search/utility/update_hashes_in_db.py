from pymongo import MongoClient
from search.modules.search_config import search_config
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from search.modules.hash_engine import video_to_phashes
import time

db_name = search_config['DB_NAME']
collection_name = search_config['COLLECTION_NAME']
video_dir = search_config['VIDEO_DIR']

def process_single_doc(doc):
    filename = doc.get('filename')
    if filename:
        file_path = os.path.join(video_dir, filename)
        if os.path.exists(file_path):
            start_time = time.time()
            raw_hashes, *_ = video_to_phashes(file_path, 16)
            end_time = time.time()
            print(f"Time taken to process {filename}: {end_time - start_time:.2f} seconds")
            return doc["_id"], raw_hashes
    return doc["_id"], None  # Always return _id so we can update even failed ones

def process_and_update(docs, collection, max_workers=2):
    docs_list = list(docs)
    total_docs = len(docs_list)
    
    print(f"Using {max_workers} processes")

    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_doc, doc): doc for doc in docs_list}

        with tqdm(total=total_docs, desc="Processing & updating") as pbar:
            for future in as_completed(futures):
                try:
                    doc_id, hashes = future.result()
                    update = {"hashes": [str(h) for h in hashes]} if hashes else {"hashes": None}
                    collection.update_one({"_id": doc_id}, {"$set": update})
                except Exception as e:
                    print(f"Error processing document: {e}")
                pbar.update(1)

def main():
    client = MongoClient(search_config['MONGO_URI'])

    try:
        if collection_name not in client[db_name].list_collection_names():
            client[db_name].create_collection(collection_name)

        collection = client[db_name][collection_name]
        docs = list(collection.find({
            "$or": [
                {"hashes": {"$exists": False}},
                {"hashes": None},
                {"hashes": []}
            ]
        }))

        process_and_update(docs, collection, max_workers=1)

    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    main()
