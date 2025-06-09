import argparse
from modules.config import config
from modules.search_engine import VideoSearchEngine
from modules.vmaf_metric import evaluate_vmaf
from modules.search import HashDB
import time
import math
from collections import defaultdict
import os
import logging
import json

log_filename = "evaluation_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode='w'),  # overwrite each run
        logging.StreamHandler()  # also print to console
    ]
)

log = logging.getLogger(__name__)

def main():
    evaluated_dataset = []
    with open(config['evaluated_file'], 'r') as f:
        evaluated_dataset = json.load(f)

    evaluated_ids = {item["id"] for item in evaluated_dataset}
    print(len(evaluated_ids))

    start_time = time.time()
    engine = VideoSearchEngine(log)
    elapsed_time = time.time() - start_time
    log.info(f"loading time: {elapsed_time}")

    queryDB = HashDB(mongo_uri=config['mongo_uri'], db_name=config['db_name'], collection_name=config['collection'])

    queryData = queryDB.get_all_video_hashes(feature='downscale')

    FP = 0
    TP = 0
    FN = 0
    TOTAL = len(queryData)
    metricsTrue = defaultdict(int)
    metricsFalse = defaultdict(int)

    log.info(f'total query: {TOTAL}')

    idx = 0
    total_start_time = time.time()
    for query in queryData:
        start_time = time.time()  
        idx += 1

        if query['id'] in evaluated_ids:
            log.info(f"evaluated id: {query['id']}")
            continue

        result = engine.searchHash(query=query, top_n=1)
        elapsed_time = time.time() - start_time

        if len(result) == 0:
            FN += 1
            log.info(f'❓ search failed!!!!, id: {query['id']}, time = {elapsed_time}')
            evaluated_data = {
                "id": query['id'],
                "search": "Failed",
                "time": elapsed_time
            }
            evaluated_dataset.append(evaluated_data)
            with open(config['evaluated_file'], 'w') as f:
                json.dump(evaluated_dataset, f, indent=2)
            continue
        
        predicted_path = result[0][0]  # (upscale_path, vmaf_score, id)
        predicted_id = result[0][2]

        # trim_path = f"{config['default_video_dir']}/{query['id']}_trim.mp4"
        # vmaf_score = evaluate_vmaf(predicted_path, trim_path)
        os.remove(predicted_path)
        
        if predicted_id == query['id']:
            TP += 1
            metricsTrue[math.floor(result[0][1])] += 1
            log.info(f'✅ query id: {query['id']}, result: {predicted_path}, eval result: True, score = {result[0][1]}, time = {elapsed_time}')
        else:
            FP += 1
            metricsFalse[math.floor(result[0][1])] += 1
            log.info(f'❌ query id: {query['id']}, result: {predicted_path}, eval result: False, score = {result[0][1]}, time = {elapsed_time}')
        
        evaluated_data = {
            "id": query['id'],
            "search": "True" if predicted_id == query['id'] else "False",
            "result": predicted_path,
            "score": result[0][1],
            "time": elapsed_time
        }

        evaluated_dataset.append(evaluated_data)
        with open(config['evaluated_file'], 'w') as f:
            json.dump(evaluated_dataset, f, indent=2)

    total_elapsed_time = time.time() - total_start_time

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    accuracy = TP / TOTAL if TOTAL else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    log.info(f"\n========== Evaluation Result ==========")
    log.info(f"🕒 Time Elapsed: {total_elapsed_time:.2f} seconds")
    log.info(f"✅ True Positives: {TP}")
    log.info(f"❌ False Positives: {FP}")
    log.info(f"❓ False Negatives: {FN}")
    log.info(f"📊 Precision: {precision:.4f}")
    log.info(f"📊 Recall: {recall:.4f}")
    log.info(f"📊 Accuracy: {accuracy:.4f}")
    log.info(f"📊 F1 Score: {f1:.4f}")
    log.info(f"\n📈 VMAF Score Distribution for True:")
    for score in sorted(metricsTrue.keys()):
        log.info(f"  Score {score}: {metricsTrue[score]}")

    log.info(f"\n📈 VMAF Score Distribution for False:")
    for score in sorted(metricsFalse.keys()):
        log.info(f"  Score {score}: {metricsFalse[score]}")
    
if __name__ == "__main__":
    main()
