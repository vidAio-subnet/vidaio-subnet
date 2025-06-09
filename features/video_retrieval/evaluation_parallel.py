import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from modules.config import config
from modules.search_engine import VideoSearchEngine
from modules.vmaf_metric import evaluate_vmaf
from modules.search import HashDB
from collections import defaultdict
import os, math, time, logging

log_filename = "evaluation_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

def process_query(query_idx_pair):
    query, idx = query_idx_pair
    try:
        engine = VideoSearchEngine(log=log)
        start_time = time.time()
        result = engine.searchHash(query=query, top_n=1)
        elapsed = time.time() - start_time

        if len(result) == 0:
            return {"type": "FN", "id": query["id"], "idx": idx, "vmaf": None, "time": elapsed}

        predicted_path, predicted_score, predicted_id = result[0]
        # trim_path = f"{config['default_video_dir']}/{query['id']}_trim.mp4"
        # vmaf_score = evaluate_vmaf(predicted_path, trim_path)
        os.remove(predicted_path)

        correct = predicted_id == query['id']
        return {
            "type": "TP" if correct else "FP",
            "id": query["id"],
            "predicted": predicted_path,
            "idx": idx,
            "vmaf": math.floor(predicted_score),
            "time": elapsed
        }
    except Exception as e:
        return {"type": "ERROR", "id": query["id"], "idx": idx, "error": str(e)}

def main():
    start = time.time()
    queryDB = HashDB(mongo_uri=config['mongo_uri'], db_name=config['db_name'], collection_name=config['collection'])
    queryData = queryDB.get_all_video_hashes(feature='downscale')

    log.info(f"Total queries: {len(queryData)}")
    TP, FP, FN = 0, 0, 0
    metricsTrue, metricsFalse = defaultdict(int), defaultdict(int)

    tasks = list(enumerate(queryData, 1))
    # with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_query, (query, idx)) for idx, query in tasks]
        for future in as_completed(futures):
            result = future.result()
            if result["type"] == "FN":
                FN += 1
                log.info(f'❓ search failed! id: {result["id"]}, idx: {result["idx"]}')
            elif result["type"] == "TP":
                TP += 1
                metricsTrue[result["vmaf"]] += 1
                log.info(f'✅ query id: {result["id"]}, result: {result["predicted"]}, down score = {result["vmaf"]:.2f}, time = {result["time"]:.2f}')
            elif result["type"] == "FP":
                FP += 1
                metricsFalse[result["vmaf"]] += 1
                log.info(f'❌ query id: {result["id"]}, result: {result["predicted"]}, down score = {result["vmaf"]:.2f}, time = {result["time"]:.2f}')
            elif result["type"] == "ERROR":
                ERR += 1
                log.error(f'❌ ERROR for query id {result["id"]} ({result["id"]}): {result["error"]}')

    total = len(queryData)
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    accuracy = TP / total if total else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    elapsed = time.time() - start

    log.info(f"\n========== Evaluation Result ==========")
    log.info(f"🕒 Time Elapsed: {elapsed:.2f} seconds")
    log.info(f"✅ True Positives: {TP}")
    log.info(f"❌ False Positives: {FP}")
    log.info(f"❓ False Negatives: {FN}")
    log.info(f"💥 Errors: {ERR}")
    log.info(f"📊 Precision: {precision:.4f}")
    log.info(f"📊 Recall: {recall:.4f}")
    log.info(f"📊 Accuracy: {accuracy:.4f}")
    log.info(f"📊 F1 Score: {f1:.4f}")

    log.info("\n📈 VMAF Distribution (TP):")
    for k in sorted(metricsTrue): log.info(f"  Score {k}: {metricsTrue[k]}")
    log.info("\n📈 VMAF Distribution (FP):")
    for k in sorted(metricsFalse): log.info(f"  Score {k}: {metricsFalse[k]}")

if __name__ == "__main__":
    main()
