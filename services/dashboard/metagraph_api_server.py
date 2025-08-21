import os
import time
import sqlite3
import uvicorn
import bittensor as bt
from fastapi import FastAPI
from typing import Dict, Tuple
from contextlib import contextmanager
from vidaio_subnet_core import CONFIG
from services.dashboard.model import MinerInfo
from loguru import logger

DATABASE_PATH = "video_subnet_validator.db"

@contextmanager
def get_database_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = connect_to_database(DATABASE_PATH)
        yield conn
    finally:
        if conn:
            conn.close()

def connect_to_database(db_path: str) -> sqlite3.Connection:
    """Connect to the SQLite database"""
    if not db_path:
        raise ValueError("Database path cannot be empty")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        return conn
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to connect to database {db_path}: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error connecting to database {db_path}: {e}")

def get_table_info(conn: sqlite3.Connection, table_name: str) -> list:
    """Get information about the table structure"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return cursor.fetchall()

def get_miner_statistics(conn: sqlite3.Connection) -> Tuple[int, int, int, Dict[str, int]]:
    """Get comprehensive miner statistics in a single query"""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total_count,
            SUM(CASE WHEN processing_task_type = 'compression' THEN 1 ELSE 0 END) as compression_count,
            SUM(CASE WHEN processing_task_type = 'upscaling' THEN 1 ELSE 0 END) as upscaling_count,
            processing_task_type,
            COUNT(*) as type_count
        FROM miner_metadata 
        GROUP BY processing_task_type
        ORDER BY type_count DESC
    """)
    
    results = cursor.fetchall()
    
    total_count = 0
    compression_count = 0
    upscaling_count = 0
    task_type_distribution = {}
    
    for row in results:
        if row[3] is not None:
            task_type_distribution[row[3]] = row[4]
            if row[3] == 'compression':
                compression_count = row[4]
            elif row[3] == 'upscaling':
                upscaling_count = row[4]
            total_count += row[4]
    
    return compression_count, upscaling_count, total_count, task_type_distribution

def get_processing_task_types() -> Dict[str, str]:
    """Get a dictionary mapping UID to processing_task_type from the miner_metadata table"""
    with get_database_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT uid, processing_task_type 
            FROM miner_metadata 
            WHERE uid IS NOT NULL AND processing_task_type IS NOT NULL
        """)
        
        uid_to_task_type = {}
        for row in cursor.fetchall():
            uid, task_type = row
            uid_to_task_type[str(uid)] = task_type
        
        return uid_to_task_type

def get_miner_counts_by_task():
    """Main function to execute the miner counting script"""
    try:
        print("Connecting to database...")
        
        with get_database_connection() as conn:
            print(f"Database connected successfully: {DATABASE_PATH}")
            print("-" * 50)
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='miner_metadata'
            """)
            
            if not cursor.fetchone():
                error_msg = "miner_metadata table not found in the database"
                print(f"Error: {error_msg}")
                raise ValueError(error_msg)
            
            print("Table structure for miner_metadata:")
            table_info = get_table_info(conn, "miner_metadata")
            for column in table_info:
                print(f"  {column[1]} ({column[2]})")
            print("-" * 50)
            
            compression_count, upscaling_count, total_count, task_type_distribution = get_miner_statistics(conn)
            
            print("All unique processing_task_type values:")
            for task_type, count in task_type_distribution.items():
                print(f"  {task_type}: {count} miners")
            print("-" * 50)
            
            print("MINER COUNT SUMMARY:")
            print(f"Compression miners: {compression_count}")
            print(f"Upscaling miners:  {upscaling_count}")
            print(f"Total miners:      {total_count}")
            print("-" * 50)
            
            if total_count > 0:
                compression_pct = (compression_count / total_count) * 100
                upscaling_pct = (upscaling_count / total_count) * 100
                print(f"Compression miners: {compression_pct:.1f}%")
                print(f"Upscaling miners:  {upscaling_pct:.1f}%")
            
            return (compression_count, upscaling_count)
            
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        raise FileNotFoundError(f"Database file not found: {DATABASE_PATH}")
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        raise sqlite3.Error(f"Database operation failed: {e}")
    except ValueError as e:
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise Exception(f"Unexpected error occurred: {e}")

config = CONFIG.bandwidth
threshold = config.min_stake

MAX_RETRIES = 3
RETRY_DELAY = 1

app = FastAPI(title="VidaIO Subnet Dashboard API", version="1.0.0")

@app.get("/health")
def health_check():
    """Health check endpoint to verify API is running"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "VidaIO Subnet Dashboard API"
    }

def get_filtered_miners(threshold=threshold):
    """Get filtered miners with optimized database operations - reads DB once per request"""
    miner_info = MinerInfo()
    subtensor = bt.subtensor()
    metagraph = subtensor.metagraph(netuid=85)
    
    incentives = metagraph.I
    trusts = metagraph.T
    emissions = metagraph.E
    hotkeys = metagraph.hotkeys
    stakes = metagraph.S
    uids = metagraph.uids
    daily_rewards = emissions * 20
    
    uid_to_task_type = get_processing_task_types()
    
    for uid, incentive, trust, emission, hotkey, stake, daily_reward in zip(
        uids, incentives, trusts, emissions, hotkeys, stakes, daily_rewards
    ):
        if stake < threshold:
            processing_task_type = uid_to_task_type.get(str(uid), "unknown")
            
            miner_info.append(
                miner_uid=uid,
                miner_hotkey=hotkey,
                trust=trust,
                incentive=incentive,
                emission=emission,
                daily_reward=daily_reward,
                processing_task_type=processing_task_type
            )
    
    return miner_info

@app.get("/miner_counts")
def get_miner_counts():
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Attempting to get miner counts (attempt {attempt + 1}/{MAX_RETRIES})")
            compression_count, upscaling_count = get_miner_counts_by_task()
            
            if compression_count is None or upscaling_count is None:
                raise ValueError("Database query returned None values")
            
            if not isinstance(compression_count, int) or not isinstance(upscaling_count, int):
                raise ValueError("Database query returned non-integer values")
            
            if compression_count < 0 or upscaling_count < 0:
                raise ValueError("Miner counts cannot be negative")
            
            print(f"Successfully retrieved miner counts: compression={compression_count}, upscaling={upscaling_count}")
            
            return {
                "status": "success",
                "compression_count": compression_count,
                "upscaling_count": upscaling_count,
                "total_count": compression_count + upscaling_count,
                "attempt": attempt + 1,
                "timestamp": time.time()
            }
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            
            if attempt == MAX_RETRIES - 1:
                print(f"All {MAX_RETRIES} attempts failed. Returning error response.")
                return {
                    "status": "error",
                    "message": f"Failed after {MAX_RETRIES} attempts: {str(e)}",
                    "compression_count": 0,
                    "upscaling_count": 0,
                    "total_count": 0,
                    "attempt": attempt + 1,
                    "timestamp": time.time()
                }
            
            # Wait before retrying
            print(f"Waiting {RETRY_DELAY} seconds before retry...")
            time.sleep(RETRY_DELAY)
            continue

@app.get("/miner_info")
def get_miner_info():
    try:
        miner_info = get_filtered_miners()
        return {
            "status": "success",
            "data": miner_info.to_dict(),
            "timestamp": time.time()
        }
    except Exception as e:
        print(f"Error getting miner info: {e}")
        return {
            "status": "error",
            "message": f"Failed to retrieve miner information: {str(e)}",
            "data": {"miners": []},
            "timestamp": time.time()
        }

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting dashboard service")
    logger.info(f"Dashboard service running on http://{CONFIG.dashboard.host}:{CONFIG.dashboard.port}")

    uvicorn.run(app, host=CONFIG.dashboard.host, port=CONFIG.dashboard.port)
