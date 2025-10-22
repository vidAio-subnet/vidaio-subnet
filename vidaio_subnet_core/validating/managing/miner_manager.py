import bittensor as bt
import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from loguru import logger
import numpy as np
from .sql_schemas import MinerMetadata, MinerPerformanceHistory, Base
from .serving_counter import ServingCounter
from ...global_config import CONFIG
from ...utilities.rate_limit import build_rate_limit
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import desc
import os
from pathlib import Path
import requests
import hashlib
import time

class MinerManager:
    def __init__(self, uid, wallet, metagraph):
        logger.info(f"Initializing MinerManager with uid: {uid}")
        self.uid = uid
        self.wallet = wallet
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = metagraph
        self.config_url = CONFIG.sql.url
        self.local_db_path = "video_subnet_validator.db"
        logger.info(f"Connecting to Redis at {CONFIG.redis.host}:{CONFIG.redis.port}")
        self.redis_client = redis.Redis(
            host=CONFIG.redis.host, port=CONFIG.redis.port, db=CONFIG.redis.db
        )
        
        # Clean up old database files before connecting to new database
        # self._cleanup_old_database_files()
        
        # Download database from URL if needed
        # db_url = self._download_database_from_url()
        
        logger.info(f"Creating SQL engine")
        self.engine = create_engine("sqlite:///video_subnet_validator.db")
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        logger.info("Initializing serving counters")
        if len(self.metagraph.uids) < 256:
            uids = list(range(256)) 
            self.initialize_serving_counter(uids)
        else:
            self.initialize_serving_counter(self.metagraph.uids)

        self.UPSCALING_BONUS_THRESHOLD = 0.32
        self.PENALTY_F_THRESHOLD = 0.07 
        self.PENALTY_Q_THRESHOLD = 0.5  
        
        self.BONUS_MAX = 0.15  
        self.PENALTY_F_MAX = 0.20  
        self.PENALTY_Q_MAX = 0.25  
        
        # Compression-specific constants
        self.COMPRESSION_BONUS_THRESHOLD = 0.74  # S_f > 0.55 for compression bonus
        self.COMPRESSION_PENALTY_F_THRESHOLD = 0.4  # S_f < 0.25 for compression penalty
        self.COMPRESSION_VMAF_MARGIN = 0.0  # VMAF_score < VMAF_threshold + 5 for VMAF penalty
        
        self.COMPRESSION_BONUS_MAX = 0.15  # +15% max bonus
        self.COMPRESSION_PENALTY_F_MAX = 0.20  # -20% max penalty
        self.COMPRESSION_PENALTY_VMAF_MAX = 0.30  # -30% max VMAF penalty
        
        # Compression scoring weights
        self.COMPRESSION_RATE_WEIGHT = 0.8  # w_c
        self.COMPRESSION_VMAF_WEIGHT = 0.2  # w_vmaf
        
        self.MIN_CONTENT_LENGTH = 5.0 
        self.TARGET_CONTENT_LENGTH = 30.0  
        
        self.QUALITY_WEIGHT = 0.5
        self.LENGTH_WEIGHT = 0.5 
        
        self.df = pd.DataFrame(columns=['uid', 'hotkey', 'score'])
        self.df = self.df.set_index('uid')

        logger.success("MinerManager initialization complete")
        
    def _cleanup_old_database_files(self):
        """Delete old local database files if they exist before connecting to new database URL"""
        old_db_files = [
            "video_subnet_validator.db",
            "video_subnet_validator.db-journal",
            "video_subnet_validator.db-wal", 
            "video_subnet_validator.db-shm"
        ]
        
        for db_file in old_db_files:
            db_path = Path(db_file)
            if db_path.exists():
                try:
                    db_path.unlink()
                    logger.info(f"🗑️ Deleted old local database file: {db_file}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to delete old database file {db_file}: {e}")
        
        # Also check for any other .db files in the current directory
        current_dir = Path(".")
        for db_file in current_dir.glob("*.db"):
            if db_file.name.startswith("video_subnet") or db_file.name.startswith("validator"):
                try:
                    db_file.unlink()
                    logger.info(f"🗑️ Deleted old database file: {db_file.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to delete old database file {db_file.name}: {e}")
        
        logger.info("✅ Database cleanup completed")
        
    def _download_database_from_url(self) -> str:
        """Download database file from URL if the config URL is a HTTP/HTTPS URL"""
        
        # Check if the URL is a HTTP/HTTPS download URL
        if not self.config_url.startswith(('http://', 'https://')):
            logger.info(f"Database URL is not HTTP/HTTPS, using as direct connection: {self.config_url}")
            return self.config_url
        
        # Extract filename from URL or use default
        try:
            filename = self.config_url.split('/')[-1]
            if not filename.endswith('.db'):
                filename = "video_subnet_validator.db"
        except:
            filename = "video_subnet_validator.db"
        
        local_db_path = Path(filename)
        
        logger.info(f"📥 Downloading database from URL: {self.config_url}")
        logger.info(f"📁 Local database path: {local_db_path}")
        
        if os.path.exists("video_subnet_validator.db"):
            os.remove("video_subnet_validator.db")

        try:
            # Download the database file
            response = requests.get(self.config_url, timeout=60)
            response.raise_for_status()
            
            # Write the downloaded content to local file
            with open(local_db_path, 'wb') as f:
                f.write(response.content)
            
            # Calculate file hash for verification
            file_hash = hashlib.md5(response.content).hexdigest()
            file_size = len(response.content)
            
            logger.info(f"✅ Database downloaded successfully")
            logger.info(f"📊 File size: {file_size:,} bytes")
            logger.info(f"🔐 MD5 hash: {file_hash}")
            
            # Return SQLite connection string for the downloaded file
            return f"sqlite:///{local_db_path}"
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to download database from URL: {e}")
            raise RuntimeError(f"Database download failed: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error downloading database: {e}")
            raise RuntimeError(f"Database download failed: {e}")
        
    def initialize_serving_counter(self, uids: list[int]):
        rate_limit = build_rate_limit(self.metagraph, self.uid)
        logger.info(f"Creating serving counters for {len(uids)} UIDs")
        self.serving_counters = {
            uid: ServingCounter(
                rate_limit=rate_limit,
                uid=uid,
                redis_client=self.redis_client,
            )
            for uid in uids
        }
        logger.debug(
            f"Serving counters initialized with rate limit: {rate_limit}"
        )

    def query(self, uids: list[int] = []) -> dict[int, MinerMetadata]:
        query = self.session.query(MinerMetadata)
        if uids:
            query = query.filter(MinerMetadata.uid.in_(uids))
        result = {miner.uid: miner for miner in query.all()}
        return result

    def step_synthetics_upscaling(
        self,
        round_id: str,
        total_uids: List[int],
        hotkeys: List[str],
        vmaf_scores: List[float],
        pieapp_scores: List[float],
        scores: List[float],
        length_scores: List[float],
        final_scores: List[float],
        content_lengths: List[float],
        content_type: str = "video"
    ) -> None:
        """
        Process scores from a synthetic mining step and update miner records
        """
        session = self.session

        acc_scores = []
        applied_multipliers = []

        try:
            for i, uid in enumerate(total_uids):
                hotkey = hotkeys[i]
                vmaf_score = vmaf_scores[i]
                pieapp_score = pieapp_scores[i]
                s_q = scores[i]
                s_l = length_scores[i]
                s_f = final_scores[i]
                content_length = content_lengths[i]
                
                miner = session.query(MinerMetadata).filter(MinerMetadata.uid == uid).first()
                
                is_new_miner = False
                task_changed = False

                if not miner:
                    miner = MinerMetadata(
                        processing_task_type="upscaling",
                        uid=uid,
                        hotkey=hotkey,
                        accumulate_score=0.0,
                        bonus_multiplier=1.0,
                        penalty_f_multiplier=1.0,
                        penalty_q_multiplier=1.0,
                        total_multiplier=1.0,
                        performance_tier="New Miner"
                    )
                    session.add(miner)
                    is_new_miner = True
                elif miner.hotkey != hotkey:
                    bt.logging.info(f"Hotkey change detected for UID {uid}: {miner.hotkey} -> {hotkey}")
                    miner.hotkey = hotkey
                    miner.accumulate_score = 0
                    miner.processing_task_type = "upscaling"
                    
                    miner.bonus_multiplier = 1.0
                    miner.penalty_f_multiplier = 1.0
                    miner.penalty_q_multiplier = 1.0
                    miner.total_multiplier = 1.0

                    miner.avg_s_q = 0.0
                    miner.avg_s_l = 0.0
                    miner.avg_s_f = 0.0
                    miner.avg_content_length = 0.0
                    miner.avg_compression_rate = 0.0

                    miner.bonus_count = 0
                    miner.penalty_f_count = 0
                    miner.penalty_q_count = 0

                    miner.total_rounds_completed = 0
                    miner.performance_tier = "New Miner"
                    
                    miner.success_rate = 1
                    miner.longest_content_processed = 0

                    session.query(MinerPerformanceHistory).filter(
                        MinerPerformanceHistory.uid == uid
                    ).delete()
                    
                    is_new_miner = True
                
                elif miner.processing_task_type != "upscaling":
                    miner.processing_task_type = "upscaling"
                    task_changed = True

                    miner.accumulate_score = 0.05
                    miner.bonus_multiplier = 1.0
                    miner.penalty_f_multiplier = 1.0
                    miner.penalty_q_multiplier = 1.0
                    miner.total_multiplier = 1.0

                    miner.avg_s_q = 0.0
                    miner.avg_s_l = 0.0
                    miner.avg_s_f = 0.0
                    miner.avg_content_length = 0.0
                    miner.avg_compression_rate = 0.0

                    miner.bonus_count = 0
                    miner.penalty_f_count = 0
                    miner.penalty_q_count = 0

                    miner.success_rate = 1
                    miner.longest_content_processed = 0

                    miner.total_rounds_completed = 0
                    miner.performance_tier = "New Miner"

                    is_new_miner = True

                    session.query(MinerPerformanceHistory).filter(
                        MinerPerformanceHistory.uid == uid
                    ).delete()

                success = s_f > 0.08

                self._add_performance_record(
                    session,
                    uid, 
                    round_id,
                    vmaf_score,
                    pieapp_score,
                    s_q, 
                    s_l, 
                    s_f, 
                    content_length, 
                    "synthetic",
                    success,
                    processed_task_type="upscaling"
                )
                
                if not is_new_miner:
                    self._update_miner_metadata(session, miner)
                
                applied_multiplier = miner.total_multiplier
                score_with_multiplier = s_f * applied_multiplier
                
                if s_f != -100:
                    miner.accumulate_score = (
                        miner.accumulate_score * CONFIG.score.decay_factor
                        + score_with_multiplier * (1 - CONFIG.score.decay_factor)
                    )
                    miner.accumulate_score = max(0, miner.accumulate_score)

                acc_scores.append(miner.accumulate_score)
                applied_multipliers.append(applied_multiplier)

                miner.total_rounds_completed += 1
                miner.last_update_timestamp = datetime.now()
                
                if miner.total_rounds_completed > 0:
                    success_count = session.query(MinerPerformanceHistory).filter(
                        MinerPerformanceHistory.uid == uid,
                        MinerPerformanceHistory.success == True
                    ).count()
                    miner.success_rate = success_count / miner.total_rounds_completed
                
                if content_length > miner.longest_content_processed:
                    miner.longest_content_processed = content_length
            
            session.commit()
            logger.success(f"Updated metadata for {len(total_uids)} miners")
            
            return acc_scores, applied_multipliers

        except Exception as e:
            session.rollback()
            bt.logging.error(f"Error in step_synthetic: {e}")
            raise
        finally:
            session.close()

    def step_synthetics_compression(
        self,
        round_id: str,
        total_uids: List[int],
        hotkeys: List[str],
        vmaf_scores: List[float],
        final_scores: List[float],
        content_lengths: List[float],
        vmaf_threshold: float,
        compression_rates: List[float],
        content_type: str = "video"
    ) -> None:
        """
        Process scores from a synthetic compression mining step and update miner records
        """
        session = self.session

        acc_scores = []
        applied_multipliers = []

        try:
            for i, uid in enumerate(total_uids):
                hotkey = hotkeys[i]
                vmaf_score = vmaf_scores[i]
                s_f = final_scores[i]
                content_length = content_lengths[i]
                compression_rate = compression_rates[i]
                
                miner = session.query(MinerMetadata).filter(MinerMetadata.uid == uid).first()
                
                is_new_miner = False
                task_changed = False

                if not miner:
                    miner = MinerMetadata(
                        processing_task_type="compression",
                        uid=uid,
                        hotkey=hotkey,
                        accumulate_score=0.0,
                        bonus_multiplier=1.0,
                        penalty_f_multiplier=1.0,
                        total_multiplier=1.0,
                        performance_tier="New Miner"
                    )
                    session.add(miner)
                    is_new_miner = True
                elif miner.hotkey != hotkey:
                    bt.logging.info(f"Hotkey change detected for UID {uid}: {miner.hotkey} -> {hotkey}")
                    miner.hotkey = hotkey
                    miner.accumulate_score = 0
                    miner.processing_task_type = "compression"
                    
                    miner.bonus_multiplier = 1.0
                    miner.penalty_f_multiplier = 1.0
                    miner.penalty_q_multiplier = 1.0
                    miner.total_multiplier = 1.0

                    miner.avg_s_q = 0.0
                    miner.avg_s_l = 0.0
                    miner.avg_s_f = 0.0
                    miner.avg_content_length = 0.0
                    miner.avg_compression_rate = 0.0

                    miner.bonus_count = 0
                    miner.penalty_f_count = 0
                    miner.penalty_q_count = 0

                    miner.total_rounds_completed = 0
                    miner.performance_tier = "New Miner"
                    
                    miner.success_rate = 1
                    miner.longest_content_processed = 0

                    session.query(MinerPerformanceHistory).filter(
                        MinerPerformanceHistory.uid == uid
                    ).delete()
                    
                    is_new_miner = True
                
                elif miner.processing_task_type != "compression":
                    miner.processing_task_type = "compression"
                    task_changed = True

                    miner.accumulate_score = 0.05
                    miner.bonus_multiplier = 1.0
                    miner.penalty_f_multiplier = 1.0
                    miner.penalty_q_multiplier = 1.0
                    miner.total_multiplier = 1.0

                    miner.avg_s_q = 0.0
                    miner.avg_s_l = 0.0
                    miner.avg_s_f = 0.0
                    miner.avg_content_length = 0.0
                    miner.avg_compression_rate = 0.0

                    miner.bonus_count = 0
                    miner.penalty_f_count = 0
                    miner.penalty_q_count = 0

                    miner.success_rate = 1
                    miner.longest_content_processed = 0
                    miner.last_update_timestamp = datetime.now()

                    miner.total_rounds_completed = 0
                    miner.performance_tier = "New Miner"

                    session.query(MinerPerformanceHistory).filter(
                        MinerPerformanceHistory.uid == uid
                    ).delete()

                    task_changed = True
                
                success = s_f > 0.08

                self._add_performance_record(
                    session,
                    uid, 
                    round_id,
                    vmaf_score,
                    0,
                    0,  # s_l not used in compression
                    0,
                    s_f, 
                    content_length, 
                    "synthetic",
                    success,
                    processed_task_type="compression",
                    compression_rate=compression_rate,
                    vmaf_threshold=vmaf_threshold
                )
                
                if not is_new_miner:
                    self._update_miner_metadata_compression(session, miner)
                
                applied_multiplier = miner.total_multiplier 
                score_with_multiplier = s_f * applied_multiplier if compression_rate < 1 else -10 
                
                if s_f != -100:
                    miner.accumulate_score = (
                        miner.accumulate_score * CONFIG.score.decay_factor
                        + score_with_multiplier * (1 - CONFIG.score.decay_factor)
                    )
                    miner.accumulate_score = max(0, miner.accumulate_score)

                acc_scores.append(miner.accumulate_score)
                applied_multipliers.append(applied_multiplier)

                miner.total_rounds_completed += 1
                miner.last_update_timestamp = datetime.now()
                
                if miner.total_rounds_completed > 0:
                    success_count = session.query(MinerPerformanceHistory).filter(
                        MinerPerformanceHistory.uid == uid,
                        MinerPerformanceHistory.success == True
                    ).count()
                    miner.success_rate = success_count / miner.total_rounds_completed
                
                if content_length > miner.longest_content_processed:
                    miner.longest_content_processed = content_length
            
            session.commit()
            logger.success(f"Updated compression metadata for {len(total_uids)} miners")
            
            return acc_scores, applied_multipliers

        except Exception as e:
            session.rollback()
            bt.logging.error(f"Error in step_synthetics_compression: {e}")
            raise
        finally:
            session.close()    

    def _add_performance_record(
        self,
        session: Session,
        uid: int,
        round_id: str,
        vmaf_score: float,
        pie_app_score: float,
        s_q: float,
        s_l: float,
        s_f: float,
        content_length: float,
        content_type: str,
        success: bool,
        processed_task_type: str,
        compression_rate: float = 0.0,
        vmaf_threshold: float = 0.0
    ) -> None:
        """
        Add a new performance record and prune history if needed
        """
        record = MinerPerformanceHistory(
            uid=uid,
            round_id=round_id,
            vmaf_score=vmaf_score,
            pie_app_score=pie_app_score,
            s_q=s_q,
            s_l=s_l,
            s_f=s_f,
            content_length=content_length,
            content_type=content_type,
            success=success,
            processed_task_type=processed_task_type,
            compression_rate=compression_rate,
            vmaf_threshold=vmaf_threshold
        )
        session.add(record)
        session.flush()
        
        self._prune_history(session, uid)

    def _prune_history(self, session: Session, uid: int) -> None:
        """
        Ensure only 10 most recent records are kept per miner
        """
        records = session.query(MinerPerformanceHistory).filter(
            MinerPerformanceHistory.uid == uid
        ).order_by(desc(MinerPerformanceHistory.timestamp)).all()
        
        if len(records) > 10:
            for record in records[10:]:
                session.delete(record)

    def _update_miner_metadata(self, session: Session, miner: MinerMetadata) -> None:
        """
        Recalculate and update miner metadata based on recent performance
        """
        recent_records = session.query(MinerPerformanceHistory).filter(
            MinerPerformanceHistory.uid == miner.uid
        ).order_by(desc(MinerPerformanceHistory.timestamp)).limit(10).all()
        
        if not recent_records:
            return
        
        miner.avg_s_q = sum(r.s_q for r in recent_records) / len(recent_records)
        miner.avg_s_l = sum(r.s_l for r in recent_records) / len(recent_records)
        miner.avg_s_f = sum(r.s_f for r in recent_records) / len(recent_records)
        miner.avg_content_length = sum(r.content_length for r in recent_records) / len(recent_records)
        
        miner.bonus_count = sum(1 for r in recent_records if r.s_f > self.UPSCALING_BONUS_THRESHOLD)
        miner.penalty_f_count = sum(1 for r in recent_records if r.s_f < self.PENALTY_F_THRESHOLD)
        miner.penalty_q_count = sum(1 for r in recent_records if r.s_q < self.PENALTY_Q_THRESHOLD)
        
        miner.bonus_multiplier = 1.0 + (miner.bonus_count / 10) * self.BONUS_MAX
        miner.penalty_f_multiplier = 1.0 - (miner.penalty_f_count / 10) * self.PENALTY_F_MAX
        miner.penalty_q_multiplier = 1.0 - (miner.penalty_q_count / 10) * self.PENALTY_Q_MAX
        
        miner.total_multiplier = (
            miner.bonus_multiplier * 
            miner.penalty_f_multiplier * 
            miner.penalty_q_multiplier
        )
        
        miner.performance_tier = self._calculate_performance_tier(miner.avg_s_f)
        
        if recent_records:
            most_recent = recent_records[0]
            most_recent.applied_multiplier = miner.total_multiplier
            
    def _update_miner_metadata_compression(self, session: Session, miner: MinerMetadata) -> None:
        """
        Recalculate and update miner metadata based on recent compression performance
        """
        recent_records = session.query(MinerPerformanceHistory).filter(
            MinerPerformanceHistory.uid == miner.uid,
            MinerPerformanceHistory.processed_task_type == "compression"
        ).order_by(desc(MinerPerformanceHistory.timestamp)).limit(10).all()
        
        if not recent_records:
            return
        
        miner.avg_s_f = sum(r.s_f for r in recent_records) / len(recent_records)
        miner.avg_content_length = sum(r.content_length for r in recent_records) / len(recent_records)
        miner.avg_compression_rate = sum(r.compression_rate for r in recent_records) / len(recent_records)
        
        # Compression-specific bonus and penalty calculations
        miner.bonus_count = sum(1 for r in recent_records if r.s_f > self.COMPRESSION_BONUS_THRESHOLD)
        miner.penalty_f_count = sum(1 for r in recent_records if r.s_f < self.COMPRESSION_PENALTY_F_THRESHOLD)
        
        # Calculate multipliers
        miner.bonus_multiplier = 1.0 + (miner.bonus_count / 10) * self.COMPRESSION_BONUS_MAX
        miner.penalty_f_multiplier = 1.0 - (miner.penalty_f_count / 10) * self.COMPRESSION_PENALTY_F_MAX
        
        # Total multiplier for compression
        miner.total_multiplier = (
            miner.bonus_multiplier * 
            miner.penalty_f_multiplier
        )
        
        miner.performance_tier = self._calculate_performance_tier_compression(miner.avg_s_f)
        
        if recent_records:
            most_recent = recent_records[0]
            most_recent.applied_multiplier = miner.total_multiplier

    def _calculate_performance_tier(self, avg_s_f: float) -> str:
        """
        Determine performance tier based on average S_F score
        """
        if avg_s_f > 0.4:
            return "Elite"
        elif avg_s_f > 0.3:
            return "Outstanding"
        elif avg_s_f > 0.25:
            return "High Performance"
        elif avg_s_f > 0.2:
            return "Good Performance"
        elif avg_s_f > 0.1:
            return "Average"
        elif avg_s_f > 0.07:
            return "Below Average"
        else:
            return "Poor Performance"

    def _calculate_performance_tier_compression(self, avg_s_f: float) -> str:
        """
        Determine performance tier based on average S_F score for compression
        """
        if avg_s_f > 0.4:
            return "Elite"
        elif avg_s_f > 0.3:
            return "Outstanding"
        elif avg_s_f > 0.25:
            return "High Performance"
        elif avg_s_f > 0.2:
            return "Good Performance"
        elif avg_s_f > 0.1:
            return "Average"
        elif avg_s_f > 0.07:
            return "Below Average"
        else:
            return "Poor Performance"

    def check_database_connection(self):
        local_db_path = "vidaio_subnet_validator.db"
        if not os.path.exists(local_db_path):
            try:
                response = requests.get(self.config_url, timeout=60)
                response.raise_for_status()
                with open(self.local_db_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                logger.error(f"Unexpected error connect to database")

    def get_miner_stats(self, uid: int) -> Dict[str, Any]:
        """
        Get comprehensive stats for a miner
        """
        session = self.Session()
        try:
            metadata = session.query(MinerMetadata).filter(MinerMetadata.uid == uid).first()
            if not metadata:
                return {"error": "Miner not found"}
            
            history = session.query(MinerPerformanceHistory).filter(
                MinerPerformanceHistory.uid == uid
            ).order_by(desc(MinerPerformanceHistory.timestamp)).limit(10).all()
            
            metadata_dict = {
                "uid": metadata.uid,
                "hotkey": metadata.hotkey,
                "accumulate_score": metadata.accumulate_score,
                "bonus_multiplier": metadata.bonus_multiplier,
                "penalty_f_multiplier": metadata.penalty_f_multiplier,
                "penalty_q_multiplier": metadata.penalty_q_multiplier,
                "total_multiplier": metadata.total_multiplier,
                "avg_s_q": metadata.avg_s_q,
                "avg_s_l": metadata.avg_s_l,
                "avg_s_f": metadata.avg_s_f,
                "avg_content_length": metadata.avg_content_length,
                "bonus_count": metadata.bonus_count,
                "penalty_f_count": metadata.penalty_f_count,
                "penalty_q_count": metadata.penalty_q_count,
                "performance_tier": metadata.performance_tier,
                "total_rounds_completed": metadata.total_rounds_completed,
                "success_rate": metadata.success_rate,
                "longest_content_processed": metadata.longest_content_processed,
                "last_update": metadata.last_update_timestamp.isoformat()
            }
            
            history_list = []
            for record in history:
                history_list.append({
                    "round_id": record.round_id,
                    "timestamp": record.timestamp.isoformat(),
                    "s_q": record.s_q,
                    "s_l": record.s_l,
                    "s_f": record.s_f,
                    "content_length": record.content_length,
                    "content_type": record.content_type,
                    "applied_multiplier": record.applied_multiplier
                })
            
            return {
                "metadata": metadata_dict,
                "history": history_list
            }
            
        finally:
            session.close()



    def get_all_miners_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all miners for monitoring and analysis
        """
        session = self.Session()
        try:
            miners = session.query(MinerMetadata).all()
            result = []
            
            for miner in miners:
                result.append({
                    "uid": miner.uid,
                    "hotkey": miner.hotkey,
                    "accumulate_score": miner.accumulate_score,
                    "performance_tier": miner.performance_tier,
                    "total_multiplier": miner.total_multiplier,
                    "avg_s_f": miner.avg_s_f,
                    "total_rounds": miner.total_rounds_completed
                })
            
            return result
        finally:
            session.close()

    def analyze_system_performance(self) -> Dict[str, Any]:
        """
        Analyze overall system performance and score distribution
        """
        session = self.Session()
        try:
            miners = session.query(MinerMetadata).all()
            
            if not miners:
                return {
                    "total_miners": 0,
                    "tier_distribution": {},
                    "average_scores": {"avg_s_q": 0, "avg_s_l": 0, "avg_s_f": 0},
                    "average_multipliers": {"avg_bonus": 1.0, "avg_penalty_f": 1.0, "avg_penalty_q": 1.0, "avg_total": 1.0}
                }
            
            tier_counts = {}
            for miner in miners:
                tier = miner.performance_tier
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            avg_scores = {
                "avg_s_q": sum(m.avg_s_q for m in miners) / len(miners),
                "avg_s_l": sum(m.avg_s_l for m in miners) / len(miners),
                "avg_s_f": sum(m.avg_s_f for m in miners) / len(miners)
            }
            
            avg_multipliers = {
                "avg_bonus": sum(m.bonus_multiplier for m in miners) / len(miners),
                "avg_penalty_f": sum(m.penalty_f_multiplier for m in miners) / len(miners),
                "avg_penalty_q": sum(m.penalty_q_multiplier for m in miners) / len(miners),
                "avg_total": sum(m.total_multiplier for m in miners) / len(miners)
            }
            
            return {
                "total_miners": len(miners),
                "tier_distribution": tier_counts,
                "average_scores": avg_scores,
                "average_multipliers": avg_multipliers
            }
        finally:
            session.close()


    def step_organic_upscaling(self, scores: list[float], total_uids: list[int], round_id: str = None):
        """
        Process scores from organic mining step and update miner records
        Organic scoring should be more lenient than synthetic scoring
        """
        logger.info(f"Updating organic scores for {len(total_uids)} miners")
        
        if round_id is None:
            round_id = f"organic_{int(time.time())}"
        
        acc_scores = []
        applied_multipliers = []

        try:
            for uid, score in zip(total_uids, scores):
                # Skip processing if score is -1 (skipped)
                if score == -1:
                    logger.debug(f"Skipping UID {uid} due to score -1")
                    # Get current miner state for return values
                    miner = self.query([uid]).get(uid, None)
                    if miner:
                        acc_scores.append(miner.accumulate_score)
                        applied_multipliers.append(miner.total_multiplier)
                    else:
                        acc_scores.append(0.0)
                        applied_multipliers.append(1.0)
                    continue
                
                miner = self.query([uid]).get(uid, None)
                
                if miner is None:
                    logger.info(f"Creating new metadata record for UID {uid}")
                    miner = MinerMetadata(
                        uid=uid,
                        hotkey="",  # Will be updated when synthetic scoring runs
                        accumulate_score=0.0,
                        bonus_multiplier=1.0,
                        penalty_f_multiplier=1.0,
                        penalty_q_multiplier=1.0,
                        total_multiplier=1.0,
                        performance_tier="New Miner"
                    )
                    self.session.add(miner)
                
                # Convert organic scores to synthetic-like format

                acc_score = 0.0
                current_score = miner.accumulate_score

                if score == 3.0:
                    organic_s_f = 0.4
                    organic_s_q = 0.5
                    organic_s_l = 0.5
                    success = True

                    boost_percentage = 0.03
                    boost_amount = current_score * boost_percentage
                    acc_score = current_score + boost_amount

                elif score == 2.0:
                    organic_s_f = 0.3
                    organic_s_q = 0.5
                    organic_s_l = 0.5
                    success = True

                    acc_score = miner.accumulate_score

                elif score == 1.0:  # Failure
                    organic_s_f = 0.2  # Moderate success score
                    organic_s_q = 0.5   # Moderate quality score
                    organic_s_l = 0.5   # Moderate length score
                    success = False

                    panelty_percentage = 0.05
                    panelty_amount = current_score * panelty_percentage
                    acc_score = current_score - panelty_amount

                elif score == 0.0:  # Failure
                    organic_s_f = 0.0
                    organic_s_q = 0.0
                    organic_s_l = 0.0
                    success = False

                    panelty_percentage = 0.15  # Deduct 15% of current score
                    panelty_amount = current_score * panelty_percentage
                    acc_score = current_score - panelty_amount

                # Add performance record for organic scoring
                self._add_performance_record(
                    self.session,
                    uid,
                    round_id,
                    vmaf_score=0.0,  # Not applicable for organic
                    pie_app_score=0.0, # Not applicable for organic
                    s_q=organic_s_q,
                    s_l=organic_s_l,
                    s_f=organic_s_f,
                    content_length=10.0,  # Default length for organic
                    content_type="organic",
                    success=success,
                    processed_task_type='upscaling'
                )
                
                self._update_miner_metadata(self.session, miner)
                
                acc_scores.append(acc_score)
                miner.accumulate_score = acc_score
                applied_multipliers.append(1.0)
                
                miner.total_rounds_completed += 1
                miner.last_update_timestamp = datetime.now()
                
                # Update success rate
                if miner.total_rounds_completed > 0:
                    success_count = self.session.query(MinerPerformanceHistory).filter(
                        MinerPerformanceHistory.uid == uid,
                        MinerPerformanceHistory.success == True
                    ).count()
                    miner.success_rate = success_count / miner.total_rounds_completed
            
            self.session.commit()
            logger.success(f"Updated organic metadata for {len(total_uids)} miners")
            
            return acc_scores, applied_multipliers
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error in step_organics: {e}")
            raise
        finally:
            self.session.close()

    def step_organic_compression(self, scores: list[float], total_uids: list[int], vmaf_scores: list[float], compression_rates: list[float], vmaf_thresholds: list[float], round_id: str = None):
        """
        Process scores from organic mining step and update miner records
        Organic scoring should be more lenient than synthetic scoring
        """
        logger.info(f"Updating organic scores for {len(total_uids)} miners")
        
        if round_id is None:
            round_id = f"organic_{int(time.time())}"
        
        acc_scores = []
        applied_multipliers = []

        try:
            for uid, score, vmaf_score, compression_rate, vmaf_threshold in zip(total_uids, scores, vmaf_scores, compression_rates, vmaf_thresholds):
                # Skip processing if score is -1 (skipped)
                if score == -1:
                    logger.debug(f"Skipping UID {uid} due to score -1 (skipped)")
                    # Get current miner state for return values
                    miner = self.query([uid]).get(uid, None)
                    if miner:
                        acc_scores.append(miner.accumulate_score)
                        applied_multipliers.append(miner.total_multiplier)
                    else:
                        acc_scores.append(0.0)
                        applied_multipliers.append(1.0)
                    continue
                
                miner = self.query([uid]).get(uid, None)
                
                if miner is None:
                    logger.info(f"Creating new metadata record for UID {uid}")
                    miner = MinerMetadata(
                        uid=uid,
                        hotkey="",  # Will be updated when synthetic scoring runs
                        accumulate_score=0.0,
                        bonus_multiplier=1.0,
                        penalty_f_multiplier=1.0,
                        penalty_q_multiplier=1.0,
                        total_multiplier=1.0,
                        performance_tier="New Miner"
                    )
                    self.session.add(miner)

                success = score > 0.5

                # Add performance record for organic scoring
                self._add_performance_record(
                    self.session,
                    uid,
                    round_id,
                    vmaf_score=vmaf_score,  
                    pie_app_score=0.0, 
                    s_q=0,
                    s_l=0,
                    s_f=score,
                    content_length=10.0,
                    content_type="organic",             
                    success=success,
                    processed_task_type='compression',
                    compression_rate=compression_rate,
                    vmaf_threshold=vmaf_threshold
                )
                
                # Update miner metadata (recalculate multipliers, etc.)
                self._update_miner_metadata(self.session, miner)
                
                # Apply multiplier to organic score
                applied_multiplier = miner.total_multiplier
                score_with_multiplier = score * applied_multiplier if compression_rate < 1 else -10 
                
                # Accumulate score with decay factor (same as synthetic)
                miner.accumulate_score = (
                    miner.accumulate_score * CONFIG.score.decay_factor
                    + score_with_multiplier * (1 - CONFIG.score.decay_factor)
                )
                miner.accumulate_score = max(0, miner.accumulate_score)
                
                acc_scores.append(miner.accumulate_score)
                applied_multipliers.append(applied_multiplier)
                
                # Update miner statistics
                miner.total_rounds_completed += 1
                miner.last_update_timestamp = datetime.now()
                
                # Update success rate
                if miner.total_rounds_completed > 0:
                    success_count = self.session.query(MinerPerformanceHistory).filter(
                        MinerPerformanceHistory.uid == uid,
                        MinerPerformanceHistory.success == True
                    ).count()
                    miner.success_rate = success_count / miner.total_rounds_completed
            
            self.session.commit()
            logger.success(f"Updated organic metadata for {len(total_uids)} miners")
            
            return acc_scores, applied_multipliers
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error in step_organics: {e}")
            raise
        finally:
            self.session.close()

    def consume(self, uids: list[int]) -> list[int]:
        logger.info(f"Consuming {len(uids)} UIDs")
        filtered_uids = [uid for uid in uids if self.serving_counters[uid].increment()]
        logger.info(f"Filtered to {len(filtered_uids)} UIDs after rate limiting")

        return filtered_uids

    def get_miner_task_info(self) -> tuple[List[int], List[str], List[float]]:
        """
        Get uid, processing_task_type, and avg_content_length for all miners
        Returns three lists: (uids, processing_task_types, avg_content_lengths)
        """
        session = self.session
        try:
            miners = session.query(
                MinerMetadata.uid,
                MinerMetadata.processing_task_type,
                MinerMetadata.avg_content_length
            ).all()
            
            uids = [miner.uid for miner in miners]
            processing_task_types = [miner.processing_task_type for miner in miners]
            avg_content_lengths = [miner.avg_content_length for miner in miners]
            
            return uids, processing_task_types, avg_content_lengths
            
        except Exception as e:
            logger.error(f"Error getting miner task info: {e}")
            return [], [], []
        finally:
            session.close()

    def get_miner_processing_task_types(self, uids: List[int]) -> Dict[int, str]:
        """
        Get processing_task_type for specific UIDs from the database.
        Returns a dictionary mapping UID to processing_task_type.
        If a miner doesn't exist or has no processing_task_type, it won't be in the result.
        """
        session = self.session
        try:
            miners = session.query(
                MinerMetadata.uid,
                MinerMetadata.processing_task_type
            ).filter(
                MinerMetadata.uid.in_(uids),
                MinerMetadata.processing_task_type.isnot(None)
            ).all()
            
            uid_to_task_type = {miner.uid: miner.processing_task_type for miner in miners}
            
            return uid_to_task_type
            
        except Exception as e:
            logger.error(f"Error getting miner processing task types: {e}")
            return {}
        finally:
            session.close()

    @property
    def weights(self):
        """
        Calculate weights for reward distribution with 60% for compression and 40% for upscaling.
        Within each task type, rewards are distributed proportionally based on accumulated scores.
        """
        # Collect miners by task type
        compression_miners = []
        upscaling_miners = []

        self.check_database_connection()

        for uid, miner in self.query().items():
            # Exclude validator from getting weights set
            if miner.accumulate_score == -1 or self.metagraph.validator_permit[uid]: 
                continue
            if miner.processing_task_type == "compression":
                compression_miners.append((uid, miner.accumulate_score))
            elif miner.processing_task_type == "upscaling":
                upscaling_miners.append((uid, miner.accumulate_score))
        
        # Initialize result arrays
        uids = []
        scores = []
        compression_count = len(compression_miners)
        upscaling_count = len(upscaling_miners)
        alloc_comp, alloc_up = 0.6, 0.4

        if compression_count == 0 and upscaling_count > 0:
            alloc_comp, alloc_up = 0.0, 1.0
        elif upscaling_count == 0 and compression_count > 0:
            alloc_comp, alloc_up = 1.0, 0.0

        # Process compression miners (60% of total rewards)
        if compression_miners:
            compression_uids, compression_scores = zip(*compression_miners)
            compression_scores = np.array(compression_scores)
            
            # Normalize compression scores and apply 60% allocation
            if compression_scores.sum() > 0:
                compression_weights = (compression_scores / compression_scores.sum()) * alloc_comp
            else:
                # If no scores, distribute 60% equally among compression miners
                compression_weights = np.full(len(compression_scores), alloc_comp / len(compression_scores))
            
            uids.extend(compression_uids)
            scores.extend(compression_weights)
        
        # Process upscaling miners (40% of total rewards)
        if upscaling_miners:
            upscaling_uids, upscaling_scores = zip(*upscaling_miners)
            upscaling_scores = np.array(upscaling_scores)
            
            # Normalize upscaling scores and apply 40% allocation
            if upscaling_scores.sum() > 0:
                upscaling_weights = (upscaling_scores / upscaling_scores.sum()) * alloc_up
            else:
                # If no scores, distribute 40% equally among upscaling miners
                upscaling_weights = np.full(len(upscaling_scores), alloc_up / len(upscaling_scores))
            
            uids.extend(upscaling_uids)
            scores.extend(upscaling_weights)
        
        # Convert to numpy arrays and sort by UID
        uids = np.array(uids)
        scores = np.array(scores)
        
        # Sort by UID for consistent ordering
        sorted_indices = np.argsort(uids)
        uids = uids[sorted_indices]
        scores = scores[sorted_indices]
        
        # Log distribution summary
        compression_count = len(compression_miners)
        upscaling_count = len(upscaling_miners)
        compression_total_weight = sum(scores[i] for i, uid in enumerate(uids) if uid in [c[0] for c in compression_miners])
        upscaling_total_weight = sum(scores[i] for i, uid in enumerate(uids) if uid in [u[0] for u in upscaling_miners])
        
        logger.info(f"Reward distribution: {compression_count} compression miners ({compression_total_weight:.3f} weight), "
                   f"{upscaling_count} upscaling miners ({upscaling_total_weight:.3f} weight)")
        
        return uids, scores
