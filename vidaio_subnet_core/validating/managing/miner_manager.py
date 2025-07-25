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
        logger.info(f"Connecting to Redis at {CONFIG.redis.host}:{CONFIG.redis.port}")
        self.redis_client = redis.Redis(
            host=CONFIG.redis.host, port=CONFIG.redis.port, db=CONFIG.redis.db
        )
        
        # Clean up old database files before connecting to new database
        self._cleanup_old_database_files()
        
        # Download database from URL if needed
        db_url = self._download_database_from_url()
        
        logger.info(f"Creating SQL engine with URL: {db_url}")
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        logger.info("Initializing serving counters")
        if len(self.metagraph.uids) < 256:
            uids = list(range(256)) 
            self.initialize_serving_counter(uids)
        else:
            self.initialize_serving_counter(self.metagraph.uids)

        self.BONUS_THRESHOLD = 0.3
        self.PENALTY_F_THRESHOLD = 0.07 
        self.PENALTY_Q_THRESHOLD = 0.5  
        
        self.BONUS_MAX = 0.15  
        self.PENALTY_F_MAX = 0.20  
        self.PENALTY_Q_MAX = 0.25  
        
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
        config_url = CONFIG.sql.url
        
        # Check if the URL is a HTTP/HTTPS download URL
        if not config_url.startswith(('http://', 'https://')):
            logger.info(f"Database URL is not HTTP/HTTPS, using as direct connection: {config_url}")
            return config_url
        
        # Extract filename from URL or use default
        try:
            filename = config_url.split('/')[-1]
            if not filename.endswith('.db'):
                filename = "video_subnet_validator.db"
        except:
            filename = "video_subnet_validator.db"
        
        local_db_path = Path(filename)
        
        logger.info(f"📥 Downloading database from URL: {config_url}")
        logger.info(f"📁 Local database path: {local_db_path}")
        
        try:
            # Download the database file
            response = requests.get(config_url, timeout=60)
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
            logger.error(f"💡 Please check if the URL is accessible: {config_url}")
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

    def step_synthetics(
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
                if not miner:
                    miner = MinerMetadata(
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
                    
                    miner.bonus_multiplier = 1.0
                    miner.penalty_f_multiplier = 1.0
                    miner.penalty_q_multiplier = 1.0
                    miner.total_multiplier = 1.0

                    miner.avg_s_q = 0.0
                    miner.avg_s_l = 0.0
                    miner.avg_s_f = 0.0
                    miner.avg_content_length = 0.0

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
                    content_type,
                    success
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
        success: bool
    ) -> None:
        """
        Add a new performance record and prune history if needed
        """
        new_record = MinerPerformanceHistory(
            uid=uid,
            round_id=round_id,
            timestamp=datetime.now(),
            vmaf_score=vmaf_score,
            pie_app_score=pie_app_score,
            s_q=s_q,
            s_l=s_l,
            s_f=s_f,
            content_length=content_length,
            content_type=content_type,
            applied_multiplier=1.0,  
            success=success
        )
        session.add(new_record)
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
        
        miner.bonus_count = sum(1 for r in recent_records if r.s_f > self.BONUS_THRESHOLD)
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


    def step_organics(self, scores: list[float], total_uids: list[int], round_id: str = None):
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
                
                # Convert organic scores to synthetic-like format
                if score == 1.0:  # Success
                    organic_s_f = 0.3  # Moderate success score
                    organic_s_q = 0.5   # Moderate quality score
                    organic_s_l = 0.5   # Moderate length score
                    success = True
                elif score == 0.0:  # Failure
                    # Deduct from current accumulated score for failure
                    deduction_factor = 0.1  # Deduct 10% of current score
                    current_score = miner.accumulate_score
                    deduction = current_score * deduction_factor
                    organic_s_f = -deduction  # Negative score to deduct
                    organic_s_q = 0.0
                    organic_s_l = 0.0
                    success = False
                else:
                    # Unknown score value
                    organic_s_f = 0.0
                    organic_s_q = 0.0
                    organic_s_l = 0.0
                    success = False
                
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
                    content_length=20.0,  # Default length for organic
                    content_type="organic",
                    success=success
                )
                
                # Update miner metadata (recalculate multipliers, etc.)
                self._update_miner_metadata(self.session, miner)
                
                # Apply multiplier to organic score
                applied_multiplier = miner.total_multiplier
                score_with_multiplier = organic_s_f * applied_multiplier
                
                # Accumulate score with decay factor (same as synthetic)
                if organic_s_f != -100:  # Not a system error
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
                
                # Update longest content processed (organic default)
                if 20.0 > miner.longest_content_processed:
                    miner.longest_content_processed = 20.0
            
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

    @property
    def weights(self):
        uids = []
        scores = []
        
        # Collect uids and scores
        for uid, miner in self.query().items():
            uids.append(uid)
            if miner.accumulate_score == -1:
                continue
            else:
                scores.append(miner.accumulate_score)

        scores = np.array(scores)

        scores = scores / scores.sum()

        sorted_indices = np.argsort(uids)  
        uids = np.array(uids)[sorted_indices]  
        scores = scores[sorted_indices]  

        return uids, scores
