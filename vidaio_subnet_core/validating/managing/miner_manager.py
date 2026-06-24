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
from sqlalchemy import desc, text
import os
from pathlib import Path
import requests
import hashlib
import time
import ipaddress

class MinerManager:
    def __init__(self, uid, config, wallet, metagraph):
        logger.info(f"Initializing MinerManager with uid: {uid}")
        self.uid = uid
        self.wallet = wallet
        self.dendrite = bt.Dendrite(wallet=self.wallet)

        self.config = config
        self.subtensor = bt.Subtensor(config=self.config)
        self.burn_proportion = 0.95   # 95% of miner emissions burnt

        # Task allocations and rank-based distribution within each task pool.
        self.compression_emission_allocation = 0.60
        self.upscaling_emission_allocation = 0.40
        self.emission_rank_shares = [0.60, 0.20, 0.10, 0.06, 0.04]

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
        self._migrate_miner_metadata_table()
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.sync_miner_chain_metadata()
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
        self.COMPRESSION_BONUS_THRESHOLD = 0.74  # S_f > 0.74 for compression bonus
        self.COMPRESSION_PENALTY_F_THRESHOLD = 0.4  # S_f < 0.4 for compression penalty
        
        self.COMPRESSION_BONUS_MAX = 0.15  # +15% max bonus
        self.COMPRESSION_PENALTY_F_MAX = 0.20  # -20% max penalty
        # Compression scoring weights
        self.COMPRESSION_RATE_WEIGHT = 0.7  # w_c
        self.COMPRESSION_VMAF_WEIGHT = 0.3  # w_vmaf
        
        self.MIN_CONTENT_LENGTH = 5.0 
        self.TARGET_CONTENT_LENGTH = 30.0  
        
        self.QUALITY_WEIGHT = 0.5
        self.LENGTH_WEIGHT = 0.5 
        
        self.df = pd.DataFrame(columns=['uid', 'hotkey', 'score'])
        self.df = self.df.set_index('uid')

        logger.success("MinerManager initialization complete")

    def _migrate_miner_metadata_table(self) -> None:
        """Add miner chain metadata columns without recreating existing SQLite data."""
        migrations = {
            "coldkey": "ALTER TABLE miner_metadata ADD COLUMN coldkey VARCHAR(64) NOT NULL DEFAULT ''",
            "ip_address": "ALTER TABLE miner_metadata ADD COLUMN ip_address VARCHAR(64) NOT NULL DEFAULT ''",
            "port": "ALTER TABLE miner_metadata ADD COLUMN port INTEGER NOT NULL DEFAULT 0",
        }

        with self.engine.begin() as connection:
            existing_columns = {
                row[1] for row in connection.execute(text("PRAGMA table_info(miner_metadata)"))
            }
            for column_name, statement in migrations.items():
                if column_name not in existing_columns:
                    logger.info(f"Applying miner metadata migration: add {column_name}")
                    connection.execute(text(statement))

    def _axon_ip_address(self, axon) -> str:
        ip_value = None
        for attr in ("ip_str", "external_ip", "ip"):
            candidate = getattr(axon, attr, None)
            if callable(candidate):
                try:
                    candidate = candidate()
                except Exception as e:
                    logger.debug(f"Unable to resolve axon {attr}: {e}")
                    candidate = None

            if candidate is not None and candidate != "":
                ip_value = candidate
                break

        if ip_value is None:
            return ""
        if isinstance(ip_value, int):
            try:
                return str(ipaddress.ip_address(ip_value))
            except ValueError:
                return str(ip_value)

        ip_text = str(ip_value).strip()
        if ip_text.startswith("/ipv"):
            parts = ip_text.split("/", 2)
            if len(parts) == 3:
                ip_text = parts[2]

        if ip_text.startswith("["):
            closing_bracket = ip_text.find("]")
            if closing_bracket != -1:
                return ip_text[1:closing_bracket]

        try:
            return str(ipaddress.ip_address(ip_text))
        except ValueError:
            host, separator, port = ip_text.rpartition(":")
            if separator and port.isdigit():
                return host
            return ip_text

    def _chain_metadata_for_uid(self, uid: int) -> dict[str, Any]:
        axon = self.metagraph.axons[uid]
        hotkey = ""
        coldkey = ""
        if uid < len(getattr(self.metagraph, "hotkeys", [])):
            hotkey = self.metagraph.hotkeys[uid]
        if uid < len(getattr(self.metagraph, "coldkeys", [])):
            coldkey = self.metagraph.coldkeys[uid]

        return {
            "hotkey": hotkey or getattr(axon, "hotkey", "") or "",
            "coldkey": coldkey or getattr(axon, "coldkey", "") or "",
            "ip_address": self._axon_ip_address(axon),
            "port": int(getattr(axon, "port", 0) or 0),
        }

    def _apply_chain_metadata(self, miner: MinerMetadata, uid: int, fallback_hotkey: str = "") -> None:
        try:
            metadata = self._chain_metadata_for_uid(uid)
        except Exception as e:
            logger.warning(f"Unable to fetch metagraph metadata for UID {uid}: {e}")
            metadata = {"hotkey": fallback_hotkey, "coldkey": "", "ip_address": "", "port": 0}

        miner.hotkey = metadata["hotkey"] or fallback_hotkey or miner.hotkey
        miner.coldkey = metadata["coldkey"]
        miner.ip_address = metadata["ip_address"]
        miner.port = metadata["port"]

    def _new_miner_metadata(
        self,
        uid: int,
        processing_task_type: str | None = None,
        fallback_hotkey: str = "",
        **kwargs,
    ) -> MinerMetadata:
        try:
            metadata = self._chain_metadata_for_uid(uid)
        except Exception as e:
            logger.warning(f"Unable to fetch metagraph metadata for new UID {uid}: {e}")
            metadata = {"hotkey": fallback_hotkey, "coldkey": "", "ip_address": "", "port": 0}

        return MinerMetadata(
            uid=uid,
            hotkey=metadata["hotkey"] or fallback_hotkey,
            coldkey=metadata["coldkey"],
            ip_address=metadata["ip_address"],
            port=metadata["port"],
            processing_task_type=processing_task_type,
            **kwargs,
        )

    def sync_miner_chain_metadata(self, uids: List[int] | None = None) -> None:
        session = self.session
        try:
            query = session.query(MinerMetadata)
            if uids:
                query = query.filter(MinerMetadata.uid.in_(uids))

            updated = 0
            for miner in query.all():
                self._apply_chain_metadata(miner, miner.uid, miner.hotkey)
                updated += 1

            if updated:
                session.commit()
                logger.info(f"Synced metagraph metadata for {updated} miner records")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to sync miner chain metadata: {e}")

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
                    miner = self._new_miner_metadata(
                        processing_task_type="upscaling",
                        uid=uid,
                        fallback_hotkey=hotkey,
                        accumulate_score=0.0,
                        bonus_multiplier=1.0,
                        penalty_f_multiplier=1.0,
                        penalty_q_multiplier=1.0,
                        total_multiplier=1.0,
                        performance_tier="New Miner"
                    )
                    session.add(miner)
                    logger.info(f"New upscaling miner detected for UID {uid}: {hotkey}")
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

                    miner.accumulate_score = 0
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

                self._apply_chain_metadata(miner, uid, hotkey)
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
        vmaf_thresholds: List[float],
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
                    miner = self._new_miner_metadata(
                        processing_task_type="compression",
                        uid=uid,
                        fallback_hotkey=hotkey,
                        accumulate_score=0.0,
                        bonus_multiplier=1.0,
                        penalty_f_multiplier=1.0,
                        total_multiplier=1.0,
                        performance_tier="New Miner"
                    )
                    session.add(miner)
                    logger.info(f"New compression miner detected for UID {uid}: {hotkey}")
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

                    miner.accumulate_score = 0
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
                
                self._apply_chain_metadata(miner, uid, hotkey)
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
                    vmaf_threshold=vmaf_thresholds[i]
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
        
        if len(records) > CONFIG.score.max_performance_records:
            for record in records[CONFIG.score.max_performance_records:]:
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
                if score == -1 or score >= 2.0:
                    logger.debug(f"Skipping UID {uid} due to score -1 or score >= 2")
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
                    miner = self._new_miner_metadata(
                        uid=uid,
                        processing_task_type="upscaling",
                        accumulate_score=0.0,
                        bonus_multiplier=1.0,
                        penalty_f_multiplier=1.0,
                        penalty_q_multiplier=1.0,
                        total_multiplier=1.0,
                        performance_tier="New Miner"
                    )
                    self.session.add(miner)

                self._apply_chain_metadata(miner, uid, miner.hotkey)
                
                # Convert organic scores to synthetic-like format

                acc_score = 0.0
                current_score = miner.accumulate_score

                # Disabled positive scoring for organics

                # if score == 3.0:
                #     organic_s_f = 0.4
                #     organic_s_q = 0.5
                #     organic_s_l = 0.5
                #     success = True

                #     boost_percentage = 0.03
                #     boost_amount = current_score * boost_percentage
                #     acc_score = current_score + boost_amount

                # elif score == 2.0:
                #     organic_s_f = 0.3
                #     organic_s_q = 0.5
                #     organic_s_l = 0.5
                #     success = True

                #     acc_score = miner.accumulate_score

                if score == 1.0:  # Failure
                    organic_s_f = 0.2  # Moderate success score
                    organic_s_q = 0.5   # Moderate quality score
                    organic_s_l = 0.5   # Moderate length score
                    success = False

                    panelty_percentage = 0.01  # Deduct 1% of current score
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
                # Skip processing if score is -1 (skipped) or success is true
                success = score > 0.5
                if score == -1 or success:
                    logger.debug(f"Skipping UID {uid} due to score -1 or success (skipped)")
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
                    miner = self._new_miner_metadata(
                        uid=uid,
                        processing_task_type="compression",
                        accumulate_score=0.0,
                        bonus_multiplier=1.0,
                        penalty_f_multiplier=1.0,
                        penalty_q_multiplier=1.0,
                        total_multiplier=1.0,
                        performance_tier="New Miner"
                    )
                    self.session.add(miner)

                self._apply_chain_metadata(miner, uid, miner.hotkey)

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

    def get_top_hotkeys_by_task(self, task_type: str, limit: int = 20) -> List[str]:
        """
        Get top hotkeys for a given task type, ordered by accumulate_score descending.
        
        Args:
            task_type (str): Task type to filter by ('upscaling' or 'compression')
            limit (int): Number of top hotkeys to return (default 20)
            
        Returns:
            List[str]: List of hotkeys ordered by accumulate_score descending.
                       Returns an empty list if the table has no entries or no
                       entries match the given task_type.
        """
        if not task_type:
            logger.warning("get_top_hotkeys_by_task called with no task_type")
            return []

        session = self.session
        try:
            # Check if the miner_metadata table has any entries at all
            total_count = session.query(MinerMetadata).count()
            if total_count == 0:
                logger.warning("MinerMetadata table is empty, no miners registered yet")
                return []

            miners = session.query(
                MinerMetadata.hotkey
            ).filter(
                MinerMetadata.processing_task_type == task_type
            ).order_by(
                MinerMetadata.accumulate_score.desc()
            ).limit(limit).all()

            if not miners:
                logger.warning(f"No miners found for task_type={task_type} in MinerMetadata")
                return []

            return [miner.hotkey for miner in miners]
            
        except Exception as e:
            logger.error(f"Error getting top hotkeys for task {task_type}: {e}")
            return []
        finally:
            session.close()

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

    def get_burn_uid(self):
        # Get the subtensor owner hotkey
        sn_owner_hotkey = self.subtensor.query_subtensor(
            "SubnetOwnerHotkey",
            params=[self.config.netuid],
        )
        logger.info(f"SN Owner Hotkey: {sn_owner_hotkey}")

        # Get the UID of this hotkey
        sn_owner_uid = self.subtensor.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=sn_owner_hotkey,
            netuid=self.config.netuid,
        )
        logger.info(f"SN Owner UID: {sn_owner_uid}")

        return sn_owner_uid

    @property
    def weights(self):
        """
        Calculate weights with fixed task allocations and a top-heavy rank curve.
        Compression receives 60% and upscaling receives 40% of the pre-burn
        miner pool. Within each task pool, the top five miners receive 60%,
        20%, 10%, 6%, and 4% respectively; all lower ranks receive zero.
        """
        # Collect eligible miners by task type, then rank each task pool separately.
        compression_miners = []
        upscaling_miners = []

        owner_uid = self.get_burn_uid()

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

        def _apply_rank_curve(miners, allocation):
            miners.sort(key=lambda x: x[1], reverse=True)
            for rank, (uid, _) in enumerate(miners):
                uids.append(uid)
                if rank < len(self.emission_rank_shares):
                    scores.append(allocation * self.emission_rank_shares[rank])
                else:
                    scores.append(0.0)

        _apply_rank_curve(compression_miners, self.compression_emission_allocation)
        _apply_rank_curve(upscaling_miners, self.upscaling_emission_allocation)
        
        # burn self.burn_proportion fraction of miner emissions
        total_scores = sum(scores)
        scores = [x * (1 - self.burn_proportion) for x in scores]
        uids.append(owner_uid)
        scores.append(self.burn_proportion * total_scores)

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
                   f"{upscaling_count} upscaling miners ({upscaling_total_weight:.3f} weight)"
                   f"Rewards data will be distributed as: {[(int(uid), float(score)) for uid, score in zip(uids, scores)]}"
                   )
        
        return uids, scores
