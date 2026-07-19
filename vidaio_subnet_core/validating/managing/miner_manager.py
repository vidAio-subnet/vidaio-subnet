import bittensor as bt
import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from loguru import logger
import numpy as np
from .sql_schemas import (
    Base,
    MinerEmissionEpochSnapshot,
    MinerMetadata,
    MinerPerformanceHistory,
)
from .serving_counter import ServingCounter
from ...global_config import CONFIG
from ...utilities.rate_limit import build_rate_limit
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import desc, text
import os
import math
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
        self.burn_proportion = 0.0   # No miner emissions burned by default

        # Task allocations and rank-based distribution within each task pool.
        self.compression_emission_allocation = 0.80
        self.upscaling_emission_allocation = 0.20
        self.emission_rank_shares = [0.20, 0.20, 0.20, 0.20, 0.20]
        self.alpha_stake_weigh_factor = CONFIG.score.alpha_stake_weigh_factor
        self.emission_liquidation_weigh_factor = (
            CONFIG.score.emission_liquidation_weigh_factor
        )
        self.emission_liquidation_window_epochs = (
            CONFIG.score.emission_liquidation_window_epochs
        )

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
        self._migrate_miner_emission_epoch_snapshots_table()
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
            "alpha_stake": "ALTER TABLE miner_metadata ADD COLUMN alpha_stake FLOAT NOT NULL DEFAULT 0.0",
        }

        with self.engine.begin() as connection:
            existing_columns = {
                row[1] for row in connection.execute(text("PRAGMA table_info(miner_metadata)"))
            }
            for column_name, statement in migrations.items():
                if column_name not in existing_columns:
                    logger.info(f"Applying miner metadata migration: add {column_name}")
                    connection.execute(text(statement))

    def _migrate_miner_emission_epoch_snapshots_table(self) -> None:
        """Add snapshot columns without recreating existing SQLite data."""
        migrations = {
            "hotkey": "ALTER TABLE miner_emission_epoch_snapshots ADD COLUMN hotkey VARCHAR(64) NOT NULL DEFAULT ''",
            "coldkey": "ALTER TABLE miner_emission_epoch_snapshots ADD COLUMN coldkey VARCHAR(64) NOT NULL DEFAULT ''",
            "task_type": "ALTER TABLE miner_emission_epoch_snapshots ADD COLUMN task_type VARCHAR(32)",
            "epoch_block": "ALTER TABLE miner_emission_epoch_snapshots ADD COLUMN epoch_block INTEGER NOT NULL DEFAULT 0",
            "epoch_index": "ALTER TABLE miner_emission_epoch_snapshots ADD COLUMN epoch_index INTEGER NOT NULL DEFAULT 0",
            "alpha_stake": "ALTER TABLE miner_emission_epoch_snapshots ADD COLUMN alpha_stake FLOAT NOT NULL DEFAULT 0.0",
            "emission": "ALTER TABLE miner_emission_epoch_snapshots ADD COLUMN emission FLOAT NOT NULL DEFAULT 0.0",
            "timestamp": "ALTER TABLE miner_emission_epoch_snapshots ADD COLUMN timestamp DATETIME",
        }

        with self.engine.begin() as connection:
            table_exists = connection.execute(
                text(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='miner_emission_epoch_snapshots'"
                )
            ).fetchone()
            if table_exists is None:
                return

            existing_columns = {
                row[1]
                for row in connection.execute(
                    text("PRAGMA table_info(miner_emission_epoch_snapshots)")
                )
            }
            for column_name, statement in migrations.items():
                if column_name not in existing_columns:
                    logger.info(
                        "Applying miner emission snapshot migration: "
                        f"add {column_name}"
                    )
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

    def _stake_value_to_float(self, value: Any) -> float:
        if value is None:
            return 0.0

        tao_value = getattr(value, "tao", None)
        if tao_value is not None:
            value = tao_value() if callable(tao_value) else tao_value

        item = getattr(value, "item", None)
        if callable(item):
            try:
                value = item()
            except ValueError:
                return 0.0

        try:
            stake = float(value)
        except (TypeError, ValueError):
            return 0.0

        if not math.isfinite(stake):
            return 0.0
        return max(0.0, stake)

    def _alpha_stake_for_uid(self, uid: int) -> float:
        for stake_attr in ("alpha_stake", "AS"):
            try:
                alpha_stakes = getattr(self.metagraph, stake_attr, None)
            except AttributeError:
                continue

            if alpha_stakes is None:
                continue

            try:
                return self._stake_value_to_float(alpha_stakes[uid])
            except (IndexError, KeyError, TypeError) as e:
                logger.debug(
                    f"Unable to read metagraph {stake_attr} for UID {uid}: {e}"
                )

        return 0.0

    def _emission_for_uid(self, uid: int) -> float:
        for emission_attr in ("emission", "emissions", "E"):
            try:
                emissions = getattr(self.metagraph, emission_attr, None)
            except AttributeError:
                continue

            if emissions is None:
                continue

            try:
                return self._stake_value_to_float(emissions[uid])
            except (IndexError, KeyError, TypeError) as e:
                logger.debug(
                    f"Unable to read metagraph {emission_attr} for UID {uid}: {e}"
                )

        return 0.0

    def _current_epoch_block(self) -> int | None:
        try:
            return int(self.subtensor.get_current_block())
        except Exception as e:
            logger.warning(f"Unable to read current block for emission snapshot: {e}")
            return None

    def _tempo_blocks(self) -> int:
        tempo_candidates = [
            getattr(self.metagraph, "tempo", None),
            getattr(getattr(self.metagraph, "hparams", None), "tempo", None),
            CONFIG.SUBNET_TEMPO,
        ]
        for tempo_candidate in tempo_candidates:
            item = getattr(tempo_candidate, "item", None)
            if callable(item):
                try:
                    tempo_candidate = item()
                except ValueError:
                    continue
            try:
                tempo = int(tempo_candidate)
            except (TypeError, ValueError):
                continue
            if tempo > 0:
                return tempo
        return 1

    def _epoch_index_for_block(self, block: int) -> int:
        return max(0, int(block) // self._tempo_blocks())

    def _epoch_index_for_snapshot(self, snapshot: MinerEmissionEpochSnapshot) -> int:
        try:
            epoch_block = int(snapshot.epoch_block or 0)
        except (TypeError, ValueError):
            epoch_block = 0
        if epoch_block > 0:
            return self._epoch_index_for_block(epoch_block)

        try:
            return max(0, int(snapshot.epoch_index or 0))
        except (TypeError, ValueError):
            return 0

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
            "alpha_stake": self._alpha_stake_for_uid(uid),
        }

    def _apply_chain_metadata(self, miner: MinerMetadata, uid: int, fallback_hotkey: str = "") -> None:
        try:
            metadata = self._chain_metadata_for_uid(uid)
        except Exception as e:
            logger.warning(f"Unable to fetch metagraph metadata for UID {uid}: {e}")
            metadata = {
                "hotkey": fallback_hotkey,
                "coldkey": "",
                "ip_address": "",
                "port": 0,
                "alpha_stake": 0.0,
            }

        miner.hotkey = metadata["hotkey"] or fallback_hotkey or miner.hotkey
        miner.coldkey = metadata["coldkey"]
        miner.ip_address = metadata["ip_address"]
        miner.port = metadata["port"]
        miner.alpha_stake = metadata["alpha_stake"]

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
            metadata = {
                "hotkey": fallback_hotkey,
                "coldkey": "",
                "ip_address": "",
                "port": 0,
                "alpha_stake": 0.0,
            }

        return MinerMetadata(
            uid=uid,
            hotkey=metadata["hotkey"] or fallback_hotkey,
            coldkey=metadata["coldkey"],
            ip_address=metadata["ip_address"],
            port=metadata["port"],
            alpha_stake=metadata["alpha_stake"],
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

    def _snapshot_window_epochs(self) -> int:
        try:
            return max(1, int(self.emission_liquidation_window_epochs))
        except (TypeError, ValueError):
            return 10

    def _prune_miner_emission_epoch_snapshots(
        self,
        session: Session,
        current_epoch_index: int,
    ) -> None:
        window_epochs = self._snapshot_window_epochs()
        oldest_retained_epoch = max(0, current_epoch_index - window_epochs + 1)
        deleted = (
            session.query(MinerEmissionEpochSnapshot)
            .filter(MinerEmissionEpochSnapshot.epoch_index < oldest_retained_epoch)
            .delete(synchronize_session=False)
        )
        if deleted:
            logger.info(
                "Pruned miner emission epoch snapshots older than "
                f"epoch_index={oldest_retained_epoch}: deleted {deleted} rows"
            )

    def _normalize_miner_emission_epoch_snapshot_indexes(
        self,
        session: Session,
    ) -> None:
        def is_older_snapshot(
            candidate: MinerEmissionEpochSnapshot,
            current: MinerEmissionEpochSnapshot,
        ) -> bool:
            candidate_block = int(candidate.epoch_block or 0)
            current_block = int(current.epoch_block or 0)
            if candidate_block > 0 and current_block > 0:
                return candidate_block < current_block
            if candidate_block > 0:
                return True
            if current_block > 0:
                return False

            candidate_timestamp = candidate.timestamp or datetime.max
            current_timestamp = current.timestamp or datetime.max
            if candidate_timestamp != current_timestamp:
                return candidate_timestamp < current_timestamp
            return (candidate.id or 0) < (current.id or 0)

        snapshots = session.query(MinerEmissionEpochSnapshot).all()
        snapshots_by_uid_epoch: dict[tuple[int, int], MinerEmissionEpochSnapshot] = {}
        duplicate_snapshots = []
        normalized_updates = 0

        for snapshot in snapshots:
            normalized_epoch_index = self._epoch_index_for_snapshot(snapshot)
            key = (snapshot.uid, normalized_epoch_index)
            existing = snapshots_by_uid_epoch.get(key)
            if existing is None:
                snapshots_by_uid_epoch[key] = snapshot
                continue

            if is_older_snapshot(snapshot, existing):
                duplicate_snapshots.append(existing)
                snapshots_by_uid_epoch[key] = snapshot
            else:
                duplicate_snapshots.append(snapshot)

        for snapshot in duplicate_snapshots:
            session.delete(snapshot)
        if duplicate_snapshots:
            session.flush()

        duplicate_ids = {snapshot.id for snapshot in duplicate_snapshots}
        for (uid, normalized_epoch_index), snapshot in snapshots_by_uid_epoch.items():
            if snapshot.id in duplicate_ids:
                continue
            if snapshot.epoch_index != normalized_epoch_index:
                snapshot.epoch_index = normalized_epoch_index
                normalized_updates += 1

        if duplicate_snapshots or normalized_updates:
            logger.info(
                "Normalized miner emission snapshot epoch indexes: "
                f"updated={normalized_updates}, "
                f"deleted_duplicate_rows={len(duplicate_snapshots)}"
            )

    def record_miner_emission_epoch_snapshots(
        self,
        miners_by_uid: Dict[int, MinerMetadata],
    ) -> None:
        current_block = self._current_epoch_block()
        if current_block is None:
            return

        current_epoch_index = self._epoch_index_for_block(current_block)
        session = self.session
        inserted_snapshot_count = 0
        retained_existing_count = 0

        try:
            for uid, miner in miners_by_uid.items():
                if miner.processing_task_type not in ("compression", "upscaling"):
                    continue

                snapshot = (
                    session.query(MinerEmissionEpochSnapshot)
                    .filter(
                        MinerEmissionEpochSnapshot.uid == uid,
                        MinerEmissionEpochSnapshot.epoch_index == current_epoch_index,
                    )
                    .first()
                )
                if snapshot is not None:
                    retained_existing_count += 1
                    continue

                snapshot = MinerEmissionEpochSnapshot(
                    uid=uid,
                    epoch_index=current_epoch_index,
                )
                session.add(snapshot)

                snapshot.hotkey = miner.hotkey or ""
                snapshot.coldkey = miner.coldkey or ""
                snapshot.task_type = miner.processing_task_type
                snapshot.epoch_block = current_block
                snapshot.alpha_stake = self._stake_value_to_float(miner.alpha_stake)
                snapshot.emission = self._emission_for_uid(uid)
                snapshot.timestamp = datetime.now()
                inserted_snapshot_count += 1

            self._normalize_miner_emission_epoch_snapshot_indexes(session)
            self._prune_miner_emission_epoch_snapshots(
                session,
                current_epoch_index,
            )
            session.commit()
            if inserted_snapshot_count or retained_existing_count:
                logger.info(
                    "Recorded miner emission snapshots at "
                    f"block={current_block}, "
                    f"epoch_index={current_epoch_index}, "
                    f"inserted={inserted_snapshot_count}, "
                    f"retained_existing={retained_existing_count}, "
                    f"window_epochs={self._snapshot_window_epochs()}"
                )
        except Exception as e:
            session.rollback()
            logger.error(f"Unable to record miner emission epoch snapshots: {e}")

    def _empty_emission_liquidation_stats(
        self,
        uid: int,
        status: str,
        snapshot_count: int = 0,
    ) -> dict[str, Any]:
        return {
            "uid": uid,
            "snapshot_count": snapshot_count,
            "status": status,
            "hotkey": "",
            "coldkey": "",
            "first_epoch_index": None,
            "last_epoch_index": None,
            "first_alpha_stake": 0.0,
            "last_alpha_stake": 0.0,
            "alpha_stake_delta": 0.0,
            "first_excluded_emission": 0.0,
            "total_emission": 0.0,
            "retained_emission": 0.0,
            "liquidated_emission": 0.0,
            "liquidated_proportion": None,
            "retained_proportion": None,
            "emission_samples": "",
            "alpha_stake_samples": "",
        }

    def recent_emission_liquidation_stats(
        self,
        uids: List[int],
    ) -> dict[int, dict[str, Any]]:
        stats: dict[int, dict[str, Any]] = {}
        if not uids:
            return stats

        window_epochs = self._snapshot_window_epochs()
        for uid in uids:
            snapshots_desc = (
                self.session.query(MinerEmissionEpochSnapshot)
                .filter(MinerEmissionEpochSnapshot.uid == uid)
                .order_by(
                    MinerEmissionEpochSnapshot.epoch_block.desc(),
                    MinerEmissionEpochSnapshot.epoch_index.desc(),
                )
                .limit(window_epochs)
                .all()
            )
            snapshots = list(reversed(snapshots_desc))
            if len(snapshots) < 2:
                stats[uid] = self._empty_emission_liquidation_stats(
                    uid,
                    "new_or_insufficient_history",
                    len(snapshots),
                )
                continue

            first_snapshot = snapshots[0]
            last_snapshot = snapshots[-1]
            first_epoch_index = self._epoch_index_for_snapshot(first_snapshot)
            last_epoch_index = self._epoch_index_for_snapshot(last_snapshot)
            first_excluded_emission = max(
                0.0,
                float(first_snapshot.emission or 0.0),
            )
            emission_samples = ", ".join(
                f"{self._epoch_index_for_snapshot(snapshot)}:"
                f"{max(0.0, float(snapshot.emission or 0.0)):.6f}"
                for snapshot in snapshots
            )
            alpha_stake_samples = ", ".join(
                f"{self._epoch_index_for_snapshot(snapshot)}:"
                f"{float(snapshot.alpha_stake or 0.0):.6f}"
                for snapshot in snapshots
            )
            # Snapshots are boundary samples. The first row's emission belongs
            # to the interval before the first alpha baseline in this window,
            # so only emissions after that baseline are comparable.
            total_emission = sum(
                max(0.0, float(snapshot.emission or 0.0))
                for snapshot in snapshots[1:]
            )
            alpha_stake_delta = max(
                0.0,
                float(last_snapshot.alpha_stake or 0.0)
                - float(first_snapshot.alpha_stake or 0.0),
            )

            if total_emission <= 0.0:
                stats[uid] = self._empty_emission_liquidation_stats(
                    uid,
                    "no_recent_emission",
                    len(snapshots),
                )
                stats[uid].update(
                    {
                        "hotkey": last_snapshot.hotkey or "",
                        "coldkey": last_snapshot.coldkey or "",
                        "first_epoch_index": first_epoch_index,
                        "last_epoch_index": last_epoch_index,
                        "first_alpha_stake": float(first_snapshot.alpha_stake or 0.0),
                        "last_alpha_stake": float(last_snapshot.alpha_stake or 0.0),
                        "alpha_stake_delta": alpha_stake_delta,
                        "first_excluded_emission": first_excluded_emission,
                        "emission_samples": emission_samples,
                        "alpha_stake_samples": alpha_stake_samples,
                    }
                )
                continue

            retained_emission = min(alpha_stake_delta, total_emission)
            liquidated_emission = max(0.0, total_emission - retained_emission)
            liquidated_proportion = liquidated_emission / total_emission
            retained_proportion = retained_emission / total_emission

            stats[uid] = {
                "uid": uid,
                "snapshot_count": len(snapshots),
                "status": "ok",
                "hotkey": last_snapshot.hotkey or "",
                "coldkey": last_snapshot.coldkey or "",
                "first_epoch_index": first_epoch_index,
                "last_epoch_index": last_epoch_index,
                "first_alpha_stake": float(first_snapshot.alpha_stake or 0.0),
                "last_alpha_stake": float(last_snapshot.alpha_stake or 0.0),
                "alpha_stake_delta": alpha_stake_delta,
                "first_excluded_emission": first_excluded_emission,
                "total_emission": total_emission,
                "retained_emission": retained_emission,
                "liquidated_emission": liquidated_emission,
                "liquidated_proportion": liquidated_proportion,
                "retained_proportion": retained_proportion,
                "emission_samples": emission_samples,
                "alpha_stake_samples": alpha_stake_samples,
            }

        return stats

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

                #     increase_percentage = 0.03
                #     increase_amount = current_score * increase_percentage
                #     acc_score = current_score + increase_amount

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

    def _format_log_table(self, rows: List[dict[str, Any]]) -> str:
        if not rows:
            return ""

        columns = list(rows[0].keys())
        column_widths = {
            column: max(len(str(column)), *(len(str(row[column])) for row in rows))
            for column in columns
        }
        header = " | ".join(
            str(column).ljust(column_widths[column]) for column in columns
        )
        separator = "-+-".join("-" * column_widths[column] for column in columns)
        body = "\n".join(
            " | ".join(
                str(row[column]).ljust(column_widths[column]) for column in columns
            )
            for row in rows
        )
        return f"{header}\n{separator}\n{body}"

    def _log_alpha_stake_weighing(
        self,
        task_type: str,
        rows: List[dict[str, Any]],
        weigh_factor: float,
        total_alpha_stake: float,
        base_total: float,
        raw_total: float,
        total_preserving_scale: float,
    ) -> None:
        if not rows:
            return

        logger.info(
            f"Alpha stake weighing details for {task_type} task pool: "
            f"factor={weigh_factor:.4f}, total_alpha_stake={total_alpha_stake:.6f}, "
            f"base_pre_burn_total={base_total:.10f}, raw_weighted_total={raw_total:.10f}, "
            f"normalization_scale={total_preserving_scale:.10f}, "
            f"post_burn_scale={(1 - self.burn_proportion):.4f}\n"
            f"{self._format_log_table(rows)}"
        )

    def _log_emission_liquidation_weighing(
        self,
        task_type: str,
        rows: List[dict[str, Any]],
        weigh_factor: float,
        known_signal_count: int,
        fallback_retained_proportion: float,
        base_total: float,
        raw_total: float,
        total_preserving_scale: float,
    ) -> None:
        if not rows:
            return

        logger.info(
            f"Emission liquidation weighing details for {task_type} task pool: "
            f"factor={weigh_factor:.4f}, "
            f"window_epochs={self._snapshot_window_epochs()}, "
            f"known_signal_count={known_signal_count}, "
            f"fallback_retained_proportion={fallback_retained_proportion:.6f}, "
            f"base_pre_burn_total={base_total:.10f}, "
            f"raw_weighted_total={raw_total:.10f}, "
            f"normalization_scale={total_preserving_scale:.10f}, "
            f"post_burn_scale={(1 - self.burn_proportion):.4f}\n"
            f"{self._format_log_table(rows)}"
        )

    def _weigh_scores_by_alpha_stake(
        self,
        ranked_scores: List[tuple[int, float, float]],
        task_type: str = "unknown",
    ) -> List[tuple[int, float]]:
        try:
            weigh_factor = max(0.0, float(self.alpha_stake_weigh_factor))
        except (TypeError, ValueError):
            weigh_factor = 0.0

        weighted_scores = [(uid, score) for uid, score, _ in ranked_scores]
        # Only the rank recipients with non-zero emissions participate in the
        # per-task alpha stake weighing; lower ranks keep their zero score.
        nonzero_scores = [
            (index, uid, score, max(0.0, alpha_stake))
            for index, (uid, score, alpha_stake) in enumerate(ranked_scores)
            if score > 0.0
        ]
        if not nonzero_scores:
            return weighted_scores

        total_alpha_stake = sum(
            alpha_stake for _, _, _, alpha_stake in nonzero_scores
        )
        base_total = sum(score for _, _, score, _ in nonzero_scores)
        raw_scores = []
        log_rows = []

        for index, uid, score, alpha_stake in nonzero_scores:
            alpha_share = (
                alpha_stake / total_alpha_stake
                if total_alpha_stake > 0.0
                else 0.0
            )
            raw_multiplier = 1.0 + weigh_factor * alpha_share
            raw_score = score * raw_multiplier
            raw_scores.append((index, uid, raw_score, alpha_share, raw_multiplier))

        raw_total = sum(raw_score for _, _, raw_score, _, _ in raw_scores)
        total_preserving_scale = base_total / raw_total if raw_total > 0.0 else 1.0

        for index, uid, raw_score, alpha_share, raw_multiplier in raw_scores:
            normalized_score = raw_score * total_preserving_scale
            if weigh_factor > 0.0 and total_alpha_stake > 0.0:
                weighted_scores[index] = (uid, normalized_score)

            base_score = ranked_scores[index][1]
            alpha_stake = max(0.0, ranked_scores[index][2])
            log_rows.append(
                {
                    "rank": index + 1,
                    "uid": uid,
                    "alpha_stake": f"{alpha_stake:.6f}",
                    "alpha_share": f"{alpha_share * 100:.4f}%",
                    "base_pre_burn": f"{base_score:.10f}",
                    "base_final": f"{base_score * (1 - self.burn_proportion):.10f}",
                    "raw_multiplier": f"{raw_multiplier:.10f}",
                    "raw_weighted": f"{raw_score:.10f}",
                    "normalized_pre_burn": f"{normalized_score:.10f}",
                    "normalized_final": (
                        f"{normalized_score * (1 - self.burn_proportion):.10f}"
                    ),
                }
            )

        self._log_alpha_stake_weighing(
            task_type=task_type,
            rows=log_rows,
            weigh_factor=weigh_factor,
            total_alpha_stake=total_alpha_stake,
            base_total=base_total,
            raw_total=raw_total,
            total_preserving_scale=total_preserving_scale,
        )

        return weighted_scores

    def _weigh_scores_by_emission_liquidation(
        self,
        ranked_scores: List[tuple[int, float]],
        task_type: str = "unknown",
        emission_liquidation_stats: dict[int, dict[str, Any]] | None = None,
    ) -> List[tuple[int, float]]:
        try:
            weigh_factor = max(0.0, float(self.emission_liquidation_weigh_factor))
        except (TypeError, ValueError):
            weigh_factor = 0.0

        weighted_scores = list(ranked_scores)
        nonzero_scores = [
            (index, uid, score)
            for index, (uid, score) in enumerate(ranked_scores)
            if score > 0.0
        ]
        if not nonzero_scores:
            return weighted_scores

        stats_by_uid = emission_liquidation_stats or {}
        fallback_retained_proportion = 0.5
        known_retained_proportions = []
        for _, uid, _ in nonzero_scores:
            stats = stats_by_uid.get(uid, {})
            retained_proportion = stats.get("retained_proportion")
            if retained_proportion is None:
                continue
            known_retained_proportions.append(
                min(1.0, max(0.0, float(retained_proportion)))
            )

        known_signal_count = len(known_retained_proportions)

        base_total = sum(score for _, _, score in nonzero_scores)
        raw_scores = []
        log_rows = []

        for index, uid, score in nonzero_scores:
            stats = stats_by_uid.get(
                uid,
                self._empty_emission_liquidation_stats(uid, "missing_snapshot"),
            )
            retained_proportion = stats.get("retained_proportion")
            if retained_proportion is None:
                retained_signal = fallback_retained_proportion
                signal_source = "assumed_50pct_liquidated"
            else:
                retained_signal = min(1.0, max(0.0, float(retained_proportion)))
                signal_source = "history"

            raw_multiplier = 1.0 + weigh_factor * retained_signal
            raw_score = score * raw_multiplier
            raw_scores.append(
                (
                    index,
                    uid,
                    raw_score,
                    retained_signal,
                    raw_multiplier,
                    stats,
                    signal_source,
                )
            )

        raw_total = sum(raw_score for _, _, raw_score, _, _, _, _ in raw_scores)
        total_preserving_scale = base_total / raw_total if raw_total > 0.0 else 1.0

        for (
            index,
            uid,
            raw_score,
            retained_signal,
            raw_multiplier,
            stats,
            signal_source,
        ) in raw_scores:
            normalized_score = raw_score * total_preserving_scale
            if weigh_factor > 0.0 and raw_total > 0.0:
                weighted_scores[index] = (uid, normalized_score)

            base_score = ranked_scores[index][1]
            liquidated_proportion = stats.get("liquidated_proportion")
            retained_proportion = stats.get("retained_proportion")
            log_rows.append(
                {
                    "rank": index + 1,
                    "uid": uid,
                    "snapshots": stats.get("snapshot_count", 0),
                    "status": stats.get("status", "missing_snapshot"),
                    "epoch_first": stats.get("first_epoch_index", "n/a"),
                    "epoch_last": stats.get("last_epoch_index", "n/a"),
                    "alpha_first": f"{stats.get('first_alpha_stake', 0.0):.6f}",
                    "alpha_last": f"{stats.get('last_alpha_stake', 0.0):.6f}",
                    "alpha_delta": f"{stats.get('alpha_stake_delta', 0.0):.6f}",
                    "first_excluded_emission": (
                        f"{stats.get('first_excluded_emission', 0.0):.6f}"
                    ),
                    "settled_emission": f"{stats.get('total_emission', 0.0):.6f}",
                    "retained_emission": f"{stats.get('retained_emission', 0.0):.6f}",
                    "liquidated_emission": f"{stats.get('liquidated_emission', 0.0):.6f}",
                    "liquidated_pct": (
                        f"{liquidated_proportion * 100:.4f}%"
                        if liquidated_proportion is not None
                        else "n/a"
                    ),
                    "retained_pct": (
                        f"{retained_proportion * 100:.4f}%"
                        if retained_proportion is not None
                        else "n/a"
                    ),
                    "retained_signal": f"{retained_signal:.6f}",
                    "signal_source": signal_source,
                    "epoch_emissions": stats.get("emission_samples", ""),
                    "epoch_alpha_stakes": stats.get("alpha_stake_samples", ""),
                    "base_pre_burn": f"{base_score:.10f}",
                    "raw_multiplier": f"{raw_multiplier:.10f}",
                    "raw_weighted": f"{raw_score:.10f}",
                    "normalized_pre_burn": f"{normalized_score:.10f}",
                    "normalized_final": (
                        f"{normalized_score * (1 - self.burn_proportion):.10f}"
                    ),
                }
            )

        self._log_emission_liquidation_weighing(
            task_type=task_type,
            rows=log_rows,
            weigh_factor=weigh_factor,
            known_signal_count=known_signal_count,
            fallback_retained_proportion=fallback_retained_proportion,
            base_total=base_total,
            raw_total=raw_total,
            total_preserving_scale=total_preserving_scale,
        )

        return weighted_scores

    def _apply_rank_curve(
        self,
        miners: List[tuple[int, float, float]],
        allocation: float,
        task_type: str = "unknown",
        emission_liquidation_stats: dict[int, dict[str, Any]] | None = None,
    ) -> List[tuple[int, float]]:
        miners.sort(key=lambda x: x[1], reverse=True)
        ranked_scores = []
        for rank, (uid, _, alpha_stake) in enumerate(miners):
            if rank < len(self.emission_rank_shares):
                score = allocation * self.emission_rank_shares[rank]
            else:
                score = 0.0
            ranked_scores.append((uid, score, alpha_stake))

        alpha_weighted_scores = self._weigh_scores_by_alpha_stake(
            ranked_scores,
            task_type,
        )
        return self._weigh_scores_by_emission_liquidation(
            alpha_weighted_scores,
            task_type,
            emission_liquidation_stats,
        )

    @property
    def weights(self):
        """
        Calculate weights with fixed task allocations and top-five shares.
        Compression receives 80% and upscaling receives 20% of the pre-burn
        miner pool. Within each task pool, the top five miners start from 20%
        each, then optional alpha stake weighing reallocates those non-zero
        shares and optional emission liquidation weighing reallocates them
        again within the same task pool. All lower ranks receive zero.
        """
        # Collect eligible miners by task type, then rank each task pool separately.
        compression_miners = []
        upscaling_miners = []

        self.check_database_connection()
        self.sync_miner_chain_metadata()

        miners_by_uid = self.query()
        self.record_miner_emission_epoch_snapshots(miners_by_uid)

        for uid, miner in miners_by_uid.items():
            # Every row in miner_metadata is treated as a miner, regardless of
            # whether its UID currently has a validator permit.
            if miner.accumulate_score == -1:
                continue
            alpha_stake = miner.alpha_stake or 0.0
            if miner.processing_task_type == "compression":
                compression_miners.append((uid, miner.accumulate_score, alpha_stake))
            elif miner.processing_task_type == "upscaling":
                upscaling_miners.append((uid, miner.accumulate_score, alpha_stake))

        emission_liquidation_stats = self.recent_emission_liquidation_stats(
            [uid for uid, _, _ in compression_miners + upscaling_miners]
        )
        
        # Initialize result arrays
        uids = []
        scores = []

        for uid, score in self._apply_rank_curve(
            compression_miners,
            self.compression_emission_allocation,
            "compression",
            emission_liquidation_stats,
        ):
            uids.append(uid)
            scores.append(score)
        for uid, score in self._apply_rank_curve(
            upscaling_miners,
            self.upscaling_emission_allocation,
            "upscaling",
            emission_liquidation_stats,
        ):
            uids.append(uid)
            scores.append(score)
        
        # Apply the configured burn only when it assigns a positive weight.
        total_scores = sum(scores)
        burn_weight = self.burn_proportion * total_scores
        scores = [x * (1 - self.burn_proportion) for x in scores]
        if burn_weight > 0.0:
            uids.append(self.get_burn_uid())
            scores.append(burn_weight)

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
        compression_uids = {uid for uid, _, _ in compression_miners}
        upscaling_uids = {uid for uid, _, _ in upscaling_miners}
        compression_total_weight = sum(
            scores[i] for i, uid in enumerate(uids) if uid in compression_uids
        )
        upscaling_total_weight = sum(
            scores[i] for i, uid in enumerate(uids) if uid in upscaling_uids
        )
        
        logger.info(
            f"Reward distribution: {compression_count} compression miners "
            f"({compression_total_weight:.3f} weight), {upscaling_count} "
            f"upscaling miners ({upscaling_total_weight:.3f} weight), alpha "
            f"stake weigh factor: {self.alpha_stake_weigh_factor:.3f}, "
            f"emission liquidation weigh factor: "
            f"{self.emission_liquidation_weigh_factor:.3f}, "
            f"emission liquidation window epochs: "
            f"{self._snapshot_window_epochs()}. "
            f"Rewards data will be distributed as: "
            f"{[(int(uid), float(score)) for uid, score in zip(uids, scores)]}"
        )
        
        return uids, scores
