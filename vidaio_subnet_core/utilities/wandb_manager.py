import os
import re
import time
import hashlib
import asyncio
import datetime
import threading
from urllib.parse import quote, urlsplit, urlunsplit
import wandb
from dotenv import load_dotenv
from loguru import logger
from vidaio_subnet_core import __version__ as version


load_dotenv()


DEFAULT_ROTATION_DAYS = 0
DEFAULT_HEARTBEAT_SECONDS = 300
DEFAULT_CONSOLE_CHUNK_SECONDS = 60 * 60
DEFAULT_CONSOLE_CHUNK_BYTES = 25 * 1024 * 1024


class WandbManager:
    def __init__(self, validator=None):
        self.wandb = None
        self.wandb_start = None
        self.validator = validator
        self.process_start_time = time.time()
        self.rotation_days = self._get_env_int(
            "WANDB_RUN_ROTATION_DAYS", DEFAULT_ROTATION_DAYS
        )
        self.heartbeat_seconds = self._get_env_int(
            "WANDB_HEARTBEAT_SECONDS", DEFAULT_HEARTBEAT_SECONDS
        )
        self.console_chunk_seconds = self._get_env_int(
            "WANDB_CONSOLE_CHUNK_SECONDS", DEFAULT_CONSOLE_CHUNK_SECONDS
        )
        self.console_chunk_bytes = self._get_env_int(
            "WANDB_CONSOLE_CHUNK_BYTES", DEFAULT_CONSOLE_CHUNK_BYTES
        )
        self.run_id = self._get_run_id()
        self.run_name = os.getenv("WANDB_RUN_NAME") or f"validator-{self.validator.uid}"
        self._lock = threading.Lock()
        self.enabled = self._is_enabled()

        if self.enabled:
            self.init_wandb("startup")
        elif self.validator.config.wandb.off:
            logger.warning("Running validators without Wandb. Recommend to add Wandb!")
        else:
            logger.warning("WANDB_API_KEY is not set; running validators without Wandb.")

    def _is_enabled(self):
        return not self.validator.config.wandb.off and bool(os.getenv("WANDB_API_KEY"))

    def _get_env_int(self, name, default):
        raw_value = os.getenv(name)
        if raw_value is None:
            return default

        try:
            return int(raw_value)
        except ValueError:
            logger.warning(f"Invalid {name}={raw_value!r}; using default {default}.")
            return default

    def _get_run_id(self):
        configured_run_id = os.getenv("WANDB_RUN_ID")
        if configured_run_id:
            return self._sanitize_run_id(configured_run_id)

        network = self._sanitize_run_id(str(self.validator.config.subtensor.network))
        hotkey_hash = hashlib.sha1(
            self.validator.wallet.hotkey.ss58_address.encode("utf-8")
        ).hexdigest()[:12]
        return f"validator-{network}-{self.validator.config.netuid}-{hotkey_hash}"

    def _sanitize_run_id(self, value):
        value = re.sub(r"[/\\#?%:\s]+", "-", value.strip())
        value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
        return value.strip("-") or "validator"

    def _get_public_run_url(self, wandb_entity, wandb_project):
        run_url = getattr(self.wandb, "url", None)
        if run_url:
            parsed = urlsplit(run_url)
            return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))

        entity = quote(wandb_entity, safe="")
        project = quote(wandb_project, safe="")
        run_id = quote(self.run_id, safe="")
        return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"

    def _build_settings(self):
        try:
            return wandb.Settings(
                silent=True,
                console_multipart=True,
                console_chunk_max_seconds=self.console_chunk_seconds,
                console_chunk_max_bytes=self.console_chunk_bytes,
            )
        except Exception as e:
            logger.debug(
                "Wandb multipart console chunk settings are unsupported in this SDK; "
                f"falling back without chunk controls ({type(e).__name__})."
            )

        try:
            return wandb.Settings(silent=True, console_multipart=True)
        except Exception as e:
            logger.warning(
                "Wandb multipart console logs are unsupported in this SDK; "
                f"falling back to silent mode only ({type(e).__name__})."
            )

        try:
            return wandb.Settings(silent=True)
        except Exception as e:
            logger.warning(
                f"Wandb silent setting is unsupported in this SDK ({type(e).__name__})."
            )
            return None

    def init_wandb(self, reason="manual"):
        with self._lock:
            self._init_wandb(reason)

    def _init_wandb(self, reason):
        """Creates a new wandb run for validator logs."""
        logger.debug("Init wandb")

        self._finish_locked()

        self.wandb_start = datetime.date.today()
        wandb_project = self.validator.config.wandb.project_name
        wandb_entity = self.validator.config.wandb.entity
        os.environ.setdefault("WANDB_SILENT", "true")
        logger.info("Initializing wandb entity and project.")
        try:
            init_kwargs = {
                "id": self.run_id,
                "resume": "allow",
                "name": self.run_name,
                "project": wandb_project,
                "entity": wandb_entity,
                "config": {
                    "uid": self.validator.uid,
                    "hotkey": self.validator.wallet.hotkey.ss58_address,
                    "version": version,
                    "type": "validator",
                    "wandb_run_id": self.run_id,
                    "wandb_rotation_days": self.rotation_days,
                    "wandb_init_reason": reason,
                },
                "allow_val_change": True,
            }
            settings = self._build_settings()
            if settings is not None:
                init_kwargs["settings"] = settings

            self.wandb = wandb.init(**init_kwargs)
        except Exception as e:
            self.wandb = None
            self.wandb_start = None
            logger.error(f"Failed to initialize Wandb: {e}")
            return

        logger.info(f"Init Wandb run {self.run_id}: {self.run_name} ({reason})")
        logger.info(
            f"Wandb run URL: {self._get_public_run_url(wandb_entity, wandb_project)}"
        )

    def _finish_locked(self):
        if self.wandb is None and wandb.run is None:
            return

        try:
            if self.wandb is not None:
                self.wandb.finish()
            elif wandb.run is not None:
                wandb.finish()
        except Exception as e:
            logger.warning(f"Failed to finish Wandb run cleanly: {e}")
        finally:
            self.wandb = None

    def finish(self):
        with self._lock:
            self._finish_locked()

    def _should_rotate(self):
        if self.rotation_days <= 0 or self.wandb_start is None:
            return False

        return datetime.date.today() >= self.wandb_start + datetime.timedelta(
            days=self.rotation_days
        )

    def maintain(self):
        if not self.enabled:
            return

        with self._lock:
            if self.wandb is None or wandb.run is None:
                logger.warning("Wandb run is missing; reinitializing.")
                self._init_wandb("missing-run")
                return

            if self._should_rotate():
                logger.info(f"Rotating Wandb run after {self.rotation_days} day(s).")
                self._init_wandb("scheduled-rotation")
                return

            try:
                self.wandb.log(
                    {
                        "validator_heartbeat": 1,
                        "validator_uptime_seconds": int(
                            time.time() - self.process_start_time
                        ),
                    }
                )
            except Exception as e:
                logger.warning(f"Wandb heartbeat failed; reinitializing run: {e}")
                self._init_wandb("heartbeat-failed")

    async def run_maintenance(self):
        if not self.enabled:
            return

        sleep_seconds = max(self.heartbeat_seconds, 60)
        while True:
            await asyncio.sleep(sleep_seconds)
            try:
                self.maintain()
            except Exception:
                logger.exception("Unexpected Wandb maintenance error.")
