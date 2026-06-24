import os
import re
import time
import hashlib
import asyncio
import datetime
import threading
import uuid
from pathlib import Path
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
DEFAULT_CONSOLE_MODE = "redirect"
DEFAULT_CONSOLE_MULTIPART = False
DEFAULT_LOG_LEVEL = "DEBUG"
LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{function}:{line} - {message}"
)


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
        self.console_mode = os.getenv("WANDB_CONSOLE", DEFAULT_CONSOLE_MODE)
        self.console_multipart = self._get_env_bool(
            "WANDB_CONSOLE_MULTIPART", DEFAULT_CONSOLE_MULTIPART
        )
        self.run_group = self._get_run_group()
        self.run_id = None
        self.run_name = os.getenv("WANDB_RUN_NAME") or f"validator-{self.validator.uid}"
        self.log_level = os.getenv("WANDB_LOG_LEVEL", DEFAULT_LOG_LEVEL)
        self.wandb_output_sink_id = None
        self.wandb_output_log_path = None
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

    def _get_env_bool(self, name, default):
        raw_value = os.getenv(name)
        if raw_value is None:
            return default

        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False

        logger.warning(f"Invalid {name}={raw_value!r}; using default {default}.")
        return default

    def _get_run_id(self):
        configured_run_id = os.getenv("WANDB_RUN_ID")
        if configured_run_id:
            return self._sanitize_run_id(configured_run_id)

        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%d-%H%M%S"
        )
        suffix = uuid.uuid4().hex[:8]
        return self._sanitize_run_id(f"{self.run_group}-{timestamp}-{suffix}")

    def _get_run_group(self):
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

        if self._is_non_syncing_wandb_mode():
            return None

        entity = quote(wandb_entity, safe="")
        project = quote(wandb_project, safe="")
        run_id = quote(self.run_id, safe="")
        return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"

    def _build_settings(self):
        settings_attempts = [
            {
                "silent": True,
                "console": self.console_mode,
            },
            {
                "silent": True,
            },
        ]

        if self.console_multipart:
            settings_attempts.insert(
                0,
                {
                    "silent": True,
                    "console": self.console_mode,
                    "console_multipart": True,
                    "console_chunk_max_seconds": self.console_chunk_seconds,
                    "console_chunk_max_bytes": self.console_chunk_bytes,
                },
            )
            settings_attempts.insert(
                1,
                {
                    "silent": True,
                    "console": self.console_mode,
                    "console_multipart": True,
                },
            )

        for settings_kwargs in settings_attempts:
            try:
                return wandb.Settings(**settings_kwargs)
            except Exception as e:
                logger.debug(
                    "Wandb settings are unsupported in this SDK; "
                    f"falling back ({type(e).__name__})."
                )

        return None

    def _get_wandb_mode(self):
        disabled = os.getenv("WANDB_DISABLED", "").strip().lower()
        if disabled in {"1", "true", "yes", "on"}:
            return "disabled"

        env_mode = os.getenv("WANDB_MODE")
        if env_mode:
            return env_mode.strip().lower()

        try:
            settings = getattr(self.wandb, "settings", None) or getattr(
                self.wandb, "_settings", None
            )
            mode = getattr(settings, "mode", None) or getattr(settings, "_mode", None)
            if mode:
                return str(mode).lower()
        except Exception:
            pass

        return "online"

    def _is_non_syncing_wandb_mode(self):
        return self._get_wandb_mode() in {"disabled", "dryrun", "offline"}

    def init_wandb(self, reason="manual"):
        with self._lock:
            self._init_wandb(reason)

    def _init_wandb(self, reason):
        """Creates a new wandb run for validator logs."""
        logger.debug("Init wandb")

        self._finish_locked()

        self.run_id = self._get_run_id()
        self.wandb_start = datetime.date.today()
        wandb_project = self.validator.config.wandb.project_name
        wandb_entity = self.validator.config.wandb.entity
        os.environ.setdefault("WANDB_SILENT", "true")
        logger.info("Initializing wandb entity and project.")
        try:
            init_kwargs = {
                "id": self.run_id,
                "name": self.run_name,
                "project": wandb_project,
                "entity": wandb_entity,
                "group": self.run_group,
                "config": {
                    "uid": self.validator.uid,
                    "hotkey": self.validator.wallet.hotkey.ss58_address,
                    "version": version,
                    "type": "validator",
                    "wandb_run_id": self.run_id,
                    "wandb_run_group": self.run_group,
                    "wandb_rotation_days": self.rotation_days,
                    "wandb_init_reason": reason,
                    "wandb_console": self.console_mode,
                    "wandb_console_multipart": self.console_multipart,
                },
                "allow_val_change": True,
            }
            if os.getenv("WANDB_RUN_ID"):
                init_kwargs["resume"] = "allow"

            settings = self._build_settings()
            if settings is not None:
                init_kwargs["settings"] = settings

            self.wandb = wandb.init(**init_kwargs)
            self._install_log_sink_locked()
        except Exception as e:
            self.wandb = None
            self.wandb_start = None
            logger.error(f"Failed to initialize Wandb: {e}")
            return

        logger.info(f"Init Wandb run {self.run_id}: {self.run_name} ({reason})")
        logger.info(
            "Wandb mode: "
            f"{self._get_wandb_mode()}; console capture: {self.console_mode}; "
            f"multipart: {self.console_multipart}"
        )
        run_url = self._get_public_run_url(wandb_entity, wandb_project)
        if run_url:
            logger.info(f"Wandb run URL: {run_url}")
        else:
            logger.warning(
                "Wandb is initialized in a non-syncing mode; remote logs will not "
                "update until local runs are synced."
            )
        try:
            self._log_heartbeat_locked("init")
        except Exception as e:
            logger.warning(f"Wandb startup heartbeat failed: {e}")

    def _install_log_sink_locked(self):
        self._remove_log_sink_locked()

        run_dir = getattr(self.wandb, "dir", None)
        if not run_dir:
            logger.warning("Wandb run directory is unavailable; log file sync disabled.")
            return

        self.wandb_output_log_path = Path(run_dir) / "output.log"
        self.wandb_output_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.wandb_output_log_path.touch(exist_ok=True)
        self.wandb_output_sink_id = logger.add(
            self.wandb_output_log_path,
            format=LOG_FORMAT,
            level=self.log_level,
            colorize=False,
            enqueue=False,
            backtrace=False,
            diagnose=False,
        )

        try:
            self.wandb.save(str(self.wandb_output_log_path), policy="live")
        except Exception as e:
            logger.warning(f"Failed to register Wandb output.log for live sync: {e}")

        logger.info(f"Wandb output log sink: {self.wandb_output_log_path}")

    def _remove_log_sink_locked(self):
        if self.wandb_output_sink_id is None:
            return

        try:
            logger.remove(self.wandb_output_sink_id)
        except Exception as e:
            logger.warning(f"Failed to remove Wandb output log sink: {e}")
        self.wandb_output_sink_id = None

    def _finish_locked(self):
        if self.wandb is None and wandb.run is None:
            return

        try:
            self._remove_log_sink_locked()
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

    def _log_heartbeat_locked(self, event):
        self.wandb.log(
            {
                "validator_heartbeat": 1,
                "validator_uptime_seconds": int(time.time() - self.process_start_time),
                "validator_wandb_event": event,
            }
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
                self._log_heartbeat_locked("heartbeat")
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
