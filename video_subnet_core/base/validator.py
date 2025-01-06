import bittensor as bt
from abc import ABC, abstractmethod
import threading
import structlog

logger = structlog.get_logger()


class BaseValidator(ABC):
    def __init__(self, config: bt.Config):
        self.config = config

    def init_bittensor(self):
        self.subtensor = bt.subtensor(config=self.config)
        self.wallet = bt.wallet(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.axon = bt.axon(config=self.config)
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

    def chain_sync(self):
        self.metagraph.sync()

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            logger.debug(
                "base_validator", "Starting validator in background thread."
            )
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            logger.debug("base_validator", "Started")

    def stop_run_thread(self):
        if self.is_running:
            logger.debug(
                "base_validator", "Stopping validator in background thread."
            )
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            logger.debug("base_validator", "Stopped")

    def run_in_background_thread(self):
        if not self.is_running:
            logger.debug(
                "base_validator", "Starting validator in background thread."
            )
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            logger.debug("base_validator", "Started")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.is_running:
            logger.debug("base_validator", "Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            logger.debug("base_validator", "Stopped")

    @abstractmethod
    def start_epoch(self): ...

    @abstractmethod
    def set_weights(self): ...
