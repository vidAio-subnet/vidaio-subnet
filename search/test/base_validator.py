import os
import asyncio
import argparse
import traceback
import bittensor as bt
import time
import threading
from .config import add_common_config
from abc import abstractmethod, ABC
from loguru import logger


class BaseValidator(ABC):
    def __init__(self):
        self.config = self.get_config()
        print(self.config)
        self.setup_logging()
        self.last_update = 0
        self.current_block = 0
        self.uid = 1
        self.is_running = False
        self.should_exit = False
        self.loop = asyncio.get_event_loop()

    def get_config(self):
        parser = argparse.ArgumentParser()
        parser = add_common_config(parser)
        config = bt.config(parser)
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey_str,
                config.netuid,
                "validator",
            )
        )
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        bt.logging.enable_default()
        bt.logging.enable_info()

        if self.config.logging.debug:
            bt.logging.enable_debug()
        if self.config.logging.trace:
            bt.logging.enable_trace()
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        logger.info(
            f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.network} with config:"
        )
        logger.info(self.config)
        pass

    @abstractmethod
    async def start_synthetic_epoch(self):
        pass

    @abstractmethod
    async def start_organic_loop(self):
        pass

    async def run_synthetic(self):
        logger.info("Starting validator synthetic loop.")
        while not self.should_exit:
            try:
                await self.start_synthetic_epoch()
            except Exception as e:
                logger.error(f"Forward error: {e}")
                traceback.print_exc()

            try:
                self.resync_metagraph()
            except Exception as e:
                logger.error(f"Resync metagraph error: {e}")
                traceback.print_exc()

            except KeyboardInterrupt:
                self.axon.stop()
                logger.success("Validator killed by keyboard interrupt.")
                exit()

    async def run_organic(self):
        logger.info("Starting validator organic loop.")
        while not self.should_exit:
            try:
                await self.start_organic_loop()
            except Exception as e:
                logger.error(f"Forward error: {e}")
                traceback.print_exc()

            try:
                self.resync_metagraph()
            except Exception as e:
                logger.error(f"Resync metagraph error: {e}")
                traceback.print_exc()

            except KeyboardInterrupt:
                self.axon.stop()
                logger.success("Validator killed by keyboard interrupt.")
                exit()

    @abstractmethod
    def set_weights(self):
        pass

    def resync_metagraph(self):
        self.metagraph.sync()

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            logger.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            logger.debug("Started")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            logger.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            logger.debug("Stopped")
