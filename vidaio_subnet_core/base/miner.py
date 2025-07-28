import bittensor as bt
from abc import ABC, abstractmethod
import threading
from loguru import logger
import time
import traceback
from vidaio_subnet_core.protocol import VideoUpscalingProtocol, LengthCheckProtocol, VideoCompressionProtocol, TaskWarrantProtocol
import argparse
from .config import add_common_config
import os


class BaseMiner(ABC):
    def __init__(self   ):
        self.config = self.get_config()
        self.init_bittensor()

    def init_bittensor(self):
        self.subtensor = bt.subtensor(config=self.config)
        logger.info(f"Subtensor: {self.subtensor}")
        self.wallet = bt.wallet(config=self.config)
        logger.info(f"Wallet: {self.wallet}")
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        logger.info(f"Metagraph: {self.metagraph}")
        self.axon = bt.axon(config=self.config)
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        logger.info(f"Running Miner with UID {self.uid}")
        self.axon.attach(
            forward_fn=self.forward_upscaling_requests,
            blacklist_fn=self.blacklist_upscaling_requests,
            priority_fn=self.priority_upscaling_requests,
        ).attach(
            forward_fn=self.forward_length_check_requests,
            blacklist_fn=self.blacklist_length_check_requests,
            priority_fn=self.priority_length_check_requests,
        ).attach(
            forward_fn=self.forward_compression_requests,
            blacklist_fn=self.blacklist_compression_requests,
            priority_fn=self.priority_compression_requests,
        ).attach(
            forward_fn=self.forward_task_warrant_requests,
            blacklist_fn=self.blacklist_task_warrant_requests,
            priority_fn=self.priority_task_warrant_requests,
        )

        self.check_registered()
        self.block = self.subtensor.get_current_block()
        self.my_subnet_uid = self.metagraph.hotkeys.index(
            self.wallet.hotkey.ss58_address
        )
        logger.info(f"Running Miner on uid: {self.my_subnet_uid} at block {self.block}")
        self.should_exit: bool = False
        self.is_running: bool = False
        self.step = 0
    
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

    
    def chain_sync(self):
        logger.info("resync_metagraph()")
        self.metagraph.sync(subtensor=self.subtensor)

    @abstractmethod
    async def forward_upscaling_requests(self, synapse: VideoUpscalingProtocol) -> bt.Synapse: ...

    @abstractmethod
    async def blacklist_upscaling_requests(self, synapse: VideoUpscalingProtocol) -> bool: ...

    @abstractmethod
    async def priority_upscaling_requests(self, synapse: VideoUpscalingProtocol) -> float: ...

    @abstractmethod
    async def forward_length_check_requests(self, synapse: LengthCheckProtocol) -> bt.Synapse: ...

    @abstractmethod
    async def blacklist_length_check_requests(self, synapse: LengthCheckProtocol) -> bool: ...

    @abstractmethod
    async def priority_length_check_requests(self, synapse: LengthCheckProtocol) -> float: ...

    @abstractmethod
    async def forward_compression_requests(self, synapse: VideoCompressionProtocol) -> bt.Synapse: ...

    @abstractmethod
    async def blacklist_compression_requests(self, synapse: VideoCompressionProtocol) -> bool: ...

    @abstractmethod
    async def priority_compression_requests(self, synapse: VideoCompressionProtocol) -> float: ...

    @abstractmethod
    async def forward_task_warrant_requests(self, synapse: TaskWarrantProtocol) -> bt.Synapse: ...

    @abstractmethod
    async def blacklist_task_warrant_requests(self, synapse: TaskWarrantProtocol) -> bool: ...

    @abstractmethod
    async def priority_task_warrant_requests(self, synapse: TaskWarrantProtocol) -> float: ...

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Starts the miner's axon, making it active on the network.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The miner continues its operations until `should_exit` is set to True or an external interruption occurs.
        During each epoch of its operation, the miner waits for new blocks on the Bittensor network, updates its
        knowledge of the network (metagraph), and sets its weights. This process ensures the miner remains active
        and up-to-date with the network's latest state.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that miner is registered on the network.
        self.sync()

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        logger.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        # Start  starts the miner's axon, making it active on the network.
        self.axon.start()

        logger.info(f"Miner starting at block: {self.block}")

        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while not self.should_exit:
                while (
                    self.block - self.metagraph.last_update[self.uid]
                    < self.config.neuron.epoch_length
                ):
                    # Wait before checking again.
                    time.sleep(1)

                    # Check if we should exit.
                    if self.should_exit:
                        break

                # Sync metagraph and potentially set weights.
                self.sync()
                self.step += 1

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            logger.success("Miner killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            logger.error(traceback.format_exc())

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()

        if self.should_sync_metagraph():
            self.chain_sync()

    def check_registered(self):
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            logger.error(
                f"\nYour Miner: {self.wallet} is not registered to the subnet{self.config.netuid} \nRun 'btcli register' and try again."
            )
            exit()

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return False
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length

    def run_in_background_thread(self):
        """
        Starts the miner's operations in a separate background thread.
        This is useful for non-blocking operations.
        """
        if not self.is_running:
            logger.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            logger.debug("Started")

    def stop_run_thread(self):
        """
        Stops the miner's operations that are running in the background thread.
        """
        if self.is_running:
            logger.debug("Stopping miner in background thread.")
            self.should_exit = True
            if self.thread is not None:
                self.thread.join(5)
            self.is_running = False
            logger.debug("Stopped")

    def __enter__(self):
        """
        Starts the miner's operations in a background thread upon entering the context.
        This method facilitates the use of the miner in a 'with' statement.
        """
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the miner's background operations upon exiting the context.
        This method facilitates the use of the miner in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        self.stop_run_thread()
