import bittensor as bt
from abc import ABC, abstractmethod
import threading
from loguru import logger


class BaseMiner(ABC):
    def __init__(self, config: bt.Config):
        self.config = config

    def init_bittensor(self):
        self.subtensor = bt.subtensor(config=self.config)
        logger.info(f"Subtensor: {self.subtensor}")
        self.wallet = bt.wallet(config=self.config)
        logger.info(f"Wallet: {self.wallet}")
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        logger.info(f"Metagraph: {self.metagraph}")
        self.axon = bt.axon(config=self.config)
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            logger.error(
                f"\nYour Miner: {self.wallet} is not registered to the subnet{self.config.netuid} \nRun 'btcli register' and try again."
            )
            exit()
        else:
            self.my_subnet_uid = self.metagraph.hotkeys.index(
                self.wallet.hotkey.ss58_address
            )
            logger.info(f"Running Miner on uid: {self.my_subnet_uid}")
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

    def chain_sync(self):
        bt.logging.info("resync_metagraph()")
        self.metagraph.sync(subtensor=self.subtensor)

    @abstractmethod
    async def forward(self, synapse: bt.Synapse) -> bt.Synapse: ...

    @abstractmethod
    async def blacklist(self, synapse: bt.Synapse) -> bool: ...

    @abstractmethod
    async def priority(self, synapse: bt.synapse) -> float: ...
    