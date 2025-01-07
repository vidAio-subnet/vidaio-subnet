import bittensor as bt
from abc import ABC, abstractmethod
import threading
from loguru import logger


class BaseMiner(ABC):
    def __init__(self, config: bt.Config):
        self.config = config

    def init_bittensor(self):
        self.subtensor = bt.subtensor(config=self.config)
        self.wallet = bt.wallet(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.axon = bt.axon(config=self.config)
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

    def chain_sync(self):
        self.metagraph.sync()

    @abstractmethod
    async def forward(self, synapse: bt.Synapse) -> bt.Synapse: ...

    @abstractmethod
    async def blacklist(self, synapse: bt.Synapse) -> bool: ...
