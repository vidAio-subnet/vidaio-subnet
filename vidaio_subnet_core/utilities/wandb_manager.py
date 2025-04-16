import os
import datetime
import wandb
from dotenv import load_dotenv
from loguru import logger
from vidaio_subnet_core import __version__ as version


load_dotenv()

class WandbManager:
    def __init__(self, validator=None):
        self.wandb = None
        self.wandb_start = datetime.date.today()
        self.validator = validator
        
        if not self.validator.config.wandb.off:
            if os.getenv("WANDB_API_KEY"):
                self.init_wandb()
        else:
            logger.warning("Running validators without Wandb. Recommend to add Wandb!")
            
    def init_wandb(self):
        logger.debug("Init wandb")
        
        """Creates a new wandb for validators logs"""
        
        self.wandb_start = datetime.date.today()
        current = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        
        name = f"validator-{self.validator.uid}--{version}--{current}"
        wandb_project = self.validator.config.wandb.project_name
        wandb_entity = self.validator.config.wandb.entity
        logger.info("Initializing wandb entity and project.")
        self.wandb = wandb.init(
            anonymous="must",
            name=name,
            project=wandb_project,
            entity=wandb_entity,
            config={
                "uid":self.validator.uid,
                "hotkey":self.validator.wallet.hotkey.ss58_address,
                "version":version,
                "type":"validator",
            },
            allow_val_change=True
        )
        
        logger.info(f"Init a new Wandb: {name}")
