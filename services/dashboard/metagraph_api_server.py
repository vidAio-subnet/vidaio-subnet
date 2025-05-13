import bittensor as bt
import uvicorn
from fastapi import FastAPI
from typing import List, Dict, Any
from dataclasses import dataclass, field
from vidaio_subnet_core import CONFIG

config = CONFIG.bandwidth
threshold = config.min_stake

@dataclass
class MinerInfo:
    miners: List[Dict[str, Any]] = field(default_factory=list)
    
    def append(self, miner_uid, miner_hotkey, trust, incentive, emission, daily_reward):
        self.miners.append({
            "miner_uid": int(miner_uid),
            "miner_hotkey": miner_hotkey,
            "trust": float(trust),
            "incentive": float(incentive),
            "emission": float(emission),
            "daily_reward": float(daily_reward)
        })
    
    def to_dict(self):
        return {
            "miners": self.miners
        }

app = FastAPI()

def get_filtered_miners(threshold=threshold):
    miner_info = MinerInfo()
    subtensor = bt.subtensor()
    metagraph = subtensor.metagraph(netuid=85)
    
    incentives = metagraph.I
    trusts = metagraph.T
    emissions = metagraph.E
    hotkeys = metagraph.hotkeys
    stakes = metagraph.S
    uids = metagraph.uids
    daily_rewards = emissions * 20
    
    for uid, incentive, trust, emission, hotkey, stake, daily_reward in zip(
        uids, incentives, trusts, emissions, hotkeys, stakes, daily_rewards
    ):
        if stake < threshold:
            miner_info.append(
                miner_uid=uid,
                miner_hotkey=hotkey,
                trust=trust,
                incentive=incentive,
                emission=emission,
                daily_reward=daily_reward
            )
    
    return miner_info

@app.get("/miner_info")
def get_miner_info():

    miner_info = get_filtered_miners()
    return miner_info.to_dict()

if __name__ == "__main__":
    host = config.host
    port = config.port
    uvicorn.run(app, host=host, port=port)