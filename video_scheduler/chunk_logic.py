import random
import yaml

from redis_utils import push_organic_chunk, push_synthetic_chunk


def read_synthetic_urls(yaml_path: str):
    """
    Read the synthetic video URLs from a YAML config file.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("synthetic_videos", [])


def insert_organic_chunk_to_redis(r, url: str):
    """
    Insert a single organic chunk URL directly into the Redis queue.
    """
    push_organic_chunk(r, url)

