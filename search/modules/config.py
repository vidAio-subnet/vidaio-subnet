import yaml

def load_config(path="search_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
config = load_config()