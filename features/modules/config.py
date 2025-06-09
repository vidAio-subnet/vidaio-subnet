import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
config = load_config()