import yaml

def load_search_config(path="search_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
search_config = load_search_config()