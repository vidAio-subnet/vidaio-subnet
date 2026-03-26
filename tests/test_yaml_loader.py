import yaml

class DuplicatePreservingLoader(yaml.SafeLoader):
    pass

def construct_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    
    # Check for duplicates
    keys = set()
    has_duplicate = False
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in keys:
            has_duplicate = True
            break
        keys.add(key)
        
    if has_duplicate:
        mapping = []
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            value = loader.construct_object(value_node, deep=deep)
            mapping.append({key: value})
        return mapping
    else:
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            value = loader.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

DuplicatePreservingLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    construct_mapping)

yaml_string = """
Image:
  from_base: parachutes/python:3.12
  set_user: root
  run_command:
    - uv pip install --upgrade setuptools wheel
    - uv pip install huggingface_hub==0.19.4 minio
    - apt-get update && apt-get install -y ffmpeg
  set_user: chutes
  set_workdir: /app

NodeSelector:
  gpu_count: 1
  min_vram_gb_per_gpu: 16

Chute:
  shutdown_after_seconds: 600
  concurrency: 2
  max_instances: 5
  scaling_threshold: 0.5
"""

config = yaml.load(yaml_string, Loader=DuplicatePreservingLoader)
print('Config root type:', type(config))
print('Image node type:', type(config.get('Image')))
print('Image node content:\\n', config.get('Image'))
