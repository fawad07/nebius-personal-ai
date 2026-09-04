import yaml

def load_config(path: str) -> dict:
    """
    Load YAML configuration file and return a dictionary.
    """
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")

