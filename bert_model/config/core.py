from strictyaml import YAML , load
from pathlib import Path
import sys

file = Path(__file__).resolve()
parent , root = file.parent , file.parents[1]
sys.path.append(str(root))

import bert_model
PACKAGE_ROOT = Path(bert_model.__file__).resolve().parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "dataset"
TRAINED_MODEL_DIR = PACKAGE_ROOT/"trained_models"


def find_config_file() -> Path:
    """
    Locate the configuration file
    """
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

def fetch_config_from_yaml(cfg_path:Path=None)-> YAML:
    """Parse YAML  containing the package configuration"""
    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path,'r') as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path:{cfg_path}")


config = fetch_config_from_yaml()