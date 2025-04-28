from pathlib import Path

from config.config_model import write_example_config


def create_example_config(output_path=Path(".config/example_config.yaml")):
    write_example_config(output_path)
