# %% Import
from pathlib import Path

import typer
from typing_extensions import Annotated

from config.config_model import write_example_config
from scripts.rl import rl_train
from utils.logger import print_stats

app = typer.Typer()


# %%
@app.command(name="create-config", help="Create an example configuration file")
def create_example_config(
    output_path: Annotated[
        Path, typer.Option(help="Path to the output example config yaml file")
    ] = Path(".config/example_config.yaml"),
):
    write_example_config(output_path)


# %% Define main
@app.command(name="train", help="Train the RL aggregator using the configuration file")
def run_rl(
    config_path: Annotated[
        Path, typer.Argument(help="Path to the configuration yaml file")
    ],
):
    rl_train(
        config_path=config_path,
    )


# %% Define main
if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    print_stats()
    app()
