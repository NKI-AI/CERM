from typing import Union, List
import csv
from pathlib import Path
from enum import Enum
from omegaconf import DictConfig
from logging import getLogger
from mra.pnp_hyperparam.search_space import Hypers

logger = getLogger(__name__)


def find_config_entry(config: DictConfig, entry_name: str):
    entry = config
    for level in entry_name.split("."):
        entry = entry[level]
    return entry


def write_to_log_file(
    file_loc: Union[str, Path],
    hyper_config: Enum,
    config: DictConfig,
    score: Union[List[float], float],
    unique_id: str,
):
    file_path = Path(file_loc)

    # First time: write header with variable names
    if not file_path.exists():
        header = [hyper.value for hyper in hyper_config]
        header.append("unique_id")
        header.append("score(s)")
        with open(str(file_loc), "a") as fd:
            wr = csv.writer(fd)
            wr.writerow(header)

    # Write current run results
    try:
        row = [find_config_entry(config, hyper.value) for hyper in hyper_config]
        row.append(unique_id)
        row.append(score)

        with open(str(file_loc), "a") as fd:
            wr = csv.writer(fd)
            wr.writerow(row)

    # We don't want the hyperparam search to stop if writing fails
    except:
        logger.error("Writing config and score to log file failed")


def pre_search_checks(config: DictConfig) -> None:
    if not config.get("log_file"):
        raise RuntimeError(
            f"log_file should be specified in the config (config.log_file) for hyperparam search"
        )


def process_score(config: DictConfig, score: Union[List[float], float], unique_id: str):
    # write hyperparams and score
    write_to_log_file(config.log_file, Hypers, config, score, unique_id)
