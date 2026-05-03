import logging
import os
from datetime import datetime
import sys


def setup_logging(parameter, extra="_", root_log_dir="./logs"):
    alg_name = parameter.alg_name
    task_name = parameter.task
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    log_dir = os.path.join(root_log_dir, f"{task_name}_{alg_name}", date_str, time_str)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"train_{extra}.log")

    logger = logging.getLogger("unrl")
    logger.setLevel(logging.DEBUG)
    logger.propagate = True

    fh = logging.FileHandler(log_file_path, mode="w")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.info(f"Logging to {log_file_path}")
    logger = logging.getLogger("priv.setup")
    logger.info("Logger has been set up.")
    return log_file_path
