# logging_config.py

import logging
from torch import distributed as dist
import os
from datetime import datetime

# levels are DEBUG, INFO, WARNING, ERROR, CRITICAL


def configure_logging(
    enable_logging: bool = False,
    log_dir: str = ".",
    prefix: str = "train",
):
    """
    Configure logging to console + time-stamped log file in `log_dir` (rank 0 only).
    """

    # reduce asyncio noise
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # --------- choose log level ---------
    level = logging.DEBUG if enable_logging else logging.INFO

    # --------- rank > 0: only console, no file ---------
    if dist.is_initialized() and dist.get_rank() != 0:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
            force=True,
        )
        return

    # --------- rank 0: console + file ---------
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{prefix}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    handlers = [
        logging.StreamHandler(),              # console
        logging.FileHandler(log_path, "w"),   # file
    ]

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )

    logging.info(f"Logging to file: {log_path}")