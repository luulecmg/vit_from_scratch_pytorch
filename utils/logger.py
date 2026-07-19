import logging
from pathlib import Path
from typing import Union

class ColoredFormatter(logging.Formatter):
    """A command line formatter with different colors for each level."""

    def __init__(self):
        super().__init__()
        reset = "\033[0m"
        colors = {
            logging.DEBUG: f"{reset}\033[36m",  # cyan,
            logging.INFO: f"{reset}\033[32m",  # green
            logging.WARNING: f"{reset}\033[33m",  # yellow
            logging.ERROR: f"{reset}\033[31m",  # red
            logging.CRITICAL: f"{reset}\033[35m",  # magenta
        }
        fmt_str = "{color}%(levelname)s %(asctime)s %(process)d %(filename)s:%(lineno)4d:{reset} %(message)s"
        self.formatters = {
            level: logging.Formatter(fmt_str.format(color=color, reset=reset))
            for level, color in colors.items()
        }
        self.default_formatter = self.formatters[logging.INFO]

    def format(self, record):
        formatter = self.formatters.get(record.levelno, self.default_formatter)
        return formatter.format(record)

def get_logger(name: str = "trainer", run_dir: Union[Path, None] = None):
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    # file
    if run_dir is not None:
        log_file_path = run_dir / f"train.log"

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(ColoredFormatter())
        logger.addHandler(file_handler)
    
    return logger
