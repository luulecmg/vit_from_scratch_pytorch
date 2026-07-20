import logging
from pathlib import Path

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

def get_logger(logger_name: str, log_dir: str):
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # remove other loggers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []
    
    # console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    # file
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file_path = Path(log_dir) / f"train.log"

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logging.Formatter(
        "%(levelname)s %(asctime)s %(process)d %(filename)s:%(lineno)4d: %(message)s"
        ))
        logger.addHandler(file_handler)
    
    return logger
