import logging
import sys

def get_logger(name):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if not log.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(
            "debug.log",
            mode="a",
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        log.addHandler(console_handler)
        log.addHandler(file_handler)

    return log