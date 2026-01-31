# logger.py
import logging
from pathlib import Path
from typing import Optional


class RunLogger:
    def __init__(
        self,
        log_path: Path,
        name: str = "combined-eval",
        level: int = logging.INFO,
        also_console: bool = True
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logger_name = f"{name}:{self.log_path}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(level)
        self._logger.propagate = False

        if not self._logger.handlers:
            fmt = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

            file_handler = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(fmt)
            self._logger.addHandler(file_handler)

            if also_console:
                stream_handler = logging.StreamHandler()
                stream_handler.setLevel(level)
                stream_handler.setFormatter(fmt)
                self._logger.addHandler(stream_handler)

    def info(self, msg: str): self._logger.info(msg)
    def debug(self, msg: str): self._logger.debug(msg)
    def warning(self, msg: str): self._logger.warning(msg)
    def error(self, msg: str): self._logger.error(msg)

    def exception(self, msg: str):
        self._logger.exception(msg)

    def log(self, message: str, level: str = "info"):
        level = level.lower()
        if level == "debug":
            self.debug(message)
        elif level == "info":
            self.info(message)
        elif level == "warning":
            self.warning(message)
        elif level == "error":
            self.error(message)
        elif level == "exception":
            self.exception(message)
        else:
            self.info(message)
