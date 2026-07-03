"""Logging helpers for ApiaViz runs."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

import torch


def model_logger(args):
    now = datetime.now()
    model_kind = "ann"
    output_folder = os.path.join(args.output_dir, model_kind, now.strftime("%d%m%y-%H-%M-%S"))
    os.makedirs(output_folder, exist_ok=True)

    logger = logging.getLogger("apiaviz")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(f"{output_folder}/apiaviz.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)-8s - %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    logger.info("")
    logger.info("ApiaViz")
    logger.info("Computational visual backbone and route-navigation tools")
    logger.info("")

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        logger.info(f"CUDA available: True -- Current device: {torch.cuda.get_device_name(current_device)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS available: True -- Current device: MPS")
    else:
        logger.info("CUDA available: False -- Current device: CPU")

    logger.info("")
    logger.info(f"Evaluating '{getattr(args, 'eval_dataset', '?')}'")

    logger.info("")
    return logger, output_folder
