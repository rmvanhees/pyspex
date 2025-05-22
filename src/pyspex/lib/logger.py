#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2022-2025 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
#
"""Provide function to start the logger for spx1_monitor."""

from __future__ import annotations

__all__ = ["start_logger"]

from importlib.resources import files
from logging.config import dictConfig

import yaml


def start_logger() -> None:
    """Initialize logger for pyspex."""
    yaml_fl = files("pyspex.Data").joinpath("logger_setup.yaml")
    if not yaml_fl.is_file():
        raise FileNotFoundError(f"{yaml_fl} not found")

    with yaml_fl.open("r", encoding="ascii") as fid:
        try:
            config_data = yaml.safe_load(fid)
        except yaml.YAMLError as exc:
            raise RuntimeError("failed to read YAML file") from exc

    dictConfig(config_data)
