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
"""Read settings from a YAML file."""

from __future__ import annotations

__all__ = ["conf_from_yaml"]

from pathlib import Path

import yaml


# - main function -----------------------------------
def conf_from_yaml(file_path: Path | str) -> dict:
    """Read settings from a YAML file: `file_path`.

    Parameters
    ----------
    file_path :  Path | str
       full path of YAML file

    Returns
    -------
    dict
       content of the configuration file

    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"{file_path} not found")

    with file_path.open("r", encoding="ascii") as fid:
        try:
            settings = yaml.safe_load(fid)
        except yaml.parser.ParserError as exc:
            raise RuntimeError(f"Failed to parse {file_path}") from exc

    return settings
