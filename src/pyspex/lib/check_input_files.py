#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2022-2024 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Check validity and format of SPEXone level-0 products."""

from __future__ import annotations

__all__ = ["check_input_files"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataclasses import dataclass


def check_input_files(config: dataclass) -> dataclass:
    """Check SPEXone level-0 files on existence and format.

    Parameters
    ----------
    config :  dataclass
       Dataclass that contains the settings of the L0-L1A processor

    Returns
    -------
    dataclass
       fields 'l0_format' {'raw', 'st3', 'dsb'} and 'l0_list' are updated.

    Raises
    ------
    FileNotFoundError
       If files are not found on the system.
    TypeError
       If determined file type differs from value supplied by user.

    """
    file_list = config.l0_list
    if file_list[0].suffix == ".H":
        if not file_list[0].is_file():
            raise FileNotFoundError(file_list[0])
        data_dir = file_list[0].parent
        file_stem = file_list[0].stem
        file_list = (
            sorted(data_dir.glob(file_stem + ".[0-9]"))
            + sorted(data_dir.glob(file_stem + ".?[0-9]"))
            + sorted(data_dir.glob(file_stem + "_hk.[0-9]"))
        )
        if not file_list:
            raise FileNotFoundError(file_stem + ".[0-9]")

        config.l0_format = "raw"
        config.l0_list = file_list
    elif file_list[0].suffix == ".ST3":
        if not file_list[0].is_file():
            raise FileNotFoundError(file_list[0])
        config.l0_format = "st3"
        config.l0_list = [file_list[0]]
    elif file_list[0].suffix == ".spx":
        file_list_out = []
        for flname in file_list:
            if not flname.is_file():
                raise FileNotFoundError(flname)

            if flname.suffix == ".spx":
                file_list_out.append(flname)

        if not file_list_out:
            raise FileNotFoundError(file_list)
        config.l0_format = "dsb"
        config.l0_list = file_list_out
    else:
        raise TypeError("Input files not recognized as SPEXone level-0 data")

    return config
