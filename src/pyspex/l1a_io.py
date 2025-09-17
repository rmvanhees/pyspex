#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2025 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Contains the class `L1Aio` to write PACE/SPEXone data in level-1A format."""

from __future__ import annotations

__all__ = ["L1Aio"]

import logging
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np

from .lib.attrs_def import attrs_def
from .lib.l1a_def import init_l1a
from .lib.tlm_utils import convert_hk

if TYPE_CHECKING:
    import datetime as dt

    from numpy.typing import NDArray


# - global parameters -------------------
module_logger = logging.getLogger("pyspex.l1a_io")

MCP_TO_SEC = 1e-7
ONE_DAY = 24 * 60 * 60


# - local functions ---------------------
def _binning_table_(img_hk: NDArray) -> NDArray[np.uint8]:
    """Return binning table identifier (zero for full-frame images)."""
    if "REG_FULL_FRAME" not in img_hk.dtype.names:
        module_logger.warning("can not determine binning table identifier")
        return np.full(len(img_hk), 0xFF, dtype="u1")

    # REG_FULL_FRAME:
    # 0: science = binning (no line skipping)
    # 1: diagnostic = full frame
    # 2: science = binning with line skipping
    uval, counts = np.unique(img_hk["REG_FULL_FRAME"] & 0x3, return_counts=True)
    if uval.size > 1:
        module_logger.warning("value of REG_FULL_FRAME not unique")
    full_frame = (uval[0] if uval.size == 1 else uval[counts.argmax()]).astype("u1")

    # REG_CMV_OUTPUTMODE
    # 0: 16 channels - not used
    # 1: 8 channels - Science mode
    # 2: 3 channels - not used
    # 3: 2 channels - Dignostic mode
    uval, counts = np.unique(img_hk["REG_CMV_OUTPUTMODE"] & 0x3, return_counts=True)
    if uval.size > 1:
        module_logger.warning("value of REG_CMV_OUTPUTMODE not unique")
    cmv_outputmode = (uval[0] if uval.size == 1 else uval[counts.argmax()]).astype("u1")

    # Diagnostic mode
    if full_frame == 1:
        if cmv_outputmode != 3:
            raise KeyError("Diagnostic mode with REG_CMV_OUTPUTMODE != 3")
        return np.zeros(len(img_hk), dtype="u1")

    # Science mode
    if full_frame == 2:
        if cmv_outputmode != 1:
            raise KeyError("Science mode with REG_CMV_OUTPUTMODE != 1")
        bin_tbl_start = img_hk["REG_BINNING_TABLE_START"]
        res = np.full(len(img_hk), 0xFF, dtype="u1")
        mask = ((img_hk["REG_FULL_FRAME"] & 0x3) == 2) & (
            (img_hk["REG_CMV_OUTPUTMODE"] & 0x3) == 1
        )
        res[mask] = 1 + (bin_tbl_start[mask] - 0x80000000) // 0x400000
        return res

    raise KeyError("REG_FULL_FRAME not equal to 1 or 2")


def _digital_offset_(img_hk: NDArray) -> int | NDArray[np.int32]:
    """Return digital offset including ADC offset [count]."""
    buff = img_hk["DET_OFFSET"].astype("i4")
    if np.isscalar(buff):
        if buff >= 8192:
            buff -= 16384
    else:
        buff[buff >= 8192] -= 16384

    return buff + 70


def _exposure_time_(img_hk: NDArray) -> NDArray[float]:
    """Return exposure time in seconds [float]."""
    return 1.29e-05 * (0.43 * img_hk["DET_FOTLEN"][:] + img_hk["DET_EXPTIME"][:])


def _nr_coadditions_(img_hk: NDArray) -> NDArray[np.uint8]:
    """Return number of coadditions."""
    return img_hk["REG_NCOADDFRAMES"]


# - class L1Aio -------------------------
class L1Aio:
    """Class to create SPEXone level-1A products.

    Parameters
    ----------
    product :  str
       Name of the SPEXone level-1A product
    ref_date :  dt.datetime
       Date of the first detector image
    dims :  dict
       Dimensions of the datasets, default values::

          number_of_images : None     # number of image frames
          samples_per_image : 184000  # depends on binning table
          hk_packets : None           # number of HK tlm-packets

    compression : bool, default=False
       Use compression on dataset /science_data/detector_images

    """

    dset_stored: ClassVar[dict[str, int]] = {
        "/science_data/detector_images": 0,
        "/science_data/detector_telemetry": 0,
        "/image_attributes/binning_table": 0,
        "/image_attributes/digital_offset": 0,
        "/image_attributes/nr_coadditions": 0,
        "/image_attributes/exposure_time": 0,
        "/image_attributes/icu_time_sec": 0,
        "/image_attributes/icu_time_subsec": 0,
        "/image_attributes/image_time": 0,
        "/image_attributes/timedelta_centre": 0,
        "/image_attributes/image_ID": 0,
        "/engineering_data/NomHK_telemetry": 0,
        # '/engineering_data/DemHK_telemetry': 0,
        "/engineering_data/temp_detector": 0,
        "/engineering_data/temp_housing": 0,
        "/engineering_data/temp_radiator": 0,
        "/engineering_data/HK_tlm_time": 0,
    }

    def __init__(
        self: L1Aio,
        product: Path | str,
        ref_date: dt.datetime,
        dims: dict,
        compression: bool = False,
    ) -> None:
        """Initialize access to a SPEXone level-1A product."""
        self.product: Path = Path(product) if isinstance(product, str) else product

        # initialize private class-attributes
        self.__epoch = ref_date

        # initialize level-1A product
        self.fid = init_l1a(product, ref_date, dims, compression)
        for key in self.dset_stored:
            self.dset_stored[key] = 0

    def __iter__(self: L1Aio) -> None:
        """Allow iteration."""
        for attr in sorted(self.__dict__):
            if not attr.startswith("__"):
                yield attr

    def __enter__(self: L1Aio) -> Self:
        """Initiate the context manager."""
        return self

    def __exit__(self: L1Aio, *args: object) -> bool:
        """Exit the context manager."""
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self: L1Aio) -> None:
        """Close product and check if required datasets are filled with data."""
        if self.fid is None:
            return

        # check if at least one dataset is updated
        if self.fid.dimensions["number_of_images"].size == 0:
            self.fid.close()
            self.fid = None
            return

        # check of all required dataset their sizes
        self.check_stored(allow_empty=True)
        self.fid.close()
        self.fid = None

    # ---------- PUBLIC FUNCTIONS ----------
    @property
    def epoch(self: L1Aio) -> dt.datetime:
        """Provide epoch for SPEXone."""
        return self.__epoch

    def get_dim(self: L1Aio, name: str) -> int:
        """Get size of a netCDF4 dimension."""
        return self.fid.dimensions[name].size

    # ----- ATTRIBUTES --------------------
    def get_attr(self: L1Aio, name: str, ds_name: str | None = None) -> str | None:
        """Read data of an attribute.

        Global or attached to a group or variable.

        Parameters
        ----------
        name : str
           name of the attribute
        ds_name : str, default=None
           name of dataset to which the attribute is attached

        Returns
        -------
        scalar or array_like
           value of attribute 'name', global or attached to dataset 'ds_name'

        """
        if ds_name is None:
            res = self.fid.getncattr(name)
        elif ds_name in self.fid.groups or ds_name in self.fid.variables:
            res = self.fid[ds_name].getncattr(name)
        else:
            return None

        return res.decode() if isinstance(res, bytes) else res

    def set_attr(
        self: L1Aio,
        name: str,
        value: np.scalar | NDArray,
        *,
        ds_name: str | None = None,
    ) -> None:
        """Write data to an attribute.

        Global or attached to a group or variable.

        Parameters
        ----------
        name : string
           name of the attribute
        value : scalar, array_like
           value or values to be written
        ds_name : str, default=None
           name of group or dataset to which the attribute is attached
           **Use group name without starting '/'**

        """
        if ds_name is None:
            if isinstance(value, str):
                self.fid.setncattr(name, np.bytes_(value))
            else:
                self.fid.setncattr(name, value)
        else:
            grp_name = str(PurePosixPath(ds_name).parent)
            var_name = str(PurePosixPath(ds_name).name)
            if grp_name != ".":
                if (
                    var_name not in self.fid[grp_name].groups
                    and var_name not in self.fid[grp_name].variables
                ):
                    raise KeyError(f"ds_name {ds_name} not in product")
            elif var_name not in self.fid.groups and var_name not in self.fid.variables:
                raise KeyError(f"ds_name {ds_name} not in product")

            if isinstance(value, str):
                self.fid[ds_name].setncattr(name, np.bytes_(value))
            else:
                self.fid[ds_name].setncattr(name, value)

    # ----- VARIABLES --------------------
    def get_dset(self: L1Aio, name: str) -> None:
        """Read data of a netCDF4 variable.

        Parameters
        ----------
        name : str
           name of dataset

        Returns
        -------
        scalar or array_like
           value of dataset 'name'

        """
        grp_name = str(PurePosixPath(name).parent)
        var_name = str(PurePosixPath(name).name)
        if grp_name != ".":
            if var_name not in self.fid[grp_name].variables:
                raise KeyError(f"dataset {name} not in level-1A product")
        elif var_name not in self.fid.variables:
            raise KeyError(f"dataset {name} not in level-1A product")

        return self.fid[name][:]

    def set_dset(self: L1Aio, name: str, value: np.scalar | NDArray) -> None:
        """Write data to a netCDF4 variable.

        Parameters
        ----------
        name : str
           Name of level-1A dataset
        value : scalar or array_like
           Value or values to be written

        """
        value = np.asarray(value)
        grp_name = str(PurePosixPath(name).parent)
        var_name = str(PurePosixPath(name).name)
        if grp_name != ".":
            if var_name not in self.fid[grp_name].variables:
                raise KeyError(f"dataset {name} not in level-1A product")
        elif var_name not in self.fid.variables:
            raise KeyError(f"dataset {name} not in level-1A product")

        self.fid[name][...] = value
        self.dset_stored[name] += 1 if value.shape == () else value.shape[0]

    # -------------------------
    def fill_global_attrs(self: L1Aio, inflight: bool = False) -> None:
        """Define global attributes in the SPEXone level-1A products.

        Parameters
        ----------
        inflight :  bool, default=False
           Measurements performed on-ground or inflight

        """
        dict_attrs = attrs_def(inflight)
        dict_attrs["product_name"] = self.product.name
        for key, value in dict_attrs.items():
            if value is not None:
                self.fid.setncattr(key, value)

    # - L1A specific functions ------------------------
    def check_stored(self: L1Aio, allow_empty: bool = False) -> None:
        """Check variables with the same first dimension have equal sizes.

        Parameters
        ----------
        allow_empty :  bool, default=False
           Allow variables to be empty

        """
        warn_str = (
            "SPEX level-1A format check [WARNING]:"
            ' size of variable "{:s}" is wrong, only {:d} elements'
        )

        # check image datasets
        dim_sz = self.get_dim("number_of_images")
        key_list = [
            x
            for x in self.dset_stored
            if (x.startswith("/science_data") or x.startswith("/image_attributes"))
        ]
        res = np.array([self.dset_stored[key] for key in key_list])
        if allow_empty:
            indx = ((res > 0) & (res != dim_sz)).nonzero()[0]
        else:
            indx = np.nonzero(res != dim_sz)[0]
        for ii in indx:
            print(warn_str.format(key_list[ii], res[ii]))

        # check house-keeping datasets
        dim_sz = self.get_dim("hk_packets")
        key_list = [x for x in self.dset_stored if x.startswith("/engineering_data")]
        res = []
        for key in key_list:
            res.append(self.dset_stored[key])
        res = np.array(res)
        if allow_empty:
            indx = np.nonzero((res > 0) & (res != dim_sz))[0]
        else:
            indx = np.nonzero(res != dim_sz)[0]
        for ii in indx:
            print(warn_str.format(key_list[ii], res[ii]))

    # ---------- PUBLIC FUNCTIONS ----------
    def fill_science(
        self: L1Aio,
        img_data: NDArray[np.uint16],
        img_hk: NDArray,
        img_id: NDArray[np.uint16],
    ) -> None:
        """Write Science data and housekeeping telemetry (Science).

        Parameters
        ----------
        img_data : NDArray[np.uint16]
           Detector image data
        img_hk : NDArray
           Structured array with all Science telemetry parameters
        img_id : NDArray[np.uint16]
           Detector frame counter modulo 0x3FFF

        Notes
        -----
        Adds detector_telemetry data to the group /science_data

        Parameters: binning_table, digital_offset, exposure_time
        and nr_coadditions are extracted from the telemetry packets and writen
        in the group /image_attributes

        """
        if len(img_hk) == 0:
            return

        self.set_dset("/science_data/detector_images", img_data)
        self.set_dset("/science_data/detector_telemetry", img_hk)
        self.set_dset("/image_attributes/image_ID", img_id)
        self.set_dset("/image_attributes/binning_table", _binning_table_(img_hk))
        self.set_dset("/image_attributes/digital_offset", _digital_offset_(img_hk))
        self.set_dset("/image_attributes/exposure_time", _exposure_time_(img_hk))
        self.set_dset("/image_attributes/nr_coadditions", _nr_coadditions_(img_hk))

    def fill_nomhk(self: L1Aio, nomhk_data: NDArray) -> None:
        """Write nominal house-keeping telemetry packets (NomHK).

        Parameters
        ----------
        nomhk_data : NDArray
           Structured array with all NomHK telemetry parameters

        Notes
        -----
        Writes nomhk_data as TM_telemetry in group /engineering_data

        Parameters: temp_detector and temp_housing are extracted and converted
        to Kelvin and writen to the group /engineering_data

        """
        if len(nomhk_data) == 0:
            return

        self.set_dset("/engineering_data/NomHK_telemetry", nomhk_data)

        if np.all(nomhk_data["TS1_DEM_N_T"] == 0):
            self.set_dset(
                "/engineering_data/temp_detector", np.full(nomhk_data.size, 18.33)
            )
        else:
            self.set_dset(
                "/engineering_data/temp_detector",
                convert_hk("TS1_DEM_N_T", nomhk_data["TS1_DEM_N_T"]),
            )

        if np.all(nomhk_data["TS2_HOUSING_N_T"] == 0):
            self.set_dset(
                "/engineering_data/temp_housing", np.full(nomhk_data.size, 19.61)
            )
        else:
            self.set_dset(
                "/engineering_data/temp_housing",
                convert_hk("TS2_HOUSING_N_T", nomhk_data["TS2_HOUSING_N_T"]),
            )

        if np.all(nomhk_data["TS3_RADIATOR_N_T"] == 0):
            self.set_dset(
                "/engineering_data/temp_radiator", np.full(nomhk_data.size, 0.6)
            )
        else:
            self.set_dset(
                "/engineering_data/temp_radiator",
                convert_hk("TS3_RADIATOR_N_T", nomhk_data["TS3_RADIATOR_N_T"]),
            )
