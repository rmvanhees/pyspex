#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2023 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Contains the class `OrbitCounter` to build a revolution-counter database."""

from __future__ import annotations

__all__ = ["OrbitCounter"]

from pathlib import Path

import h5py
import numpy as np


class OrbitCounter:
    """Class to build a revolution-counter database from PACE_HKT products.

    Parameters
    ----------
    db_name : Path | str, optional
        name of the revolution-counter database
    """

    def __init__(self: OrbitCounter, db_name: str | Path | None = None) -> None:
        """..."""
        # ToDo: option to let counter start at a higher number
        self._counter: int = 0
        self._time_start: np.datetime64 | None = None
        self.db_name = None

        if db_name is not None:
            name = db_name if isinstance(db_name, Path) else Path(db_name)
            self.db_name = name.with_suffix(".h5")
        else:
            self.db_name = Path("pace_orbit_counter.h5")

    @property
    def orb_counter(self: OrbitCounter) -> int:
        """Return current value of the orbit-revolution counter."""
        return self._counter

    @property
    def orb_time_start(self: OrbitCounter) -> np.datetime64 | None:
        """Return timestamp of start of current orbit revolution."""
        return self._time_start

    def from_hkt(self: OrbitCounter, hkt_file: str | Path) -> bool:
        """Check if PACE_HKT product contains a start of a orbit revolution.

        Parameters
        ----------
        hkt_file: str | Path
           name of PACE_HKT product

        Returns
        -------
        bool
           True if start of new orbit revolution found in HKT product else False
        """
        with h5py.File(hkt_file) as fid:
            dset = fid["navigation_data/orb_time"]
            dt0 = np.datetime64("T".join(dset.attrs["units"].decode().split(" ")[2:]))
            z_vel = fid["navigation_data/orb_vel"][:, 2]
            indx0 = np.diff(np.signbit(z_vel)).nonzero()[0]
            if indx0.size == 0:
                return False
            # print(indx0)
            z_pos = fid["navigation_data/orb_pos"][indx0, 2]

            indx1 = (z_pos > 0).nonzero()[0]
            if indx1.size == 0:
                return False
            # print(indx1)
            indices = indx0[indx1]
            for indx in indices:
                x0 = dset[indx]
                x1 = dset[indx + 1]
                y0 = z_vel[indx]
                y1 = z_vel[indx + 1]
                frac = (x1 - x0) * y0 / (y0 - y1)
                self._counter += 1
                self._time_start = dt0 + np.timedelta64(int(1e6 * (x0 + frac)), "us")

        return True

    def db_init(self: OrbitCounter, hkt_file: str | Path) -> None:
        """Initialize new database for orbit-revolution counters.

        Parameters
        ----------
        htk_file: str | Path
          name of first (oldest) PACE_HKT product
        """
        with h5py.File(hkt_file) as fid:
            dset = fid["navigation_data/orb_time"]
            dt0 = np.datetime64("T".join(dset.attrs["units"].decode().split(" ")[2:]))
            z_time = dt0 + np.timedelta64(int(1e6 * dset[0]), "us")

        with h5py.File(self.db_name, "w") as fid:
            dset = fid.create_dataset("orb_counter", (1,), maxshape=(None,), dtype=int)
            dset[0] = self._counter

            dtype = np.dtype("S26")
            dset = fid.create_dataset(
                "orb_time_start", (1,), maxshape=(None,), dtype=dtype
            )
            dset[0] = str(z_time)

    def db_update(self: OrbitCounter) -> None:
        """Update database with new orbit-revolution data."""
        # ToDo: keep the database in chronological order
        if not self.db_name.is_file():
            raise FileNotFoundError

        with h5py.File(self.db_name, "r+") as fid:
            dset = fid["orb_counter"]
            if self._counter < np.min(dset):
                return
            if self._counter > np.max(dset):
                dset.resize((dset.size + 1,))
                dset[-1] = self._counter
                dset = fid["orb_time_start"]
                dset.resize((dset.size + 1,))
                dset[-1] = str(self._time_start)
            else:
                ii = 0
                for orb in dset[:]:
                    if self._counter == orb:
                        break
                    ii += 1
                dset = fid["orb_time_start"]
                dset[ii] = str(self._time_start)

    def revolution(
        self: OrbitCounter,
        count: int,
    ) -> list[np.datetime64, np.datetime64 | None] | None:
        """Return timestamp at start and end of this orbit-revolution.

        Parameters
        ----------
        count :  int
           value of orbit-revolution counter

        Returns
        -------
        list[np.datetime64, np.datetime64] | None
            begin and end time of orbit-revolution
        """
        res = None
        with h5py.File(self.db_name, "r") as fid:
            orb_time = fid["orb_time_start"][:]
            orb_counter = fid["orb_counter"][:]

        if count > orb_counter[-1]:
            return None
        if count == orb_counter[-1]:
            return [
                np.datetime64(orb_time[-1].decode()),
                None,
            ]

        ii = 0
        for orb in orb_counter[:]:
            if orb == count:
                break
            ii += 1

        return [
            np.datetime64(orb_time[ii].decode()),
            np.datetime64(orb_time[ii + 1].decode()),
        ]

    def find_on_time(
        self: OrbitCounter,
        timestamp: np.datetime64,
    ) -> list[int, np.datetime64, float | None] | None:
        """Return counter, start time and orbit phase for given timestamp.

        Parameters
        ----------
        timestamp :  np.datetime64

        Returns
        -------
        list[int, np.datetime64, float | None] | None
        """
        res = None
        with h5py.File(self.db_name, "r") as fid:
            orb_start = fid["orb_time_start"][:].astype("datetime64")
            ii = 0
            for tstamp in orb_start[1:]:
                if timestamp < tstamp:
                    break
                ii += 1
            else:
                res = [fid["orb_counter"][ii], orb_start[ii], None]

            if res is None:
                res = [
                    fid["orb_counter"][ii],
                    orb_start[ii],
                    (timestamp - orb_start[ii]) / (orb_start[ii + 1] - orb_start[ii]),
                ]
        return res


def main() -> None:
    """..."""
    data_dir = Path("/data/richardh/SPEXone/ocal/pace-sds/pace_hkt/V1.0/2022/02")
    file_list = sorted(data_dir.glob("**/PACE.*.HKT.nc"))
    odb = OrbitCounter()
    odb.db_init(file_list[0])
    for flname in file_list:
        if not odb.from_hkt(flname):
            continue

        odb.db_update()
        print(flname, odb.orb_counter, odb.orb_time_start)

    print(odb.revolution(00))
    print(odb.revolution(10))
    print(odb.revolution(30))
    print(odb.revolution(40))
    print(odb.find_on_time(np.datetime64("2022-02-01T00:00:00.000000")))
    print(odb.find_on_time(np.datetime64("2022-02-02T09:09:00.000000")))
    print(odb.find_on_time(np.datetime64("2022-02-03T00:00:00.000000")))

    total = 0
    for ii in range(50):
        res = odb.revolution(ii + 1)
        if res is None or res[1] is None:
            break
        total += (res[1] - res[0]).astype(int)
        print(ii, res[1] - res[0])
    print(total / (ii * 1e6))


if __name__ == "__main__":
    main()
