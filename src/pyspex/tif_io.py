#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2022 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""
Contains the class `TIFio` to read simulated SPEXone measurements.
"""
__all__ = ['TIFio']

from pathlib import Path

import numpy as np
try:
    import pytiff
except ModuleNotFoundError:
    FOUND_PYTIFF = False
else:
    FOUND_PYTIFF = True

# - global parameters ------------------------------


# - local functions --------------------------------


# - class TIFio -------------------------
if FOUND_PYTIFF:
    class TIFio():
        """
        This class can be used to read SPEXone instrument simulator output

        Parameters
        ----------
        hdr_file : str
        inp_tif : bool
        lineskip : bool

        Examples
        --------

        >>>  tif = TIFio(filename)
        >>>  print(tif.header())
        >>>  print(tif.tags()[0])
        >>>  print(tif.images().shape)

        """
        def __init__(self, hdr_file: str, inp_tif=False, lineskip=False):
            """Initialize TIFio object
            """
            if not Path(hdr_file).is_file():
                raise FileNotFoundError(f'file {hdr_file} not found')

            # initialize class-attributes
            self.filename = Path(hdr_file)
            self.dir_name = Path(hdr_file).parent
            self.__stem = Path(hdr_file).stem
            self.__header = None
            self.inp_tif = inp_tif
            self.lineskip = lineskip

        def __repr__(self):
            class_name = type(self).__name__
            return f'{class_name}({self.filename.name!r})'

        # --------------------------------------------------
        def header(self) -> dict:
            """Read header as dictionary.
            """
            res = {}
            with self.filename.open(encoding='ascii', errors='ignore') as fp:
                res['history'] = fp.readline()[:-1]
                _ = fp.readline()
                for line in fp:
                    if line == '-\n':
                        break
                    key, value = line[:-1].split(':')
                    if value.strip(' ') != 'None':
                        res[key.rstrip(' ')] = value.strip(' ')

                # read Spectral data stimulus
                line = fp.readline()
                buff = line[:-1].split('[')
                key = buff[0].rstrip(' ')
                if key == 'Spectral data stimulus':
                    ds_sets = buff[1].replace(']', '').split(', ')

                    ds_dict = {}
                    for name in ds_sets:
                        line = fp.readline()
                        if not line:
                            break
                        ds_dict[name] = np.array(line[:-1].split(','),
                                                 dtype=float)

                    if ds_dict:
                        res[key] = ds_dict

            self.__header = res
            return res

        # --------------------------------------------------
        def tags(self) -> dict:
            """return TIFF tags as dictionary.
            """
            if self.__header is None:
                self.header()
            n_frame = int(self.__header['Number of measurements'])

            res = []
            for num in range(n_frame):
                tif_path = self.dir_name / f'{self.__stem}_{num}.tif'
                with pytiff.Tiff(str(tif_path)) as handle:
                    res.append(handle.read_tags())

            return res

        # --------------------------------------------------
        def images(self, n_frame=None):
            """Return TIFF data as numpy array.

            Parameters
            ----------
            n_frame :  int, optional
            Distribute coadded signal (32-bit) over n_frame images (16-bit)
            """
            if self.__header is None:
                self.header()

            if self.inp_tif:
                tif_path = self.dir_name / f'{self.__stem}_inp.tif'
                with pytiff.Tiff(str(tif_path)) as handle:
                    data = handle[:]

                if n_frame is None:
                    n_frame = 1 + (data.max() // 0xFFFF)
                elif n_frame < 1 + (data.max() // 0xFFFF):
                    print('[WARNING]: n_frame too small'
                          ' - precision will be lost')

                if n_frame == 1:
                    return data.astype('u2')

                frames = np.zeros((n_frame,) + data.shape, dtype='u2')
                frames += (data // n_frame)
                diff = data - np.sum(frames, axis=0)
                for img in frames:
                    mask = diff > 0
                    img[mask] += 1
                    diff[mask] -= 1

                return frames

            # convert regular TIFF files
            n_frame = int(self.__header['Number of measurements'])

            if self.lineskip:
                tif_fmt = str(self.dir_name / '{}_lineskip_{}.tif')
            else:
                tif_fmt = str(self.dir_name / '{}_{}.tif')

            res = []
            for num in range(n_frame):
                with pytiff.Tiff(tif_fmt.format(self.__stem, num)) as handle:
                    res.append(handle[:])

            return np.array(res)
