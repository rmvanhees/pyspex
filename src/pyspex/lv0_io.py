"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Various functions to read SPEXone level 0 data and write level 0 packages
to a level-1A product.

Copyright (c) 2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime, timedelta, timezone

import numpy as np

from pyspex.lib.tmtc_def import tmtc_dtype
from pyspex.lv1_io import L1Aio


# - global parameters ------------------------------
EPOCH_1958 = datetime(1958, 1, 1, tzinfo=timezone.utc)
EPOCH_1970 = datetime(1970, 1, 1, tzinfo=timezone.utc)

FULLFRAME_BYTES = 2 * 2048 * 2048


# - local functions --------------------------------
def ap_id(hdr: np.ndarray) -> int:
    """
    Returns Telemetry APID, the range 0x320 to 0x351 is available to SPEXone

    The following values are recognized:
        0x350 : Science
        0x320 : NomHk
        0x322 : DemHk
        0x331 : TcAccept
        0x332 : TcReject
        0x333 : TcExecute
        0x334 : TcFail
        0x335 : EventRp
    """
    return hdr['type'] & 0x7FF


def grouping_flag(hdr: np.ndarray) -> int:
    """
    Returns grouping flag

    The 2-byte flag is encoded as follows:
        00 continuation packet-data segment
        01 first packet-data segment
        10 last packet-data segment
        11 packet-data unsegmented
    """
    return (hdr['sequence'] >> 14) & 0x3


def sequence(hdr: np.ndarray) -> int:
    """
    Returns sequence counter, rollover to zero at 0x3FFF
    """
    return hdr['sequence'] & 0x3FFF


def packet_length(hdr: np.ndarray) -> int:
    """
    Returns size of secondary header + user data - 1 in bytes

    Notes
    -----
    We always read the primary header and secondary header at once.

    Value range: 7 - 16375
    """
    return hdr['length']

def dtype_packet_hdr(file_format: str) -> np.dtype:
    """
    Return numpy dtype of the packet headers

    Parameters
    ----------
    file_format: str
       File format of level 0 products: raw, dsb or st3
       'raw' data has no file header and standard CCSDS packet headers
       'st3' data has no file header and ITOS + spacewire + CCSDS packet headers
       'dsb' data has a cFE file header and spacewire + CCSDS packet headers
    """
    if file_format == 'raw':
        return np.dtype([('type', '>u2'),
                         ('sequence', '>u2'),
                         ('length', '>u2'),
                         ('tai_sec', '>u4'),
                         ('sub_sec', '>u2')])

    if file_format == 'dsb':
        return np.dtype([('spacewire', 'u1', (2,)),
                         ('type', '>u2'),
                         ('sequence', '>u2'),
                         ('length', '>u2'),
                         ('tai_sec', '>u4'),
                         ('sub_sec', '>u2')])

    if file_format == 'st3':
        return np.dtype([('itos_hdr', '>u2', (8,)),
                         ('spacewire', 'u1', (2,)),
                         ('type', '>u2'),
                         ('sequence', '>u2'),
                         ('length', '>u2'),
                         ('tai_sec', '>u4'),
                         ('sub_sec', '>u2')])

    return None


def dtype_tmtc(hdr: np.dtype) -> np.dtype:
    """
    Return numpy dtype of a telemetry message or command
    """
    return {0x320: np.dtype([('hdr', hdr.dtype),
                             ('hk', tmtc_dtype(0x320))]),
            0x322: np.dtype([('hdr', hdr.dtype),
                             ('hk', tmtc_dtype(0x322))]),
            0x331: np.dtype([('hdr', hdr.dtype),
                             ('TcPacketId', '>u2'),
                             ('TcSeqControl', '>u2')]),
            0x332: np.dtype([('hdr', hdr.dtype),
                             ('TcPacketId', '>u2'),
                             ('TcSeqControl', '>u2'),
                             ('TcErrorCode', '>u2'),
                             ('RejectParameter1', '>u2'),
                             ('RejectParameter2', '>u2')]),
            0x333: np.dtype([('hdr', hdr.dtype),
                             ('TcPacketId', '>u2'),
                             ('TcSeqControl', '>u2')]),
            0x334: np.dtype([('hdr', hdr.dtype),
                             ('TcPacketId', '>u2'),
                             ('TcSeqControl', '>u2'),
                             ('TcErrorCode', '>u2'),
                             ('FailParameter1', '>u2'),
                             ('FailParameter2', '>u2')])}.get(ap_id(hdr), None)


def read_lv0_data(args) -> tuple:
    """
    Read level 0 data and return Science and telemetry data
    """
    hdr_dtype = dtype_packet_hdr(args.file_format)
    scihk_dtype = tmtc_dtype(0x350)
    icutm_dtype = np.dtype([('tai_sec', '>u4'),
                            ('sub_sec', '>u2')])

    # read level 0 headers and CCSDS data of Science and TmTc data
    ccsds_sci = ()
    ccsds_hk = ()
    for flname in args.file_list:
        with open(flname, 'rb') as fp:
            offs = 0
            ccsds_data = fp.read()

            # read cFE file header
            if args.file_format == 'dsb':
                # read cFE file header
                cfe_dtype = np.dtype([
                    ('ContentType', 'S4'),
                    ('SubType', 'S4'),
                    ('FileHeaderLength', '>u4'),
                    ('SpacecraftID', 'S4'),
                    ('ProcessorID', '>u4'),
                    ('InstrumentID', 'S4'),
                    ('TimeSec', '>u4'),
                    ('TimeSubSec', '>u4'),
                    ('Filename', 'S32'),
                ])
                cfe_hdr = np.frombuffer(ccsds_data, count=1, offset=offs,
                                        dtype=cfe_dtype)[0]
                # Now we can check the values of the cFE File header
                # or even write these values to the L1A product
                if args.verbose:
                    print('[INFO] cFE File header: ', cfe_hdr)
                offs += cfe_dtype.itemsize

            # read CCSDS header and user data
            while offs < len(ccsds_data):
                hdr = np.frombuffer(ccsds_data, count=1, offset=offs,
                                    dtype=hdr_dtype)[0]
                # copy the full CCSDS package
                if args.debug:
                    print(ap_id(hdr), grouping_flag(hdr),
                          hdr_dtype.itemsize, hdr['length'], offs)
                    offs += hdr_dtype.itemsize + hdr['length'] - 5
                elif ap_id(hdr) == 0x350:                   # Science APID
                    nbytes = hdr['length'] - 5
                    if grouping_flag(hdr) == 1:
                        buff = np.empty(1, dtype=np.dtype([
                            ('hdr', hdr_dtype),
                            ('hk', scihk_dtype),
                            ('icu_tm', icutm_dtype),
                            ('frame', 'O')]))
                        buff['hdr'] = hdr
                        offs += hdr_dtype.itemsize
                        buff['hk'] = np.frombuffer(ccsds_data,
                                                   count=1, offset=offs,
                                                   dtype=scihk_dtype)[0]
                        offs += scihk_dtype.itemsize
                        buff['icu_tm'] = np.frombuffer(ccsds_data,
                                                       count=1, offset=offs,
                                                       dtype=icutm_dtype)[0]
                        offs += icutm_dtype.itemsize
                        nbytes -= (scihk_dtype.itemsize + icutm_dtype.itemsize)
                    else:
                        buff = np.empty(1, dtype=np.dtype([
                            ('hdr', hdr_dtype),
                            ('frame', 'O')]))
                        buff['hdr'] = hdr
                        offs += hdr_dtype.itemsize

                    buff['frame'][0] = np.frombuffer(ccsds_data,
                                                     count=nbytes // 2,
                                                     offset=offs, dtype='>u2')
                    ccsds_sci += ({'hdr': hdr, 'data': buff.copy()},)
                    offs += nbytes
                elif 0x320 <= ap_id(hdr) < 0x335:           # other valid APIDs
                    buff = np.frombuffer(ccsds_data, count=1, offset=offs,
                                         dtype=dtype_tmtc(hdr))[0]
                    ccsds_hk += ({'hdr': hdr, 'data': buff},)
                    offs += dtype_tmtc(hdr).itemsize
                else:
                    offs += hdr_dtype.itemsize + hdr['length'] - 5
        del ccsds_data

    if args.verbose:
        print(f'[INFO] number of Science packages: {len(ccsds_sci)}')
        print(f'[INFO] number of Engineering packages: {len(ccsds_hk)}')

    return ccsds_sci, ccsds_hk


def dump_lv0_data(args, ccsds_sci, ccsds_hk) -> None:
    """
    Perform an ASCII dump of level 0 data
    """
    # dump header information of the Science packages
    flname = args.datapath / (args.file_list[0].stem + '.dump')
    with open(flname, 'w', encoding='ascii') as fp:
        fp.write('APID Grouping Counter Length'
                 ' ICUSWVER MPS_ID  IMRLEN     ICU_SEC ICU_SUBSEC\n')
        for segment in ccsds_sci:
            hdr = segment['hdr']
            data = segment['data'][0]
            if grouping_flag(hdr) == 1:
                fp.write(f"{ap_id(hdr):4x} {grouping_flag(hdr):8d}"
                         f" {sequence(hdr):7d} {packet_length(hdr):6d}"
                         f" {data['hk']['ICUSWVER']:8x}"
                         f" {data['hk']['MPS_ID']:6d}"
                         f" {data['hk']['IMRLEN']:7d}"
                         f" {data['icu_tm']['tai_sec']:11d}"
                         f" {data['icu_tm']['sub_sec']:10d}\n")
            else:
                fp.write(f'{ap_id(hdr):4x} {grouping_flag(hdr):8d}'
                         f' {sequence(hdr):7d} {packet_length(hdr):6d}\n')

    # dump header information of the nominal house-keeping packages
    flname = args.datapath / (args.file_list[0].stem + '_hk.dump')
    with open(flname, 'w', encoding='ascii') as fp:
        fp.write('APID Grouping Counter Length     TAI_SEC    SUB_SEC'
                 ' ICUSWVER MPS_ID TcSeqControl TcErrorCode\n')
        for segment in ccsds_hk:
            hdr = segment['hdr']
            data = segment['data']
            if ap_id(hdr) == 0x320:
                fp.write(f"{ap_id(hdr):4x} {grouping_flag(hdr):8d}"
                         f" {sequence(hdr):7d} {packet_length(hdr):6d}"
                         f" {hdr['tai_sec']:11d} {hdr['sub_sec']:10d}"
                         f" {data['hk']['ICUSWVER']:8x}"
                         f" {data['hk']['MPS_ID']:6d}\n")
            elif ap_id(hdr) in (0x332, 0x334):
                fp.write(f"{ap_id(hdr):4x} {grouping_flag(hdr):8d}"
                         f" {sequence(hdr):7d} {packet_length(hdr):6d}"
                         f" {hdr['tai_sec']:11d} {hdr['sub_sec']:10d}"
                         f" {-1:8x} {-1:6d} {data['TcSeqControl']:12d}"
                         f" {bin(data['TcErrorCode'])}\n")
            else:
                fp.write(f"{ap_id(hdr):4x} {grouping_flag(hdr):8d}"
                         f" {sequence(hdr):7d} {packet_length(hdr):6d}"
                         f" {hdr['tai_sec']:11d} {hdr['sub_sec']:10d}\n")


def select_lv0_data(args, ccsds_sci, ccsds_hk) -> tuple:
    """
    read Science packages and collect all detector read-outs
    """
    ii = 0
    for segment in ccsds_sci:
        if grouping_flag(segment['hdr']) == 1:
            break
        ii += 1

    if ii > 0:
        print(f'[WARNING]: found first valid segment at {ii}')

    frame = ()
    science = ()
    for segment in ccsds_sci[ii:]:
        hdr = segment['hdr']
        if grouping_flag(hdr) == 1:
            buff = segment['data']
            frame = (buff['frame'][0],)
        else:
            frame += (segment['data']['frame'][0],)
        if grouping_flag(hdr) == 2:
            buff['frame'][0] = np.concatenate(frame)
            if args.select == 'all':
                science += (buff.copy(),)
            elif (args.select == 'binned'
                  and buff['hk']['IMRLEN'][0] < FULLFRAME_BYTES):
                science += (buff.copy(),)
            elif (args.select == 'fullFrame'
                  and buff['hk']['IMRLEN'][0] == FULLFRAME_BYTES):
                science += (buff.copy(),)

    science = np.concatenate(science)
    mps_list = np.unique(science['hk']['MPS_ID']).tolist()
    if args.verbose:
        print(f'[INFO]: list of unique MPS {mps_list}')

    if ccsds_hk:
        nomhk = np.concatenate(
            [(x['data'],) for x in ccsds_hk if ap_id(x['hdr']) == 0x320
             and x['data']['hk']['MPS_ID'] in mps_list])
    else:
        nomhk = np.array(())

    return science, nomhk


def get_science_timestamps(science):
    """
    Return timestamps of the Science packets
    """
    if science['hk']['ICUSWVER'][0] > 0x123:
        img_sec = science['icu_tm']['tai_sec']
        img_subsec = science['icu_tm']['sub_sec']
        return (img_sec, img_subsec)

    img_sec = science['hdr']['tai_sec']
    img_subsec = science['hdr']['sub_sec']
    if science['hk']['ICUSWVER'][0] == 0x123:
        # fix bug in sub-seconds
        us100 = np.round(10000 * img_subsec.astype(float) / 65536)
        buff = us100 + img_sec - 10000
        us100 = buff.astype('u8') % 10000
        img_subsec = ((us100 << 16) // 10000).astype('u2')
    return (img_sec, img_subsec)


def get_nomhk_timestamps(nomhk):
    """
    Return timestamps of the telemetry packets
    """
    nomhk_sec = nomhk['hdr']['tai_sec']
    nomhk_subsec = nomhk['hdr']['sub_sec']
    if nomhk['hk']['ICUSWVER'][0] == 0x123:
        # fix bug in sub-seconds
        us100 = np.round(10000 * nomhk_subsec.astype(float) / 65536)
        buff = us100 + nomhk_sec - 10000
        us100 = buff.astype('u8') % 10000
        nomhk_subsec = ((us100 << 16) // 10000).astype('u2')
    return (nomhk_sec, nomhk_subsec)


def get_l1a_name(args, science) -> str:
    """
    Generate name of Level-1A product, using the following filename conventions

    Inflight
    --------
    L1A file name format, following the NASA ... naming convention:
       PACE_SPEXone[_TTT].YYYYMMDDTHHMMSS.L1A.Vnn.nc
    where
       TTT is an optional data type (e.g., for the calibration data files)
       YYYYMMDDTHHMMSS is time stamp of the first image in the file
       nn file-version number
    for example
    [Science Product] PACE_SPEXone.20230115T123456.L1A.V01.nc
    [Calibration Product] PACE_SPEXone_CAL.20230115T123456.L1A.V01.nc
    [Monitoring Products] PACE_SPEXone_DARK.20230115T123456.L1A.V01.nc

    OCAL
    ----
    L1A file name format:
       SPX1_OCAL_<msm_id>_L1A_YYYYMMDDTHHMMSS_yyyymmddThhmmss_vvvv.nc
    where
       msm_id is the measurement identifier
       YYYYMMDDTHHMMSS is time stamp of the first image in the file
       yyyymmddThhmmss is the creation time (UTC) of the product
       vvvv is the version number of the product starting at 0001
    """
    img_sec, _ = get_science_timestamps(science)
    if args.file_format != 'raw':
        # inflight product name
        # ToDo: detect Diagnostic DARK measurements
        prod_type = '_CAL' if args.select == 'fullFrame' else ''
        #sensing_start = EPOCH_1958 + timedelta(seconds=int(img_sec[0]))
        sensing_start = EPOCH_1970 + timedelta(seconds=int(img_sec[0]))

        return (f'PACE_SPEXone{prod_type}'
                f'.{sensing_start.strftime("%Y%m%dT%H%M%S"):15s}.L1A'
                f'.V{args.file_version:02d}.nc')

    # OCAL product name
    sensing_start = EPOCH_1970 + timedelta(seconds=int(img_sec[0]))

    # determine measurement identifier
    msm_id = args.file_name.stem
    try:
        new_date = datetime.strptime(
            msm_id[-22:], '%y-%j-%H:%M:%S.%f').strftime('%Y%m%dT%H%M%S.%f')
    except ValueError:
        pass
    else:
        msm_id = msm_id[:-22] + new_date

    return (f'SPX1_OCAL_{msm_id}_L1A'
            f'_{sensing_start.strftime("%Y%m%dT%H%M%S"):15s}'
            f'_{datetime.utcnow().strftime("%Y%m%dT%H%M%S"):15s}'
            f'_{args.file_version:04d}.nc')

def write_lv0_data(args, science, nomhk) -> None:
    """
    Write level 0 packages to a level-1A product
    """
    dims = {'number_of_images': science.size,
            'samples_per_image': science['hk']['IMRLEN'].max() // 2,
            'hk_packets': nomhk.size,
            'SC_records': None}

    # generate name of the level-1A product
    prod_name = get_l1a_name(args, science)

    # Generate and fill L1A product
    with L1Aio(args.datapath / prod_name, dims=dims) as l1a:
        # write image data, detector telemetry and image attributes
        img_data = np.empty((science.size, dims['samples_per_image']),
                            dtype=float)
        for ii, data in enumerate(science['frame']):
            img_data[ii, :data.size] = data
        l1a.fill_science(img_data, science['hk'],
                         np.bitwise_and(science['hdr']['sequence'], 0x3fff))
        del img_data
        img_sec, img_subsec = get_science_timestamps(science)
        l1a.fill_time(img_sec, img_subsec, group='image_attributes')

        # write engineering data
        if nomhk.size > 0:
            l1a.fill_nomhk(nomhk['hk'])
            nomhk_sec, nomhk_subsec = get_nomhk_timestamps(nomhk)
            l1a.fill_time(nomhk_sec, nomhk_subsec, group='engineering_data')

        # if demhk.size > 0:
        #    l1a.fill_demhk(demhk['hk'])

        # write global attributes
        if nomhk.size > 0:
            l1a.set_attr('icu_sw_version', nomhk[0]['hk']['ICUSWVER'])
        if args.file_format == 'raw':
            l1a.fill_global_attrs(inflight=False)
        else:
            l1a.fill_global_attrs(inflight=True)
        l1a.set_attr('input_files',
                     [x.name for x in args.file_list])
