"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Class to convert TAI to UTC and visa versa

Copyright (c) 2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime

# - global parameters ------------------------------
EPOCH_1900 = int(datetime.fromisoformat("1900-01-01T00+00:00").timestamp())
EPOCH_1958 = int(datetime.fromisoformat("1958-01-01T00+00:00").timestamp())


# - class Clocks -------------------------
class Clocks:
    """
    Convert TAI to UTC or visa versa
    """
    def __init__(self):
        """
        Initializes class Clocks with table to convert TAI to UTC

        Notes
        -----
        Information obtained from IERS Bulletin C62

        origin: https://www.ietf.org/timezones/data/leap-seconds.list

        Valid until:  28 June 2022
        """
        # This table contains 3 columns:
        #  1) epoch as a number of seconds since 1 January 1900, 00:00:00
        #  2) number of seconds that must be added to UTC to compute TAI
        #  3) UTC time corresponding to the epoch in the first column
        self.table = [
            (2272060800, 10, "1972-01-01T00+00:00"),    # 1 Jan 1972
            (2287785600, 11, "1972-07-01T00+00:00"),    # 1 Jul 1972
            (2303683200, 12, "1973-01-01T00+00:00"),    # 1 Jan 1973
            (2335219200, 13, "1974-01-01T00+00:00"),    # 1 Jan 1974
            (2366755200, 14, "1975-01-01T00+00:00"),    # 1 Jan 1975
            (2398291200, 15, "1976-01-01T00+00:00"),    # 1 Jan 1976
            (2429913600, 16, "1977-01-01T00+00:00"),    # 1 Jan 1977
            (2461449600, 17, "1978-01-01T00+00:00"),    # 1 Jan 1978
            (2492985600, 18, "1979-01-01T00+00:00"),    # 1 Jan 1979
            (2524521600, 19, "1980-01-01T00+00:00"),    # 1 Jan 1980
            (2571782400, 20, "1981-07-01T00+00:00"),    # 1 Jul 1981
            (2603318400, 21, "1982-07-01T00+00:00"),    # 1 Jul 1982
            (2634854400, 22, "1983-07-01T00+00:00"),    # 1 Jul 1983
            (2698012800, 23, "1985-07-01T00+00:00"),    # 1 Jul 1985
            (2776982400, 24, "1988-01-01T00+00:00"),    # 1 Jan 1988
            (2840140800, 25, "1990-01-01T00+00:00"),    # 1 Jan 1990
            (2871676800, 26, "1991-01-01T00+00:00"),    # 1 Jan 1991
            (2918937600, 27, "1992-07-01T00+00:00"),    # 1 Jul 1992
            (2950473600, 28, "1993-07-01T00+00:00"),    # 1 Jul 1993
            (2982009600, 29, "1994-07-01T00+00:00"),    # 1 Jul 1994
            (3029443200, 30, "1996-01-01T00+00:00"),    # 1 Jan 1996
            (3076704000, 31, "1997-07-01T00+00:00"),    # 1 Jul 1997
            (3124137600, 32, "1999-01-01T00+00:00"),    # 1 Jan 1999
            (3345062400, 33, "2006-01-01T00+00:00"),    # 1 Jan 2006
            (3439756800, 34, "2009-01-01T00+00:00"),    # 1 Jan 2009
            (3550089600, 35, "2012-07-01T00+00:00"),    # 1 Jul 2012
            (3644697600, 36, "2015-07-01T00+00:00"),    # 1 Jul 2015
            (3692217600, 37, "2017-01-01T00+00:00"),    # 1 Jan 2017
            (3865363200, None, "2022-06-28T00+00:00")]  # expiring date

        # use epoch since 1970.0 instead 1900.0
        self.table = [(x + EPOCH_1900, y, z) for x, y, z in self.table]

    def to_tai(self, timestamp: int) -> int:
        """
        Return TAI timestamp for given UTC timestamp

        Parameters
        ----------
        timestamp :  int
           timestamp in UTC (epoch 1970.0)

        Return
        ------
        int :  TAI timestamp (epoch 1958.0)
        """
        for iers_epoch, iers_offs, _ in self.table[::-1]:
            if timestamp >= iers_epoch:
                break

        if iers_offs is None:
            raise ValueError('update your leap second list through'
                             ' IERS Bulletin C62')

        return timestamp + iers_offs - EPOCH_1958

    def to_utc(self, timestamp: int) -> int:
        """
        Return UTC timestamp for given TAI timestamp

        Parameters
        ----------
        timestamp :  int
           timestamp in UTC (epoch 1958.0)

        Return
        ------
        int :  UTC timestamp (epoch 1970.0)
        """
        timestamp += EPOCH_1958
        for iers_epoch, iers_offs, _ in self.table[::-1]:
            if timestamp >= iers_epoch:
                break

        if iers_offs is None:
            raise ValueError('update your leap second list through'
                             ' IERS Bulletin C62')

        return timestamp - iers_offs

    def test(self) -> None:
        """
        Perform unit test on class Clocks
        """
        for res in self.table:
            dt_str = datetime.fromisoformat(res[2])
            if res[0] != int(dt_str.timestamp()):
                raise ValueError(f'Detected error in table at epoch={res[0]}')

        print('INFO: current information on leap seconds expires'
              f' at {self.table[-1][2]}')


def main():
    """
    main function
    """
    clocks = Clocks()
    clocks.test()

    for xx in [489023999, 489024000, 489024001]:
        print(f'{xx}'
              f' -> {datetime.utcfromtimestamp(xx).isoformat()}'
              f' -> {clocks.to_tai(xx)}')

    for xx in [1120176021, 1120176022, 1120176023, 1120176024]:
        print(f'{xx} -> {clocks.to_utc(xx)}')

    # check timestamp after the leap seconds file has expires
    timestamp = datetime.fromisoformat('2022-06-28T00+00:00').timestamp()
    clocks.to_tai(timestamp)


if __name__ == '__main__':
    main()