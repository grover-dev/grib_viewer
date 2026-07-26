"""GRIB fixtures shaped like the real thing.

The bugs this suite guards against only appear on ERA5's awkward shapes -- a
0..360 longitude grid, accumulations carried on a (reference time + step) ladder
rather than a plain time axis -- so the fixtures reproduce those rather than
writing tidy files that would pass no matter what.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import eccodes as ec
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _write(fh, *, paramid, values, lat0, lat1, lon0, lon1, date, hour, step=0):
    """Write one regular_ll message. step > 0 makes it an accumulation."""
    nj, ni = values.shape
    h = ec.codes_grib_new_from_samples("regular_ll_sfc_grib1")
    ec.codes_set(h, "paramId", paramid)
    ec.codes_set(h, "Ni", ni)
    ec.codes_set(h, "Nj", nj)
    ec.codes_set(h, "latitudeOfFirstGridPointInDegrees", lat0)
    ec.codes_set(h, "latitudeOfLastGridPointInDegrees", lat1)
    ec.codes_set(h, "longitudeOfFirstGridPointInDegrees", lon0)
    ec.codes_set(h, "longitudeOfLastGridPointInDegrees", lon1)
    ec.codes_set(h, "iDirectionIncrementInDegrees", (lon1 - lon0) / (ni - 1))
    ec.codes_set(h, "jDirectionIncrementInDegrees", abs(lat0 - lat1) / (nj - 1))
    ec.codes_set(h, "jScansPositively", 0)
    ec.codes_set(h, "iScansNegatively", 0)
    ec.codes_set(h, "dataDate", date)
    ec.codes_set(h, "hour", hour)
    if step:
        # an accumulation: the reference time is NOT the hour it describes
        ec.codes_set(h, "stepUnits", 1)          # hours
        ec.codes_set(h, "startStep", step - 1)
        ec.codes_set(h, "endStep", step)
    ec.codes_set_values(h, values.astype(float).ravel())
    ec.codes_write(h, fh)
    ec.codes_release(h)


@pytest.fixture(scope="session")
def global_grib(tmp_path_factory) -> Path:
    """Global 0..360 grid, 4 hourly steps, instantaneous fields only.

    Longitudes run 0..350 like real ERA5, so anything that assumes -180..180 is
    caught. Values encode their own longitude, which lets a test assert that a
    rolled or cropped grid still carries the right column in the right place.
    """
    path = tmp_path_factory.mktemp("grib") / "global.grib"
    nj, ni = 19, 36                      # 10 degree grid
    lats = np.linspace(90, -90, nj)
    lons = np.linspace(0, 350, ni)       # 0..360 frame, NOT -180..180
    lon_g, lat_g = np.meshgrid(lons, lats)

    with open(path, "wb") as fh:
        for t, hour in enumerate([0, 6, 12, 18]):
            # u10 = longitude, v10 = latitude: values identify their own cell
            _write(fh, paramid=165, values=lon_g, lat0=90, lat1=-90, lon0=0, lon1=350,
                   date=20250101, hour=hour)
            _write(fh, paramid=166, values=lat_g, lat0=90, lat1=-90, lon0=0, lon1=350,
                   date=20250101, hour=hour)
            _write(fh, paramid=167, values=np.full((nj, ni), 273.15 + t),
                   lat0=90, lat1=-90, lon0=0, lon1=350, date=20250101, hour=hour)
    return path


@pytest.fixture(scope="session")
def stepped_grib(tmp_path_factory) -> Path:
    """Instantaneous + accumulated fields, ERA5-style.

    t2m sits on a plain hourly axis (00..05Z on 2025-01-02).
    ssrd is an accumulation on a step ladder: reference 2025-01-01T18Z with steps
    6..11h, which lands on the SAME valid hours (00..05Z) but reaches them via
    time + step. A loader that reads the reference time instead of the valid time
    puts the two fields on different axes -- the exact bug this guards.
    """
    path = tmp_path_factory.mktemp("grib") / "stepped.grib"
    nj, ni = 11, 21
    vals = np.arange(nj * ni, dtype=float).reshape(nj, ni)

    with open(path, "wb") as fh:
        for hour in range(6):  # instantaneous: valid 2025-01-02 00..05Z
            _write(fh, paramid=167, values=vals + hour, lat0=60, lat1=50, lon0=-5, lon1=5,
                   date=20250102, hour=hour)
        for k, step in enumerate(range(6, 12)):  # accumulation: 18Z + 6..11h -> 00..05Z
            _write(fh, paramid=169, values=vals * 1000 + k, lat0=60, lat1=50, lon0=-5, lon1=5,
                   date=20250101, hour=18, step=step)
    return path


@pytest.fixture(scope="session")
def single_frame_grib(tmp_path_factory) -> Path:
    """One valid time only, with an accumulation carrying a scalar step.

    A blanket squeeze() eats the length-1 time axis and the scalar step coord,
    which silently strands the accumulation on its reference hour.
    """
    path = tmp_path_factory.mktemp("grib") / "one.grib"
    nj, ni = 11, 21
    vals = np.arange(nj * ni, dtype=float).reshape(nj, ni)
    with open(path, "wb") as fh:
        _write(fh, paramid=167, values=vals, lat0=60, lat1=50, lon0=-5, lon1=5,
               date=20250102, hour=3)
        _write(fh, paramid=169, values=vals * 2, lat0=60, lat1=50, lon0=-5, lon1=5,
               date=20250101, hour=18, step=9)  # 18Z + 9h = 2025-01-02T03Z
    return path
