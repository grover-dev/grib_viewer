"""Regression tests for the GRIB layer.

Every test here corresponds to a bug that actually shipped and had to be fixed
on real ERA5 data. The docstrings say what went wrong, so a future change that
reintroduces one gets an explanation rather than just a red X.
"""

from __future__ import annotations

import numpy as np
import pytest

import grib_utils as gu


# ---------------------------------------------------------------------------
# Time: valid time vs reference time
# ---------------------------------------------------------------------------


def test_accumulation_lands_on_valid_time_not_reference_time(stepped_grib):
    """BUG: accumulations were read on their reference time.

    ssrd is stored as 2025-01-01T18Z + steps 6..11h; t2m is stored directly at
    2025-01-02T00..05Z. Both describe the same hours. Reading the reference time
    put ssrd on a separate, wrong axis (and inflated the timeline).
    """
    fields = gu.open_grib(str(stepped_grib))
    t2m_times = fields["t2m"]["time"].values
    ssrd_times = fields["ssrd"]["time"].values

    expected = np.array([np.datetime64(f"2025-01-02T{h:02d}:00") for h in range(6)])
    np.testing.assert_array_equal(t2m_times.astype("datetime64[m]"), expected)
    np.testing.assert_array_equal(ssrd_times.astype("datetime64[m]"), expected)

    # and therefore one shared timeline, not two
    assert len(gu.available_times(fields)) == 6


def test_single_frame_keeps_its_time_axis(single_frame_grib):
    """BUG: squeeze() ate the length-1 time axis and the scalar step coord.

    A one-frame file left the accumulation stranded on its reference hour (18Z)
    while the instantaneous field sat at 03Z -- so a 1-frame file reported 2.
    """
    fields = gu.open_grib(str(single_frame_grib))
    times = gu.available_times(fields)

    assert len(times) == 1, "a one-frame file must have exactly one frame"
    assert times[0].astype("datetime64[m]") == np.datetime64("2025-01-02T03:00")
    for name, da in fields.items():
        assert "time" in da.dims, f"{name} lost its time dimension"
        assert da.sizes["time"] == 1


def test_phantom_times_are_dropped(stepped_grib):
    """The xarray view of a step ladder must not invent times the file lacks.

    cfgrib presents (time x step) as a rectangle; folding it onto valid time can
    produce slots no message backs. The message headers are ground truth.
    """
    fields = gu.open_grib(str(stepped_grib))
    from_messages = set(gu.scan_times(str(stepped_grib)).astype("datetime64[m]").tolist())
    from_fields = set(gu.available_times(fields).astype("datetime64[m]").tolist())
    assert from_fields == from_messages


# ---------------------------------------------------------------------------
# Longitude frames
# ---------------------------------------------------------------------------


def test_global_grid_is_rolled_onto_180(global_grib):
    """A global 0..360 grid must be rolled, or the Atlantic is split in two."""
    fields = gu.open_grib(str(global_grib))
    lon = fields["u10"][gu.xdim_of(fields["u10"])].values

    assert lon.min() < 0, "global grid should be re-centred onto -180..180"
    assert lon.max() <= 180
    assert np.all(np.diff(lon) > 0), "longitudes must stay monotonic"


def test_rolling_moves_the_values_with_the_coordinate(global_grib):
    """BUG risk: rewriting the coordinate without rolling the data.

    u10 is seeded with each cell's own longitude, so a cell at -170 must hold the
    value 190 (its original 0..360 longitude). If the values weren't rolled with
    the coord, it would hold something else entirely.
    """
    fields = gu.open_grib(str(global_grib))
    u10 = fields["u10"].isel(time=0)

    for lon_180, want_360 in [(-170.0, 190.0), (-10.0, 350.0), (0.0, 0.0), (170.0, 170.0)]:
        got = float(u10.sel(latitude=0, longitude=lon_180, method="nearest"))
        assert got == pytest.approx(want_360), f"value at {lon_180} should be {want_360}"


def test_regional_grid_across_the_dateline_is_left_alone():
    """BUG: wrapping a Pacific window into -180..180 tore it in half.

    A 170..190 window is contiguous in its own frame. Wrapping and sorting it
    scatters it into two clumps with a hole between, and imshow (which assumes
    uniform spacing) then smears it across the whole map.
    """
    import xarray as xr

    lons = np.arange(170.0, 190.1, 2.5)
    da = xr.DataArray(
        np.zeros((1, 5, len(lons))),
        dims=("time", "latitude", "longitude"),
        coords={"time": [np.datetime64("2025-01-01")],
                "latitude": np.linspace(20, -20, 5), "longitude": lons},
    )
    out = gu.normalize_lons(da)
    got = out["longitude"].values

    np.testing.assert_allclose(got, lons)  # untouched
    spacing = np.diff(got)
    assert np.allclose(spacing, spacing[0]), "spacing must stay uniform (no hole)"


# ---------------------------------------------------------------------------
# Area cropping (message level)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bbox, expect_lons",
    [
        ((10, 40, -10, 10), [10, 20, 30, 40]),          # plain box, no seam
        ((-20, 20, -10, 10), [-20, -10, 0, 10, 20]),    # crosses Greenwich (0/360 seam)
        ((170, -170, -10, 10), [170, 180, 190]),        # crosses the dateline (W > E)
    ],
    ids=["plain", "greenwich", "dateline"],
)
def test_crop_handles_every_seam(global_grib, tmp_path, bbox, expect_lons):
    """BUG class: a 0..360 source cropped against a -180..180 box.

    An ordinary European box is ALREADY a seam crossing in the source. The values
    identify their own longitude, so a scrambled gather is detectable, not just a
    wrong extent.

    Note the frames differ by case: a Greenwich-crossing crop comes back on
    -20..20, while a dateline-crossing crop legitimately stays on 170..190 (see
    normalize_lons). So each cell is checked against the longitude it is *labelled
    with*, whatever frame that is -- the value must equal that cell's original
    0..360 longitude.
    """
    dst = tmp_path / "crop.grib"
    gu.write_subset(str(global_grib), str(dst), bbox=bbox)

    u10 = gu.open_grib(str(dst))["u10"].isel(time=0)
    lon = u10[gu.xdim_of(u10)].values

    assert np.all(np.diff(lon) > 0), "cropped longitudes must be monotonic"
    np.testing.assert_allclose(sorted(lon), sorted(expect_lons))

    row = u10.sel(latitude=0, method="nearest").values
    for x, value in zip(lon, row):
        assert value == pytest.approx(x % 360), f"column at {x} carries the wrong data"


def test_crop_narrows_the_grid(global_grib, tmp_path):
    dst = tmp_path / "small.grib"
    gu.write_subset(str(global_grib), str(dst), bbox=(0, 30, 0, 30))

    before = gu.open_grib(str(global_grib))["u10"]
    after = gu.open_grib(str(dst))["u10"]
    assert after.sizes["latitude"] < before.sizes["latitude"]
    assert after.sizes["longitude"] < before.sizes["longitude"]
    assert after.sizes["time"] == before.sizes["time"]  # time untouched


# ---------------------------------------------------------------------------
# Time subsetting (message level)
# ---------------------------------------------------------------------------


def test_write_subset_keeps_only_requested_times(global_grib, tmp_path):
    times = gu.scan_times(str(global_grib))
    keep = {times[1].item(), times[3].item()}
    dst = tmp_path / "t.grib"

    gu.write_subset(str(global_grib), str(dst), keep=set(times[[1, 3]].tolist()))

    got = gu.scan_times(str(dst))
    assert len(got) == 2
    np.testing.assert_array_equal(got, times[[1, 3]])


def test_write_subset_refuses_to_write_nothing(global_grib, tmp_path):
    with pytest.raises(SystemExit):
        gu.write_subset(str(global_grib), str(tmp_path / "x.grib"),
                        keep={np.datetime64("1999-01-01")})


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_coord_tag_formats_timedeltas_as_hours():
    """BUG: layers came out named 'cdir_step10800000000000 nanoseconds'."""
    assert gu.coord_tag("step", np.timedelta64(3, "h")) == "step3h"
    assert gu.coord_tag("isobaricInhPa", np.float64(850.0)) == "isobaricInhPa850"


def test_derived_wind_speed(global_grib):
    fields = gu.derive_fields(gu.open_grib(str(global_grib)))
    assert "ws10" in fields
    ws = fields["ws10"].isel(time=0)
    u = fields["u10"].isel(time=0)
    v = fields["v10"].isel(time=0)
    np.testing.assert_allclose(ws.values, np.hypot(u.values, v.values))


def test_open_grib_is_lazy(global_grib):
    """open_grib must not pull data into memory: a 16 GB file has to be openable.

    Reading the whole cube to describe the file is what OOM-killed grib_info.
    """
    fields = gu.open_grib(str(global_grib))
    assert all(hasattr(f.data, "dask") for f in fields.values()), \
        "fields should be dask-backed (a recipe), not materialized numpy"
