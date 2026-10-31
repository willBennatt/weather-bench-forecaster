"""Forcing functions for the WB2 forecaster.

Forcings are inputs the model receives but never predicts. They are either known analytically at
any time (solar radiation, time of day/year) or fixed (land-sea mask, orography, latitude), so they
are available at every step of a rollout, however far into the future.
"""
import numpy as np
import torch

SOLAR_CONSTANT = 1361.0  # W/m^2
DAYS_IN_YEAR = 365.24  # days
AXIAL_TILT = 23.44  # degrees

FORCING_NAMES = [
    "toa_t", "toa_t+6h",  # solar radiation at the start and end of the step, divided by SOLAR_CONSTANT
    "sin_hour", "cos_hour", "sin_year", "cos_year",  # time of day (UTC) and time of year
    "land_sea_mask", "orography", "sin_lat",  # static fields
]


def day_of_year(time):
    """Fractional days since 1 Jan 00:00 UTC of the same year (0 at the start of the year)."""
    time = np.asarray(time, dtype="datetime64[s]")
    return (time - time.astype("datetime64[Y]")) / np.timedelta64(1, "D")


def hour_of_day(time):
    """Fractional UTC hour (0-24)."""
    time = np.asarray(time, dtype="datetime64[s]")
    return (time - time.astype("datetime64[D]")) / np.timedelta64(1, "h")


def solar_declination(day):
    """Solar declination in radians: the latitude where the sun is directly overhead at noon.

    The simple form δ = -23.44° cos(2π(doy+10)/365) assumes a circular orbit and is off by up to ~2°
    around the equinoxes. This version corrects for the elliptical orbit (eccentricity 0.0167,
    perihelion around 3 Jan): δ = asin(sin(-23.44°) cos(2π(doy+10)/365 + 2·0.0167 sin(2π(doy-2)/365))).
    """
    orbit_angle = 2 * np.pi * (day + 10) / DAYS_IN_YEAR + 2 * 0.0167 * np.sin(2 * np.pi * (day - 2) / DAYS_IN_YEAR)
    return np.arcsin(np.sin(np.radians(-AXIAL_TILT)) * np.cos(orbit_angle))


def equation_of_time(day):
    """Minutes the sun runs ahead of clock time, from Earth's tilt and elliptical orbit (up to ±16 min)."""
    b = 2 * np.pi * (day - 81) / 364
    return 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)


def toa(lat, lon, time):
    """Top-of-atmosphere incoming solar radiation in W/m^2, shape (T, Lat, Lon).

    Given by TOA = S₀ · (1 + 0.033 cos(2π·doy/365)) · max(0, cos zenith), where
    cos zenith = sin φ sin δ + cos φ cos δ cos h and the hour angle h is 0 at local solar noon.

    `lat` and `lon` are 1D arrays in degrees and `time` is a datetime64 array (or a single time).
    """
    time = np.atleast_1d(time)
    day = day_of_year(time)[:, None, None]  # (T, 1, 1) so it broadcasts over the grid
    solar_decl = solar_declination(day)
    # Local solar time: UTC + 1 hour per 15° east, corrected by the equation of time.
    solar_time = hour_of_day(time)[:, None, None] + np.asarray(lon)[None, None, :] / 15 + equation_of_time(day) / 60
    hour_angle = np.radians((solar_time - 12) * 15)  # degrees to radians, 0 at solar noon
    lat_rad = np.radians(np.asarray(lat))[None, :, None]  # (1, Lat, 1)
    cos_zenith = np.sin(lat_rad) * np.sin(solar_decl) + np.cos(lat_rad) * np.cos(solar_decl) * np.cos(hour_angle)
    cos_zenith = np.clip(cos_zenith, 0, None)  # no negative TOA: the sun is below the horizon
    return SOLAR_CONSTANT * (1 + 0.033 * np.cos(2 * np.pi * day / DAYS_IN_YEAR)) * cos_zenith


def time_encodings(time):
    """sin/cos of the UTC hour and of the time of year, shape (T, 4).

    Encoding a cycle as (sin, cos) puts 23:00 next to 00:00 and 31 Dec next to 1 Jan.
    """
    time = np.atleast_1d(time)
    hour = 2 * np.pi * hour_of_day(time) / 24
    year = 2 * np.pi * day_of_year(time) / DAYS_IN_YEAR
    return np.stack([np.sin(hour), np.cos(hour), np.sin(year), np.cos(year)], axis=1)


def static_fields(ds):
    """Land-sea mask, normalized orography and sin(latitude), shape (3, Lat, Lon)."""
    def grid(name):
        # WB2 stores (longitude, latitude); models expect (latitude, longitude).
        return ds[name].transpose("latitude", "longitude").values.astype(np.float64)

    orography = grid("geopotential_at_surface")
    orography = (orography - orography.mean()) / orography.std()
    sin_lat = np.broadcast_to(np.sin(np.radians(ds.latitude.values))[:, None], orography.shape)
    return np.stack([grid("land_sea_mask"), orography, sin_lat])


class Forcings:
    """All forcing channels (FORCING_NAMES) for the 6-hour step starting at each of `times`.

    Computed on demand rather than stored: for 1979-2015 a precomputed (T, 9, 32, 64) array would
    take ~4 GB, while computing one step takes microseconds. Indexing returns a (9, Lat, Lon) tensor,
    so an instance can be passed as `forcing_tensor` to OneStepDataset. `at(times)` computes a
    batch for any times, e.g. the valid times during a rollout.
    """

    names = FORCING_NAMES

    def __init__(self, times, ds, step_hours=6):
        self.times = np.asarray(times, dtype="datetime64[ns]")
        self.lat, self.lon = ds.latitude.values, ds.longitude.values
        self.step = np.timedelta64(step_hours, "h")
        self.static = static_fields(ds)

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        return torch.from_numpy(self.at(self.times[idx])[0])

    def at(self, times):
        """Forcings for steps starting at `times`, shape (T, 9, Lat, Lon), float32."""
        times = np.atleast_1d(times)
        shape = (len(times), len(self.lat), len(self.lon))
        channels = [
            toa(self.lat, self.lon, times) / SOLAR_CONSTANT,
            toa(self.lat, self.lon, times + self.step) / SOLAR_CONSTANT,
            *[np.broadcast_to(enc[:, None, None], shape) for enc in time_encodings(times).T],
            *[np.broadcast_to(field, shape) for field in self.static],
        ]
        return np.stack(channels, axis=1).astype(np.float32)
