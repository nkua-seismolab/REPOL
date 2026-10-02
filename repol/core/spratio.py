"""S/P amplitude ratio measurement.

Adapted from https://github.com/mingzhaochina/DiTing-FOCALFLOW
"""

from __future__ import annotations

import logging

import numpy as np
from obspy import Stream, UTCDateTime

from repol.config import SPRatioConfig

logger = logging.getLogger(__name__)


def _amplitude_picker(
    trace: np.ndarray,
    pickindex: float,
    df: float,
    s_delay: float,
    p_window: float,
    s_window: float,
) -> tuple[float, float]:
    """Peak amplitudes in the P and S search windows of a summed trace."""
    p_seg = trace[int((pickindex - 1) * df) : int((pickindex + p_window) * df)]
    s_seg = trace[int((pickindex + s_delay) * df) : int((pickindex + s_delay + s_window) * df)]
    if len(p_seg) == 0 or len(s_seg) == 0:
        raise ValueError("Empty P or S amplitude window")
    return max(p_seg), max(s_seg)


def measure_sp_ratio(
    traces: Stream,
    pick_time: UTCDateTime,
    distance_km: float,
    config: SPRatioConfig,
    s_pick_time: UTCDateTime | None = None,
) -> float | None:
    """Measure the S/P amplitude ratio on the vector sum of three components.

    The S window is placed at the observed S arrival when provided (and
    enabled), otherwise at the theoretical S delay from vp/vs travel times.
    Returns None when the measurement is not possible.
    """
    s_delay = None
    if config.use_observed_s and s_pick_time is not None:
        s_delay = float(s_pick_time - pick_time)
        if s_delay <= config.min_sp_time:
            logger.warning("Observed S-P time too short (%.2f s); using theoretical delay", s_delay)
            s_delay = None
    if s_delay is None:
        s_delay = distance_km / config.vs - distance_km / config.vp

    starttime = pick_time - config.pre_pick
    endtime = starttime + 50.0 + distance_km / 8.0
    # make sure the S search window is fully covered
    min_end = pick_time + s_delay + config.s_window + 5.0
    endtime = max(endtime, min_end)

    tr = (
        traces.slice(starttime=starttime - 10, endtime=endtime + 10)
        .detrend("demean")
        .integrate()
        .detrend("linear")
        .slice(starttime=starttime, endtime=endtime)
    )
    if len(tr) < 3:
        logger.warning("Could not sum traces: fewer than 3 components available")
        return None

    df = tr[0].stats.sampling_rate
    # seconds of the pick relative to the filtered (cut) trace start
    pickindex = float(pick_time - starttime) - config.cut_time

    trfil = tr.filter("highpass", freq=config.highpass).slice(
        starttime=starttime + config.cut_time, endtime=endtime
    )
    try:
        sumtracefil = np.sqrt(trfil[0].data ** 2 + trfil[1].data ** 2 + trfil[2].data ** 2)
    except (IndexError, ValueError):
        logger.warning("Could not sum traces. Maybe one trace is missing or corrupt.")
        return None
    max_amp = max(sumtracefil)
    if max_amp == 0:
        logger.warning("Null summed trace; cannot measure S/P")
        return None
    sumtracefil = sumtracefil / max_amp * 100

    try:
        p, s = _amplitude_picker(
            sumtracefil, pickindex, df, s_delay, config.p_window, config.s_window
        )
    except ValueError as exc:
        logger.warning("Amplitude picking failed: %s", exc)
        return None
    if p == 0:
        logger.warning("Zero P amplitude; cannot measure S/P")
        return None
    return s / p
