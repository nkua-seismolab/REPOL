"""SDS waveform retrieval, QC and preprocessing."""

from __future__ import annotations

import logging

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from obspy.clients.filesystem.sds import Client as SDSClient

from repol.config import WaveformConfig

logger = logging.getLogger(__name__)


def warm_up_processing() -> None:
    """Trigger ObsPy's lazy entry-point imports once, single-threaded.

    detrend/filter/taper/decimate import their backends on first use;
    concurrent first calls from worker threads deadlock on the import lock.
    """
    tr = Trace(data=np.random.default_rng(0).normal(size=256))
    tr.stats.sampling_rate = 100.0
    st = Stream([tr])
    st.detrend("linear").detrend("demean")
    st.taper(0.001, type="cosine")
    st.filter("bandpass", freqmin=1.0, freqmax=10.0, zerophase=True)
    st.filter("highpass", freq=1.0)
    st.decimate(factor=2, strict_length=False, no_filter=True)
    logger.debug("ObsPy processing backends warmed up")


class WaveformError(Exception):
    """Raised when waveform data are missing or unusable."""


class WaveformFetcher:
    """Retrieves all components of a pick's stream from an SDS archive."""

    def __init__(self, archive: str, config: WaveformConfig):
        self.client = SDSClient(archive)
        self.config = config
        logger.info("Initialized SDS client with archive: %s", archive)

    def fetch(self, seed_id: str, pick_time: UTCDateTime) -> Stream:
        net, sta, loc, cha = seed_id.split(".")
        st = self.client.get_waveforms(
            network=net,
            station=sta,
            location=loc,
            channel=cha[:-1] + "?",
            starttime=pick_time - self.config.t_before,
            endtime=pick_time + self.config.t_after,
        )
        if not st:
            raise WaveformError(f"No waveform data for {seed_id} at {pick_time}")
        return st


def ensure_sampling_rate(st: Stream, target: float, decimate: bool) -> Stream:
    """Bring a stream to the target sampling rate, or raise WaveformError."""
    sps = st[0].stats.sampling_rate
    if sps == target:
        return st
    if sps < target:
        raise WaveformError(f"Sampling rate {sps} Hz is below the target {target} Hz")
    if not decimate:
        raise WaveformError(
            f"Sampling rate {sps} Hz exceeds the target {target} Hz and decimation is disabled"
        )
    if sps % target:
        raise WaveformError(f"Sampling rate {sps} Hz is not an integer multiple of {target} Hz")
    st.decimate(factor=int(sps // target), strict_length=False, no_filter=True)
    return st


def preprocess(st: Stream, filt_bounds: list[float] | None = None) -> Stream:
    """Detrend, optionally bandpass, and taper a stream in place."""
    st.detrend("linear").detrend("demean")
    if filt_bounds:
        st.filter("bandpass", freqmin=filt_bounds[0], freqmax=filt_bounds[1], zerophase=True)
    st.taper(0.001, type="cosine")
    return st


def model_window(
    st: Stream, pick_time: UTCDateTime, half_window: float, n_samples: int = 128
) -> Stream:
    """Slice the classification window around the pick."""
    win = st.slice(pick_time - half_window, pick_time + half_window)
    if not win or len(win[0].data) < n_samples:
        raise WaveformError("Not enough samples around the pick for the model window")
    return win
