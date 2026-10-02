"""Tests for S/P ratio measurement on synthetic waveforms."""

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from repol.config import SPRatioConfig
from repol.core.spratio import measure_sp_ratio

SPS = 100.0
T0 = UTCDateTime(2026, 1, 1, 0, 0, 0)


def make_stream(p_offset, s_offset, duration=120.0, p_amp=1.0, s_amp=5.0, components="ZNE"):
    """3-component stream with sine bursts at the P and S offsets (seconds)."""
    n = int(duration * SPS)
    t = np.arange(n) / SPS
    rng = np.random.default_rng(1)
    traces = []
    for comp in components:
        data = rng.normal(scale=0.01, size=n)
        for offset, amp in ((p_offset, p_amp), (s_offset, s_amp)):
            idx = (t >= offset) & (t < offset + 1.0)
            data[idx] += amp * np.sin(2 * np.pi * 5 * (t[idx] - offset))
        tr = Trace(data=data)
        tr.stats.sampling_rate = SPS
        tr.stats.starttime = T0
        tr.stats.channel = f"HH{comp}"
        traces.append(tr)
    return Stream(traces)


def test_theoretical_s_window():
    config = SPRatioConfig()
    dist = 30.0  # km -> theoretical S delay = 30/3.5 - 30/6 = ~3.57 s
    p_offset = 20.0
    s_delay = dist / config.vs - dist / config.vp
    st = make_stream(p_offset, p_offset + s_delay)
    sp = measure_sp_ratio(st, T0 + p_offset, dist, config)
    assert sp is not None
    assert sp > 1.0


def test_observed_s_window():
    config = SPRatioConfig()
    p_offset = 20.0
    s_offset = 28.0  # far from the theoretical delay for dist=30 km
    st = make_stream(p_offset, s_offset)
    sp = measure_sp_ratio(st, T0 + p_offset, 30.0, config, s_pick_time=T0 + s_offset)
    assert sp is not None
    assert sp > 1.0


def test_observed_s_before_p_falls_back_to_theoretical():
    config = SPRatioConfig()
    dist = 30.0
    p_offset = 20.0
    s_delay = dist / config.vs - dist / config.vp
    st = make_stream(p_offset, p_offset + s_delay)
    # bogus observed S before P must not crash; theoretical delay still works
    sp = measure_sp_ratio(st, T0 + p_offset, dist, config, s_pick_time=T0 + p_offset - 5)
    assert sp is not None


def test_missing_component_returns_none():
    config = SPRatioConfig()
    st = make_stream(20.0, 25.0, components="ZN")
    assert measure_sp_ratio(st, T0 + 20.0, 30.0, config) is None


@pytest.mark.parametrize("use_observed", [True, False])
def test_use_observed_s_flag(use_observed):
    config = SPRatioConfig(use_observed_s=use_observed)
    p_offset = 20.0
    s_offset = 28.0
    st = make_stream(p_offset, s_offset)
    sp = measure_sp_ratio(st, T0 + p_offset, 30.0, config, s_pick_time=T0 + s_offset)
    assert sp is not None
