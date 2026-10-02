"""Conversion between SeisComP Pick objects and core boundary dataclasses.

obspy/numpy and the core models are imported lazily inside the functions
that need them, so the pure helpers do not require those dependencies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from seiscomp import datamodel

from repol import __version__ as repol_version

if TYPE_CHECKING:
    from obspy import UTCDateTime

    from repol.core.models import PickResult, PickTask

logger = logging.getLogger(__name__)

POLARITY_MAP = {
    "U": datamodel.POSITIVE,
    "D": datamodel.NEGATIVE,
    "x": datamodel.UNDECIDABLE,
}

ONSET_MAP = {
    "I": datamodel.IMPULSIVE,
    "E": datamodel.EMERGENT,
    "x": datamodel.UNDECIDABLE,
}


def seed_id(pick) -> str:
    wfid = pick.waveformID()
    return f"{wfid.networkCode()}.{wfid.stationCode()}.{wfid.locationCode()}.{wfid.channelCode()}"


def pick_time(pick) -> UTCDateTime:
    from obspy import UTCDateTime

    return UTCDateTime(pick.time().value().toString("%Y-%m-%dT%H:%M:%S.%f"))


def has_polarity(pick) -> bool:
    try:
        pick.polarity()
        return True
    except Exception:
        return False


def arrival_index(origin) -> dict:
    """Map pickID -> Arrival for all arrivals of an origin."""
    return {origin.arrival(i).pickID(): origin.arrival(i) for i in range(origin.arrivalCount())}


def _arrival_phase(arrival) -> str:
    try:
        return arrival.phase().code()
    except Exception:
        return ""


def hypo_distance_km(origin, arrival) -> float | None:
    import numpy as np
    from obspy.geodetics.base import degrees2kilometers

    try:
        epi_km = degrees2kilometers(arrival.distance())
        depth_km = origin.depth().value()
    except Exception:
        return None
    return float(np.sqrt(epi_km**2 + depth_km**2))


def observed_s_times(origin, picks) -> dict[str, UTCDateTime]:
    """Map NET.STA -> earliest observed S arrival time for an origin."""
    arrivals = arrival_index(origin)
    s_times: dict[str, UTCDateTime] = {}
    for pick in picks:
        arrival = arrivals.get(pick.publicID())
        if arrival is None or not _arrival_phase(arrival).upper().startswith("S"):
            continue
        wfid = pick.waveformID()
        station = f"{wfid.networkCode()}.{wfid.stationCode()}"
        t = pick_time(pick)
        if station not in s_times or t < s_times[station]:
            s_times[station] = t
    return s_times


def is_p_pick(pick, arrival) -> bool:
    """A pick is treated as P if its arrival phase (or phase hint) starts with P."""
    phase = _arrival_phase(arrival) if arrival is not None else ""
    if not phase:
        try:
            phase = pick.phaseHint().code()
        except Exception:
            phase = ""
    return phase.upper().startswith("P")


def build_task(pick, origin, s_times: dict[str, UTCDateTime], use_observed_s: bool) -> PickTask:
    """Build the core-facing task for one P pick."""
    from repol.core.models import PickTask

    arrivals = arrival_index(origin)
    arrival = arrivals.get(pick.publicID())
    distance = hypo_distance_km(origin, arrival) if arrival is not None else None

    wfid = pick.waveformID()
    station = f"{wfid.networkCode()}.{wfid.stationCode()}"
    s_time = s_times.get(station) if use_observed_s else None

    return PickTask(
        pick_id=pick.publicID(),
        seed_id=seed_id(pick),
        pick_time=pick_time(pick),
        hypo_distance_km=distance,
        s_pick_time=s_time,
    )


def apply_result(pick, result: PickResult) -> bool:
    """Write a PickResult onto a Pick. Returns True if the pick was modified."""
    if not result.ok:
        return False

    import json

    from obspy import UTCDateTime

    modified = False
    if result.polarity is not None:
        pick.setPolarity(POLARITY_MAP[result.polarity])
        modified = True
    if result.onset is not None:
        pick.setOnset(ONSET_MAP[result.onset])
        modified = True
    if modified:
        # First prepare the SP ratio value (if available)
        sp_ratio_value = f"{result.sp_ratio:.3f}" if result.sp_ratio is not None else None
        stored_at_time = UTCDateTime.utcnow().isoformat()
        comment_payload = {
            "provenance": {
                "software": "REPOL",
                "version": repol_version,
                "stored_at": stored_at_time,
            },
            "sp_ratio": sp_ratio_value,
        }
        comment_text = json.dumps(comment_payload, allow_nan=True, ensure_ascii=True)

        comment_id = f"{pick.publicID()}/comment/repol"
        for index in range(pick.commentCount()):
            comment = pick.comment(index)
            if comment.id() == comment_id:
                comment.setText(comment_text)
                comment.update()
                break
        else:
            comment = datamodel.Comment()
            comment.setText(comment_text)
            comment.setId(comment_id)
            pick.add(comment)

    return modified
