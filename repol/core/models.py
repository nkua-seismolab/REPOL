"""Boundary dataclasses exchanged between the client and core modules."""

from __future__ import annotations

from dataclasses import dataclass

from obspy import UTCDateTime


@dataclass
class PickTask:
    """Everything the core needs to process one P pick."""

    pick_id: str
    seed_id: str  # NET.STA.LOC.CHA
    pick_time: UTCDateTime
    hypo_distance_km: float | None = None
    s_pick_time: UTCDateTime | None = None


@dataclass
class PickResult:
    """Outcome of processing one pick. Codes: polarity U/D/x, onset I/E/x."""

    pick_id: str
    polarity: str | None = None
    onset: str | None = None
    sp_ratio: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
