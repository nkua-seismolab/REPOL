"""Tests for batched pick processing (no SDS archive or TF model required)."""

import numpy as np
from obspy import Stream, Trace, UTCDateTime

from repol.config import Config
from repol.core.models import PickTask
from repol.core.processor import PickProcessor
from repol.core.waveforms import WaveformError

SPS = 100.0
T0 = UTCDateTime(2026, 1, 1, 0, 0, 0)
PICK_TIME = T0 + 20.0


def make_stream(duration=60.0, components="ZNE"):
    n = int(duration * SPS)
    rng = np.random.default_rng(7)
    traces = []
    for comp in components:
        tr = Trace(data=rng.normal(size=n))
        tr.stats.sampling_rate = SPS
        tr.stats.starttime = T0
        tr.stats.channel = f"HH{comp}"
        traces.append(tr)
    return Stream(traces)


class _StubFetcher:
    """Returns a synthetic stream, or raises for seed ids marked bad."""

    def fetch(self, seed_id, pick_time):
        if "BAD" in seed_id:
            raise WaveformError(f"No waveform data for {seed_id}")
        return make_stream()


class _StubClassifier:
    n_samples = 128

    def predict_batch(self, model_inputs):
        return [("U", "I")] * model_inputs.shape[0]

    def predict(self, model_input):
        return self.predict_batch(model_input)[0]


def make_processor(config=None):
    processor = PickProcessor.__new__(PickProcessor)
    processor.config = config or Config()
    processor.fetcher = _StubFetcher()
    processor.classifier = _StubClassifier()
    return processor


def make_task(pick_id, station="STA", distance=30.0):
    return PickTask(
        pick_id=pick_id,
        seed_id=f"HL.{station}..HHZ",
        pick_time=PICK_TIME,
        hypo_distance_km=distance,
    )


def test_process_batch_mixed_success_and_failure():
    processor = make_processor()
    tasks = [make_task("p1"), make_task("p2", station="BAD"), make_task("p3")]
    results = processor.process_batch(tasks)

    assert set(results) == {"p1", "p2", "p3"}
    for pick_id in ("p1", "p3"):
        assert results[pick_id].ok
        assert results[pick_id].polarity == "U"
        assert results[pick_id].onset == "I"
    assert not results["p2"].ok
    assert "No waveform data" in results["p2"].error


def test_process_batch_empty():
    assert make_processor().process_batch([]) == {}


def test_process_batch_no_distance_skips_sp():
    processor = make_processor()
    results = processor.process_batch([make_task("p1", distance=None)])
    assert results["p1"].ok
    assert results["p1"].sp_ratio is None


def test_process_single_matches_batch():
    processor = make_processor()
    task = make_task("p1")
    single = processor.process(task)
    batch = processor.process_batch([task])["p1"]
    assert (single.polarity, single.onset) == (batch.polarity, batch.onset)
