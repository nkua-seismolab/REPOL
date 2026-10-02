"""Per-pick and batched orchestration of the scientific pipeline."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from obspy import Stream

from repol.config import Config
from repol.core.models import PickResult, PickTask
from repol.core.polarity import PolarityClassifier, augment
from repol.core.spratio import measure_sp_ratio
from repol.core.waveforms import (
    WaveformFetcher,
    ensure_sampling_rate,
    model_window,
    preprocess,
    warm_up_processing,
)

logger = logging.getLogger(__name__)


class PickProcessor:
    """Runs waveform retrieval, polarity classification and S/P measurement.

    Errors are contained per pick: process() and process_batch() always
    return PickResults, carrying an error message instead of raising.
    """

    def __init__(self, config: Config):
        self.config = config
        self.fetcher = WaveformFetcher(config.sds.archive, config.waveform)
        self.classifier = PolarityClassifier(
            config.model.path,
            config.waveform.target_sampling_rate,
            config.waveform.model_half_window,
        )
        warm_up_processing()
        # first predict builds the TF graph; do it here, not in the event path
        self.classifier.predict_batch(np.zeros((1, self.classifier.n_samples, 2)))
        logger.info("Processing pipeline warmed up")

    def process(self, task: PickTask) -> PickResult:
        try:
            model_input, st_all = self._prepare(task)
            polarity, onset = self.classifier.predict(model_input)
        except Exception as exc:
            logger.warning("Pick %s failed: %s", task.pick_id, exc)
            return PickResult(pick_id=task.pick_id, error=str(exc))
        return PickResult(
            pick_id=task.pick_id,
            polarity=polarity,
            onset=onset,
            sp_ratio=self._measure_sp(task, st_all),
        )

    def process_batch(self, tasks: list[PickTask]) -> dict[str, PickResult]:
        """Process picks in parallel: threaded fetch, one batched inference,
        threaded S/P. Runs entirely on core objects (thread-safe to call
        from a worker thread)."""
        results: dict[str, PickResult] = {}
        if not tasks:
            return results
        workers = self.config.processing.fetch_workers
        t0 = time.perf_counter()

        logger.info("Batch: fetching %d picks with %d workers", len(tasks), workers)
        prepared: list[tuple[PickTask, np.ndarray, Stream]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for task, outcome in zip(tasks, pool.map(self._try_prepare, tasks)):
                if isinstance(outcome, Exception):
                    logger.warning("Pick %s failed: %s", task.pick_id, outcome)
                    results[task.pick_id] = PickResult(pick_id=task.pick_id, error=str(outcome))
                else:
                    prepared.append((task, *outcome))
        logger.info(
            "Batch: fetched %d/%d picks in %.1fs",
            len(prepared),
            len(tasks),
            time.perf_counter() - t0,
        )

        if not prepared:
            return results

        t1 = time.perf_counter()
        batch = np.vstack([model_input for _, model_input, _ in prepared])
        classes = self.classifier.predict_batch(batch)
        logger.info("Batch: classified %d picks in %.1fs", len(prepared), time.perf_counter() - t1)

        t2 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            sp_ratios = list(pool.map(lambda item: self._measure_sp(item[0], item[2]), prepared))
        logger.info("Batch: measured S/P in %.1fs", time.perf_counter() - t2)

        for (task, _, _), (polarity, onset), sp_ratio in zip(prepared, classes, sp_ratios):
            results[task.pick_id] = PickResult(
                pick_id=task.pick_id,
                polarity=polarity,
                onset=onset,
                sp_ratio=sp_ratio,
            )
        return results

    def _try_prepare(self, task: PickTask):
        try:
            return self._prepare(task)
        except Exception as exc:
            return exc

    def _prepare(self, task: PickTask) -> tuple[np.ndarray, Stream]:
        """Fetch and preprocess one pick; returns (model input, raw stream)."""
        cfg = self.config.waveform
        st_all = self.fetcher.fetch(task.seed_id, task.pick_time)

        # copy so preprocessing does not alter the raw data used for S/P
        st_z = st_all.select(component="Z").copy()
        if not st_z:
            raise ValueError(f"No vertical component for {task.seed_id}")

        ensure_sampling_rate(st_z, cfg.target_sampling_rate, cfg.decimate)
        preprocess(st_z, cfg.filter)
        window = model_window(
            st_z, task.pick_time, cfg.model_half_window, self.classifier.n_samples
        )
        return augment(window, self.classifier.n_samples), st_all

    def _measure_sp(self, task: PickTask, st_all: Stream) -> float | None:
        if not self.config.spratio.enabled:
            return None
        if task.hypo_distance_km is None:
            logger.warning("Pick %s: no distance available, skipping S/P", task.pick_id)
            return None
        try:
            return measure_sp_ratio(
                st_all,
                task.pick_time,
                task.hypo_distance_km,
                self.config.spratio,
                s_pick_time=task.s_pick_time,
            )
        except Exception as exc:
            logger.warning("Pick %s: S/P measurement failed: %s", task.pick_id, exc)
            return None
