"""Wires the scclient listener to the core pick processor.

dispatch runs on the main thread (SeisComP objects), the processor's
process_batch runs on the listener's worker thread (core objects only),
and finalize runs on the main thread again.
"""

from __future__ import annotations

import logging

from seiscomp import datamodel

from repol.config import Config
from repol.scclient import OUTPUT_GROUP, pick_io

logger = logging.getLogger(__name__)


def make_dispatch(config: Config):
    """Build the main-thread callable that turns an event into PickTasks."""

    def dispatch(app, event, origin, picks):
        arrivals = pick_io.arrival_index(origin)
        s_times = pick_io.observed_s_times(origin, picks)

        # select the P picks to process
        targets = []
        for pick in picks:
            if not pick_io.is_p_pick(pick, arrivals.get(pick.publicID())):
                continue
            if pick_io.has_polarity(pick) and not config.seiscomp.reprocess:
                logger.debug("Pick %s already has a polarity, skipping", pick.publicID())
                continue
            targets.append(pick)
        logger.info("Processing %d of %d picks", len(targets), len(picks))

        tasks = [
            pick_io.build_task(pick, origin, s_times, config.spratio.use_observed_s)
            for pick in targets
        ]
        # the pick objects wait on the main thread until finalize
        return tasks, targets

    return dispatch


def make_finalize(config: Config):
    """Build the main-thread callable that writes results back to SeisComP."""

    def finalize(app, context, results):
        targets = context

        # attach picks to a parent before enabling the notifier so that
        # modifications are tracked as UPDATE operations
        ep = datamodel.EventParameters()
        for pick in targets:
            ep.add(pick)

        modified = 0
        datamodel.Notifier.Enable()
        try:
            for pick in targets:
                result = results.get(pick.publicID())
                if result is None:
                    continue
                if pick_io.apply_result(pick, result):
                    pick.update()
                    modified += 1
                    logger.info(
                        "Pick %s (%s): polarity=%s onset=%s sp=%s",
                        pick.publicID(),
                        pick_io.seed_id(pick),
                        result.polarity,
                        result.onset,
                        f"{result.sp_ratio:.4f}" if result.sp_ratio is not None else "N/A",
                    )
        finally:
            msg = datamodel.Notifier.GetMessage()
            datamodel.Notifier.Disable()

        if modified:
            app.send_message(OUTPUT_GROUP, msg)
        else:
            logger.info("No picks were modified, nothing to send")

    return finalize
