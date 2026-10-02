"""Worker-process entry points for the processing pipeline.

Application.run() holds the GIL almost continuously, starving any Python
thread in the same process. Processing therefore runs in a separate
process with its own GIL; only picklable PickTask/PickResult objects
cross the boundary. Heavy imports (TensorFlow, ObsPy) happen only here,
keeping the parent process light.
"""

from __future__ import annotations

import logging

from repol.config import Config
from repol.core.models import PickResult, PickTask

_processor = None


def init_worker(config: Config) -> None:
    """ProcessPoolExecutor initializer: build the pipeline in the child."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper(), logging.DEBUG),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from repol.core.processor import PickProcessor

    global _processor
    _processor = PickProcessor(config)


def run_batch(tasks: list[PickTask]) -> dict[str, PickResult]:
    return _processor.process_batch(tasks)
