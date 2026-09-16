"""
main.py

PSD-HP (Papulosquamous Skin Disease Harmonization Protocol) entry point.

Running

    python main.py

from the PSD_HP/ project root is the only command required to execute
the framework. This script's only responsibilities are: load the
configuration, set up the top-level logger, instantiate the
harmonization pipeline, run it, and report a clear final status to
the console. All real logic lives in config/, utils/, modules/, and
pipeline/ — main.py deliberately contains no business logic of its
own, so it never needs to change as later PSD-HP phases add modules.
"""

from __future__ import annotations

import sys

from config.config import get_config
from utils.logger import get_module_logger
from pipeline.harmonization_pipeline import HarmonizationPipeline


def main() -> int:
    """Run the PSD-HP harmonization pipeline end-to-end.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 if an unhandled error
        occurred anywhere in the pipeline.
    """
    config = get_config()
    logger = get_module_logger("main", config.logs_dir, config.log_level)

    logger.info("PSD-HP (Papulosquamous Skin Disease Harmonization Protocol) starting")
    logger.info(f"Project root: {config.project_root}")
    logger.info(f"Configured datasets: {[d.name for d in config.datasets]}")
    logger.info(f"Target classes: {config.target_classes}")

    try:
        pipeline = HarmonizationPipeline(config)
        pipeline.run()
    except Exception as exc:  # noqa: BLE001 — top-level catch-all is intentional here
        logger.exception(f"PSD-HP pipeline terminated with an unhandled error: {exc}")
        return 1

    logger.info(
        "PSD-HP finished. See reports/dataset_validation_report.txt "
        "and logs/pipeline.log for details."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
