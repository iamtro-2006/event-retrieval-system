"""Run GPU speed inference in a fresh process, without Decord/OmniShotCut imports."""
from __future__ import annotations

import faulthandler
import json
import logging
import sys
from pathlib import Path
from src.utils.pipeline_progress import StageProgress, verbose_logging

RESULT_PREFIX = "__SPEED_LABELS__="


def main():
    faulthandler.enable(all_threads=True)
    level = logging.INFO if verbose_logging() else logging.WARNING
    logging.basicConfig(level=level, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)
    logger = logging.getLogger("speed_worker")
    logger.setLevel(level)
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.handlers[:] = [handler]
    request = json.loads(sys.stdin.read())
    from .speed import ShotSpeedPredictor
    # Import-order regression guard: never load CPU video libraries in this worker.
    forbidden = [name for name in ("decord", "av", "cv2", "omnishotcut") if name in sys.modules]
    if forbidden:
        raise RuntimeError(f"Speed worker unexpectedly loaded video libraries: {forbidden}")
    logger.info("[SPEED-WORKER] Fresh process; no Decord/PyAV/OpenCV/OmniShotCut loaded")
    with StageProgress("SPEED model", 1, unit="model") as progress:
        predictor = ShotSpeedPredictor(request["config"], logger)
        progress.update()
    results = {}
    try:
        for item in request["videos"]:
            path = Path(item["path"])
            results[str(path)] = predictor.predict(path, item["windows"], item["fps"])
    finally:
        predictor.close()
    print(RESULT_PREFIX + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
