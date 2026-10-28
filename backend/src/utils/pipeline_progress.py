"""Progress that survives stdout pipes without printing a new bar on every update."""
from __future__ import annotations

import json
import os
import sys
from time import monotonic

PROGRESS_PREFIX = "__CAPTION_PROGRESS__="


def verbose_logging() -> bool:
    return os.environ.get("CAPTION_VERBOSE", "0") == "1"


class StageProgress:
    def __init__(self, stage, total, *, video="", unit="step", initial=0):
        self.stage, self.total, self.video, self.unit = stage, total, str(video), unit
        self.n, self.postfix, self.last_emit = initial, "", 0.0
        self.disabled = os.environ.get("CAPTION_PROGRESS", "1") == "0" or verbose_logging()
        self.protocol = os.environ.get("CAPTION_PROGRESS_PROTOCOL") == "1"
        self.bar = None
        self.closed = False
        if not self.disabled and not self.protocol:
            from tqdm import tqdm
            self.bar = tqdm(total=total, initial=initial, desc=f"{stage} {video}".strip(),
                            unit=unit, dynamic_ncols=True, mininterval=.5, file=sys.stderr)
        self._emit("running", force=True)

    def _emit(self, status, force=False):
        if self.disabled or not self.protocol:
            return
        now = monotonic()
        if not force and now - self.last_emit < .5:
            return
        self.last_emit = now
        print(PROGRESS_PREFIX + json.dumps(dict(
            key=f"{self.stage}:{self.video}", stage=self.stage, video=self.video,
            total=self.total, n=self.n, unit=self.unit, postfix=self.postfix, status=status,
        )), flush=True)

    def update(self, count=1, postfix=None):
        self.n += count
        if postfix is not None:
            self.postfix = str(postfix)
        if self.bar is not None:
            if postfix is not None:
                self.bar.set_postfix_str(self.postfix, refresh=False)
            self.bar.update(count)
        self._emit("running")

    def close(self, failed=False):
        if not self.closed:
            self.closed = True
            self._emit("error" if failed else "done", force=True)
            if self.bar is not None:
                self.bar.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        self.close(failed=kind is not None)


def write_caption(message):
    from tqdm import tqdm
    tqdm.write(message, file=sys.stdout)
