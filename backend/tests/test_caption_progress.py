"""Exercise progress through a real subprocess pipe without CUDA/model downloads."""
import ast
import io
import json
import os
import re
import subprocess
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.utils.pipeline_progress import PROGRESS_PREFIX, StageProgress


class CaptionProgressTests(unittest.TestCase):
    def test_protocol_reports_real_counts_and_error_does_not_complete(self):
        output = io.StringIO()
        with patch.dict(os.environ, CAPTION_PROGRESS="1", CAPTION_PROGRESS_PROTOCOL="1", CAPTION_VERBOSE="0"), redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                with StageProgress("test", 5, video="001", unit="shot") as progress:
                    progress.update(2)
                    raise RuntimeError("failure")
        events = [json.loads(line[len(PROGRESS_PREFIX):]) for line in output.getvalue().splitlines()]
        self.assertEqual(events[0]["n"], 0)
        self.assertEqual(events[-1]["n"], 2)
        self.assertEqual(events[-1]["total"], 5)
        self.assertEqual(events[-1]["status"], "error")

    def runner(self):
        notebook = Path(__file__).resolve().parents[1] / "notebooks/extract_caption.ipynb"
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        source = next("".join(c["source"]) for c in cells if "".join(c["source"]).startswith("# 2."))
        function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "run")
        scope = dict(subprocess=subprocess, json=json, re=re)
        exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook-runner", "exec"), scope)
        return scope["run"]

    def run_child(self, fail=False):
        bars, messages = [], []
        class Bar:
            def __init__(self, **kwargs):
                self.n, self.desc, self.disable = kwargs.get("initial", 0), kwargs["desc"], False
                bars.append(self)
            def update(self, count): self.n += count
            def set_postfix_str(self, *args, **kwargs): pass
            def set_description(self, description): self.desc = description
            def refresh(self): pass
            def close(self): self.disable = True
            @staticmethod
            def write(message): messages.append(message)
        fake_tqdm = types.ModuleType("tqdm")
        fake_auto = types.ModuleType("tqdm.auto")
        fake_auto.tqdm = Bar
        code = (
            "from src.utils.pipeline_progress import StageProgress\n"
            "print('2026-10-29 01:00:00 INFO hidden chatter', flush=True)\n"
            "print('WARNING keep this warning', flush=True)\n"
            "with StageProgress('CAPTION', 5, video='001') as progress:\n"
            "    progress.update(2)\n"
            + ("    raise ValueError('visible failure')\n" if fail else "    progress.update(3)\n")
        )
        env = {**os.environ, "CAPTION_PROGRESS": "1", "CAPTION_PROGRESS_PROTOCOL": "1", "CAPTION_VERBOSE": "0"}
        runner = self.runner()
        with patch.dict(sys.modules, {"tqdm": fake_tqdm, "tqdm.auto": fake_auto}), redirect_stdout(io.StringIO()):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "visible failure"):
                    runner([sys.executable, "-u", "-c", code], env=env)
            else:
                runner([sys.executable, "-u", "-c", code], env=env)
        return bars, messages

    def test_notebook_pipe_redraws_one_bar_and_preserves_warning(self):
        bars, messages = self.run_child()
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].n, 5)
        self.assertTrue(bars[0].disable)
        self.assertTrue(any("keep this warning" in message for message in messages))
        self.assertFalse(any("hidden chatter" in message for message in messages))
        self.assertFalse(any(PROGRESS_PREFIX in message for message in messages))

    def test_notebook_pipe_error_keeps_traceback_and_incomplete_progress(self):
        bars, messages = self.run_child(fail=True)
        self.assertEqual(bars[0].n, 2)
        self.assertIn("FAILED", bars[0].desc)
        self.assertTrue(any("ValueError: visible failure" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
