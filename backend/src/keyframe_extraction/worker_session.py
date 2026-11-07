"""A sequential persistent VLM subprocess; the notebook never imports GPU models."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

RESULT_PREFIX = '__VLM_RESULT__='


class VlmWorkerSession:
    def __init__(self, command, on_line=None):
        self.command = [str(x) for x in command]
        self.on_line = on_line or (lambda line: print(line, end='', flush=True))
        self.process = None

    def __enter__(self):
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1)
        return self

    def extract(self, config, config_path):
        config_path = Path(config_path)
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError('VLM worker is not running; rerun the processing cell')
        request = str(config_path)
        try:
            process.stdin.write(json.dumps(dict(config=request)) + '\n')
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError('VLM worker exited before accepting the video') from exc
        for line in process.stdout:
            if not line.startswith(RESULT_PREFIX):
                self.on_line(line)
                continue
            result = json.loads(line[len(RESULT_PREFIX):])
            if result.get('config') != request:
                raise RuntimeError('VLM worker acknowledged the wrong video')
            if not result.get('ok'):
                raise RuntimeError(f'VLM extraction failed: {result.get("error", "unknown error")}')
            return
        raise RuntimeError(f'VLM worker exited without completing {request} '
                           f'(exit code {process.wait()})')

    def close(self, abort=False):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if abort and process.poll() is None:
                process.terminate()
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass  # The child may already have crashed; still reap it below.
            try:
                code = process.wait(timeout=30 if not abort else 5)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
        finally:
            process.stdout.close()
        if code and not abort:
            raise RuntimeError(f'VLM worker exited with code {code}')

    def __exit__(self, exc_type, exc, tb):
        self.close(abort=exc_type is not None)
