"""Embed canonical VLM/V3C sources into the standalone vlm_keyframes notebook.

Run from backend: python scripts/keyframe_extraction/build_vlm_keyframes.py
The repository is the source of truth; no files are recovered from an old bundle.
"""
from pathlib import Path
import ast
import base64
import io
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    'configs/kf_extraction.yaml',
    'src/embedding_extraction/__init__.py',
    'src/embedding_extraction/models/__init__.py',
    'src/embedding_extraction/models/backends/base.py',
    'src/embedding_extraction/models/backends/perception_encoder.py',
    'src/embedding_extraction/models/encoder.py',
    'src/embedding_extraction/models/frame_loader.py',
    'src/embedding_extraction/models/gpu_encoder.py',
    'src/embedding_extraction/models/registry.py',
    'src/keyframe_extraction/__init__.py',
    'src/keyframe_extraction/model_cache.py',
    'src/keyframe_extraction/models/__init__.py',
    'src/keyframe_extraction/models/export.py',
    'src/keyframe_extraction/models/omni_detector.py',
    'src/keyframe_extraction/models/p3.py',
    'src/keyframe_extraction/models/quality.py',
    'src/keyframe_extraction/nvdec_input.py',
    'src/keyframe_extraction/pipeline/__init__.py',
    'src/keyframe_extraction/pipeline/extract_keyframes.py',
    'src/keyframe_extraction/remote_jobs.py',
    'src/keyframe_extraction/worker_session.py',
    'src/utils/config.py',
    'src/utils/device.py',
    'src/utils/frame_sampling.py',
    'src/utils/gpu_video.py',
    'src/utils/logger.py',
    'src/utils/omni_progress.py',
    'src/utils/seed.py',
    'src/utils/video_io.py',
)


def rebuild():
    path = ROOT / 'notebooks/vlm_keyframes.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']

    def find_cell(marker):
        matches = [cell for cell in cells if marker in ''.join(cell['source'])]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one notebook cell containing {marker!r}')
        return matches[0]

    files = {name: (ROOT / name).read_bytes() for name in SOURCES}
    files['vlm_worker.py'] = (ROOT / 'scripts/keyframe_extraction/vlm_worker.py').read_text(encoding='utf-8').encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, content)

    cell = find_cell('SOURCE_ZIP = ')
    source = re.sub(r"SOURCE_ZIP = '[^']+'",
                    "SOURCE_ZIP = '" + base64.b64encode(output.getvalue()).decode() + "'",
                    ''.join(cell['source']))
    cell['source'] = source.splitlines(keepends=True)

    cell = find_cell('BATCH_SOURCE = ')
    batch = (ROOT / 'src/keyframe_extraction/remote_jobs.py').read_text(encoding='utf-8')
    source = re.sub(r'BATCH_SOURCE = .*?\nexec\(BATCH_SOURCE\)',
                    lambda _: 'BATCH_SOURCE = ' + repr(batch) + '\nexec(BATCH_SOURCE)',
                    ''.join(cell['source']), flags=re.DOTALL)
    cell['source'] = source.splitlines(keepends=True)

    cell = find_cell('from src.keyframe_extraction.worker_session import VlmWorkerSession')
    cell['source'] = (ROOT / 'scripts/keyframe_extraction/vlm_run_cell.py').read_text(
        encoding='utf-8').splitlines(keepends=True)
    for cell in cells:
        ast.parse(''.join(cell['source']))
        cell['outputs'] = []
        cell['execution_count'] = None
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    rebuild()
