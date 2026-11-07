"""Notebook processing cell: one resident GPU worker across all video batches."""
sys.path.insert(0, str(SOURCE_ROOT))
from src.keyframe_extraction.worker_session import VlmWorkerSession

worker_path = SOURCE_ROOT/'vlm_worker.py'
bars = {}


def show_worker_line(line):
    if re.search(r'\d+%\|', line):
        stage = line.split(':', 1)[0]
        payload = {'text/plain': line.rstrip()}
        if stage not in bars:
            bars[stage] = display(payload, raw=True, display_id=True)
        else:
            bars[stage].update(payload, raw=True)
    else:
        print(line, end='', flush=True)


def extract_video(config, config_path):
    bars.clear()
    session.extract(config, config_path)


UPLOAD_DONE = globals().get('UPLOAD_DONE', set())
pending = [job for job in jobs if (job['group'], job['video_id']) not in UPLOAD_DONE]
try:
    if pending:
        with VlmWorkerSession(
                [PYTHON, '-u', worker_path, '--serve', '--source-root', SOURCE_ROOT],
                on_line=show_worker_line) as session:
            for job, result in process_video_batches(
                    pending, VIDEO_BATCH_SIZE, JOBS_ROOT, CONFIG, rclone, RCLONE_CONFIG, RCLONE_REMOTE,
                    DOWNLOAD_ROOT_FOLDER_ID, UPLOAD_ROOT_FOLDER_ID, run, extract_video,
                    sftp_config=SFTP_CONFIG, sftp_remote=SFTP_REMOTE, upload_workers=UPLOAD_WORKERS):
                UPLOAD_DONE.add((job['group'], job['video_id']))
                print(f'Uploaded {job["video_id"]}: {result["keyframes"]} images; local job removed.', flush=True)
except Exception:
    print(f'Batch stopped; unfinished jobs retained under {JOBS_ROOT}. Re-run this cell to retry.', flush=True)
    raise
print(f'Completed {len(UPLOAD_DONE.intersection((job["group"], job["video_id"]) for job in jobs))}/{len(jobs)}')
