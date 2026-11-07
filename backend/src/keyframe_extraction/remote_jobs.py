"""V3C SFTP / LHE Drive download, ZIP, upload and isolated cleanup."""
import csv
from pathlib import Path
import shutil
import zipfile
import subprocess
from concurrent.futures import ThreadPoolExecutor

V3C_LIMITS = {'V3C1': (1, 7475), 'V3C2': (7476, 17235), 'V3C3': (17236, 28450)}
VIDEO_EXTENSIONS = frozenset({
    '.mp4', '.mov', '.mkv', '.avi', '.webm', '.wmv', '.asf', '.m4v',
    '.mpg', '.mpeg', '.mpe', '.mpv', '.m1v', '.m2v', '.ts', '.mts',
    '.m2ts', '.m2t', '.mxf', '.vob', '.flv', '.f4v', '.3gp', '.3g2',
    '.ogv', '.ogg', '.rm', '.rmvb', '.dv', '.divx', '.qt', '.nut', '.nsv', '.amv',
})
VIDEO_EXTENSION_PRIORITY = {ext: rank for rank, ext in enumerate(('.mp4', '.mov', '.mkv', '.webm', '.avi'))}


def select_sftp_jobs(rclone, config_path, remote, collections, ranges, id_lists=None,
                     video_subdir='videos', completed=(), command=None):
    """V3C SFTP layout: collection/videos/00001/00001.<video extension>.

    List in chunks of 200, fall back to individual folders, prefer MP4 then MOV.
    Extensions discover candidates only; ffprobe/NVDEC check the actual codec.
    Explicit ID lists override ranges; clamp selections to collection bounds.
    Only metadata is listed here; download one source video at a time later.
    """
    command = command or subprocess.run
    jobs = []
    if not remote or ':' in remote or video_subdir != 'videos':
        raise ValueError('Invalid SFTP remote or video subdirectory')

    def listing(*args):
        return command([str(rclone), '--config', str(config_path), 'lsf', *args],
                       capture_output=True, text=True, check=False)

    for collection in dict.fromkeys(collections):
        if collection not in V3C_LIMITS:
            raise ValueError(f'Unknown V3C collection: {collection}')
        lower, upper = V3C_LIMITS[collection]
        explicit = (id_lists or {}).get(collection)
        if explicit:
            wanted = sorted({int(n) for n in explicit if lower <= int(n) <= upper})
        else:
            first, last = map(int, ranges[collection])
            if first > last:
                raise ValueError(f'Invalid range: {collection}')
            wanted = range(max(first, lower), min(last, upper) + 1)
        ids = [f'{n:05d}' for n in wanted if f'{collection}/{n:05d}' not in completed]
        root = f'{collection}/{video_subdir}'
        by_id = {}
        for start in range(0, len(ids), 200):
            chunk = ids[start:start + 200]
            include = '/{' + ','.join(chunk) + '}/*' if len(chunk) > 1 else f'/{chunk[0]}/*'
            result = listing('-R', '--files-only', '--include', include, f'{remote}:{root}/')
            if result.returncode:
                print(f'{collection}: recursive listing failed; trying individual folders')
                continue
            for entry in result.stdout.splitlines():
                path = Path(entry.strip())
                if len(path.parts) == 2 and path.parent.name in chunk:
                    by_id.setdefault(path.parent.name, []).append(path.name)
        for video_id in ids:
            if video_id not in by_id:
                result = listing('--files-only', f'{remote}:{root}/{video_id}/')
                if result.returncode:
                    raise RuntimeError(f'Cannot list SFTP folder {collection}/{video_id}: {result.stderr}')
                by_id[video_id] = [entry.strip() for entry in result.stdout.splitlines()]
            names = [name for name in by_id[video_id]
                     if Path(name).name == name and Path(name).stem == video_id
                     and Path(name).suffix.lower() in VIDEO_EXTENSIONS]
            if not names:
                print(f'{collection}/{video_id}: no matching video file; skipped. '
                      f'Files seen: {by_id[video_id][:10]}')
                continue
            filename = min(names, key=lambda n: (
                VIDEO_EXTENSION_PRIORITY.get(Path(n).suffix.lower(), len(VIDEO_EXTENSION_PRIORITY)),
                n.lower(), n))
            jobs.append(dict(source_name=filename, video_id=video_id, group='V3C',
                             collection=collection, source_kind='sftp',
                             remote_path=f'{root}/{video_id}/{filename}'))
    return jobs


def _valid_job(job):
    name, video_id = job['source_name'], job['video_id']
    if Path(name).name != name or Path(name).stem != video_id:
        return False
    if job['group'] == 'LHE':
        return name in set(range_names(1, 75)) and job.get('source_kind', 'drive') == 'drive'
    if job['group'] != 'V3C' or job.get('source_kind') != 'sftp':
        return False
    collection = job.get('collection')
    if collection not in V3C_LIMITS or len(video_id) != 5 or not video_id.isascii() or not video_id.isdigit():
        return False
    lower, upper = V3C_LIMITS[collection]
    return (lower <= int(video_id) <= upper and Path(name).suffix.lower() in VIDEO_EXTENSIONS
            and job.get('remote_path') == f'{collection}/videos/{video_id}/{name}')


def range_names(start_id, end_id):
    if (type(start_id) is not int or type(end_id) is not int
            or not 1 <= start_id <= end_id <= 75):
        raise ValueError('Range must satisfy 1 <= START_VIDEO_ID <= END_VIDEO_ID <= 75')
    return [f'LHE{index:02d}.mp4' for index in range(start_id, end_id + 1)]


def select_remote_jobs(entries, start_id, end_id):
    """Require every requested file exactly once before starting any downloads."""
    names = range_names(start_id, end_id)
    selected = {name: [] for name in names}
    for entry in entries:
        name = entry.get('Name')
        if not entry.get('IsDir', False) and name in selected:
            if entry.get('Path', name) != name:
                raise ValueError(f'Expected source files directly inside the Drive root: {name}')
            selected[name].append(entry)
    jobs = []
    for name in names:
        matches = selected[name]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {name} in source Drive folder; found {len(matches)}')
        size = matches[0].get('Size', 0)
        if type(size) is not int or size <= 0:
            raise ValueError(f'Invalid source video size: {name}')
        jobs.append(dict(source_name=name, video_id=Path(name).stem, group='LHE', source_bytes=size))
    return jobs


def cleanup_video(work_dir, jobs_root):
    """Delete only the isolated per-video workspace, never input/Drive/shared runtime."""
    work_dir, jobs_root = Path(work_dir), Path(jobs_root).resolve()
    resolved = work_dir.resolve()
    relative = resolved.relative_to(jobs_root)
    group, video_id = relative.parts if len(relative.parts) == 2 else ('', '')
    valid_id = (video_id in {Path(name).stem for name in range_names(1, 75)} if group == 'LHE'
                else group == 'V3C' and len(video_id) == 5 and video_id.isascii()
                and video_id.isdigit() and 1 <= int(video_id) <= 28450)
    if (work_dir.is_symlink() or work_dir.parent.is_symlink() or not valid_id
            or resolved.parent != jobs_root/group):
        raise ValueError(f'Refusing cleanup outside the per-video workspace: {resolved}')
    shutil.rmtree(resolved)


def download_remote_video(job, jobs_root, rclone, config_path, remote,
                          download_root_id, upload_root_id, run,
                          sftp_config=None, sftp_remote=None):
    """Download and validate one source; leave its isolated workspace in place."""
    is_sftp = job.get('source_kind') == 'sftp'
    if (not _valid_job(job) or not upload_root_id
            or (not is_sftp and (not download_root_id or download_root_id == upload_root_id))
            or (is_sftp and (not sftp_config or not sftp_remote or ':' in sftp_remote))):
        raise ValueError('Invalid source job or Drive folder configuration')
    remote = remote.rstrip(':')
    if not remote or ':' in remote:
        raise ValueError('Invalid rclone remote')
    jobs_root = Path(jobs_root).resolve()
    jobs_root.mkdir(parents=True, exist_ok=True)
    work = jobs_root/job['group']/job['video_id']
    if work.is_symlink() or work.parent.is_symlink() or work.resolve().parent != jobs_root/job['group']:
        raise ValueError('Video workspace must be inside jobs_root')
    work.mkdir(parents=True, exist_ok=True)
    video = work/job['source_name']
    # rclone checks transfers and raises through run() on failure; existing downloads can resume.
    if is_sftp:
        run([rclone, '--config', sftp_config, 'copyto', f"{sftp_remote}:{job['remote_path']}", video,
             '--retries', '3', '--multi-thread-streams', '8', '--multi-thread-cutoff', '32M'])
    else:
        run([rclone, '--config', config_path, '--drive-root-folder-id', download_root_id,
             '--checksum', '--retries', '3', 'copyto', f"{remote}:{job['source_name']}", video])
    if (not video.is_file() or video.stat().st_size == 0
            or (not is_sftp and video.stat().st_size != job['source_bytes'])):
        raise ValueError(f'Download missing or wrong size: {video}')
    return work, video


def extract_downloaded_video(job, prepared, config, extract):
    work, video = prepared
    output = work/'output'
    video_config = dict(config, video=str(video), group=job['group'], output_dir=str(output))
    extract(video_config, work/'config.json')


def process_video_batches(records, video_batch_size, jobs_root, config, rclone, config_path, remote,
                          download_root_id, upload_root_id, run, extract,
                          sftp_config=None, sftp_remote=None, upload_workers=4):
    """Download/extract N; upload parallel files in ordered folder phases.

    Yield each successfully uploaded/cleaned job immediately so the notebook can
    record partial batch completion. Failed/pending workspaces remain for retry.
    """
    if type(video_batch_size) is not int or video_batch_size < 1:
        raise ValueError('VIDEO_BATCH_SIZE must be a positive integer')
    if type(upload_workers) is not int or upload_workers < 1:
        raise ValueError('UPLOAD_WORKERS must be a positive integer')
    for start in range(0, len(records), video_batch_size):
        batch = records[start:start + video_batch_size]
        print(f'Video batch {start // video_batch_size + 1}: {len(batch)} videos', flush=True)
        prepared = []
        for job in batch:
            print(f'Download: {job["source_name"]}', flush=True)
            prepared.append(download_remote_video(job, jobs_root, rclone, config_path, remote,
                download_root_id, upload_root_id, run, sftp_config, sftp_remote))
        for job, workspace in zip(batch, prepared):
            print(f'Extract: {job["source_name"]}', flush=True)
            extract_downloaded_video(job, workspace, config, extract)
        results = upload_processed_batch(batch, prepared, rclone, config_path, remote,
                                         upload_root_id, run, upload_workers)
        # All four phases must succeed before deleting even the first workspace.
        for job, workspace, result in zip(batch, prepared, results):
            cleanup_video(workspace[0], jobs_root)
            yield job, result


def upload_processed_batch(batch, prepared, rclone, config_path, remote, root_folder_id, run, workers=4):
    """ZIPs -> scenes -> embeddings -> maps; parallel copyto within each phase.

    Wait for every transfer in a folder before starting the next one. Failure
    leaves all local workspaces intact; checksums/retries support rerunning.
    """
    remote = remote.rstrip(':')
    if not remote or ':' in remote or not root_folder_id:
        raise ValueError('Set a valid rclone remote and Drive root_folder_id')
    if type(workers) is not int or workers < 1 or len(batch) != len(prepared):
        raise ValueError('Invalid upload workers or batch workspaces')
    packages = []
    for job, (work, _) in zip(batch, prepared):
        archive, mapping, scenes, embeddings, count = package_keyframes(
            work/'output', work/'archives', job)
        packages.append(dict(keyframes=archive, scenes=scenes, embeddings=embeddings,
                             map_keyframes=mapping, count=count))
    results = [dict(video=job['source_name'], keyframes=pack['count'], destinations=[])
               for job, pack in zip(batch, packages)]
    common = [rclone, '--config', config_path, '--drive-root-folder-id', root_folder_id,
              '--checksum', '--retries', '3']
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for folder, suffix in [('keyframes', '.zip'), ('scenes', '.scenes.txt'),
                               ('embeddings', '.pkl'), ('map_keyframes', '.csv')]:
            print(f'Upload folder {folder}: {len(batch)} files, up to {workers} parallel', flush=True)
            futures = []
            for job, pack, result in zip(batch, packages, results):
                destination = f'{remote}:{folder}/{job["video_id"]}{suffix}'
                result['destinations'].append(destination)
                futures.append(pool.submit(run, [*common, 'copyto', pack[folder], destination]))
            errors = []
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise RuntimeError(f'Upload folder {folder} failed ({len(errors)} files); '
                                   'all local batch files retained') from errors[0]
    return results


def package_keyframes(output_dir, archive_dir, job):
    """Flat JPG ZIP per video plus a separate canonical map; reject partial outputs."""
    root = Path(output_dir)
    group, video_id = Path(job['group']), job['video_id']
    image_dir = root/'keyframes'/group/video_id
    mapping = root/'map_keyframes'/group/f'{video_id}.csv'
    with mapping.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        required = {'keyframe_id', 'video_id', 'frame_idx', 'timestamp_sec', 'fps'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Invalid keyframe map columns: {mapping}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'No keyframes selected: {mapping}')
    expected = [f'{index:06d}.jpg' for index in range(len(rows))]
    if ([int(row['keyframe_id']) for row in rows] != list(range(len(rows)))
            or any(row['video_id'] != video_id for row in rows)):
        raise ValueError(f'Keyframe IDs/video IDs do not match: {mapping}')
    frames = [int(row['frame_idx']) for row in rows]
    if frames != sorted(set(frames)) or frames[0] < 0:
        raise ValueError(f'Invalid source frame indexes: {mapping}')
    actual = sorted(path.name for path in image_dir.iterdir() if path.is_file())
    if actual != expected or any((image_dir/name).stat().st_size == 0 for name in expected):
        raise ValueError(f'Images do not match keyframe map: {image_dir}')
    archive = Path(archive_dir)/'keyframes'/group/f'{video_id}.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix('.zip.partial')
    try:
        # JPEG is already compressed. STORE avoids spending CPU recompressing it.
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as handle:
            for name in expected:
                handle.write(image_dir/name, arcname=name)
        with zipfile.ZipFile(temporary) as handle:
            if handle.namelist() != expected or handle.testzip() is not None:
                raise ValueError(f'Invalid keyframe ZIP: {temporary}')
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    scenes = root/'scenes'/group/f'{video_id}.scenes.txt'
    embeddings = root/'embeddings'/group/f'{video_id}.pkl'
    for path in [scenes, embeddings]:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing pipeline cache: {path}')
    return archive, mapping, scenes, embeddings, len(rows)
