"""Offline checks for V3C SFTP selection and isolated sequential jobs."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import threading

from src.keyframe_extraction import remote_jobs as jobs


class RemoteJobTests(unittest.TestCase):
    def test_video_batches_download_then_extract_then_upload(self):
        records = [dict(source_name=f'{i}.mp4', video_id=str(i)) for i in range(5)]
        events = []
        def download(job, *args):
            events.append(('download', job['video_id']))
            return ('work', 'video')
        def extract(job, *args): events.append(('extract', job['video_id']))
        def upload(batch, *args):
            for job in batch:
                events.append(('upload', job['video_id']))
            return [dict(keyframes=1) for job in batch]
        with patch.object(jobs, 'download_remote_video', side_effect=download), \
                patch.object(jobs, 'extract_downloaded_video', side_effect=extract), \
                patch.object(jobs, 'cleanup_video'), \
                patch.object(jobs, 'upload_processed_batch', side_effect=upload):
            results = list(jobs.process_video_batches(records, 2, 'jobs', {}, 'rclone', 'config',
                                                      'remote', 'input', 'output', None, None))
        expected = []
        for batch in (['0', '1'], ['2', '3'], ['4']):
            for phase in ('download', 'extract', 'upload'):
                expected.extend((phase, video_id) for video_id in batch)
        self.assertEqual(events, expected)
        self.assertEqual(len(results), 5)

    def test_batch_upload_failure_retains_entire_batch_and_stops(self):
        records = [dict(source_name=f'{i}.mp4', video_id=str(i)) for i in range(3)]
        with patch.object(jobs, 'download_remote_video', return_value=('work', 'video')) as download, \
                patch.object(jobs, 'extract_downloaded_video') as extract, \
                patch.object(jobs, 'cleanup_video') as cleanup, \
                patch.object(jobs, 'upload_processed_batch', side_effect=RuntimeError('offline')):
            iterator = jobs.process_video_batches(records, 2, 'jobs', {}, 'rclone', 'config',
                                                  'remote', 'input', 'output', None, None)
            with self.assertRaisesRegex(RuntimeError, 'offline'): next(iterator)
            cleanup.assert_not_called()
            self.assertEqual(download.call_count, 2)
            self.assertEqual(extract.call_count, 2)
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                list(jobs.process_video_batches([], invalid, 'jobs', {}, 'rclone', 'config',
                                                'remote', 'input', 'output', None, None))

    def test_parallel_upload_folder_barriers_and_failure_retention(self):
        batch = [dict(source_name=f'{i}.mp4', video_id=str(i)) for i in range(2)]
        prepared = [(Path(f'work{i}'), Path(f'{i}.mp4')) for i in range(2)]
        folders = ['keyframes', 'scenes', 'embeddings', 'map_keyframes']
        for fail in (None, 'scenes'):
            with self.subTest(failed_folder=fail):
                barrier = threading.Barrier(2, timeout=5)
                lock = threading.Lock()
                completed = {folder: 0 for folder in folders}
                started = []
                def upload(command):
                    folder = command[-1].split(':', 1)[1].split('/')[0]
                    with lock:
                        for previous in folders[:folders.index(folder)]:
                            self.assertEqual(completed[previous], 2)
                        started.append(folder)
                    # Both transfers must be running concurrently to pass.
                    barrier.wait()
                    if folder == fail:
                        raise RuntimeError('network failure')
                    with lock:
                        completed[folder] += 1
                with patch.object(jobs, 'package_keyframes', return_value=('zip', 'map', 'scene', 'emb', 1)), \
                        patch.object(jobs, 'cleanup_video') as cleanup:
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'all local batch files retained'):
                            jobs.upload_processed_batch(batch, prepared, 'rclone', 'config',
                                                        'remote', 'folder-id', upload, 2)
                        self.assertEqual(started, ['keyframes'] * 2 + ['scenes'] * 2)
                    else:
                        results = jobs.upload_processed_batch(batch, prepared, 'rclone', 'config',
                                                              'remote', 'folder-id', upload, 2)
                        self.assertEqual(started, [f for f in folders for _ in range(2)])
                        self.assertEqual(len(results), 2)
                    cleanup.assert_not_called()

    def test_collection_bounds_and_mp4_preference(self):
        calls = []
        def command(args, **kwargs):
            calls.append(args)
            return SimpleNamespace(returncode=0, stderr='', stdout=(
                '07476/07476.MOV\n07476/07476.MP4\n07477/07477.mov\n07477/unrelated.mp4\n'))
        selected = jobs.select_sftp_jobs('rclone', 'sftp.conf', 'stfp', ['V3C2'],
            {'V3C2': (7470, 7477)}, command=command)
        self.assertEqual([j['source_name'] for j in selected], ['07476.MP4', '07477.mov'])
        self.assertEqual(selected[1]['remote_path'], 'V3C2/videos/07477/07477.mov')
        self.assertEqual(selected[0]['group'], 'V3C')
        self.assertEqual(len(calls), 1)
        self.assertIn('/{07476,07477}/*', calls[0])

    def test_sftp_discovers_all_configured_video_extensions(self):
        extensions = sorted(jobs.VIDEO_EXTENSIONS)
        listing = '\n'.join(f'{i:05d}/{i:05d}{ext.upper()}'
                            for i, ext in enumerate(extensions, 1))
        selected = jobs.select_sftp_jobs('rclone', 'sftp.conf', 'stfp', ['V3C1'],
            {'V3C1': (1, len(extensions))}, command=lambda *a, **k:
            SimpleNamespace(returncode=0, stderr='', stdout=listing))
        self.assertEqual(len(selected), len(extensions))
        for job in selected:
            self.assertTrue(jobs._valid_job(job), job)

    def test_sftp_ignores_sidecars_and_selects_video_deterministically(self):
        listing = ('00215/00215.srt\n00215/00215.jpg\n00215/00215.AVI\n'
                   '00437/00437.webm\n00437/00437.mkv\n00774/00774.MXF\n')
        command = lambda *a, **k: SimpleNamespace(returncode=0, stderr='', stdout=listing)
        selected = jobs.select_sftp_jobs('rclone', 'sftp.conf', 'stfp', ['V3C1'], {},
            {'V3C1': [215, 437, 774]}, command=command)
        self.assertEqual([j['source_name'] for j in selected], ['00215.AVI', '00437.mkv', '00774.MXF'])

    def test_explicit_ids_override_range_and_folder_fallback(self):
        calls = []
        def command(args, **kwargs):
            calls.append(args)
            if '-R' in args:
                return SimpleNamespace(returncode=1, stdout='', stderr='listing failed')
            return SimpleNamespace(returncode=0, stdout='07615.MOV\n', stderr='')
        selected = jobs.select_sftp_jobs('rclone', 'sftp.conf', 'stfp', ['V3C2'],
            {'V3C2': (7476, 7485)}, {'V3C2': [7615, 1]}, command=command)
        self.assertEqual([j['video_id'] for j in selected], ['07615'])
        self.assertEqual(len(calls), 2)

    def test_failed_folder_listing_stops_instead_of_silently_skipping(self):
        with self.assertRaisesRegex(RuntimeError, 'Cannot list'):
            jobs.select_sftp_jobs('rclone', 'sftp.conf', 'stfp', ['V3C1'],
                {'V3C1': (1, 1)}, command=lambda *a, **k:
                SimpleNamespace(returncode=1, stdout='', stderr='offline'))

    def test_lhe_selection_remains_available(self):
        selected = jobs.select_remote_jobs([dict(Name='LHE01.mp4', Size=12)], 1, 1)
        self.assertEqual(selected[0]['group'], 'LHE')
        self.assertTrue(jobs._valid_job(selected[0]))

    def test_upload_paths_have_no_dataset_directory(self):
        for group, video_id in [('V3C', '07476'), ('LHE', 'LHE01')]:
            calls = []
            job = dict(group=group, video_id=video_id, source_name=video_id + '.mp4')
            with patch.object(jobs, 'package_keyframes', return_value=('zip', 'map', 'scene', 'emb', 1)):
                results = jobs.upload_processed_batch([job], [(Path('work'), None)],
                    'rclone', 'drive.conf', 'iamtro', 'folder-id', calls.append)
            destinations = results[0]['destinations']
            self.assertEqual(destinations, [f'iamtro:keyframes/{video_id}.zip',
                f'iamtro:scenes/{video_id}.scenes.txt', f'iamtro:embeddings/{video_id}.pkl',
                f'iamtro:map_keyframes/{video_id}.csv'])
            self.assertEqual([call[-1] for call in calls], destinations)
            self.assertEqual(calls[-1][-2], 'map')

    def test_sftp_job_download_config_upload_and_cleanup(self):
        job = dict(source_name='07476.mov', video_id='07476', group='V3C',
                   source_kind='sftp', collection='V3C2',
                   remote_path='V3C2/videos/07476/07476.mov')
        for fail in (False, True):
            with self.subTest(upload_fails=fail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)/'jobs'
                root.mkdir()
                sentinel = root/'preserve.txt'
                sentinel.write_text('keep')
                calls = []
                def run(args):
                    calls.append(args)
                    if '--drive-root-folder-id' in args:
                        if fail: raise RuntimeError('upload failed')
                        self.assertIn('output-id', args)
                    else:
                        Path(args[args.index('copyto') + 2]).write_bytes(b'video')
                def extract(config, path):
                    self.assertEqual(config['group'], 'V3C')
                    self.assertTrue(Path(config['video']).is_file())
                with patch.object(jobs, 'package_keyframes', return_value=('zip', 'map', 'scene', 'emb', 1)):
                    def process():
                        return list(jobs.process_video_batches([job], 1, root, {}, 'rclone', 'drive.conf',
                            'iamtro', 'input-id', 'output-id', run, extract,
                            sftp_config='sftp.conf', sftp_remote='stfp'))
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'Upload folder keyframes failed'): process()
                        self.assertTrue((root/'V3C/07476/07476.mov').exists())
                    else:
                        process()
                        self.assertFalse((root/'V3C/07476').exists())
                self.assertTrue(sentinel.exists())
                self.assertEqual(calls[0][1:3], ['--config', 'sftp.conf'])
                self.assertIn('stfp:V3C2/videos/07476/07476.mov', calls[0])
                self.assertNotIn('--drive-root-folder-id', calls[0])


if __name__ == '__main__':
    unittest.main()
