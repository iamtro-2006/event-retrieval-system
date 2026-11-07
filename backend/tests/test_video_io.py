"""Source timelines, bounded NVDEC counts and conditional input transcoding."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.keyframe_extraction import nvdec_input as inputs
from src.utils import frame_sampling as sampling
from src.utils.gpu_video import CudaVideoReader


class NvdecInputTests(unittest.TestCase):
    def test_compatible_source_does_not_transcode(self):
        probe = Mock()
        with patch.object(inputs, 'probe_video_format', return_value=dict(codec_name='h264', pix_fmt='yuv420p')), \
                patch.object(inputs.subprocess, 'run') as run:
            source = Path('07497.mp4')
            self.assertEqual(inputs.prepare_nvdec_input(source, Path('jobs'), probe), source)
        probe.assert_called_once_with(source)
        run.assert_not_called()

    def test_known_codec_profile_constraints(self):
        for info in [dict(codec_name='dnxhd'), dict(codec_name='prores'),
                     dict(codec_name='h264', pix_fmt='yuv420p10le'),
                     dict(codec_name='h264', pix_fmt='yuv444p'),
                     dict(codec_name='hevc', pix_fmt='yuv422p10le'), dict(codec_name='av1')]:
            with self.subTest(info=info):
                self.assertIsNotNone(inputs.incompatibility(info, 7))
        self.assertIsNone(inputs.incompatibility(dict(codec_name='hevc', pix_fmt='yuv420p10le'), 7))

    def test_generic_errors_do_not_trigger_conversion(self):
        for message in ['CUDA out of memory', 'NVDEC decoded 9/10', 'returned error 300']:
            with self.subTest(message=message), \
                    patch.object(inputs, 'probe_video_format', return_value=dict(codec_name='h264', pix_fmt='yuv420p')), \
                    patch.object(inputs.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, message):
                    inputs.prepare_nvdec_input(Path('test.mp4'), Path('jobs'),
                                               Mock(side_effect=RuntimeError(message)))
                run.assert_not_called()

    def test_dnxhd_transcodes_preserves_id_and_reuses_valid_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = work/'08538.mov'
            source.write_bytes(b'original video')
            commands = []
            def encode(command, **kwargs):
                commands.append(command)
                Path(command[-1]).write_bytes(b'converted video')
            probe = Mock()
            with patch.object(inputs, 'probe_video_format', return_value=dict(codec_name='dnxhd', pix_fmt='yuv422p')), \
                    patch.object(inputs.subprocess, 'run', side_effect=encode), \
                    patch.object(inputs, 'validate_conversion') as validate:
                target = inputs.prepare_nvdec_input(source, work, probe)
                self.assertEqual(target, work/'transcoded/08538.mp4')
                self.assertTrue(target.is_file())
                self.assertEqual(source.read_bytes(), b'original video')
                self.assertEqual(len(commands), 1)
                self.assertIn('h264_nvenc', commands[0])
                self.assertIn('passthrough', commands[0])
                self.assertNotIn('-r', commands[0])
                self.assertNotIn('-hwaccel', commands[0])
                self.assertEqual(inputs.prepare_nvdec_input(source, work, probe), target)
                self.assertEqual(len(commands), 1)
                source.write_bytes(b'changed original source')
                inputs.prepare_nvdec_input(source, work, probe)
                self.assertEqual(len(commands), 2)
                self.assertEqual(validate.call_count, 3)

    def test_conversion_failure_leaves_source_and_cleans_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = work/'08538.mov'
            source.write_bytes(b'original')
            def encode(command, **kwargs):
                Path(command[-1]).write_bytes(b'partial')
                raise subprocess.CalledProcessError(1, command)
            with patch.object(inputs, 'probe_video_format', return_value=dict(codec_name='dnxhd')), \
                    patch.object(inputs.subprocess, 'run', side_effect=encode):
                with self.assertRaises(subprocess.CalledProcessError):
                    inputs.prepare_nvdec_input(source, work, Mock())
            self.assertEqual(source.read_bytes(), b'original')
            self.assertFalse((work/'transcoded/08538.partial.mp4').exists())
            self.assertFalse((work/'transcoded/08538.mp4').exists())

    def test_timeline_mismatch_rejected_and_valid_timestamps_accepted(self):
        for target in (np.array([0., .1]), np.array([0., .02, .066])):
            with patch.object(inputs, 'probe_video_timeline', side_effect=[
                    (np.array([0., .033, .066]), 30.), (target, 30.)]):
                with self.assertRaisesRegex(ValueError, 'changed display'):
                    inputs.validate_conversion(Path('source.mov'), Path('converted.mp4'))
        with patch.object(inputs, 'probe_video_timeline', side_effect=[
                (np.array([0., .033, .066]), 30.), (np.array([0., .033, .066]), 30.)]):
            inputs.validate_conversion(Path('source.mov'), Path('converted.mp4'))

    def test_explicit_unsupported_runtime_error_uses_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = work/'00100.mov'
            source.write_bytes(b'original')
            def encode(command, **kwargs): Path(command[-1]).write_bytes(b'converted')
            probe = Mock(side_effect=[RuntimeError('unsupported profile'), None])
            with patch.object(inputs, 'probe_video_format', return_value=dict(codec_name='mpeg4')), \
                    patch.object(inputs.subprocess, 'run', side_effect=encode), \
                    patch.object(inputs, 'validate_conversion'):
                self.assertEqual(inputs.prepare_nvdec_input(source, work, probe).stem, source.stem)
                self.assertEqual(probe.call_count, 2)



class DisplayTimelineTests(unittest.TestCase):
    def probe(self, packets, declared=None, video='test.mp4', stream=None, format_name=''):
        metadata = dict(streams=[dict(avg_frame_rate='30/1', nb_frames=str(
            len(packets) if declared is None else declared))])
        metadata['streams'][0].update(stream or {})
        metadata['format'] = dict(format_name=format_name)
        outputs = [SimpleNamespace(stdout=json.dumps(dict(packets=packets)), stderr='')]
        with patch.object(sampling.subprocess, 'check_output', return_value=json.dumps(metadata)), \
                patch.object(sampling.subprocess, 'run', side_effect=outputs) as run:
            size = Path(video).stat().st_size if Path(video).is_file() else 1
            result = sampling._probe_video_timeline.__wrapped__(str(video), size, 1)
        return result, run

    def test_regular_b_frame_pts_sorted_without_cpu_decode(self):
        result, run = self.probe([dict(pts_time=str(t), flags='___') for t in [0, .1, .033, .066]])
        np.testing.assert_allclose(result['timestamps'], [0, .033, .066, .1])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(result['discard_count'], 0)

    def test_discard_tail_demux_only_and_excluded_from_display_count(self):
        result, run = self.probe([dict(pts_time=str(t), flags=f) for t, f in
            [(0, 'K__'), (.1, '_D_'), (.033, '___'), (.066, '___')]])
        np.testing.assert_allclose(result['timestamps'], [0, .033, .066])
        self.assertEqual((result['packet_count'], result['discard_count']), (4, 1))
        self.assertEqual(run.call_count, 1)
        for call in run.call_args_list:
            self.assertNotIn('-show_frames', call.args[0])
            self.assertNotIn('-count_frames', call.args[0])

    def test_interior_discard_keeps_timestamp_gap_and_origin(self):
        result, _ = self.probe([dict(pts_time=str(t), flags=f) for t, f in
            [(10, 'K__'), (11, '_D_'), (12, '___')]])
        np.testing.assert_allclose(result['timestamps'], [0, 2])
        self.assertFalse(result['timestamps'].flags.writeable)

    def test_unexplained_packet_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Packet/frame counts differ'):
            self.probe([dict(pts_time='0', flags='K__')], declared=2)

    def test_corrupt_duplicate_and_missing_pts_rejected(self):
        cases = [[dict(pts_time='0', flags='__C')],
                 [dict(pts_time='0'), dict(pts_time='0')],
                 [dict(flags='___')]]
        for packets in cases:
            with self.subTest(packets=packets), self.assertRaises(ValueError):
                self.probe(packets)

    def test_duplicate_tail_requires_verified_avc_end_marker(self):
        stream = dict(codec_name='h264', is_avc='true', nal_length_size='4')
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / '00129.mp4'
            packets = [dict(pts_time=str(t), flags='___') for t in
                       [335.92, 336.04, 335.96, 336.00, 336.04]]
            packets[-1].update(size='5', pos='4')
            for marker in (b'\x0a', b'\x0b'):
                data = b'head\x00\x00\x00\x01' + marker
                video.write_bytes(data)
                result, run = self.probe(packets, video=video, stream=stream,
                                         format_name='mov,mp4,m4a,3gp,3g2,mj2')
                np.testing.assert_allclose(result['timestamps'], [0., .04, .08, .12])
                self.assertEqual(result['packet_count'], 5)
                self.assertEqual(result['discard_count'], 0)
                self.assertEqual(result['non_picture_count'], 1)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(video.read_bytes(), data)

            # Same size/PTS alone is not proof: a slice, unknown NAL, invalid
            # length or missing container/codec metadata must still be rejected.
            for data, overrides, fmt in [
                (b'\x00\x00\x00\x01\x01', {}, 'mp4'),
                (b'\x00\x00\x00\x01\x05', {}, 'mp4'),
                (b'\x00\x00\x00\x01\x00', {}, 'mp4'),
                (b'\x00\x00\x00\x02\x0a', {}, 'mp4'),
                (b'\x00\x00\x00\x01\x0a', {}, 'matroska'),
                (b'\x00\x00\x00\x01\x0a', dict(codec_name='hevc'), 'mp4'),
                (b'\x00\x00\x00\x01\x0a', dict(is_avc='false'), 'mp4'),
                (b'\x00\x00\x00\x01\x0a', dict(nal_length_size='0'), 'mp4'),
            ]:
                with self.subTest(data=data, overrides=overrides, fmt=fmt):
                    video.write_bytes(b'head' + data)
                    with self.assertRaisesRegex(ValueError, 'unique increasing'):
                        self.probe(packets, video=video, stream=stream | overrides, format_name=fmt)

    def test_end_marker_does_not_hide_another_duplicate_or_bad_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / 'test.mp4'
            video.write_bytes(b'\x00\x00\x00\x01\x0a')
            kwargs = dict(video=video, stream=dict(codec_name='h264', is_avc='true',
                                                  nal_length_size='4'), format_name='mp4')
            tail = dict(pts_time='1', size='5', pos='0')
            with self.assertRaisesRegex(ValueError, 'unique increasing'):
                self.probe([dict(pts_time='0'), dict(pts_time='0'), dict(pts_time='1'), tail], **kwargs)
            with self.assertRaisesRegex(ValueError, 'Packet/frame counts differ'):
                self.probe([dict(pts_time='0'), dict(pts_time='1'), tail], declared=4, **kwargs)
            # An interior packet is not covered by the narrow tail exception.
            with self.assertRaisesRegex(ValueError, 'unique increasing'):
                self.probe([dict(pts_time='0'), tail, dict(pts_time='1')], **kwargs)

    def reader(self, metadata, batch_lengths, timeline, backend='threaded', requests=None):
        class Decoder:
            def __init__(self, **kwargs): self.lengths = iter(batch_lengths)
            def get_stream_metadata(self): return SimpleNamespace(num_frames=metadata)
            def get_batch_frames(self, size):
                if requests is not None:
                    requests.append(size)
                length = next(self.lengths, 0)
                if backend == 'simple' and (length == 0 or size > length):
                    raise RuntimeError('End of stream reached while seeking')
                return [object()] * length
        class Batch:
            is_cuda = True
            dtype = torch.uint8
            ndim = 4
            shape = (1, 3, 8, 8)
            def __init__(self, items): self.items = items
            def __len__(self): return len(self.items)
        fake_nvc = SimpleNamespace(ThreadedDecoder=Decoder, SimpleDecoder=Decoder,
                                   OutputColorType=SimpleNamespace(RGBP=1))
        with patch.dict('sys.modules', PyNvVideoCodec=fake_nvc), \
                patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'current_device', return_value=0), \
                patch.object(torch.cuda, 'synchronize'), \
                patch.object(torch, 'from_dlpack', side_effect=lambda x: x), \
                patch.object(torch, 'stack', side_effect=Batch), \
                patch.object(sampling, 'video_timeline_info', return_value=timeline):
            with CudaVideoReader(Path('test.mp4'), 2, backend=backend) as reader:
                return reader.total, [(start, len(batch)) for start, batch in reader]

    def test_nvdec_allows_only_explained_discard_difference(self):
        timeline = dict(timestamps=np.array([0., 1., 2.]), packet_count=4, discard_count=1)
        self.assertEqual(self.reader(4, [2, 1], timeline), (3, [(0, 2), (2, 1)]))
        with self.assertRaisesRegex(RuntimeError, 'decoded 2/3'):
            self.reader(4, [2], timeline)
        with self.assertRaisesRegex(RuntimeError, 'decoded 4/3'):
            self.reader(4, [2, 2], timeline)
        with self.assertRaisesRegex(RuntimeError, 'disagrees'):
            self.reader(5, [2, 1], timeline)

    def test_nvdec_no_blanket_minus_one_without_discard(self):
        timeline = dict(timestamps=np.array([0., 1., 2., 3.]), packet_count=4, discard_count=0)
        with self.assertRaisesRegex(RuntimeError, 'decoded 3/4'):
            self.reader(4, [2, 1], timeline)

    def test_nvdec_end_marker_metadata_and_simple_final_batch(self):
        timeline = dict(timestamps=np.arange(8401), packet_count=8402,
                        discard_count=0, non_picture_count=1)
        requests = []
        total, batches = self.reader(8402, [2] * 4200 + [1], timeline,
                                     backend='simple', requests=requests)
        self.assertEqual(total, 8401)
        self.assertEqual(batches[-1], (8400, 1))
        self.assertEqual(sum(requests), 8401)
        self.assertEqual(requests[-1], 1)
        self.assertEqual(len(requests), 4201)
        self.assertEqual(self.reader(8402, [2] * 4200 + [1], timeline)[0], 8401)

    def test_nvdec_threaded_rejects_excess_after_expected_frames(self):
        timeline = dict(timestamps=np.arange(3), packet_count=4,
                        discard_count=0, non_picture_count=1)
        with self.assertRaisesRegex(RuntimeError, 'decoded 4/3'):
            self.reader(4, [2, 1, 1], timeline)
        with self.assertRaisesRegex(RuntimeError, 'decoded 2/3'):
            self.reader(4, [2], timeline)
        with self.assertRaisesRegex(RuntimeError, 'disagrees'):
            self.reader(5, [2, 1], timeline)



if __name__ == '__main__':
    unittest.main()
