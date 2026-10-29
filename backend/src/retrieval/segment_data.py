"""Offline CSV links and online O(1) bidirectional segment/frame lookups."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

FIELDS = ['segment_id', 'start_time', 'end_time', 'keyframe_id', 'relation', 'offset_seconds', 'selection_cost']


def choose_asr_fallback(times, values, start, end, tau=5., right_weight=2.):
    """Continuous log distance; a missing candidate has cost +infinity. Ties prefer left."""
    import bisect
    if not math.isfinite(tau) or tau <= 0 or not math.isfinite(right_weight) or right_weight < 1:
        raise ValueError('tau must be positive and right_weight must be at least one')
    left = bisect.bisect_left(times, start)-1
    right = bisect.bisect_left(times, end)
    left_cost = math.log1p((start-times[left])/tau) if left >= 0 else math.inf
    right_cost = right_weight*math.log1p((times[right]-end)/tau) if right < len(times) else math.inf
    if math.isinf(min(left_cost, right_cost)):
        return None
    if left_cost <= right_cost:
        return (*values[left], 'previous', left_cost)
    return (*values[right], 'next', right_cost)


def load_segments(root: Path, dataset: str, source: str):
    folder = root / ('captions' if source == 'caption' else 'asr') / dataset.upper()
    for path in sorted(folder.glob('*.json')):
        records = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(records, dict):
            records = records.get('segments', [])
        for number, record in enumerate(records):
            start = float(record.get('start', record.get('start_time', 0)))
            end = float(record.get('end', record.get('end_time', start)))
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                raise ValueError(f'Invalid interval: {path} row {number}')
            text = str(record.get('caption', record.get('transcript', record.get('text', ''))) or '').strip()
            if not text:
                continue
            video = str(record.get('video_id', path.stem))
            identity = f'{dataset.lower()}:{video}:{source}:{number:06d}'
            yield dict(segment_id=identity, video_id=video, source=source, text=text,
                       start_time=start, end_time=end,
                       speed=('timelapse' if 'timelapse' in text.lower() else
                              'slow_motion' if 'slow-motion' in text.lower() else 'normal')
                       if source == 'caption' else 'unknown')


def fingerprint(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_maps(root: Path, dataset: str, tau=5., right_weight=2.):
    """Use frames inside [start,end); ASR fallback minimizes asymmetric log distance."""
    if not math.isfinite(tau) or tau <= 0 or not math.isfinite(right_weight) or right_weight < 1:
        raise ValueError('Invalid fallback log parameters')
    import bisect
    import shutil
    target = root / 'map_data'
    frame_folder = target / 'map_keyframes' / dataset.upper()
    frame_folder.mkdir(parents=True, exist_ok=True)
    for old in (root / 'map_keyframes' / dataset.upper()).glob('*.csv'):
        if not (frame_folder / old.name).exists():
            shutil.copy2(old, frame_folder / old.name)
    frames = {}
    for path in sorted(frame_folder.glob('*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        values = sorted((float(r['timestamp_sec']), r['keyframe_id']) for r in rows)
        frames[path.stem] = ([v[0] for v in values], values)
    report = {}
    for source in ('asr', 'caption'):
        records = list(load_segments(root, dataset, source))
        groups = defaultdict(list)
        for segment in records:
            groups[segment['video_id']].append(segment)
        folder = target / f'map_{source}' / dataset.upper()
        folder.mkdir(parents=True, exist_ok=True)
        for video, segments in groups.items():
            if video not in frames:
                raise ValueError(f'Missing keyframe map for {dataset}/{video}')
            times, values = frames[video]
            path = folder / f'{video}.csv'
            temporary = path.with_suffix('.csv.tmp')
            with temporary.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                for segment in segments:
                    lo = bisect.bisect_left(times, segment['start_time'])
                    hi = bisect.bisect_left(times, segment['end_time'])
                    matches = [(timestamp, key, 'inside', 0.) for timestamp, key in values[lo:hi]]
                    if not matches and source == 'asr':
                        fallback = choose_asr_fallback(times, values, segment['start_time'], segment['end_time'], tau, right_weight)
                        if fallback is not None:
                            matches = [fallback]
                    if not matches:
                        matches = [(None, '', 'unmapped', '')]
                    for timestamp, key, relation, cost in matches:
                        writer.writerow({k: segment[k] for k in ('segment_id', 'start_time', 'end_time')} |
                            {'keyframe_id': key, 'relation': relation,
                             'offset_seconds': max(0., segment['start_time']-timestamp, timestamp-segment['end_time']) if timestamp is not None else '',
                             'selection_cost': cost})
            temporary.replace(path)
        # Commit manifest last; online loader rejects stale source data.
        (folder / 'manifest.json').write_text(json.dumps({'source_hash': fingerprint(records),
            'mapping_policy': 'inside_then_log_v1' if source == 'asr' else 'inside_v1',
            'fallback_tau_seconds': tau, 'fallback_right_weight': right_weight}), encoding='utf-8')
        report[source] = len(records)
    return report


class SegmentMaps:
    def __init__(self, root: Path, dataset: str, source: str, records):
        self.forward = defaultdict(list)
        self.reverse = defaultdict(list)
        self.relations = defaultdict(dict)
        folder = root / 'map_data' / f'map_{source}' / dataset.upper()
        manifest = json.loads((folder / 'manifest.json').read_text())
        if source == 'asr' and manifest.get('mapping_policy') != 'inside_then_log_v1':
            raise RuntimeError('ASR mapping policy changed; rebuild maps with scripts.build_segment_data --maps-only')
        if manifest['source_hash'] != fingerprint(records):
            raise RuntimeError(f'Stale {source} mapping; rebuild maps and segment indexes')
        known = {r['segment_id'] for r in records}
        for path in folder.glob('*.csv'):
            with path.open(encoding='utf-8-sig', newline='') as stream:
                for row in csv.DictReader(stream):
                    sid = row['segment_id']
                    if sid not in known:
                        continue
                    self.forward[sid]  # Preserve segments without frames.
                    if row['keyframe_id']:
                        key = (path.stem, int(row['keyframe_id']))
                        self.forward[sid].append(key)
                        self.reverse[key].append(sid)
                        self.relations[sid][key] = {'relation': row.get('relation', 'inside'),
                            'offset_seconds': float(row.get('offset_seconds') or 0)}
        if known != set(self.forward):
            raise RuntimeError(f'Incomplete {source} mapping')
