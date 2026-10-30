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
    """Legacy single-frame asymmetric fallback retained for compatibility."""
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


def choose_asr_fallback_frames(times, values, start, end):
    """Return nearest keyframes on both sides of an unmatched ASR segment."""
    import bisect
    left = bisect.bisect_right(times, start) - 1
    right = bisect.bisect_left(times, end)
    result = []
    if left >= 0:
        result.append((*values[left], 'previous', max(0., start - times[left])))
    if right < len(times) and right != left:
        result.append((*values[right], 'next', max(0., times[right] - end)))
    return result


def load_segment_files(root: Path, dataset: str, source: str):
    """Load segment records grouped by source JSON filename (without suffix)."""
    folder = root / ('captions' if source == 'caption' else 'asr') / dataset.upper()
    for path in sorted(folder.glob('*.json')):
        records = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(records, dict):
            records = records.get('segments', [])
        file_records = []
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
            file_records.append(dict(segment_id=identity, video_id=video, source=source, text=text,
                                     start_time=start, end_time=end,
                                     speed=('timelapse' if 'timelapse' in text.lower() else
                                            'slow_motion' if 'slow-motion' in text.lower() else 'normal')
                                     if source == 'caption' else 'unknown'))
        yield path.stem, file_records


def load_segments(root: Path, dataset: str, source: str):
    for _, records in load_segment_files(root, dataset, source):
        yield from records


def load_indexable_segments(root: Path, dataset: str, source: str):
    """Load index documents, omitting caption shots with no mapped keyframe."""
    records = list(load_segments(root, dataset, source))
    if source != 'caption' or not records:
        return records
    maps = SegmentMaps(root, dataset, source, records)
    return [record for record in records if maps.forward.get(record['segment_id'])]


def fingerprint(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_maps(root: Path, dataset: str, tau=5., right_weight=2.):
    """Map captions to in-shot frames plus nearest unassigned boundary frames.

    ASR segments map to one representative frame in every overlapping caption
    shot. Segments outside all shots map to their nearest surrounding frames.
    """
    if not math.isfinite(tau) or tau <= 0 or not math.isfinite(right_weight) or right_weight < 1:
        raise ValueError('Invalid fallback log parameters')
    import bisect
    target = root / 'map_data'
    frame_folder = root / 'map_keyframes' / dataset.upper()
    if not frame_folder.exists():
        raise FileNotFoundError(f'Missing keyframe maps for {dataset}: {frame_folder}')
    frames = {}
    for path in sorted(frame_folder.glob('*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        values = sorted((float(r['timestamp_sec']), str(r['keyframe_id'])) for r in rows)
        frames[path.stem] = ([v[0] for v in values], values)

    caption_records = list(load_segments(root, dataset, 'caption'))
    asr_records = list(load_segments(root, dataset, 'asr'))
    caption_groups = defaultdict(list)
    asr_groups = defaultdict(list)
    for segment in caption_records:
        caption_groups[segment['video_id']].append(segment)
    for segment in asr_records:
        asr_groups[segment['video_id']].append(segment)

    def write_source(source, records, groups, rows_by_video, policy):
        folder = target / f'map_{source}' / dataset.upper()
        folder.mkdir(parents=True, exist_ok=True)
        for video, segments in groups.items():
            if video not in frames:
                raise ValueError(f'Missing keyframe map for {dataset}/{video}')
            path = folder / f'{video}.csv'
            temporary = path.with_suffix('.csv.tmp')
            with temporary.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                for segment in segments:
                    matches = rows_by_video.get(segment['segment_id'], [])
                    if not matches:
                        matches = [(None, '', 'unmapped', '', '')]
                    for timestamp, key, relation, offset, cost in matches:
                        writer.writerow({k: segment[k] for k in ('segment_id', 'start_time', 'end_time')} |
                            {'keyframe_id': key, 'relation': relation,
                             'offset_seconds': offset, 'selection_cost': cost})
            temporary.replace(path)
        (folder / 'manifest.json').write_text(json.dumps({
            'source_hash': fingerprint(records), 'mapping_policy': policy,
        }), encoding='utf-8')
        return len(records)

    caption_rows = defaultdict(list)
    for video, segments in caption_groups.items():
        if video not in frames:
            raise ValueError(f'Missing keyframe map for {dataset}/{video}')
        times, values = frames[video]
        covered = set()
        for segment in segments:
            lo = bisect.bisect_left(times, segment['start_time'])
            hi = bisect.bisect_left(times, segment['end_time'])
            for timestamp, key in values[lo:hi]:
                caption_rows[segment['segment_id']].append((timestamp, key, 'inside', 0., 0.))
                covered.add(key)
        # Assign every frame outside all caption shots to the closest shot
        # boundary, preserving frames at their nearest caption in a gap.
        if segments and not values:
            raise ValueError(f'No keyframes available to represent caption shots for {dataset}/{video}')
        for timestamp, key in values:
            if key in covered:
                continue
            segment = min(segments, key=lambda item: min(
                abs(timestamp - item['start_time']), abs(timestamp - item['end_time'])))
            offset = min(abs(timestamp - segment['start_time']), abs(timestamp - segment['end_time']))
            relation = 'before' if timestamp < segment['start_time'] else 'after'
            caption_rows[segment['segment_id']].append((timestamp, key, relation, offset, offset))

        # Every shot must have a representative. First use the nearest frame
        # before and after the shot; at a collection boundary, fill from the
        # nearest remaining frame(s) so a shot is never left unmapped.
        for segment in segments:
            sid = segment['segment_id']
            if caption_rows[sid]:
                continue
            left = bisect.bisect_right(times, segment['start_time']) - 1
            right = bisect.bisect_left(times, segment['end_time'])
            candidates = []
            if left >= 0:
                candidates.append((*values[left], 'before', segment['start_time'] - times[left]))
            if right < len(values) and right != left:
                candidates.append((*values[right], 'after', times[right] - segment['end_time']))
            if len(candidates) < min(2, len(values)):
                remaining = sorted((i for i in range(len(values)) if i not in {left, right}),
                                  key=lambda i: min(abs(times[i]-segment['start_time']),
                                                    abs(times[i]-segment['end_time'])))
                for i in remaining[:min(2, len(values))-len(candidates)]:
                    relation = 'before' if times[i] < segment['start_time'] else 'after'
                    offset = min(abs(times[i]-segment['start_time']), abs(times[i]-segment['end_time']))
                    candidates.append((*values[i], relation, offset))
            for timestamp, key, relation, offset in candidates:
                caption_rows[sid].append((timestamp, key, relation, offset, offset))

    caption_count = write_source('caption', caption_records, caption_groups, caption_rows,
                                 'inside_plus_nearest_boundary_and_empty_shot_two_sided_v3')

    # ASR shot representatives are selected from the frames already assigned
    # to each caption shot. Use the frame nearest the overlap midpoint.
    caption_frames_by_video = defaultdict(list)
    for segment in caption_records:
        for timestamp, key, *_ in caption_rows[segment['segment_id']]:
            caption_frames_by_video[segment['video_id']].append((segment, timestamp, key))

    asr_rows = defaultdict(list)
    for video, segments in asr_groups.items():
        if video not in frames:
            raise ValueError(f'Missing keyframe map for {dataset}/{video}')
        times, values = frames[video]
        shots = caption_groups.get(video, [])
        for segment in segments:
            overlapping = [shot for shot in shots if
                           segment['start_time'] < shot['end_time'] and
                           segment['end_time'] > shot['start_time']]
            if overlapping:
                for shot in overlapping:
                    left = max(segment['start_time'], shot['start_time'])
                    right = min(segment['end_time'], shot['end_time'])
                    midpoint = (left + right) / 2
                    representatives = [item for item in caption_frames_by_video[video]
                                       if item[0]['segment_id'] == shot['segment_id']]
                    if representatives:
                        _, timestamp, key = min(representatives, key=lambda item: abs(item[1] - midpoint))
                        asr_rows[segment['segment_id']].append((timestamp, key, 'shot',
                                                                abs(timestamp - midpoint), 0.))
                continue
            for timestamp, key, relation, offset in choose_asr_fallback_frames(
                    times, values, segment['start_time'], segment['end_time']):
                asr_rows[segment['segment_id']].append((timestamp, key, relation, offset, offset))

    asr_count = write_source('asr', asr_records, asr_groups, asr_rows,
                             'shot_representative_or_two_sided_v2')
    return {'asr': asr_count, 'caption': caption_count}


class SegmentMaps:
    def __init__(self, root: Path, dataset: str, source: str, records):
        self.forward = defaultdict(list)
        self.reverse = defaultdict(list)
        self.relations = defaultdict(dict)
        folder = root / 'map_data' / f'map_{source}' / dataset.upper()
        manifest = json.loads((folder / 'manifest.json').read_text())
        expected_policy = ('shot_representative_or_two_sided_v2' if source == 'asr'
                           else 'inside_plus_nearest_boundary_and_empty_shot_two_sided_v3')
        if manifest.get('mapping_policy') != expected_policy:
            raise RuntimeError(f'{source} mapping policy changed; rebuild maps with scripts.build_segment_data --maps-only')
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
