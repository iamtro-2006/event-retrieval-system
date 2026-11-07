"""Resolve plan scope independently of retrieval strategy and API transport."""
from dataclasses import dataclass


MODES = ('semantic', 'temporal', 'asr', 'caption', 'ocr', 'hierarchical', 'similarity', 'color')
SCOPE_MODES = {
    'dataset': list(MODES), 'videos': list(MODES), 'pool_videos': list(MODES),
    'pool_frames': ['semantic', 'hierarchical'],
}


def completed_pool(session, identifier):
    for entry in (session or {}).get('plans', []):
        if entry['id'] == identifier and entry.get('status') == 'completed':
            if entry.get('dataset', session.get('dataset')) != session.get('dataset'):
                raise ValueError('Scope pool belongs to another dataset')
            return entry
    raise ValueError('Unknown completed pool in this session')


def pool_frame_keys(pool):
    """Sequence cards already contain mapped frames; never invent neighbors."""
    keys = set()
    for row in pool.get('records', []):
        frames = row['matched_sequence'] if 'matched_sequence' in row else [row]
        for frame in frames or []:
            keys.add((str(frame['video_id']), int(frame['keyframe_id'])))
    return frozenset(keys)


@dataclass(frozen=True)
class ResolvedScope:
    videos: frozenset[str] | None
    frames: frozenset[tuple[str, int]] | None = None

    @property
    def empty(self):
        return self.videos == frozenset() or self.frames == frozenset()

    def describe(self, plan):
        return {'scope': plan.scope, 'scope_pool_id': plan.scope_pool_id,
                'video_ids': sorted(self.videos) if self.videos is not None else None,
                'frame_count': len(self.frames) if self.frames is not None else None}


def resolve_scope(plan, session=None, outer_video_ids=None):
    if plan.mode not in SCOPE_MODES[plan.scope]:
        raise ValueError(f'{plan.mode} does not support scope={plan.scope}')
    # Empty legacy UI video filters mean no outer restriction. An empty
    # intersection below remains explicitly empty and must never become global.
    outer = frozenset(map(str, outer_video_ids)) if outer_video_ids else None
    videos, frames = outer, None
    if plan.scope == 'videos':
        videos = frozenset(plan.video_ids)
    elif plan.scope in ('pool_frames', 'pool_videos'):
        frames = pool_frame_keys(completed_pool(session, plan.scope_pool_id))
        videos = frozenset(video for video, _ in frames)
        if plan.scope == 'pool_videos':
            frames = None
    if outer is not None and videos is not None:
        videos &= outer
    if frames is not None and videos is not None:
        frames = frozenset(key for key in frames if key[0] in videos)
    return ResolvedScope(videos, frames)
