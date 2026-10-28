"""Progress around upstream windows; no replacement of model or stitching logic."""


def inference_with_progress(detector, video, overlap=10, *, mode='default', fps=None, logger=None):
    from omnishotcut import engine

    original = engine.split_videos
    original_decode = engine._decode_video
    bars = []

    def progress_windows(*args, **kwargs):
        windows = original(*args, **kwargs)
        if logger is not None:
            def logged_windows():
                for index, window in enumerate(windows, 1):
                    # Upstream tuple: chunk, padding, start, valid_start, valid_end, valid_len.
                    detail = ''
                    if fps and len(window) >= 6:
                        start, valid_len = int(window[2]), int(window[5])
                        detail = f' frames=[{start},{start+valid_len}) time=[{start/fps:.3f},{(start+valid_len)/fps:.3f})s'
                    logger.info('[OMNI] %s window=%d/%d START%s', video, index, len(windows), detail)
                    yield window
                    logger.info('[OMNI] %s window=%d/%d DONE%s', video, index, len(windows), detail)
            return logged_windows()
        from tqdm import tqdm
        bar = tqdm(windows, total=len(windows), desc='OmniShotCut', unit='window',
                   ncols=100, mininterval=1.0)
        bars.append(bar)
        return bar

    def progress_decode(*args, **kwargs):
        if logger is not None:
            logger.info('[OMNI] %s FFmpeg decode/resize START (window progress follows decoding)', video)
        frames = original_decode(*args, **kwargs)
        if logger is not None:
            logger.info('[OMNI] %s FFmpeg decode/resize DONE frames=%d', video, len(frames))
        return frames

    print('OmniShotCut: upstream FFmpeg decoding/resizing; window progress starts after decode.', flush=True)
    engine.split_videos = progress_windows
    engine._decode_video = progress_decode
    try:
        return detector.inference(str(video), mode=mode, overlap=overlap)
    finally:
        engine.split_videos = original
        engine._decode_video = original_decode
        for bar in bars:
            bar.close()
