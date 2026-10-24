"""Progress around upstream windows; no replacement of model or stitching logic."""


def inference_with_progress(detector, video, overlap=10):
    from omnishotcut import engine
    from tqdm import tqdm

    original = engine.split_videos
    bars = []

    def progress_windows(*args, **kwargs):
        windows = original(*args, **kwargs)
        bar = tqdm(windows, total=len(windows), desc='OmniShotCut', unit='window',
                   ncols=100, mininterval=1.0)
        bars.append(bar)
        return bar

    print('OmniShotCut: upstream FFmpeg decoding/resizing; window progress starts after decode.', flush=True)
    engine.split_videos = progress_windows
    try:
        return detector.inference(str(video), mode='default', overlap=overlap)
    finally:
        engine.split_videos = original
        for bar in bars:
            bar.close()
